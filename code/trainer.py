# coding=utf-8
from __future__ import print_function

import os
import time
import numpy as np
import sys
import shutil

from PIL import Image
from ipdb import set_trace
from six.moves import range
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data
from torch.backends import cudnn
import torchvision
from torchvision import transforms, utils


from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.losses import words_loss, sent_loss, KL_loss
from miscc.losses import d_logistic_loss, d_r1_loss
from miscc.losses import g_nonsaturating_loss, g_path_regularize

from datasets import TextDataset, prepare_data
from model_base import RNN_ENCODER, CNN_ENCODER
from model import Generator as G_STYLE
from model import Discriminator as D_NET
from model import D_NET256
from distributed import (
	get_rank,
	reduce_loss_dict,
	reduce_sum,
	get_world_size,
	cleanup_distributed, 
)
from inception import InceptionV3
from fid import calculate_frechet_distance


# ################# Text to image task############################ #
class condGANTrainer(object):
	def __init__(self, output_dir, args):
		
		if cfg.TRAIN.FLAG:
			self.out_dir = output_dir
			self.model_dir = os.path.join(output_dir, 'Model')
			self.image_dir = os.path.join(output_dir, 'Image')
			self.log_dir = output_dir
			mkdir_p(self.model_dir)
			mkdir_p(self.image_dir)
			shutil.copy(args.cfg_file, output_dir)
			split_dir, bshuffle = 'train', True
		else:
			split_dir, bshuffle = 'test', False
		
		self.args = args
		self.batch_size = cfg.TRAIN.BATCH_SIZE
		self.max_epoch = cfg.TRAIN.MAX_EPOCH   # 800
		self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL
		self.img_size = cfg.TREE.BASE_SIZE

		self.data_set = TextDataset(
			cfg.DATA_DIR, 
			split_dir,
			base_size=self.img_size,
		)
		self.data_sampler = self.data_sampler(
			self.data_set, 
			shuffle=bshuffle, 
			distributed=args.distributed
		)
		self.data_loader = data.DataLoader(
			self.data_set, 
			batch_size=self.batch_size,
			sampler=self.data_sampler,
			drop_last=True, 
		)

		self.n_words = self.data_set.n_words
		self.ixtoword = self.data_set.ixtoword
		self.pretrained_emb = self.data_set.pretrained_emb
		self.num_batches = len(self.data_loader)

		self.path_batch_shrink = cfg.TRAIN.PATH_BATCH_SHRINK
		self.path_batch = max(1, self.batch_size // self.path_batch_shrink)
		
		if cfg.TRAIN.FLAG:
			self.path_loader = data.DataLoader(
				self.data_set, 
				batch_size=self.path_batch,
				sampler=self.data_sampler,
				drop_last=True, 
			)

			self.val_set = TextDataset(
				cfg.DATA_DIR, 
				'test',
				base_size=self.img_size,
			)
			self.val_loader = data.DataLoader(
				self.val_set, 
				batch_size=self.batch_size,
				drop_last=True, 
				shuffle=True, 
				num_workers=int(cfg.WORKERS)
			)
			
			f = np.load(cfg.MU_SIG)
			self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
			f.close()

			f = np.load(cfg.MU_SIG.replace('train', 'val'))
			self.mu_val, self.sigma_val = f['mu'][:], f['sigma'][:]
			f.close()
		

	def data_sampler(self, dataset, shuffle, distributed):
		if distributed:
			return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

		elif shuffle:
			return data.RandomSampler(dataset)

		else:
			return data.SequentialSampler(dataset)


	def sample_data(self, loader):
		while True:
			for batch in loader:
				yield batch


	def requires_grad(self, model, flag=True):
		for p in model.parameters():
			p.requires_grad = flag


	def accumulate(self, model1, model2, decay=0.999):
		par1 = dict(model1.named_parameters())
		par2 = dict(model2.named_parameters())

		for k in par1.keys():
			par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


	def build_models(self):
		def count_parameters(model):
			total_param = 0
			for name, param in model.named_parameters():
				if param.requires_grad:
					num_param = np.prod(param.size())
					# if param.dim() > 1:
					# 	print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
					# else:
					# 	print(name, ':', num_param)
					total_param += num_param
			return total_param

		device = self.args.device
		
		if cfg.TRAIN.NET_E == '' and get_rank() == 0:
			print('Error: no pretrained text-image encoders')
			return

		# -------------------------------------------
		# text encoder
		self.textEnc = RNN_ENCODER(
			self.n_words,
			nhidden=cfg.TEXT.EMBEDDING_DIM,
			pre_emb=self.pretrained_emb,
			ninput=768
		).to(device)
		
		state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
		self.textEnc.load_state_dict(state_dict)
		self.requires_grad(self.textEnc, False)
		self.textEnc.eval() 

		# -------------------------------------------
		# image encoder
		img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
		self.imageEnc = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM).to(device)
		
		state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
		self.imageEnc.load_state_dict(state_dict)
		self.requires_grad(self.imageEnc, False)
		self.imageEnc.eval() 
		
		# -------------------------------------------
		# image encoder for eval FID
		self.evalEnc = InceptionV3(output_blocks=[3], normalize_input=False).to(device)
		self.evalEnc.eval()

		if get_rank() == 0:
			print('Load text encoder from:', cfg.TRAIN.NET_E)
			print('Load image encoder from:', img_encoder_path)

		# #######################generator and discriminators############## #
		netG = G_STYLE(self.img_size).to(device)
		netD = D_NET(self.img_size).to(device)
		
		netG_ema = G_STYLE(self.img_size).to(device)
		netG_ema.eval()
		self.accumulate(netG_ema, netG, 0)

		if get_rank() == 0:
			print('G\'s trainable parameters =', count_parameters(netG))
			print('D\'s trainable parameters =', count_parameters(netD))

		g_reg_ratio = cfg.TRAIN.G_REG_EVERY / (cfg.TRAIN.G_REG_EVERY + 1)
		d_reg_ratio = cfg.TRAIN.D_REG_EVERY / (cfg.TRAIN.D_REG_EVERY + 1)
		
		optimG = optim.Adam(
			netG.parameters(),
			lr=cfg.TRAIN.GENERATOR_LR * g_reg_ratio,
			betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio)
		)
		optimD = optim.Adam(
			netD.parameters(), 
			lr=cfg.TRAIN.DISCRIMINATOR_LR * d_reg_ratio,
			betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio)
		)

		epoch = 0
		if cfg.TRAIN.NET_G != '':
			Gname = cfg.TRAIN.NET_G
			istart = Gname.rfind('_') + 1
			iend = Gname.rfind('.')
			epoch = int(Gname[istart:iend]) + 1

			ckpt = torch.load(
				Gname, 
				map_location=lambda storage, loc: storage
			)

			netG.load_state_dict(ckpt["g"])
			netD.load_state_dict(ckpt["d"])
			netG_ema.load_state_dict(ckpt["g_ema"])

			optimG.load_state_dict(ckpt["g_optim"])
			optimD.load_state_dict(ckpt["d_optim"])
			
			if get_rank() == 0:
				print("load model:", Gname)

		# ########################################################### #

		if self.args.distributed:
			netG = nn.parallel.DistributedDataParallel(
				netG, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
				# find_unused_parameters=True,
			)

			netD = nn.parallel.DistributedDataParallel(
				netD, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
				find_unused_parameters=True,
			)

		return [netG, netD, netG_ema, optimG, optimD, epoch]


	def prepare_labels(self):
		batch_size = self.batch_size
		device = self.args.device
		real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))  # (N,)
		fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))  # (N,)
		match_labels = Variable(torch.LongTensor(range(batch_size)))    # [0,1,...,9]
		real_labels = real_labels.to(device)
		fake_labels = fake_labels.to(device)
		match_labels = match_labels.to(device)

		return real_labels, fake_labels, match_labels


	def save_model(self, g_module, d_module, g_ema, g_optim, d_optim, s_name):
		torch.save(
			{
				"g": g_module.state_dict(),
				"d": d_module.state_dict(),
				"g_ema": g_ema.state_dict(),
				"g_optim": g_optim.state_dict(),
				"d_optim": d_optim.state_dict(),
			}, 
			s_name,
		)


	def adjust_dynamic_range(self, data, drange_in, drange_out):
		if drange_in != drange_out:
			scale = (np.float32(drange_out[1]) - np.float32(drange_out[0])) / (np.float32(drange_in[1]) - np.float32(drange_in[0]))
			bias = (np.float32(drange_out[0]) - np.float32(drange_in[0]) * scale)
			data = data * scale + bias
		return data


	def convert_to_img(self, im, drange=[0, 1]):
		assert im.ndim == 2 or im.ndim == 3
		if im.ndim == 3:
			if im.shape[0] == 1:
				im = im[0] # grayscale CHW => HW
			else:
				im = im.transpose(1, 2, 0) # CHW -> HWC
		# [-1, 1] --> [0, 255]
		# im = (im + 1.0) * 127.5
		# im = im.astype(np.uint8)
		im = self.adjust_dynamic_range(im, drange, [0,255])
		im = np.rint(im).clip(0, 255).astype(np.uint8)
		return Image.fromarray(im)
	

	def make_noise(self, batch, latent_dim, device):
		return torch.randn(batch, latent_dim, device=device)


	def save_grid_captions(self, grid_cap, filename):
		print("Saving real captions")
		n_sample = len(grid_cap)
		save_caps = []
		for i in range(n_sample):
			cap = [
				self.ixtoword[_].encode('ascii', 'ignore').decode('ascii') 
				for _ in grid_cap[i].data.cpu().numpy()
			]
			save_caps.append(' '.join(cap).replace('END','') + '\n\n')

		fullpath = os.path.join(self.image_dir, filename)
		with open(fullpath, 'w') as f:
			f.writelines(save_caps)


	def save_grid_images(self, images, filename):
		n_sample = images.size(0)

		utils.save_image(
			images,
			f"{self.image_dir}/{filename}",
			nrow=int(n_sample ** 0.5),
			normalize=True,
			range=(-1, 1),
		)


	def train(self):
		device = self.args.device
		train_loader = self.sample_data(self.data_loader)
		# val_loader = self.sample_data(self.val_loader) if self.val_loader
		path_loader = self.sample_data(self.path_loader)

		netG, netD, netG_ema, optimG, optimD, start_epoch = self.build_models()

		real_labels, fake_labels, match_labels = self.prepare_labels()
		# # (N,), (N,), [0,1,...,N]

		batch_size = self.batch_size

		mean_path_length = 0
		mean_path_length_avg = 0

		d_loss_val = 0
		g_loss_val = 0
		r1_loss = torch.tensor(0.0, device=device)
		path_loss = torch.tensor(0.0, device=device)
		path_lengths = torch.tensor(0.0, device=device)
		loss_dict = {}

		if self.args.distributed:
			g_module = netG.module
			d_module = netD.module 
			g_ema = netG_ema.module
		else:
			g_module = netG 
			d_module = netD
			g_ema = netG_ema
		
		accum = 0.5 ** (32 / (10 * 1000))

		# # save grid real images
		# if get_rank() == 0:
		# 	n_sample = self.args.n_sample
			
		# 	train_samples = self.data_set.get_grid_data(n_sample)
		# 	train_imgs, train_caps, train_caplens, _, _ = prepare_data(train_samples)
			
		# 	train_hid = self.textEnc.init_hidden(n_sample)
		# 	train_outs, train_states = self.textEnc(
		# 		train_caps, train_caplens, train_hid
		# 	)
		# 	train_outs = train_outs.detach()
		# 	train_states = train_states.detach()

		# 	self.save_grid_images(train_imgs, 'real_train.png')
		# 	self.save_grid_captions(train_caps, 'real_caps_train.txt')

		# 	val_samples = self.val_set.get_grid_data(n_sample)
		# 	val_imgs, val_caps, val_caplens, _, _ = prepare_data(
		# 		val_samples
		# 	)

		# 	val_hid = self.textEnc.init_hidden(n_sample)
		# 	val_outs, val_states = self.textEnc(
		# 		val_caps, val_caplens, val_hid
		# 	)

		# 	val_outs = val_outs.detach()
		# 	val_states = val_states.detach()

		# 	self.save_grid_images(val_imgs, 'real_val.png')
		# 	self.save_grid_captions(val_caps, 'real_caps_val.txt')

		gen_iters = 0
		best_fid, best_ep = None, None
		for epoch in range(start_epoch, self.max_epoch):
			if self.args.distributed:
				self.data_sampler.set_epoch(epoch)

			start_t = time.time()
			elapsed = 0
			step = 0
			while step < self.num_batches:
				start_step = start_t = time.time()
				
				######################################################
				# (1) Prepare training data and Compute text embeddings
				######################################################
				data = next(train_loader)
				real_img, caps, cap_lens, class_ids, keys = prepare_data(data)

				hidden = self.textEnc.init_hidden(batch_size)
				# outputs: N x nef x seq_len
				# states: N x nef
				outputs, states = self.textEnc(caps, cap_lens, hidden)
				outputs, states = outputs.detach(), states.detach()

				#######################################################
				# (2) Update D network
				######################################################
				self.requires_grad(g_module, False)
				self.requires_grad(d_module, True)
				
				# print('states: [%.4f, %.4f]' % (states.min(), states.max()))
				fake_img,  _ = g_module(states)

				loss_d, real_pred, fake_pred = d_logistic_loss(
					d_module, real_img, fake_img, states, real_labels, fake_labels
				)
				
				loss_dict["d"] = loss_d
				loss_dict["real_score"] = real_pred.mean()
				loss_dict["fake_score"] = fake_pred.mean()
				
				# backward and update parameters
				d_module.zero_grad()
				loss_d.backward()
				optimD.step()

				d_reg_every = cfg.TRAIN.D_REG_EVERY
				r1 = cfg.TRAIN.R1
				d_regularize = gen_iters % d_reg_every == 0
				
				if d_regularize:
					real_img.requires_grad = True
					r1_loss, real_pred = d_r1_loss(d_module, real_img, states)

					d_module.zero_grad()
					(r1 / 2 * r1_loss * d_reg_every + 0 * real_pred[0]).backward()

					optimD.step()
				
				loss_dict["r1"] = r1_loss

				#######################################################
				# (3) Update G network: maximize log(D(G(z)))
				######################################################
				self.requires_grad(g_module, True)
				self.requires_grad(d_module, False)
				
				fake_img, _ = g_module(states)
				
				loss_g = g_nonsaturating_loss(
					d_module, fake_img, states, real_labels
				)
				
				loss_dict["g"] = loss_g
				
				# backward and update parameters
				g_module.zero_grad()
				loss_g.backward()
				optimG.step()

				g_reg_every = cfg.TRAIN.G_REG_EVERY
				path_regularzie = cfg.TRAIN.PATH_REGULARIZE
				g_regularize = gen_iters % g_reg_every == 0
				
				if g_regularize:
					pl_data = next(path_loader)
					_, pl_caps, pl_cap_lens, _, _ = prepare_data(pl_data)
					
					path_batch = self.path_batch
					pl_caps = pl_caps[:path_batch]
					pl_cap_lens = pl_cap_lens[:path_batch]

					pl_hidden = self.textEnc.init_hidden(path_batch)
					_, pl_states = self.textEnc(pl_caps, pl_cap_lens, pl_hidden)
					pl_states = pl_states.detach()

					pl_fake_img, _, _, pl_dlatents = \
						g_module(pl_states, return_latents=True)

					path_loss, mean_path_length, path_lengths = g_path_regularize(
						pl_fake_img, pl_dlatents, mean_path_length
					)

					g_module.zero_grad()
					weighted_path_loss = path_regularzie * g_reg_every * path_loss

					if self.path_batch_shrink: 
						weighted_path_loss += 0 * pl_fake_img[0, 0, 0, 0]   # ??

					weighted_path_loss.backward()

					optimG.step()

					mean_path_length_avg = (
						reduce_sum(mean_path_length).item() / get_world_size()
					)

				loss_dict["path"] = path_loss
				loss_dict["path_length"] = path_lengths.mean()

				self.accumulate(g_ema, g_module, accum)

				loss_reduced = reduce_loss_dict(loss_dict)

				d_loss_val = loss_reduced["d"].mean().item()
				g_loss_val = loss_reduced["g"].mean().item()
				r1_val = loss_reduced["r1"].mean().item()
				path_loss_val = loss_reduced["path"].mean().item()
				real_score_val = loss_reduced["real_score"].mean().item()
				fake_score_val = loss_reduced["fake_score"].mean().item()
				path_length_val = loss_reduced["path_length"].mean().item()

				elapsed += (time.time() - start_step)
				display_gap = 100
				if get_rank() == 0:
					if gen_iters % display_gap == 0:  # 100
						print('Epoch [{}/{}] Step [{}/{}] Time [{:.2f}s]'.format(
							epoch, self.max_epoch, step, self.num_batches, elapsed/display_gap
						))
						elapsed = 0
						print(
							f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
							f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
						)
						print('[%.4f, %.4f]' %(fake_img.min(), fake_img.max()))
						print('-' * 40)
						
					if self.wandb and self.args.wandb:
						self.wandb.log(
							{
								"Generator": g_loss_val,
								"Discriminator": d_loss_val,
								"R1": r1_val,
								"Path Length Regularization": path_loss_val,
								"Mean Path Length": mean_path_length,
								"Real Score": real_score_val,
								"Fake Score": fake_score_val,
								"Path Length": path_length_val,
							}
						)

				step += 1
				gen_iters += 1
	
			end_t = time.time()

			if get_rank() == 0:
				print(cfg.CONFIG_NAME)
				print('''[%d/%d] Loss_D: %.4f Loss_G: %.4f Time: %.2fs''' % (
					epoch, self.max_epoch, d_loss_val, g_loss_val, end_t - start_t))
				
				with torch.no_grad():
					g_ema.eval()
					train_sample, _ = g_ema(train_states)
					val_sample, _ = g_ema(val_states)

					print("Saving fake images for epoch%d..." % (epoch))
					self.save_grid_images(
						train_sample, 
						f"fake_{str(epoch).zfill(4)}_train.png"
					)
					self.save_grid_images(
						val_sample, 
						f"fake_{str(epoch).zfill(4)}_val.png"
					)
					
					# validation
					fid1, fid2 = self.eval(g_ema)

					if best_fid is None or fid2 < best_fid:
						best_fid, best_ep = fid2, epoch
						self.save_model(
							g_module, d_module, g_ema, optimG, optimD, 
							f"{self.model_dir}/ckpt_best.pth"
						)
					print(
						f"FID(train/val): {fid1:.4f} / {fid2:.4f}, "
						f"best FID: {best_fid:.4f} at ep{best_ep}"
					)

					metric_file = os.path.join(self.out_dir, f'fid3k2.txt')
					with open(metric_file, 'a') as f:
						f.write(
							f'epoch-{str(epoch).zfill(4)}\t\t'
							f'fid3k2 {fid1:.4f} / {fid2:.4f}\n'
						)
				
				print('-' * 89)

				if epoch % self.snapshot_interval == 0 or epoch == self.max_epoch:
					print('Saving models...')
					self.save_model(
						g_module, d_module, g_ema, optimG, optimD, 
						f"{self.model_dir}/ckpt_{str(epoch).zfill(4)}.pth"
					)


	def eval(self, g_module):
		batch_size = self.batch_size
		n_batch = self.args.n_val // batch_size
		
		# generated images
		act = []
		data_iter = iter(self.val_loader)
		for i in tqdm(range(n_batch)):
			try:
				data = data_iter.next()
			except:
				data_iter = iter(self.val_loader)
				data = data_iter.next()

			_, caps, cap_lens, _, _ = prepare_data(data)

			hidden = self.textEnc.init_hidden(batch_size)
			_, sent_emb = self.textEnc(caps, cap_lens, hidden)
			sent_emb = sent_emb.detach()

			# Generate fake images
			fake_imgs, _ = g_module(sent_emb)

			pred = self.evalEnc(fake_imgs)[0] 
			act.append(pred.view(pred.shape[0], -1).to("cpu"))
		
		act = torch.cat(act, 0).numpy()
		mu = np.mean(act, axis=0)
		sigma = np.cov(act, rowvar=False)
		
		fid_train = calculate_frechet_distance(
			self.mu_train, self.sigma_train, mu, sigma
		)
		fid_val = calculate_frechet_distance(
			self.mu_val, self.sigma_val, mu, sigma
		)
		
		return fid_train, fid_val


	def sampling(self):
		# for single GPU

		model_dir = cfg.TRAIN.NET_G
		if model_dir == '':
			print('Error: the path for morels is not found!')
		else:
			# Build and load the generator
			device = self.args.device

			print('Load G from:', model_dir)
			ckpt = torch.load(model_dir)
			g_ema = G_STYLE(self.img_size).to(device)
			g_ema.load_state_dict(ckpt['g_ema'])
			g_ema.eval()

			# load text encoder
			print('Load text encoder from:', cfg.TRAIN.NET_E)
			state_dict = torch.load(cfg.TRAIN.NET_E)
			self.textEnc = RNN_ENCODER(
				self.n_words,
				nhidden=cfg.TEXT.EMBEDDING_DIM,
				pre_emb=self.pretrained_emb,
				ninput=768
			).to(device)
			self.textEnc.load_state_dict(state_dict)
			self.textEnc.eval()

			# the path to save generated images
			s_tmp = model_dir[:model_dir.rfind('.pth')]
			save_dir = '%s/%s' % (s_tmp, 'valid')
			mkdir_p(save_dir)
			
			batch_size = self.batch_size
			latent_dim = cfg.GAN.W_DIM
			data_loader = self.sample_data(self.data_loader)

			cnt = 0 
			flag = True 
			while flag:
				data = next(data_loader)
				real_img, caps, cap_lens, _, keys = prepare_data(data)

				hidden = self.textEnc.init_hidden(batch_size)
				_, states = self.textEnc(caps, cap_lens, hidden)
				states = states.detach()
				
				fake_img, _ = g_ema(states)

				for img, cap, key in zip(fake_img, caps, keys):
					cnt += 1
					
					# save image
					img_name = f"{key}_{str(cnt).zfill(6)}.png"
					save_path = f"{save_dir}/{img_name}"
					utils.save_image(
						img,
						save_path,
						nrow=1,
						normalize=True,
						range=(-1, 1),
					)

					# # save text
					# save_cap = [self.ixtoword[idx.item()] for idx in cap]
					# save_cap = ' '.join(save_cap).replace('END', '').strip()
					# save_path = save_path.replace('.png', '.txt')
					# with open(save_path, 'w') as f:
					# 	f.write(save_cap)

					if cnt % 2500 == 0:
						print(f"{str(cnt)} imgs saved")

					if cnt >= 30000:  # 30000
						flag = False
						break
