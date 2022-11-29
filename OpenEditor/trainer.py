# coding=utf-8
from __future__ import print_function

import os
import time
import numpy as np
import sys
import shutil
import copy

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
from tensorboardX import SummaryWriter


from miscc.config import cfg
from miscc.utils import mkdir_p
from miscc.utils import build_super_images, build_super_images2
from miscc.utils import weights_init, load_params, copy_G_params
from miscc.losses import words_loss, sent_loss, KL_loss
from miscc.losses import d_logistic_loss, d_r1_loss
from miscc.losses import g_nonsaturating_loss, g_path_regularize
from miscc.losses import g_perceptual_loss, g_perceptual_loss2
from miscc.losses import arcface_loss

from datasets import TextDataset, prepare_data
from model_base import RNN_ENCODER, CNN_ENCODER, resnet_face18
from model import Generator as G_STYLE
from model import Discriminator as D_NET
from model import Manipulater as EDIT_NET
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
			self.log_dir = os.path.join(output_dir, 'Code_backup')
			mkdir_p(self.model_dir)
			mkdir_p(self.image_dir)
			mkdir_p(self.log_dir)
			
			self.writer = SummaryWriter(output_dir)

			# shutil.copy(args.cfg_file, self.log_dir)
			# bkfiles = ['datasets', 'main', 'trainer', 'model', 'model_base', 'miscc/losses']
			# for _file in bkfiles:
			# 	shutil.copy(f'../code/{_file}.py', self.log_dir)

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
		self.ixtoword = self.data_set.ixtoword  # dict for idx to word
		self.wordtoix = self.data_set.wordtoix
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

			path = '../../eval/FID/faceV1.0_train.npz'
			f = np.load(path)
			self.mu_train, self.sigma_train = f['mu'][:], f['sigma'][:]
			f.close()

			path = '../../eval/FID/faceV1.0_val.npz'
			f = np.load(path)
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
		
		# ******************  Pre-trained Enc. / G / D  *************************
		if get_rank() == 0:
			if cfg.TRAIN.NET_E == '' or cfg.TRAIN.NET_G == '':
				print('Error: no pretrained Encoder / Generator / Discriminator')
				return

		# init and load text encoder
		self.textEnc = RNN_ENCODER(
			self.n_words,
			nhidden=cfg.TEXT.EMBEDDING_DIM,
			pre_emb=self.pretrained_emb,
			ninput=768
		).to(device)
		
		state_dict = torch.load(cfg.TRAIN.NET_E, map_location=lambda storage, loc: storage)
		self.textEnc.load_state_dict(state_dict)
		self.requires_grad(self.textEnc, False)
		self.textEnc.eval()  # disable BatchNormalization & Dropout

		# init and load image encoder
		img_encoder_path = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
		self.imageEnc = CNN_ENCODER(
			cfg.TEXT.EMBEDDING_DIM
		).to(device)
		
		state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
		self.imageEnc.load_state_dict(state_dict)
		self.requires_grad(self.imageEnc, False)
		self.imageEnc.eval()  # disable BatchNormalization & Dropout

		if get_rank() == 0:
			print('Load text encoder:', cfg.TRAIN.NET_E)
			print('Load image encoder:', img_encoder_path)

		
		netG = G_STYLE(self.img_size).to(device)
		netD = D_NET(self.img_size).to(device)
		
		epoch = 0
		Gname = cfg.TRAIN.NET_G
		istart = Gname.rfind('_') + 1
		iend = Gname.rfind('.')
		epoch = int(Gname[istart:iend]) + 1

		ckpt = torch.load(Gname, map_location=lambda storage, loc: storage)
		# netG.load_state_dict(ckpt["g"], strict=False)
		netG_ema.load_state_dict(ckpt["g_ema"], strict=True)
		self.requires_grad(netG, False)
		netG_ema.eval()

		netD.load_state_dict(ckpt["d"], strict=True)
		self.requires_grad(netD, False)
		netD.eval()

		if get_rank() == 0:
			print('Load G / D from:', Gname)

		# Trainable Manipulater
		Editor = EDIT_NET(netG.size, netG.channels, netG.n_latent).to(device)

		if get_rank() == 0:
			print('Editor\'s trainable parameters =', count_parameters(Editor))
			# print('D\'s trainable parameters =', count_parameters(netD))

		optimEdit = optim.Adam(
			Editor.parameters(),
			lr=cfg.TRAIN.MANIPULATER_LR,
			betas=(0.9, 0.999)
		)

		# ########################################################### #

		if self.args.distributed:
			Editor = nn.parallel.DistributedDataParallel(
				Editor, 
				device_ids=[self.args.local_rank],
				output_device=self.args.local_rank,
				broadcast_buffers=False,
				find_unused_parameters=True,
			)

		return [netG_ema, netD, Editor, optimEdit, epoch]


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
				# "d": d_module.state_dict(),
				"g_ema": g_ema.state_dict(),
				"g_optim": g_optim.state_dict(),
				# "d_optim": d_optim.state_dict(),
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
		print(f"Saving {filename}")
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
		print(f"Saving {filename}")
		n_sample = images.size(0)
		
		utils.save_image(
			images,
			f"{self.image_dir}/{filename}",
			nrow=int(n_sample ** 0.5),
			normalize=True,
			range=(-1, 1),
		)


	def save_sample(self, split='train'):
		n_sample = self.args.n_sample
		dataset = self.data_set if split == 'train' else self.val_set
		
		samples = dataset.get_grid_data(n_sample)
		
		imgs, caps1, caplens1, caps2, caplens2, \
			_, _, idx1, idx2 = prepare_data(samples) 
		
		hidden1 = self.textEnc.init_hidden(n_sample)
		word1, sent1 = self.textEnc(caps1, caplens1, hidden1)
		word1, sent1 = word1.detach(), sent1.detach()

		hidden2 = self.textEnc.init_hidden(n_sample)
		word2, sent2 = self.textEnc(caps2, caplens2, hidden2)
		word2, sent2 = word2.detach(), sent2.detach()

		_, recover_idx = torch.sort(idx2, 0)
		word2 = word2[recover_idx][idx1]
		sent2 = sent2[recover_idx][idx1]
		caps2 = caps2[recover_idx][idx1]

		self.save_grid_images(imgs, f'real_{split}.png')
		self.save_grid_captions(caps1, f'real_{split}_caps.txt')
		self.save_grid_captions(caps2, f'real_{split}_edit.txt')

		return word1, sent1, word2, sent2


	def train(self):
		device = self.args.device
		
		train_loader = self.sample_data(self.data_loader)
		path_loader = self.sample_data(self.path_loader)

		netG_ema, netD, Editor, optimEdit, start_epoch = self.build_models()
		# start_epoch = 0 # todo finetune
		
		self.eval_cnn = InceptionV3(
			output_blocks=[3],
			normalize_input=False,
		).to(device)
		self.eval_cnn.eval()

		real_labels, fake_labels, match_labels = self.prepare_labels()
		# # (N,), (N,), [0,1,...,N]

		batch_size = self.batch_size

		loss_dict = {}

		if self.args.distributed:
			g_ema = netG_ema.module
			d_module = netD.module
			editor = Editor.module
		else:
			g_ema = netG_ema 
			d_module = netD
			editor = Editor

		# save grid real images
		if get_rank() == 0:
			_, train_sent_src, _, train_sent_tar = self.save_sample('train')

			_, val_sent_src, _, val_sent_tar = self.save_sample('val')

			fake_src_caps = self.data_set.examples['src'].squeeze()
			fake_tar_caps = self.data_set.examples['tar'].squeeze()
			self.save_grid_captions(fake_src_caps, 'fake_test_caps.txt')
			self.save_grid_captions(fake_tar_caps, 'fake_test_edit.txt')


		gen_iters = 0
		best_fid, best_ep = None, None
		for epoch in range(start_epoch, self.max_epoch):
			if self.args.distributed:
				self.data_sampler.set_epoch(epoch)

			ep_start = time.time()
			step = 0
			while step < self.num_batches:
				######################################################
				# (1) Prepare training data and Compute text embeddings
				######################################################

				data = next(train_loader)
				real_img, caps_src, caplens_src, caps_tar, caplens_tar, \
					class_ids, keys, sort_idx1, sort_idx2 = prepare_data(data)

				hid_src = self.textEnc.init_hidden(batch_size)
				# word_src: N x nef x seq_len
				# sent_src: N x nef
				word_src, sent_src = self.textEnc(caps_src, caplens_src, hid_src)
				word_src, sent_src = word_src.detach(), sent_src.detach()

				hid_tar = self.textEnc.init_hidden(batch_size)
				word_tar, sent_tar = self.textEnc(caps_tar, caplens_tar, hid_tar)
				word_tar, sent_tar = word_tar.detach(), sent_tar.detach()

				# make sure the (sent_src, sent_tar) matched
				_, recover_idx = torch.sort(sort_idx2, 0)
				word_tar = word_tar[recover_idx][sort_idx1]
				sent_tar = sent_tar[recover_idx][sort_idx1]
				caps_tar = caps_tar[recover_idx][sort_idx1]

				#######################################################
				# (2) Update Manipulation Module
				####################################################### 
				self.requires_grad(editor, True)
				
				# generating images
				b1_fake_img, _, feat_src = g_ema(sent_src)

				b2_fake_img, gate, alpha, feat_edited = editor(sent_src, sent_tar, g_ema)

				# entropy
				if gate is not None:
					gate_entropy = -(gate * torch.log2(gate)).sum()
					loss_dict["gate_entropy"] = gate_entropy
				if alpha is not None:
					alpha_entropy = -(alpha * torch.log2(alpha)).sum()
					loss_dict["alpha_entropy"] = alpha_entropy

				# generate loss
				b2_g_loss = g_nonsaturating_loss(
					d_module, b2_fake_img, sent_tar, real_labels
				)
				loss_dict["b2_g"] = b2_g_loss

				b2_region, b2_global = self.imageEnc(b2_fake_img)
				
				# match loss, WordLoss(region_edited, word_tar)
				b2_w_loss0, b2_w_loss1, _ = words_loss(
					b2_region, word_tar,
					match_labels, caplens_tar, class_ids, batch_size
				)
				b2_w_loss = (b2_w_loss0 + b2_w_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
				loss_dict["b2_word"] = b2_w_loss
				

				# identity loss1, SentLoss(global_edited, sent_src)
				b1_s_loss0, b1_s_loss1 = sent_loss(
					b2_global, sent_src,
					match_labels, class_ids, batch_size
				)
				b1_s_loss = (b1_s_loss0 + b1_s_loss1) * cfg.TRAIN.SMOOTH.LAMBDA
				loss_dict["b1_sent"] = b1_s_loss
				

				# identity loss2
				IFeat_type = 'l2'
				b2_IncepFeat_loss = g_perceptual_loss(
					self.imageEnc, b1_fake_img, b2_fake_img, IFeat_type
				) * cfg.TRAIN.SMOOTH.DELTA1

				loss_dict[f"b2_InceptFeat_{IFeat_type}"] = b2_IncepFeat_loss

				# identity loss3
				GFeat_type = 'l1'
				b2_GFeat_loss = g_perceptual_loss2(
					feat_src, feat_edited, loss_type=GFeat_type
				) * cfg.TRAIN.SMOOTH.DELTA1
				
				loss_dict[f"b2_GFeat_{GFeat_type}"] = b2_GFeat_loss

				identity_loss = b1_s_loss + b2_IncepFeat_loss + b2_GFeat_loss

				loss_dict[f"b2_identity"] = identity_loss

				g_total = b2_g_loss + b2_w_loss + identity_loss
				loss_dict[f"total"] = g_total

				# backward and update parameters
				editor.zero_grad()
				g_total.backward()
				optimEdit.step()


				loss_reduced = reduce_loss_dict(loss_dict)
				
				# For generating quality
				b2_g_loss_val = loss_reduced["b2_g"].mean().item()

				# For attribute matching
				b2_w_loss_val = loss_reduced["b2_word"].mean().item()

				# For identity presereve
				b1_s_loss_val = loss_reduced["b1_sent"].mean().item()
				IncepFeat_val = loss_reduced[f"b2_InceptFeat_{IFeat_type}"].mean().item()
				GFeat_val = loss_reduced[f"b2_GFeat_{GFeat_type}"].mean().item()
								
				# others
				gate_entropy_val = loss_reduced['gate_entropy'].mean().item()
				alpha_entropy_val = loss_reduced['alpha_entropy'].mean().item()
				identity_val = loss_reduced["b2_identity"].mean().item()
				total_val = loss_reduced["total"].mean().item()

				elapsed = time.time() - ep_start

				display_gap = 500
				if get_rank() == 0:
					if gen_iters % display_gap == 0:  # 100
						print('-' * 60)
						print(
							f"Epoch [{epoch}/{self.max_epoch}] ", 
							f"Step [{step}/{self.num_batches}] ", 
							f"Time: {elapsed / (step + 1):.2f} s"
						)
						print(
							f"total: {total_val:.4f}; "
							f"g: {b2_g_loss_val:.4f}; "
							f"b2_w: {b2_w_loss_val:.4f}; "
							f"b1_s: {b1_s_loss_val:.4f}; "
							f"Incep_{IFeat_type}: {IncepFeat_val:.4f}; "
							f"G_{GFeat_type}: {GFeat_val:.4f}; "

						)

						print('-' * 30)
						if gate is not None:
							print("train gate:\n", gate[0].squeeze().detach().to("cpu"))
							print(f"entropy of gate: {gate_entropy_val:.4f}")
							self.writer.add_scalar(
								'entropy/gate', float(alpha_entropy_val), gen_iters
							)
						if alpha is not None:
							print(f"entropy of alpha: {alpha_entropy_val:.4f}")
							print("train alpha:\n", alpha[0].squeeze().detach().to("cpu"))
							self.writer.add_scalar(
								'entropy/alpha', float(alpha_entropy_val), gen_iters
							)

						# print("-" * 30, "\ngrad: ")
						# for _name, _param in editor.named_parameters():
						# 	if _param.requires_grad:
						# 		print(_name)
						# 		print(
						# 			f"[{_param.grad.min().cpu().numpy()},"
						# 			f" {_param.grad.max().cpu()}]"
						# 		)
						print('-' * 60)

						log_info = {
							f"Total": total_val,
							f"G": b2_g_loss_val,
							f"b2_Word": b2_w_loss_val,
							f"b1_Sent": b1_s_loss_val,
							f"Incep_{IFeat_type}": IncepFeat_val,
							f"G_{GFeat_type}": GFeat_val,
							f"Identity": identity_val,
						}

						for key, value in log_info.items():
							self.writer.add_scalar(f'loss/{key}', float(value), gen_iters)
					
					if gen_iters % 5000 == 0: 
						eval_start = time.time()
						with torch.no_grad():
							g_ema.eval()
							
							train_src, _, _ = g_ema(train_sent_src)
							val_src, _, _ = g_ema(val_sent_src)
							
							train_tar, _, _ = g_ema(train_sent_tar)
							val_tar, _, _ = g_ema(val_sent_tar)
							
							train_edited, train_gate, train_alpha, _ = editor(
								train_sent_src, train_sent_tar, g_ema
							)
							val_edited, val_gate, val_alpha, _ = editor(
								val_sent_src, val_sent_tar, g_ema
							)

							train_tmp, val_tmp = [], []
							for k in range(self.args.n_sample):
								train_tmp.append(torch.cat(
									[train_src[k], train_tar[k], train_edited[k]], 
									dim=-1
								))
								val_tmp.append(torch.cat(
									[val_src[k], val_tar[k], val_edited[k]], 
									dim=-1
								))
							
							train_tmp = torch.stack(train_tmp, dim=0)
							val_tmp = torch.stack(val_tmp, dim=0)

							self.save_grid_images(
								train_tmp, 
								f"fake_{str(int(gen_iters/1000)).zfill(4)}k_train.png"
							)
							self.save_grid_images(
								val_tmp, 
								f"fake_{str(int(gen_iters/1000)).zfill(4)}k_val.png"
							)

							self.test(editor, g_ema, gen_iters)

							# validation
							fid1, fid2 = self.eval(editor, g_ema, epoch)

							if best_fid is None or fid2 < best_fid:
								best_fid, best_ep = fid2, int(gen_iters/1000)
								torch.save({
									"editor": editor.state_dict(),
									"edit_optim": optimEdit.state_dict()}, 
									f"{self.model_dir}/ckpt_best.pth"
								)

							print(
								f"FID(train/val): {fid1:.4f} / {fid2:.4f}, "
								f"best FID: {best_fid:.4f} at iter{best_ep}k"
							)

							metric_file = os.path.join(self.out_dir, f'fid3k2.txt')
							with open(metric_file, 'a') as f:
								f.write(
									f'iter-{int(gen_iters/1000)}k\t\t'
									f'fid3k2 {fid1:.4f} / {fid2:.4f}\n'
								)
						eval_end = time.time()
						print(f'Eval Times: {eval_end - eval_start:.4}s')

				step += 1
				gen_iters += 1
	
			ep_end = time.time()

			if get_rank() == 0:
				print(cfg.CONFIG_NAME)
				print(
					f'Epoch [{epoch}/{self.max_epoch}] '
					f'Loss_G2: {b2_g_loss_val:.4f} '
					f'Time: {(ep_end - ep_start):.2f}s'
				)
				
				print('-' * 89)

				if epoch % self.snapshot_interval == 0 or epoch == self.max_epoch:
					print('Saving models...')
					torch.save({
						"editor": editor.state_dict(),
						"edit_optim": optimEdit.state_dict()}, 
						f"{self.model_dir}/ckpt_{str(epoch).zfill(4)}.pth"
					)


	def eval(self, Editor, netG, epoch):
		batch_size = self.batch_size
		n_batch = self.args.n_val // batch_size
		save_dir = f"{self.image_dir}/ckpt_{str(epoch).zfill(4)}"
		mkdir_p(save_dir)

		cnt = 0
		act = []
		data_iter = iter(self.val_loader)
		# while (end + 1) < n_used_imgs:
		for i in tqdm(range(n_batch)):
			try:
				data = data_iter.next()
			except:
				data_iter = iter(self.val_loader)
				data = data_iter.next()

			_, cap_src, caplen_src, cap_tar, caplen_tar, \
				_, keys, idx_src, idx_tar = prepare_data(data)

			hid_src = self.textEnc.init_hidden(batch_size)
			_, sent_src = self.textEnc(cap_src, caplen_src, hid_src)
			sent_src = sent_src.detach()

			hid_tar = self.textEnc.init_hidden(batch_size)
			_, sent_tar = self.textEnc(cap_tar, caplen_tar, hid_tar)
			sent_tar = sent_tar.detach()

			# make sure the (sent_src, sent_tar) matched
			_, recover_idx = torch.sort(idx_tar, 0)
			sent_tar = sent_tar[recover_idx][idx_src]
			cap_tar = cap_tar[recover_idx][idx_src]

			# Generate fake images
			eidted_imgs, gate, alpha, _ = Editor(sent_src, sent_tar, netG)
			pred = self.eval_cnn(eidted_imgs)[0] 
			act.append(pred.view(pred.shape[0], -1).to("cpu"))

			if cnt < 96:
				val_src, _, _ = netG(sent_src)
				val_tar, _, _ = netG(sent_tar)
				for src_, tar_, edi_, key, csrc_, ctar_ in zip(
					val_src, val_tar, eidted_imgs, keys, cap_src, cap_tar
				): 
					val_tmp = torch.cat([src_, tar_, edi_], dim=-1)
					filename = f"{key}_{str(cnt).zfill(6)}.png"
					utils.save_image(
						val_tmp,
						f"{save_dir}/{filename}",
						nrow=1,
						normalize=True,
						range=(-1, 1),
					)
					csrc_ = [
						self.ixtoword[_].encode('ascii', 'ignore').decode('ascii') 
						for _ in csrc_.data.cpu().numpy()
					]
					ctar_ = [
						self.ixtoword[_].encode('ascii', 'ignore').decode('ascii') 
						for _ in ctar_.data.cpu().numpy()
					]
					save_caps = '\n'.join([
						' '.join(csrc_).replace('END',''), 
						' '.join(ctar_).replace('END','')
					])

					fullpath = os.path.join(save_dir, filename.replace('.png', '.json'))
					with open(fullpath, 'w') as f:
						f.writelines(save_caps)
				cnt += batch_size
		
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

	def test(self, Editor, netG, gen_iters):
		cap_src = self.data_set.examples['src']
		cap_tar = self.data_set.examples['tar']
		caplen_src = self.data_set.examples['src_len']
		caplen_tar = self.data_set.examples['tar_len']
		
		test_fig = []
		test_gate = []
		test_alpha = []
		
		for i in range(cap_src.size(0)):
			hid_src = self.textEnc.init_hidden(1)
			_, sent_src = self.textEnc(
				cap_src[i], caplen_src[i], hid_src
			)
			sent_src = sent_src.detach()

			hid_tar = self.textEnc.init_hidden(1)
			_, sent_tar = self.textEnc(
				cap_tar[i], caplen_tar[i], hid_tar
			)
			sent_tar = sent_tar.detach()

			img_src, _, _ = netG(sent_src)
			img_tar, _, _ = netG(sent_tar)
			edited, gate, alpha, _ = Editor(sent_src, sent_tar, netG)

			test_fig.append(
				torch.cat([img_src[0], img_tar[0], edited[0]], dim=-1)
			)
			# test_gate.append(gate)
			test_alpha.append(alpha)

		test_fig = torch.stack(test_fig, dim=0)
		# test_gate = torch.stack(test_gate, dim=0).squeeze(-1)
		test_alpha = torch.stack(test_alpha, dim=0).squeeze(-1)

		self.save_grid_images(
			test_fig, 
			f"fake_{str(int(gen_iters/1000)).zfill(4)}k_test.png"
		)
		# print("test gate:\n", test_gate[0].detach().to("cpu"))
		print("test alpha:\n", test_alpha[0].detach().to("cpu"))


	def sampling(self):
		model_dir = cfg.TRAIN.NET_M
		if model_dir == '':
			print('Error: the path for morels is not found!')
		else:
			# Build and load the generator
			device = self.args.device

			print('Load G from:', cfg.TRAIN.NET_G)
			ckpt = torch.load(cfg.TRAIN.NET_G)
			netG = G_STYLE(self.img_size).to(device)
			netG.load_state_dict(ckpt['g_ema'])
			
			print('Load Manipulater from:', model_dir)
			ckpt = torch.load(model_dir)
			Editor = EDIT_NET(
				netG.size, netG.channels, netG.n_latent
			).to(device)
			Editor.load_state_dict(ckpt['editor'])


			if self.args.distributed:
				netG = nn.DataParallel(netG)
				Editor = nn.DataParallel(Editor)
			netG.eval()
			Editor.eval()

			# load text encoder
			print('Load text encoder from:', cfg.TRAIN.NET_E)
			state_dict = torch.load(cfg.TRAIN.NET_E)
			text_encoder = RNN_ENCODER(
				self.n_words,
				nhidden=cfg.TEXT.EMBEDDING_DIM,
				pre_emb=self.pretrained_emb,
				ninput=768
			).to(device)
			text_encoder.load_state_dict(state_dict)
			if self.args.distributed:
				text_encoder = nn.DataParallel(text_encoder)
			text_encoder.eval()

			# the path to save generated images
			s_tmp = model_dir[:model_dir.rfind('.pth')]
			save_dir = '%s/%s' % (s_tmp, 'valid')
			mkdir_p(save_dir)
			
			batch_size = self.batch_size
			latent_dim = cfg.GAN.W_DIM
			data_loader = self.sample_data(self.data_loader)
			# set_trace()

			cnt = 0 
			flag = True 
			while flag:
				data = next(data_loader)
				_, cap_src, caplen_src, cap_tar, caplen_tar, \
				_, keys, idx_src, idx_tar = prepare_data(data)

				hid_src = text_encoder.init_hidden(batch_size)
				_, sent_src = text_encoder(cap_src, caplen_src, hid_src)
				sent_src = sent_src.detach()

				hid_tar = text_encoder.init_hidden(batch_size)
				_, sent_tar = text_encoder(cap_tar, caplen_tar, hid_tar)
				sent_tar = sent_tar.detach()

				# make sure the (sent_src, sent_tar) matched
				_, recover_idx = torch.sort(idx_tar, 0)
				sent_tar = sent_tar[recover_idx][idx_src]
				cap_tar = cap_tar[recover_idx][idx_src]

				# Generate fake images
				img_src, _, _ = netG(sent_src)
				img_tar, _, _ = netG(sent_tar)
				img_edited, gate, alpha, _ = Editor(sent_src, sent_tar, netG)

				for _src, _tar, _edit, _capsrc, _captar, key in zip(
					img_src, img_tar, img_edited, cap_src, cap_tar, keys
				):
					cnt += 1
					
					img_name = f"{key}_{str(cnt).zfill(6)}.png"
					save_path = f"{save_dir}/{img_name}"
					img = torch.stack([_src, _tar, _edit], dim=0)
					utils.save_image(
						img,
						save_path,
						nrow=3,
						normalize=True,
						range=(-1, 1),
					)

					_capsrc = ' '.join([self.ixtoword[idx.item()] for idx in _capsrc])
					_captar = ' '.join([self.ixtoword[idx.item()] for idx in _captar])
					_capsrc = _capsrc.replace('END', '').strip()
					_captar = _captar.replace('END', '').strip()
					save_path = save_path.replace('.png', '.json')
					with open(save_path, 'w') as f:
						f.write(f"{_capsrc}\n{_captar}")

					if cnt % 2500 == 0:
						print(f"{str(cnt)} imgs saved")

					if cnt >= 1000:  # 30000
						flag = False
						break

	def testing(self, description):
		from bert_serving.client import BertClient

		model_dir = cfg.TRAIN.NET_M
		if model_dir == '':
			print('Error: the path for morels is not found!')
			return 

		# Build and load the generator
		device = self.args.device

		print('Load G from:', cfg.TRAIN.NET_G)
		ckpt = torch.load(cfg.TRAIN.NET_G)
		netG = G_STYLE(self.img_size).to(device)
		netG.load_state_dict(ckpt['g_ema'])
		
		print('Load Manipulater from:', model_dir)
		ckpt = torch.load(model_dir)
		Editor = EDIT_NET(
			netG.size, netG.channels, netG.n_latent
		).to(device)
		Editor.load_state_dict(ckpt['editor'])


		if self.args.distributed:
			netG = nn.DataParallel(netG)
			Editor = nn.DataParallel(Editor)
		netG.eval()
		Editor.eval()

		# load text encoder
		print('Load text encoder from:', cfg.TRAIN.NET_E)
		state_dict = torch.load(cfg.TRAIN.NET_E)
		text_encoder = RNN_ENCODER(
			self.n_words,
			nhidden=cfg.TEXT.EMBEDDING_DIM,
			pre_emb=self.pretrained_emb,
			ninput=768
		).to(device)
		text_encoder.load_state_dict(state_dict)
		if self.args.distributed:
			text_encoder = nn.DataParallel(text_encoder)
		text_encoder.eval()

		
		def t_enc(text):
			tokens, tokens_unseen = self.data_set.tokenize(text)
			# bc = BertClient(ip='10.24.82.133')
			# cap = torch.from_numpy(copy.deepcopy(bc.encode(tokens))).cuda()
			# caplen = torch.tensor(len(cap), dtype=torch.long).cuda()
			# cap = cap.unsqueeze(0)
			# caplen = caplen.unsqueeze(0)

			cap = []
			for s in tokens:
				if s in self.wordtoix.keys():
					cap.append(self.wordtoix[s])
			# print('cap:', ' '.join([self.ixtoword[i] for i in cap]))

			cap = torch.tensor(cap, dtype=torch.long).unsqueeze(0)
			cap = cap.cuda()
			caplen = len(cap)
			caplen = torch.tensor(caplen, dtype=torch.long).unsqueeze(0)
			caplen = caplen.cuda()

			hidden = text_encoder.init_hidden(1)
			word, sent = text_encoder(cap, caplen, hidden, test=False)
			word, sent = word.detach(), sent.detach()
			
			return word, sent, tokens, tokens_unseen
		
		def t2f(text, text_man=None):
			_, sent = t_enc(text)
			if text_man is None:
				sent_man = None
				img, _, _ = netG(sent)
			else:
				_, sent_man, _, _ = t_enc(text_man)
				img, _, _, _ = Editor(sent, sent_man, netG)
			
			return img, sent, sent_man

		batch_size = 1
		latent_dim = cfg.GAN.W_DIM
		# set_trace()
		start_t = time.time()
		if os.path.isdir(description):
			from glob import glob 
			files = glob(os.path.join(description, '*.txt'))
			files.sort()

			save_dir = description
			for cnt in tqdm(range(len(files))):
				file = files[cnt]
				filename = os.path.split(file)[-1]
				savename = os.path.splitext(filename)[0]
				with open(file, 'r') as f:
					texts = f.read().split('\n')
				# text_src, text_tar = texts[0], texts[1]
				_, sent_src, _, _ = t_enc(texts[0])
				_, sent_tar, _, _ = t_enc(texts[1])
				
				img_src, _, _ = netG(sent_src)
				
				# # img_tar, _, _ = netG(sent_tar)
				# # img_com, _, _ = netG(0.5 * (sent_src + sent_tar))
				# img_man, _, _, _ = Editor(sent_src, sent_tar, netG)

				# # sulution for diversity
				# sent_rand = 2 * torch.rand(1,256) - 1
				# sent_rand = sent_rand.cuda()

				img_man, _, _, _ = Editor(sent_src, sent_tar, netG)
				
				utils.save_image(img_src, f"{save_dir}/{savename}_ours_gen_noZ.png",
						normalize=True, range=(-1, 1),
				)

				# utils.save_image(img_com, f"{save_dir}/{savename}_com.png",
				# 	normalize=True, range=(-1, 1),
				# )

				utils.save_image(img_man, f"{save_dir}/{savename}_ours_man_noZ.png",
					normalize=True, range=(-1, 1),
				)

				# utils.save_image(
				# 	torch.cat([img_src, img_com, img_man], dim=-1), 
				# 	f"{save_dir}/{savename}.png",
				# 	normalize=True, range=(-1, 1), nrow=1
				# )
			
			end_t = time.time()
			print(f'gen time: {(end_t - start_t):.4f}')

		elif os.path.isfile(description):
			save_dir = os.path.splitext(description)[0]
			mkdir_p(save_dir)

			with open(description, 'r') as f:
				# texts = [cap.strip().lower().split(' ') for cap in f.read().split('\n')]
				texts = f.read().split('\n')

			# text_tar = 'the woman has goatee'
			cnt = 1
			for text_src, text_tar in zip(texts[0::2], texts[1::2]):
	
				_, sent_src, _, _ = t_enc(text_src)
				_, sent_tar, _, _ = t_enc(text_tar)
				
				img_src, _, _ = netG(sent_src)
				# img_tar, _, _ = netG(sent_tar)

				# sent_tar = -1.0 * sent_tar
				# minus_img_tar, _, _ = netG((-1.0 * sent_tar))
				# img_name = 'B'
				utils.save_image(img_src, f"{save_dir}/{str(cnt).zfill(3)}_src.png",
						normalize=True, range=(-1, 1),
				)
				# utils.save_image(img_tar, f"{save_dir}/B.png",
				# 	normalize=True, range=(-1, 1),
				# )

				# Combination
				img_com, _, _ = netG(0.5 * (sent_src + sent_tar))
				utils.save_image(img_com, f"{save_dir}/{str(cnt).zfill(3)}_com.png",
					normalize=True, range=(-1, 1),
				)

				# Manipulation
				# img_man, _, _ = t2f(text_src, man=text_tar)
				img_man, _, _, _ = Editor(sent_src, sent_tar, netG)
				# img = torch.stack([img_src[0], img_man[0], img_tar[0]], dim=0)
				utils.save_image(img_man, f"{save_dir}/{str(cnt).zfill(3)}_man.png",
					normalize=True, range=(-1, 1),
				)

				img = torch.cat([img_src, img_com, img_man], dim=-1)
				utils.save_image(img, f"{save_dir}/{str(cnt).zfill(3)}.png",
					normalize=True, range=(-1, 1), nrow=1
				)

				cnt += 1 
		else:
			# the path to save generated images
			s_tmp = model_dir[:model_dir.rfind('.pth')]
			save_dir = '%s/%s' % (s_tmp, 'valid')
			mkdir_p(save_dir)

			_, sent_src, _, _ = t_enc(description)

			for i in tqdm(range(1)):
				img1, _, _ = netG(sent_src)

				save_path = f"{save_dir}/gen_{i:03d}.png"
				utils.save_image(
					img1, save_path, normalize=True, range=(-1, 1),
				)
				
				# # solution
				# sent_rand = 2 * torch.rand(1,256) - 1
				# sent_rand = sent_rand.cuda()
				
				# img2, _, _, _ = Editor(sent_rand, sent_src, netG)
				# save_path = f"{save_dir}/man_{i:03d}.png"
				# utils.save_image(
				# 	img2, save_path, normalize=True, range=(-1, 1),
				# )