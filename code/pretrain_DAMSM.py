# coding=utf-8
from __future__ import print_function

from miscc.utils import mkdir_p
from miscc.utils import build_super_images
from miscc.losses import sent_loss, words_loss
from miscc.config import cfg, cfg_from_file

from datasets import TextDataset
from datasets import prepare_data

from model_base import RNN_ENCODER, CNN_ENCODER

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
import shutil
from PIL import Image
from ipdb import set_trace

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

import tensorflow as tf


dir_path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.abspath(dir_path))


def add_summary_value(writer, key, value, iteration):
	summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
	writer.add_summary(summary, iteration)


UPDATE_INTERVAL = 10  # 200
def parse_args():
	parser = argparse.ArgumentParser(description='Train a DAMSM network')
	parser.add_argument('--cfg', dest='cfg_file',
						help='optional config file',
						default='cfg/DAMSM/bird.yml', type=str)
	parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
	parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
	parser.add_argument('--manualSeed', type=int, help='manual seed')
	parser.add_argument(
		'--local_rank', type=int, default=0, help='local rank for distributed training'
	)
	args = parser.parse_args()
	return args


def train(dataloader, cnn_model, rnn_model, batch_size,
		  labels, optimizer, epoch, ixtoword, image_dir):
	cnn_model.train()
	rnn_model.train()

	accu_steps = cfg.TRAIN.GRAD_ACCU_STEPS
	sub_bat_sze = int(batch_size / accu_steps)

	s_total_loss0 = 0
	s_total_loss1 = 0
	w_total_loss0 = 0
	w_total_loss1 = 0
	count = (epoch + 1) * len(dataloader)
	start_time = time.time()
	for step, data in enumerate(dataloader, 0):
		# print('step', step)
		rnn_model.zero_grad()
		cnn_model.zero_grad()

		imgs, captions, cap_lens, \
			class_ids, keys = prepare_data(data)

		for accu_step in range(accu_steps):
			# sub_imgs = [_[accu_step * sub_bat_sze:
			# 			  (accu_step+1) * sub_bat_sze] for _ in imgs]
			sub_imgs = imgs[
				accu_step * sub_bat_sze: (accu_step+1) * sub_bat_sze
			]
			sub_caps = captions[accu_step * sub_bat_sze:
								(accu_step+1) * sub_bat_sze]
			sub_keys = keys[accu_step * sub_bat_sze:
							(accu_step+1) * sub_bat_sze]
			sub_cap_lens = cap_lens[accu_step * sub_bat_sze:
									(accu_step+1) * sub_bat_sze]
			sub_class_ids = class_ids[accu_step * sub_bat_sze:
									  (accu_step+1) * sub_bat_sze]
			sub_labels = labels[0:sub_bat_sze]

			# words_features: batch_size x nef x 17 x 17
			# sent_code: batch_size x nef
			words_features, sent_code = cnn_model(sub_imgs)
			# --> batch_size x nef x 17*17
			nef, att_sze = words_features.size(1), words_features.size(2)
			# words_features = words_features.view(batch_size, nef, -1)

			hidden = rnn_model.init_hidden(sub_bat_sze)
			# words_emb: batch_size x nef x seq_len
			# sent_emb: batch_size x nef
			words_emb, sent_emb = rnn_model(sub_caps, sub_cap_lens, hidden)

			w_loss0, w_loss1, attn_maps = \
				words_loss(words_features, words_emb, sub_labels,
						   sub_cap_lens, sub_class_ids, sub_bat_sze)
			loss = w_loss0 + w_loss1
			w_total_loss0 += w_loss0.data
			w_total_loss1 += w_loss1.data

			s_loss0, s_loss1 = \
					sent_loss(sent_code, sent_emb,
							  sub_labels, sub_class_ids, sub_bat_sze)
			loss += s_loss0 + s_loss1
			s_total_loss0 += s_loss0.data
			s_total_loss1 += s_loss1.data

			if cfg.TRAIN.LOSS_REDUCTION == 'mean':
				# only mean-reduction needs be divided by grad_accu_steps
				loss /= accu_steps
			#
			loss.backward()

		#
		# `clip_grad_norm` helps prevent
		# the exploding gradient problem in RNNs / LSTMs.
		torch.nn.utils.clip_grad_norm(rnn_model.parameters(),
									  cfg.TRAIN.RNN_GRAD_CLIP)
		optimizer.step()

		if step % UPDATE_INTERVAL == 0:
			count = epoch * len(dataloader) + step

			s_cur_loss0 = s_total_loss0.item() / UPDATE_INTERVAL
			s_cur_loss1 = s_total_loss1.item() / UPDATE_INTERVAL

			w_cur_loss0 = w_total_loss0.item() / UPDATE_INTERVAL
			w_cur_loss1 = w_total_loss1.item() / UPDATE_INTERVAL

			elapsed = time.time() - start_time
			print('| epoch {:3d} | {:5d}/{:5d} batches | ms/batch {:5.2f} | '
				  's_loss {:5.2f} {:5.2f} | '
				  'w_loss {:5.2f} {:5.2f}'
				  .format(epoch, step, len(dataloader),
						  elapsed * 1000. / UPDATE_INTERVAL,
						  s_cur_loss0, s_cur_loss1,
						  w_cur_loss0, w_cur_loss1))
			# tmp = UPDATE_INTERVAL
			# log_step = epoch * len(dataloader) / tmp * tmp + step
			add_summary_value(tb_summary_writer,
							  'train_s_loss0', s_cur_loss0, count)
			add_summary_value(tb_summary_writer,
							  'train_s_loss1', s_cur_loss1, count)
			add_summary_value(tb_summary_writer,
							  'train_w_loss0', w_cur_loss0, count)
			add_summary_value(tb_summary_writer,
							  'train_w_loss1', w_cur_loss1, count)

			s_total_loss0 = 0
			s_total_loss1 = 0
			w_total_loss0 = 0
			w_total_loss1 = 0
			start_time = time.time()

			# # attention Maps
			# img_set, _ = \
			#     build_super_images(imgs[-1].cpu(), captions,
			#                        ixtoword, attn_maps, att_sze)
			# if img_set is not None:
			#     im = Image.fromarray(img_set)
			#     fullpath = '%s/attention_maps%d.png' % (image_dir, step)
			#     im.save(fullpath)
	return count


def evaluate(dataloader, cnn_model, rnn_model, batch_size):
	cnn_model.eval()
	rnn_model.eval()
	accu_steps = cfg.TRAIN.GRAD_ACCU_STEPS
	sub_bat_sze = int(batch_size / accu_steps)

	s_total_loss = 0
	w_total_loss = 0
	for step, data in enumerate(dataloader, 0):
		real_imgs, captions, cap_lens, \
				class_ids, keys = prepare_data(data)

		for accu_step in range(accu_steps):
			# sub_imgs = [_[accu_step * sub_bat_sze:
			# 			  (accu_step+1) * sub_bat_sze] for _ in real_imgs]
			sub_imgs = real_imgs[
				accu_step * sub_bat_sze: (accu_step+1) * sub_bat_sze
			]
			sub_caps = captions[accu_step * sub_bat_sze:
								(accu_step+1) * sub_bat_sze]
			sub_keys = keys[accu_step * sub_bat_sze:
							(accu_step+1) * sub_bat_sze]
			sub_cap_lens = cap_lens[accu_step * sub_bat_sze:
									(accu_step+1) * sub_bat_sze]
			sub_class_ids = class_ids[accu_step * sub_bat_sze:
									  (accu_step+1) * sub_bat_sze]
			sub_labels = labels[0: sub_bat_sze]

			words_features, sent_code = cnn_model(sub_imgs)
			# nef = words_features.size(1)
			# words_features = words_features.view(batch_size, nef, -1)

			hidden = rnn_model.init_hidden(sub_bat_sze)
			words_emb, sent_emb = rnn_model(sub_caps, sub_cap_lens, hidden)

			w_loss0, w_loss1, attn = \
				words_loss(words_features, words_emb, sub_labels,
						   sub_cap_lens, sub_class_ids, sub_bat_sze)
			w_total_loss += (w_loss0 + w_loss1).data

			s_loss0, s_loss1 = \
				sent_loss(sent_code, sent_emb,
						  sub_labels, sub_class_ids, sub_bat_sze)
			s_total_loss += (s_loss0 + s_loss1).data

		if step == 20:
			break

	if step > 0:
		s_cur_loss = s_total_loss.item() / step
		w_cur_loss = w_total_loss.item() / step
	else:
		s_cur_loss = s_total_loss.item()
		w_cur_loss = w_total_loss.item()

	return s_cur_loss, w_cur_loss


def build_models():
	# build model ############################################################
	text_encoder = RNN_ENCODER(dataset.n_words,
							   nhidden=cfg.TEXT.EMBEDDING_DIM,
							   pre_emb=dataset.pretrained_emb,
							   ninput=768)  
	image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
	labels = Variable(torch.LongTensor(range(batch_size)))
	start_epoch = 0
	if cfg.TRAIN.NET_E != '':
		state_dict = torch.load(cfg.TRAIN.NET_E)
		# model_dict = text_encoder.state_dict()
		# pretrained_dict = {k:v for (k,v) in state_dict.items() if k != 'encoder.weight'} # todo modify
		# model_dict.update(pretrained_dict)
		# text_encoder.load_state_dict(model_dict)
		text_encoder.load_state_dict(state_dict)
		print('Load ', cfg.TRAIN.NET_E)
		#
		name = cfg.TRAIN.NET_E.replace('text_encoder', 'image_encoder')
		state_dict = torch.load(name)
		image_encoder.load_state_dict(state_dict)
		print('Load ', name)

		# istart = cfg.TRAIN.NET_E.rfind('_') + 8
		# iend = cfg.TRAIN.NET_E.rfind('.')
		# start_epoch = cfg.TRAIN.NET_E[istart:iend]
		# start_epoch = int(start_epoch) + 1
		# print('start_epoch', start_epoch)
	if cfg.CUDA:
		text_encoder = text_encoder.cuda()
		image_encoder = image_encoder.cuda()
		labels = labels.cuda()

	return text_encoder, image_encoder, labels, start_epoch


if __name__ == "__main__":
	args = parse_args()
	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	
	args.device = 'cuda'

	gpus = [int(gpu) for gpu in args.gpu_id.split(',')]
	new_gpus = []
	for gid in gpus:
		if gid not in new_gpus:
			new_gpus.append(gid)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
	
	n_gpu = int(os.environ["WORLD_SIZE"]) # if "WORLD_SIZE" in os.environ else 1
	# print(os.environ["WORLD_SIZE"], os.environ['CUDA_VISIBLE_DEVICES'])
	assert n_gpu == len(new_gpus), (n_gpu, len(new_gpus))
	cfg.GPU_ID = new_gpus

	if args.data_dir != '':
		cfg.DATA_DIR = args.data_dir
	print('Using config:')
	pprint.pprint(cfg)

	if not cfg.TRAIN.FLAG:
		args.manualSeed = 3201
	elif args.manualSeed is None:
		args.manualSeed = random.randint(1, 10000)
	random.seed(args.manualSeed)
	np.random.seed(args.manualSeed)
	torch.manual_seed(args.manualSeed)
	torch.cuda.manual_seed_all(args.manualSeed)

	args.distributed = n_gpu > 1
	if args.distributed:
		torch.cuda.set_device(args.local_rank)
		torch.distributed.init_process_group(
			backend="nccl", 
			init_method="env://"
		)
		synchronize()

	# torch.cuda.set_device(cfg.GPU_ID)
	cudnn.benchmark = True
	cudnn.deterministic = True

	##########################################################################
	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y%m%d%H%M')
	output_dir = '../DAMSMencoders/%s_%s' % \
		(cfg.CONFIG_NAME, timestamp)

	model_dir = os.path.join(output_dir, 'Model')
	image_dir = os.path.join(output_dir, 'Image')
	mkdir_p(model_dir)
	mkdir_p(image_dir)
	shutil.copy(args.cfg_file, output_dir) # 保留配置文件

	tb_summary_writer = tf.summary.FileWriter(output_dir)


	# Get data loader ##################################################
	imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
	batch_size = cfg.TRAIN.BATCH_SIZE
	image_transform = transforms.Compose([
		transforms.Scale(int(imsize * 76 / 64)),
		transforms.RandomCrop(imsize),
		# transforms.RandomHorizontalFlip()
	])
	dataset = TextDataset(cfg.DATA_DIR, 'train',
						  base_size=cfg.TREE.BASE_SIZE,
						  transform=image_transform)
	print('n_data:', len(dataset.captions), len(dataset.filenames))
	print('n_words:', dataset.n_words)
	# print('n_sents_per_img:', dataset.embeddings_num)
	assert dataset
	dataloader = torch.utils.data.DataLoader(
		dataset, batch_size=batch_size, drop_last=True,
		shuffle=True, num_workers=int(cfg.WORKERS))

	# # validation data #
	dataset_val = TextDataset(cfg.DATA_DIR, 'test',
							  base_size=cfg.TREE.BASE_SIZE,
							  transform=image_transform)
	dataloader_val = torch.utils.data.DataLoader(
		dataset_val, batch_size=batch_size, drop_last=True,
		shuffle=True, num_workers=int(cfg.WORKERS))

	# Train ##############################################################
	text_encoder, image_encoder, labels, start_epoch = build_models()
	para = list(text_encoder.parameters())
	for v in image_encoder.parameters():
		if v.requires_grad:
			para.append(v)
	# optimizer = optim.Adam(para, lr=cfg.TRAIN.ENCODER_LR, betas=(0.5, 0.999))
	# At any point you can hit Ctrl + C to break out of training early.
	try:
		best_loss, best_ep = None, None
		lr = cfg.TRAIN.ENCODER_LR
		for epoch in range(start_epoch, cfg.TRAIN.MAX_EPOCH):
			optimizer = optim.Adam(para, lr=lr, betas=(0.5, 0.999))
			epoch_start_time = time.time()
			count = train(dataloader, image_encoder, text_encoder,
						  batch_size, labels, optimizer, epoch,
						  dataset.ixtoword, image_dir)
			print('-' * 89)
			if len(dataloader_val) > 0:
				s_loss, w_loss = evaluate(
					dataloader_val, image_encoder, text_encoder, batch_size
				)
				if not best_loss or s_loss+w_loss < best_loss:
					best_loss = s_loss + w_loss
					best_ep = epoch
					torch.save(image_encoder.state_dict(),
						   '%s/best_image_encoder.pth' % model_dir)
					torch.save(text_encoder.state_dict(),
							   '%s/best_text_encoder.pth' % model_dir)
					print('Save best encoders.')

				print('| end epoch {:3d} | '
					  'lr {:.5f} | '
					  'valid loss {:5.2f} {:5.2f} | '
					  'best loss {:.3f} ep{:3d}'
					  .format(epoch, lr,
							  s_loss, w_loss,
							  best_loss, best_ep))

				add_summary_value(tb_summary_writer,
								  'val_s_loss', s_loss, epoch)
				add_summary_value(tb_summary_writer,
								  'val_w_loss', w_loss, epoch)
				add_summary_value(tb_summary_writer,
								  'val_loss', s_loss + w_loss, epoch)
				tb_summary_writer.flush()

			print('-' * 89)
			if lr > cfg.TRAIN.ENCODER_LR/10.:
				lr *= 0.98

			if (epoch % cfg.TRAIN.SNAPSHOT_INTERVAL == 0 or
				epoch == cfg.TRAIN.MAX_EPOCH):
				torch.save(image_encoder.state_dict(),
						   '%s/image_encoder%d.pth' % (model_dir, epoch))
				torch.save(text_encoder.state_dict(),
						   '%s/text_encoder%d.pth' % (model_dir, epoch))
				print('Save G/Ds models.')
	except KeyboardInterrupt:
		print('-' * 89)
		print('Exiting from training early')
