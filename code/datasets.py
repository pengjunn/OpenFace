# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
from miscc.config import cfg

import torch
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms

import os
import sys
import json
import numpy as np
import pandas as pd
from PIL import Image
from ipdb import set_trace
from numpy import random
from random import sample
if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle
from ipdb import set_trace
from distributed import get_rank


def prepare_data(data):
	imgs, caps, cap_lens, cls_ids, keys = data

	# sort data by the length in a decreasing order
	sorted_cap_lens, sorted_cap_indices = \
		torch.sort(cap_lens, 0, True)

	real_imgs = Variable(imgs[sorted_cap_indices]).to('cuda')

	caps = caps[sorted_cap_indices].squeeze()  # sorted
	cls_ids = cls_ids[sorted_cap_indices].numpy()  # sorted
	keys = [keys[i] for i in sorted_cap_indices.numpy()]  # sorted

	caps = Variable(caps).to('cuda')
	sorted_cap_lens = Variable(sorted_cap_lens).to('cuda')

	return real_imgs, caps, sorted_cap_lens, cls_ids, keys


def get_imgs(img_path, bbox=None, transform=None, normalize=None):
	img = Image.open(img_path).convert('RGB')
	width, height = img.size
	if bbox is not None:
		r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
		center_x = int((2 * bbox[0] + bbox[2]) / 2)
		center_y = int((2 * bbox[1] + bbox[3]) / 2)
		y1 = np.maximum(0, center_y - r)
		y2 = np.minimum(height, center_y + r)
		x1 = np.maximum(0, center_x - r)
		x2 = np.minimum(width, center_x + r)
		img = img.crop([x1, y1, x2, y2])

	if transform is not None:
		img = transform(img)
	
	if normalize is not None:
		ret = normalize(img)
	
	return ret


class TextDataset(data.Dataset):
	def __init__(self, data_dir, split='train',
				 base_size=64, transform=None, target_transform=None):
		
		self.imsize = int(cfg.TREE.BASE_SIZE)

		self.transform = transforms.Compose([
			transforms.Resize(self.imsize),
			transforms.RandomHorizontalFlip()
		])
		
		self.norm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
		)])
		
		self.target_transform = target_transform

		self.bbox = None
		# self.data = {}
		self.data_dir = data_dir
		self.data = self.load_data(data_dir)
		
		if split == 'train':
			self.filenames = self.data[split]['filenames']
			self.keys = self.data[split]['keynames']
		else:
			self.filenames, self.keys = [], []
			for split in self.data.keys():
				if split != 'train':
					self.filenames += self.data[split]['filenames']
					self.keys += self.data[split]['keynames']

		self.captions, self.ixtoword, self.wordtoix, self.n_words, \
			self.pretrained_emb = self.load_text_data(data_dir, split)
		
		self.number_example = len(self.captions)
		split_dir = os.path.join(data_dir, split)
		self.class_id = self.load_class_id(split_dir, self.number_example)


	def load_captions(self, split):
		if split == 'train':
			captions = self.data['train']['captions']
		else:
			captions = []
			for split in self.data.keys():
				if split != 'train':
					captions += self.data[split]['captions']

		new_captions = []
		for idx, caption in enumerate(captions):
			# cap = caption.decode('utf8').replace("\ufffd\ufffd", " ")
			cap = caption.replace("\ufffd\ufffd", " ")
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(cap.lower())
			if len(tokens) == 0:
				key = self.keys[idx]
				imgname = self.filenames[idx]
				print('key: {}\n'
					  'img: {}\n'
					  'cap: {}'.formate(key, imgname, cap))
				continue
			tokens_new = []
			for t in tokens:
				t = t.encode('ascii', 'ignore').decode('ascii')
				if len(t) > 0:
					tokens_new.append(t)
			new_captions.append(tokens_new)
		return new_captions

	def build_dictionary(self, train_captions, test_captions):
		word_counts = defaultdict(float)
		captions = train_captions + test_captions
		for sent in captions:
			for word in sent:
				word_counts[word] += 1

		vocab = [w for w in word_counts if word_counts[w] >= 0]

		wordtoix = {
			'END': 0,
			'PAD': 1,
			'UNK': 2,
			'CLS': 3,
		}

		# pip install -U bert-serving-server bert-serving-client
		# bert-serving-start -model_dir /path/to/mode -num_worker=4
		try:
			from bert_serving.client import BertClient
			bc = BertClient()
		except:
			raise ImportError

		for w in vocab:
			if w not in wordtoix:
				wordtoix[w] = len(wordtoix)
		ixtoword = {v: k for k, v in wordtoix.items()}

		words = list(wordtoix.keys())
		pretrained_emb = bc.encode(words)
		train_captions_new = []
		for t in train_captions:
			rev = []
			for w in t:
				if w in wordtoix:
					rev.append(wordtoix[w])
			train_captions_new.append(rev)

		test_captions_new = []
		for t in test_captions:
			rev = []
			for w in t:
				if w in wordtoix:
					rev.append(wordtoix[w])
			test_captions_new.append(rev)

		pretrained_emb = np.array(pretrained_emb)

		return [train_captions_new, test_captions_new,
				ixtoword, wordtoix, len(ixtoword), pretrained_emb]

	def displays_cap_len(self, split, captions):
		print("%s captions length details:" % split)
		count = {}
		for cap in captions:
			count[len(cap)] = count.get(len(cap), 0) + 1
		summary = 0
		min_len = min(list(count.keys()))
		max_len = max(list(count.keys()))
		for len_ in range(min_len, max_len + 1):
			num = count[len_]
			summary += num
			percent = summary*100.0/len(captions)
			print('{:3d}: {:3d}({:.2f}%)'
				  .format(len_, num, percent))

	def load_text_data(self, data_dir, split):
		captions_pkl = cfg.TEXT.CAPTIONS_PKL
		assert captions_pkl != ''
		filepath = os.path.join(data_dir, captions_pkl)

		if not os.path.isfile(filepath):
			train_captions = self.load_captions('train')
			test_captions = self.load_captions('test')

			train_captions, test_captions, \
			ixtoword, wordtoix, n_words, \
			pretrained_emb = \
				self.build_dictionary(train_captions, test_captions)

			with open(filepath, 'wb') as f:
				pickle.dump([train_captions, test_captions,
							 ixtoword, wordtoix, pretrained_emb], f)
				print('Save to: ', filepath)
		else:
			with open(filepath, 'rb') as f:
				x = pickle.load(f, encoding='iso-8859-1')

				train_captions, test_captions = x[0], x[1]
				ixtoword, wordtoix = x[2], x[3]
				pretrained_emb = x[4]
				del x
				n_words = len(ixtoword)

				if get_rank() == 0:
					print('Load from: ', filepath)

		if split == 'train':
			# a list of list: each list contains
			# the indices of words in a sentence
			captions = train_captions
		else:  # split=='test'
			captions = test_captions
		# self.displays_cap_len(split, captions)
		return captions, ixtoword, wordtoix, n_words, pretrained_emb


	def load_class_id(self, data_dir, total_num):
		if os.path.isfile(data_dir + '/class_info.pickle'):
			with open(data_dir + '/class_info.pickle', 'rb') as f:
				class_id = pickle.load(f)
		else:
			class_id = np.arange(total_num)
		return class_id


	# def load_filenames(self, data_dir, split):
	# 	filepath = '%s/%s/filenames.pickle' % (data_dir, split)
	# 	if os.path.isfile(filepath):
	# 		with open(filepath, 'rb') as f:
	# 			filenames = pickle.load(f)
	# 		if get_rank() == 0:
	# 			print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
	# 	else:
	# 		filenames = []
	# 	return filenames


	# def load_keynames(self, data_dir, split):
	# 	key_path = os.path.join(data_dir, split, 'keynames.pickle')
	# 	if os.path.isfile(key_path):
	# 		with open(key_path, 'r') as f:
	# 			keynames = pickle.load(f)
	# 		if get_rank() == 0:
	# 			print('Load keynames from: %s (%d)' % (key_path, len(keynames)))
	# 	else:
	# 		keynames = []
	# 	return keynames


	def load_data(self, data_dir):
		# dataset: {'train':{'filenames':[], 'captions':[], 'keynames':[]}, 'val':{}, 'test':{}}
		data_path = os.path.join(data_dir, 'dataset.json')
		with open(data_path, 'r') as f:
			split_data = json.load(f)
		if get_rank() == 0:
			print('Load data from: %s' % data_path)
		return split_data


	def get_caption(self, sent_ix):
		# a list of indices for a sentence
		sent_caption = np.asarray(self.captions[sent_ix]).astype('int64')
		if (sent_caption == 0).sum() > 0:
			print('ERROR: do not need END (0) token', sent_caption)
		num_words = len(sent_caption)
		# pad with 0s (i.e., '<end>')
		x = np.zeros((cfg.TEXT.WORDS_NUM, 1), dtype='int64')
		x_len = num_words
		if num_words <= cfg.TEXT.WORDS_NUM:
			x[:num_words, 0] = sent_caption
		else:
			ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
			np.random.shuffle(ix)
			ix = ix[:cfg.TEXT.WORDS_NUM]
			ix = np.sort(ix)
			x[:, 0] = sent_caption[ix]
			x_len = cfg.TEXT.WORDS_NUM
		return x, x_len


	def get_grid_data(self, k):
		# a list of indices for a sentence
		indexs = sample(range(self.__len__()), k)
		imgs, caps, cap_lens, cls_ids = [], [], [], []
		keys = []
		for idx in indexs:
			img, cap, cap_len, cls_id, key = self.__getitem__(idx)
			imgs.append(img.numpy())
			caps.append(cap)
			cap_lens.append(cap_len)
			cls_ids.append(cls_id)
			keys.append(key)
		
		imgs = torch.tensor(imgs, dtype=torch.float32)
		caps = torch.tensor(caps, dtype=torch.int64)
		cap_lens = torch.tensor(cap_lens, dtype=torch.int64)
		cls_ids = torch.tensor(cls_ids, dtype=torch.int64)
		keys = tuple(keys)
		# set_trace()
		return [imgs, caps, cap_lens, cls_ids, keys]


	def __getitem__(self, index):
		key = self.keys[index]
		file = self.filenames[index]
		cls_id = self.class_id[index]

		bbox = None
		img_name = f'{self.data_dir}/{cfg.IMG_DIR}/{file}.jpg'
		img = get_imgs(img_name, self.bbox, self.transform, self.norm)

		cap, cap_len = self.get_caption(index)

		return img, cap, cap_len, cls_id, key


	def __len__(self):
		return self.number_example
