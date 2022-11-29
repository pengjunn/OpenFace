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
from bert_serving.client import BertClient
if sys.version_info[0] == 2:
	import cPickle as pickle
else:
	import pickle
from ipdb import set_trace
from distributed import get_rank


def prepare_data(data):
	imgs, caps, cap_lens, mani_caps, mani_cap_lens, cls_ids, keys = data

	# sort data by the length in a decreasing order
	s_cap_lens, s_cap_idx = torch.sort(cap_lens, 0, True)
	s_caps = caps[s_cap_idx].squeeze()  # sorted
	
	cls_ids = cls_ids[s_cap_idx].numpy()  # sorted
	keys = [keys[i] for i in s_cap_idx.numpy()]  # sorted
	# print('keys', type(keys), keys[-1])  # list

	real_imgs = Variable(imgs[s_cap_idx]).to('cuda')
	s_cap_lens = Variable(s_cap_lens).to('cuda')
	s_caps = Variable(s_caps).to('cuda')

	# sort manipulation captions
	s_m_cap_lens, s_m_cap_idx = torch.sort(mani_cap_lens, 0, True)
	s_m_caps = mani_caps[s_m_cap_idx].squeeze()  # sorted

	s_m_cap_lens = Variable(s_m_cap_lens).to('cuda')
	s_m_caps = Variable(s_m_caps).to('cuda')
	

	return real_imgs, s_caps, s_cap_lens, s_m_caps, s_m_cap_lens, \
		cls_ids, keys, s_cap_idx, s_m_cap_idx


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

	ret = normalize(img)
	return ret


class TextDataset(data.Dataset):
	def __init__(self, data_dir, split='train',
				 base_size=64, transform=None, target_transform=None):
		
		self.imsize = int(cfg.TREE.BASE_SIZE)

		self.transform = transforms.Compose([
			transforms.Resize(self.imsize),
			transforms.RandomHorizontalFlip() 	# RandomHorizontalFilp(p=0.5)
		])

		self.norm = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
		
		self.target_transform = target_transform
		
		self.bbox = None
		# self.data = {}
		self.data_dir = data_dir

		self.data, self.data_mani = self.load_data(data_dir)

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
		
		self.mani_captions = self.load_text_data_mani(split)

		self.number_example = len(self.captions)
		split_dir = os.path.join(data_dir, split)
		self.class_id = self.load_class_id(split_dir, self.number_example)

		self.examples = self.load_examples(data_dir)

	def load_bbox(self):
		data_dir = self.data_dir
		bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
		df_bounding_boxes = pd.read_csv(bbox_path,
										delim_whitespace=True,
										header=None).astype(int)
		#
		filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
		df_filenames = \
			pd.read_csv(filepath, delim_whitespace=True, header=None)
		filenames = df_filenames[1].tolist()
		print('Total filenames: ', len(filenames), filenames[0])
		#
		filename_bbox = {img_file[:-4]: [] for img_file in filenames}
		numImgs = len(filenames)
		for i in xrange(0, numImgs):
			# bbox = [x-left, y-top, width, height]
			bbox = df_bounding_boxes.iloc[i][1:].tolist()

			key = filenames[i][:-4]
			filename_bbox[key] = bbox
		#
		return filename_bbox

	def tokenize(self, caption):
		cap = caption.replace("\ufffd\ufffd", " ")
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(cap.lower())
		assert len(tokens) > 0, 'NULL Cap'
		
		tokens_new, zero_shot_tokens = [], []
		for t in tokens:
			t = t.encode('ascii', 'ignore').decode('ascii')
			if len(t) > 0:
				tokens_new.append(t)
				if t not in self.wordtoix.keys():
					zero_shot_tokens.append(t)
					
		return tokens_new, zero_shot_tokens
		

	def load_captions(self, split):
		if split == 'train':
			captions = self.data['train']['captions']
		else:
			captions = []
			for split in self.data.keys():
				if split != 'train':
					captions += self.data[split]['captions']

		new_captions = self.process_caps(captions)
		
		return new_captions

	def process_caps(self, captions):
		new_captions = []
		for idx, caption in enumerate(captions):
			# cap = caption.decode('utf8').replace("\ufffd\ufffd", " ")
			cap = caption.replace("\ufffd\ufffd", " ")
			tokenizer = RegexpTokenizer(r'\w+')
			tokens = tokenizer.tokenize(cap.lower())
			if len(tokens) == 0:
				res_id = self.keynames[idx]
				imgname = self.filenames[idx]
				print('res_id: {}\n'
					  'img: {}\n'
					  'cap: {}'.formate(res_id, imgname, cap))
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
				# if self.use_glove:
				#     pretrained_emb.append(spacy_tool(w).vector)
		ixtoword = {v: k for k, v in wordtoix.items()}

		words = wordtoix.keys()
		pretrained_emb = bc.encode(words) # todo modifications1
		train_captions_new = []
		for t in train_captions:
			rev = []
			for w in t:
				if w in wordtoix:
					rev.append(wordtoix[w])
			# rev.append(0)  # do not need '<end>' token
			train_captions_new.append(rev)

		test_captions_new = []
		for t in test_captions:
			rev = []
			for w in t:
				if w in wordtoix:
					rev.append(wordtoix[w])
			# rev.append(0)  # do not need '<end>' token
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
		for length, num in count.items():
			summary += num
			percent = summary*100.0/len(captions)
			print('{:3d}: {:3d}({:.2f}%)'
				  .format(length, num, percent))

	def load_text_data(self, data_dir, split):
		captions_pkl = cfg.TEXT.CAPTIONS_PKL
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


	def load_text_data_mani(self, split):
		captions_pkl = cfg.TEXT.CAPTIONS_PKL.replace(
			'.pickle', '_mani3.pickle'
		)
		filepath = os.path.join(self.data_dir, captions_pkl)

		if not os.path.isfile(filepath):
			train_manis = self.data_mani['train']['mani_caps']
			test_manis = self.data_mani['test']['mani_caps']

			train_manis = self.process_caps(train_manis)
			test_manis = self.process_caps(test_manis)

			train_manis_new = []
			for t in train_manis:
				rev = []
				for w in t:
					if w in self.wordtoix:
						rev.append(self.wordtoix[w])
				# rev.append(0)  # do not need '<end>' token
				train_manis_new.append(rev)

			test_manis_new = []
			for t in test_manis:
				rev = []
				for w in t:
					if w in self.wordtoix:
						rev.append(self.wordtoix[w])
				# rev.append(0)  # do not need '<end>' token
				test_manis_new.append(rev)

			with open(filepath, 'wb') as f:
				pickle.dump([train_manis_new, test_manis_new], f)
				print('Save to: ', filepath)
		
		else:
			with open(filepath, 'rb') as f:
				x = pickle.load(f)
				train_manis_new, test_manis_new = x[0], x[1]
				del x

			if get_rank() == 0:
				print('Load from: ', filepath)
		
		if split == 'train':
			mani_caps = train_manis_new
		else:
			mani_caps = test_manis_new

		return mani_caps


	def load_examples(self, data_dir):
		with open(os.path.join(data_dir, 'examples.json')) as f:
			examples = json.load(f)
		src_caps = self.process_caps(examples['source'])
		tar_caps = self.process_caps(examples['target'])

		src_caps_new = []
		for t in src_caps:
			rev = []
			for w in t:
				if w in self.wordtoix:
					rev.append(self.wordtoix[w])
			# rev.append(0)  # do not need '<end>' token
			src_caps_new.append(rev)

		tar_caps_new = []
		for t in tar_caps:
			rev = []
			for w in t:
				if w in self.wordtoix:
					rev.append(self.wordtoix[w])
			# rev.append(0)  # do not need '<end>' token
			tar_caps_new.append(rev)
		src_cap_max_len = max([len(cap) for cap in src_caps_new])
		tar_cap_max_len = max([len(cap) for cap in tar_caps_new])
		
		src_caps, tar_caps = [], []
		src_caplens, tar_caplens = [], []
		for src, tar in zip(src_caps_new, tar_caps_new):
			src_cap, src_caplen = self.get_caption(
				src, words_num=src_cap_max_len
			)
			tar_cap, tar_caplen = self.get_caption(
				tar, words_num=tar_cap_max_len
			)
			
			src_caps.append(
				torch.tensor(src_cap.transpose(), dtype=torch.int64)
			)
			tar_caps.append(
				torch.tensor(tar_cap.transpose(), dtype=torch.int64)
			)
			src_caplens.append(src_caplen)
			tar_caplens.append(tar_caplen)

		src_caps = torch.stack(src_caps, dim=0).to('cuda')
		tar_caps = torch.stack(tar_caps, dim=0).to('cuda')
		src_caplens = torch.tensor(src_caplens, dtype=torch.int64).to('cuda')
		tar_caplens = torch.tensor(tar_caplens, dtype=torch.int64).to('cuda')

		res = {
			'src':src_caps, 'src_len':src_caplens, 
			'tar':tar_caps, 'tar_len':tar_caplens
		}
		return res


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
		data_path = os.path.join(data_dir, 'dataset2.json')
		with open(data_path, 'r') as f:
			data = json.load(f)
		
		data_mani_path = data_path.replace('.json', '_mani3.json')
		with open(data_mani_path, 'r') as f:
			data_mani = json.load(f)

		if get_rank() == 0:
			print('Load data from: %s' % data_path)
		
		return data, data_mani


	def get_caption(self, caption, words_num=None):
		if words_num is None:
			words_num = cfg.TEXT.WORDS_NUM

		# a list of indices for a sentence
		sent_caption = np.asarray(caption).astype('int64')
		if (sent_caption == 0).sum() > 0:
			print('ERROR: do not need END (0) token', sent_caption)
		num_words = len(sent_caption)
		# pad with 0s (i.e., '<end>')
		x = np.zeros((words_num, 1), dtype='int64')
		x_len = num_words
		if num_words <= words_num:
			x[:num_words, 0] = sent_caption
		else:
			ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
			np.random.shuffle(ix)
			ix = ix[:words_num]
			ix = np.sort(ix)
			x[:, 0] = sent_caption[ix]
			x_len = words_num
		return x, x_len


	def get_grid_data(self, k):
		# a list of indices for a sentence
		indexs = sample(range(self.__len__()), k)
		imgs, caps_src, caplens_src, caps_tar, caplens_tar = [], [], [], [], []
		cls_ids, keys = [], []
		for idx in indexs:
			img, cap_src, caplen_src, cap_tar, caplen_tar, cls_id, key = \
				self.__getitem__(idx)
			
			imgs.append(img)
			caps_src.append(torch.from_numpy(cap_src))
			caps_tar.append(torch.from_numpy(cap_tar))
			caplens_src.append(caplen_src)
			caplens_tar.append(caplen_tar)
			
			cls_ids.append(cls_id)
			keys.append(key)
		
		# imgs = torch.tensor(imgs, dtype=torch.float32)
		imgs = torch.stack(imgs, dim=0)
		caps_src = torch.cat(caps_src, dim=1).transpose(0, 1)
		caps_tar = torch.cat(caps_tar, dim=1).transpose(0, 1)
		caplens_src = torch.tensor(caplens_src, dtype=torch.int32)
		caplens_tar = torch.tensor(caplens_tar, dtype=torch.int32)
		cls_ids = torch.tensor(cls_ids, dtype=torch.int32)
		keys = tuple(keys)
		
		return [imgs, caps_src, caplens_src, caps_tar, caplens_tar, cls_ids, keys]


	def __getitem__(self, index):
		key = self.filenames[index]
		cls_id = self.class_id[index]

		bbox = None
		img_name = f'{self.data_dir}/{cfg.IMG_DIR}/{self.filenames[index]}.jpg'
		img = get_imgs(img_name, bbox, self.transform, self.norm)

		cap, cap_len = self.get_caption(self.captions[index])
		mani_cap, mani_cap_len = self.get_caption(
			self.mani_captions[index], words_num=8
		)

		return img, cap, cap_len, mani_cap, mani_cap_len, cls_id, key


	def __len__(self):
		return self.number_example
