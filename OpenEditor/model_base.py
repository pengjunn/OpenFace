# coding=utf-8
from __future__ import print_function

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchvision import models

from miscc.config import cfg
from ipdb import set_trace
from distributed import get_rank


def conv1x1(in_planes, out_planes, bias=False):
	"""1x1 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=bias)


def conv3x3(in_planes, out_planes, stride=1, bias=False):
	"""3x3 convolution with padding"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


# ############## Text2Image Encoder-Decoder #######
class RNN_ENCODER(nn.Module):
	def __init__(
		self, 
		ntoken, 
		ninput=300, 
		drop_prob=0.5,
		nhidden=128, 
		nlayers=1, 
		bidirectional=True, 
		pre_emb=None
	):

		super(RNN_ENCODER, self).__init__()
		if pre_emb is None:
			pre_emb = []
		self.n_steps = cfg.TEXT.WORDS_NUM
		self.ntoken = ntoken  # size of the dictionary  # 5450
		self.ninput = ninput  # size of each embedding vector
		self.drop_prob = drop_prob  # probability of an element to be zeroed
		self.nlayers = nlayers  # Number of recurrent layers
		self.bidirectional = bidirectional
		self.rnn_type = cfg.RNN_TYPE  # LSTM
		if bidirectional:
			self.num_directions = 2
		else:
			self.num_directions = 1
		# number of features in the hidden state
		self.nhidden = nhidden // self.num_directions  # 256 // 2

		self.define_module()
		self.init_weights(pre_emb)

	def define_module(self):
		self.encoder = nn.Embedding(self.ntoken, self.ninput)
		self.drop = nn.Dropout(self.drop_prob)
		if self.rnn_type == 'LSTM':
			# dropout: If non-zero, introduces a dropout layer on
			# the outputs of each RNN layer except the last layer
			self.rnn = nn.LSTM(self.ninput, self.nhidden,  # 300, 128
							   self.nlayers, batch_first=True,
							   dropout=self.drop_prob,
							   bidirectional=self.bidirectional)
		elif self.rnn_type == 'GRU':
			self.rnn = nn.GRU(self.ninput, self.nhidden,
							  self.nlayers, batch_first=True,
							  dropout=self.drop_prob,
							  bidirectional=self.bidirectional)
		else:
			raise NotImplementedError

	def init_weights(self, pre_emb):
		initrange = 0.1
		if cfg.TEXT.USE_PRE_EMB:
			if get_rank() == 0:
				print("Use pretrained embedding")
			self.encoder.weight.data.copy_(torch.from_numpy(pre_emb))
		else:
			self.encoder.weight.data.uniform_(-initrange, initrange)
		# Do not need to initialize RNN parameters, which have been initialized
		# http://pytorch.org/docs/master/_modules/torch/nn/modules/rnn.html#LSTM
		# self.decoder.weight.data.uniform_(-initrange, initrange)
		# self.decoder.bias.data.fill_(0)

	def init_hidden(self, bsz):
		weight = next(self.parameters()).data
		if self.rnn_type == 'LSTM':
			return (Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()),
					Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_()))
		else:
			return Variable(weight.new(self.nlayers * self.num_directions, bsz, self.nhidden).zero_())

	def forward(self, captions, cap_lens, hidden, mask=None, test=False):
		# input: torch.LongTensor of size batch x n_steps
		# --> emb: batch x n_steps x ninput
		if test:
			emb = self.drop(captions)
		else:
			emb = self.drop(self.encoder(captions))
		
		# Returns: a PackedSequence object
		cap_lens = cap_lens.data.tolist()
		if type(cap_lens).__name__ != "list":
			cap_lens = [cap_lens]
		emb = pack_padded_sequence(emb, cap_lens, batch_first=True)
		# #hidden and memory (num_layers * num_directions, batch, hidden_size):
		# tensor containing the initial hidden state for each element in batch.
		# #output (batch, seq_len, hidden_size * num_directions)
		# #or a PackedSequence object:
		# tensor containing output features (h_t) from the last layer of RNN
		output, hidden = self.rnn(emb, hidden)
		# PackedSequence object
		# --> (batch, seq_len, hidden_size * num_directions)
		output = pad_packed_sequence(output, batch_first=True)[0]  # (N,128*2,T)
		# output = self.drop(output)
		# --> batch x hidden_size*num_directions x seq_len
		words_emb = output.transpose(1, 2)  # (N,T,128*2)
		# --> batch x num_directions*hidden_size
		if self.rnn_type == 'LSTM':
			sent_emb = hidden[0].transpose(0, 1).contiguous()
		else:
			sent_emb = hidden.transpose(0, 1).contiguous()
		sent_emb = sent_emb.view(-1, self.nhidden * self.num_directions)  # (N,128*2)
		return words_emb, sent_emb


class CNN_ENCODER(nn.Module):
	def __init__(self, nef):
		super(CNN_ENCODER, self).__init__()
		if cfg.TRAIN.FLAG:
			self.nef = nef  # 256
		else:
			self.nef = 256  # define a uniform ranker

		model = models.inception_v3()
		url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
		model.load_state_dict(model_zoo.load_url(url))
		for param in model.parameters():
			param.requires_grad = False
		if get_rank() == 0:
			print('Load pretrained model from ', url)
		# print(model)

		self.define_module(model)
		self.init_trainable_weights()

	def define_module(self, model):
		self.Conv2d_1a_3x3 = model.Conv2d_1a_3x3
		self.Conv2d_2a_3x3 = model.Conv2d_2a_3x3
		self.Conv2d_2b_3x3 = model.Conv2d_2b_3x3
		self.Conv2d_3b_1x1 = model.Conv2d_3b_1x1
		self.Conv2d_4a_3x3 = model.Conv2d_4a_3x3
		self.Mixed_5b = model.Mixed_5b
		self.Mixed_5c = model.Mixed_5c
		self.Mixed_5d = model.Mixed_5d
		self.Mixed_6a = model.Mixed_6a
		self.Mixed_6b = model.Mixed_6b
		self.Mixed_6c = model.Mixed_6c
		self.Mixed_6d = model.Mixed_6d
		self.Mixed_6e = model.Mixed_6e
		self.Mixed_7a = model.Mixed_7a
		self.Mixed_7b = model.Mixed_7b
		self.Mixed_7c = model.Mixed_7c

		self.emb_features = conv1x1(768, self.nef)
		self.emb_cnn_code = nn.Linear(2048, self.nef)

	def init_trainable_weights(self):
		initrange = 0.1
		self.emb_features.weight.data.uniform_(-initrange, initrange)
		self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)

	def forward(self, x):
		features = None
		# --> fixed-size input: batch x 3 x 299 x 299
		if x.shape[2] != 299 or x.shape[3] != 299:
			x = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True)(x)
		# 299 x 299 x 3
		x = self.Conv2d_1a_3x3(x)
		# 149 x 149 x 32
		x = self.Conv2d_2a_3x3(x)
		# 147 x 147 x 32
		x = self.Conv2d_2b_3x3(x)
		# 147 x 147 x 64
		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# 73 x 73 x 64
		x = self.Conv2d_3b_1x1(x)
		# 73 x 73 x 80
		x = self.Conv2d_4a_3x3(x)
		# 71 x 71 x 192

		x = F.max_pool2d(x, kernel_size=3, stride=2)
		# 35 x 35 x 192
		x = self.Mixed_5b(x)
		# 35 x 35 x 256
		x = self.Mixed_5c(x)
		# 35 x 35 x 288
		x = self.Mixed_5d(x)
		# 35 x 35 x 288

		x = self.Mixed_6a(x)
		# 17 x 17 x 768
		x = self.Mixed_6b(x)
		# 17 x 17 x 768
		x = self.Mixed_6c(x)
		# 17 x 17 x 768
		x = self.Mixed_6d(x)
		# 17 x 17 x 768
		x = self.Mixed_6e(x)
		# 17 x 17 x 768

		# image region features
		features = x
		# 17 x 17 x 768

		x = self.Mixed_7a(x)
		# 8 x 8 x 1280
		x = self.Mixed_7b(x)
		# 8 x 8 x 2048
		x = self.Mixed_7c(x)
		# 8 x 8 x 2048
		x = F.avg_pool2d(x, kernel_size=8)
		# 1 x 1 x 2048
		# x = F.dropout(x, training=self.training)
		# 1 x 1 x 2048
		x = x.view(x.size(0), -1)
		# 2048

		# global image features
		cnn_code = self.emb_cnn_code(x)  # nef
		# 512
		if features is not None:
			features = self.emb_features(features)  # 17 x 17 x nef
		return features, cnn_code
		# local, global features


class IRBlock(nn.Module):
	expansion = 1

	def __init__(
		self, inplanes, planes, stride=1, downsample=None, use_se=True
	):
		super(IRBlock, self).__init__()
		self.bn0 = nn.BatchNorm2d(inplanes)
		self.conv1 = conv3x3(inplanes, inplanes)
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.prelu = nn.PReLU()
		self.conv2 = conv3x3(inplanes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride
		self.use_se = use_se
		if self.use_se:
			self.se = SEBlock(planes)

	def forward(self, x):
		residual = x
		out = self.bn0(x)
		out = self.conv1(out)
		out = self.bn1(out)
		out = self.prelu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		if self.use_se:
			out = self.se(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.prelu(out)

		return out


class SEBlock(nn.Module):

	def __init__(self, channel, reduction=16):
		super(SEBlock, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(channel, channel // reduction), nn.PReLU(),
			nn.Linear(channel // reduction, channel), nn.Sigmoid())

	def forward(self, x):
		b, c, _, _ = x.size()
		y = self.avg_pool(x).view(b, c)
		y = self.fc(y).view(b, c, 1, 1)
		return x * y


class ResNetFace(nn.Module):

	def __init__(self, block, layers, use_se=True):
		self.inplanes = 64
		self.use_se = use_se
		super(ResNetFace, self).__init__()
		self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.prelu = nn.PReLU()
		self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
		self.bn4 = nn.BatchNorm2d(512)
		self.dropout = nn.Dropout()
		self.fc5 = nn.Linear(512 * 8 * 8, 512)
		self.bn5 = nn.BatchNorm1d(512)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d) or isinstance(
					m, nn.BatchNorm1d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(
					self.inplanes,
					planes * block.expansion,
					kernel_size=1,
					stride=stride,
					bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)
		layers = []
		layers.append(
			block(
				self.inplanes, planes, stride, downsample, use_se=self.use_se))
		self.inplanes = planes
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, use_se=self.use_se))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.prelu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)
		x = self.bn4(x)
		x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.fc5(x)
		x = self.bn5(x)

		return x

		
def resnet_face18(use_se=True, **kwargs):
	model = ResNetFace(IRBlock, [2, 2, 2, 2], use_se=use_se, **kwargs)
	return model