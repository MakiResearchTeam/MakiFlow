import tensorflow as tf
import numpy as np
from makiflow.layers import ConvLayer, MaxPoolLayer
from makiflow.save_recover.activation_converter import ActivationConverter

# Reference: https://arxiv.org/pdf/1602.07261.pdf

class StemBlock(Layer):

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='stemblock'):
		"""
		Parameters
		----------
		in_f : int
			Number of the input feature maps.
		out_f : list of int
			Numbers of the outputs feature maps of the convolutions in the block.
			out_f = [
				Conv1,
				Conv2,
				Conv3,
			]
			Where ConvX*, have double number of feature maps 

		Notes
		-----
			Suggest what size of input picture is (W x H). After the passage through this block it have size as result of code below (just copy-and-paste in ipython or somethere)
			def calculate_size(W,H):
				for i in range(5):
					d = 2 if (i+1)%2 == 1 else 1
					W = int(( W - 3 ) /d) + 1 + ( (W - 3) % d)
					H = int(( H - 3 ) /d) + 1 + ( (H - 3) % d)
				return (W,H)
			Data flow scheme: 
			                      /MaxPool->|          /Conv2(1x1)------------>Conv3--------------->|          /Conv3(*)-->|
			Conv1-->Conv1-->Conv2|          |+(concat)|                                             |+(concat)|            |+(concat)-->output
			                      \Conv3--->|          \Conv2(1x1)->Conv2(7x1)->Conv2(1x7)->Conv3-->|          \MaxPool--->|
			Where two branches are concate together:
			final_output = out_f[2]*4, number of feature maps at the end.

		"""
		assert(len(out_f) == 3)
		Layer.__init__(self)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		# Main branch
		self.conv1 = ConvLayer(kw=3,kh=3,stride=2,in_f=in_f,out_f=out_f[0],padding='VALID',activation=activation,name=name+'_conv1')
		self.conv2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[0],padding='VALID',activation=activation,name=name+'_conv2')
		self.conv3 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=activation,name=name+'conv3')

		# Split 1
		# Part left  
		self.maxpool1_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		# part right
		self.conv3_R_1 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],stride=2,padding='VALID',activation=activation,name=name+'_conv3_R_1')
		in_f_split_1 = out_f[1] + out_f[2]
		# Concat

		# Split 2
		# Part left
		self.conv4_L_1 = ConvLayer(kw=1,kh=1,in_f=in_f_split_1,out_f=out_f[1],activation=None,name=name+'_conv4_L_1')
		self.conv4_L_2 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],padding='VALID',activation=activation,name=name+'_conv4_L_2')
		# Part right
		self.conv4_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f_split_1,out_f=out_f[1],activation=None,name=name+'_conv4_R_1')
		self.conv4_R_2 = ConvLayer(kw=7,kh=1,in_f=out_f[1],out_f=out_f[1],activation=activation,name=name+'_conv4_R_2')
		self.conv4_R_3 = ConvLayer(kw=1,kh=7,in_f=out_f[1],out_f=out_f[1],activation=activation,name=name+'_conv4_R_3')
		self.conv4_R_4 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],padding='VALID',activation=activation,name=name+'_conv4_R_4')
		# Concat
		in_f_split_2 = out_f[2]*2

		# Split 3 
		# Part left
		self.conv5_L_1 = ConvLayer(kw=3,kh=3,in_f=in_f_split_2,out_f=in_f_split_2,stride=2,padding='VALID',activation=activation,name=name+'_conv5_L_1')
		# Part right
		self.maxpool5_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		# Concat

		self.layers = [
			self.conv1,
			self.conv2,
			self.conv3,
			self.maxpool1_L_1,
			self.conv3_R_1,
			self.conv4_L_1,self.conv4_L_2,
			self.conv4_R_1,self.conv4_R_2,self.conv4_R_3,self.conv4_R_4,
			self.conv5_L_1,
			self.maxpool5_L_1,
		]

		self.named_params_dict = {}

		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())


	def forward(self,X,is_training=False):
		FX = X
		FX = self.conv1.forward(FX)
		FX = self.conv2.forward(FX)
		FX = self.conv3.forward(FX)

		# Split 1 
		SX = FX 

		# Left
		SX = self.maxpool1_L_1.forward(SX)
		# Right
		FX = self.conv3_R_1.forward(FX)
		# Concat branch 1
		FX = tf.concat([FX,SX],axis=3)

		# Split 2
		SX = FX
		# Left
		SX = self.conv4_L_1.forward(SX)
		SX = self.conv4_L_2.forward(SX)
		# Right
		FX = self.conv4_R_1.forward(FX)
		FX = self.conv4_R_2.forward(FX)
		FX = self.conv4_R_3.forward(FX)
		FX = self.conv4_R_4.forward(FX)
		# Concat branch 2
		FX = tf.concat([FX,SX],axis=3)

		# Split 3 

		SX = FX
		# Left
		SX = self.conv5_L_1.forward(SX)
		# Right
		FX = self.maxpool5_L_1.forward(FX)
		# Concat branch 3
		FX = tf.concat([FX,SX],axis=3)

		return FX

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
	
	def get_params_dict(self):
		return self.named_params_dict


	def to_dict(self):
		return {
			'type': 'StemBlock',
			'params':{
				'name': self.name,
				'in_f': self.in_f,
				'out_f': self.out_f,
				'activation': ActivationConverter.activation_to_str(self.f),
			}
		}
	
