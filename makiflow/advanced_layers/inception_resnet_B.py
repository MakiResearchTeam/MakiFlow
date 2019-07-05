import tensorflow as tf
import numpy as np
from makiflow.layers import ConvLayer, ActivationLayer, MaxPoolLayer
from makiflow.save_recover.activation_converter import ActivationConverter
# Reference: https://arxiv.org/pdf/1602.07261.pdf

#NOTICE! what where filter is 1x1, where no fun activation otherwise Conv have activation

#FM at the end will be out_f[3] as the in_f

class Inception_B:

	def __init__(self,in_f,out_f=[128,160,192,1152],activation=tf.nn.relu,name='inception_b'):
		"""
        Parameters
        ----------
        in_f : int
            Number of the input feature maps relative to the first convolution in the block.
		out_f : int
			Number of the output feature maps of the convolution in the block.
		
		Notes
        -----
			out_f should be something like [128,160,192,1152] which are used by default
			Notice what out_f[3] must be similar with in_f
            Data flow scheme: 
	        (left)  /->Conv(1x1)----------------------->|
			input--|									|+(concate)->Conv(1x1,Linear)=last_conv_output
			(right) \->Conv(1x1)->Conv(1x7)->Conv(7x1)->|
			Where two branches are concate together:
            final_output = input + last_conv_output.
		"""
		assert(len(out_f) == 4)
		assert(out_f[3] == in_f)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		#left branch
		self.conv1_L_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[2],activation=None,name='conv1_L_1')
		#right branch
		self.conv1_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name='conv1_R_1')
		self.conv1_R_2 = ConvLayer(kw=1,kh=7,in_f=out_f[0],out_f=out_f[1],activation=activation,name='conv1_R_2')
		self.conv1_R_3 = ConvLayer(kw=7,kh=1,in_f=out_f[1],out_f=out_f[2],activation=activation,name='conv1_R_3')

		#after connect branches
		self.conv2_af_conn = ConvLayer(kw=1,kh=1,in_f=out_f[2]*2,out_f=out_f[3],activation=None,name='conv2_af_conn')
		self.f = activation
		self.layers=[
			self.conv1_L_1,
			self.conv1_R_1,self.conv1_R_2,self.conv1_R_3,
			self.conv2_af_conn,
		]

	def forward(self,X,is_training=False):
		FX = X
		#left branch
		LX = self.conv1_L_1.forward(FX)
		#right branch
		RX = self.conv1_R_1.forward(FX)
		RX = self.conv1_R_2.forward(RX)
		RX = self.conv1_R_3.forward(RX)

		#concate
		FX = tf.concat([LX,RX],axis=3)

		FX = self.conv2_af_conn.forward(FX)

		FX = FX + X 

		if self.f is not None:
			FX = self.f(FX)
		
		return FX

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
	
	def get_params_dict(self):
		return {'type':'Stem',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}