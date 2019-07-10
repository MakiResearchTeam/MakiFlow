import tensorflow as tf
import numpy as np
from makiflow.layers import ConvLayer, ActivationLayer
from makiflow.save_recover.activation_converter import ActivationConverter
#FM at the end will be out_f[3], same as in_f


#NOTICE! what where filter is 1x1, where no fun activation otherwise Conv have activation

# Reference: https://arxiv.org/pdf/1602.07261.pdf

class Inception_A:

	def __init__(self,in_f,out_f=[32,48,64,384],activation=tf.nn.relu,name='inception_a'):
		"""
        Parameters
        ----------
        in_f : int
            Number of the input feature maps relative to the first convolution in the block.
		out_f : list of int
			Number of the output feature maps of the convolution in the block.
		
		Notes
        -----
			out_f should be something like [32,48,64,384] which are used by default
            Data flow scheme: 
	        		/->(left) Conv(1x1)---------------|
			input->|->(mid) Conv(1x1)->Conv-----------|+(concate)->Conv(1x1)=last_conv_output
					\->(right) Conv(1x1)->Conv->Conv--|
			Where two branches are concate together:
            final_output = input + last_conv_output.
		"""
		assert(len(out_f) == 4)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		#left branch
		self.conv1_L_1 = ConvLayer(kw=1,kh=1,in_f=out_f[3],out_f=out_f[0],activation=None,name=name+'_conv1_L_1')
		
		#mid branch
		self.conv1_M_1 = ConvLayer(kw=1,kh=1,in_f=out_f[3],out_f=out_f[0],activation=None,name=name+'_conv1_M_1')
		self.conv1_M_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[0],activation=activation,name=name+'_conv1_M_2')
		#right branch
		self.conv1_R_1 =  ConvLayer(kw=1,kh=1,in_f=out_f[3],out_f=out_f[0],activation=None,name=name+'_conv1_R_1')
		self.conv1_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=activation,name=name+'_conv1_R_2')
		self.conv1_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=activation,name=name+'_conv1_R_3')

		#after connect three branch
		self.conv2_connect = ConvLayer(kw=1,kh=1,in_f=out_f[0]*2+out_f[2],out_f=out_f[3],activation=None,name=name+'_conv2_connect')

		self.layers = [
			self.conv1_L_1,
			self.conv1_M_1,self.conv1_M_2,
			self.conv1_R_1,self.conv1_R_2,self.conv1_R_3,
			self.conv2_connect,
		]

		self.named_params_dict = {}
		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())
	
	def forward(self,X,is_training=False):
		FX = X

		#Left branch
		LX = self.conv1_L_1.forward(FX)

		#Mid branch
		MX = self.conv1_M_1.forward(FX)
		MX = self.conv1_M_2.forward(MX)

		#Right branch
		RX = self.conv1_R_1.forward(FX)
		RX = self.conv1_R_2.forward(RX)
		RX = self.conv1_R_3.forward(RX)

		#connect branches

		FX = tf.concat([LX,MX,RX],axis=3)

		#connect tree conv

		FX = self.conv2_connect.forward(FX)

		#sum with skip connection

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
		return self.named_params_dict


	def to_dict(self):
		return {'type':'Inception_resnet_A_Block',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}



		