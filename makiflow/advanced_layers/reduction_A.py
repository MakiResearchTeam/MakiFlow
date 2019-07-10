import tensorflow as tf
import numpy as np
from makiflow.layers import ConvLayer, ActivationLayer, MaxPoolLayer
from makiflow.save_recover.activation_converter import ActivationConverter
#FM at the end will be in_f + out_f[2] + out_f[3]

#NOTICE! what where filter is 1x1, where no fun activation otherwise Conv have activation

# Reference: https://arxiv.org/pdf/1602.07261.pdf

class Reduction_A:
								# k  l   m    n
	def __init__(self,in_f,out_f=[256,256,384,384],activation=tf.nn.relu,name='reductuon_a'):
		"""
        Parameters
        ----------
        in_f : int
            Number of the input feature maps relative to the first convolution in the block.
		out_f : list of int
			Number of the output feature maps of the convolution in the block.
		
		Notes
        -----
			Suggest what size of input picture is (W x H)
			After the passage through this block it have size as (W1 x H1), where W1=(W-3)/2 + 1, H1=(H-3)/2 + 1
			
			out_f should be something like [256,256,384,384] which are used by default
			Notice what out_f[0],out_f[1] and out_f[2],out_f[3] are similar but it is not necessary
            Data flow scheme: 
	                                                 / MaxPool--|		   /Conv(1x1)---------------------->Conv&Activ|			 /Conv&Activ|
			Conv&Activ --> Conv&Activ --> Conv&Activ| 		    |+(concat)| 										  |+(concat)| 		    |+(concat)-->output
													 \Conv&Activ|		   \Conv(1x1)-->Conv&Activ-->Conv&Activ-->Conv|			 \MaxPool-->|
            Where three branches are concate together:
            final_output = in_f + out_f[2] + out_f[3].
		"""
		assert(len(out_f) == 4)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation		
		#left branch
		self.maxPool1_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		#mid branch
		self.conv1_M_1 = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f[3],stride=2,padding='VALID',activation=activation,name=name+'conv1_M_1')

		#Right branch
		self.conv1_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv1_R_1')
		self.conv1_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=activation,name=name+'conv1_R_2')
		self.conv1_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],stride=2,padding='VALID',activation=activation,name=name+'conv1_R_3')

		self.layers = [
			self.maxPool1_L_1,
			self.conv1_M_1,
			self.conv1_R_1,self.conv1_R_2,self.conv1_R_3,
		]
		self.named_params_dict = {}
		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())

	
	def forward(self,X,is_training=False):
		FX = X

		#left branch
		LX = self.maxPool1_L_1.forward(FX,is_training)

		#Mid branch
		MX = self.conv1_M_1.forward(FX,is_training)

		#Right branch
		RX = self.conv1_R_1.forward(FX,is_training)
		RX = self.conv1_R_2.forward(RX,is_training)
		RX = self.conv1_R_3.forward(RX,is_training)

		FX = tf.concat([LX,MX,RX],axis=3)

		return FX

	def get_params_dict(self):
		return self.named_params_dict


	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
	
	def to_dict(self):
		return {'type':'Reduction_A_block',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}