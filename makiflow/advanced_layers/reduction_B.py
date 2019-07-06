import tensorflow as tf
import numpy as np
from makiflow.layers import ConvLayer, ActivationLayer, MaxPoolLayer
from makiflow.save_recover.activation_converter import ActivationConverter
#FM at the end will be in_f + out_f[3] + out_f[1] + out_f[3]
#conv kw, kh, in_f, out_f, name, stride=1, padding='SAME', activation'
#pool ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'

class Reduction_B:

	def __init__(self,in_f,out_f=[256,288,320,384],activation=tf.nn.relu,name='reduction_b'):
		"""
        Parameters
        ----------
        in_f : int
            Number of the input feature maps relative to the first convolution in the block.
		out_f : int
			Number of the output feature maps of the convolution in the block.
		
		Notes
        -----
			Suggest what size of input picture is (W x H)
			After the passage through this block it have size as (W1 x H1), where W1=(W-3)/2 + 1, H1=(H-3)/2 + 1
			
			out_f should be something like [256,288,320,384] which are used by default
            Data flow scheme: 
	        (left)		 /-->MaxPool(3x3,stride 2 V)->|
			(mid-left)  |/-->Conv(1x1)->Conv--------->|
			input------>|							  |+(concate)=final_output
			(mid-right)	|\-->Conv(1x1)->Conv--------->|
			(right)		 \-->Conv(1x1)->Conv->Conv--->|

			Where four branches are concate together:
            final_output = (left) + (mid-left) + (mid-right) + (right).
			number of feature maps at the end will be f_end = in_f + out_f[3]+out_f[1],out_f[2]
		"""
		assert(len(out_f) == 4)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		#left branch
		self.maxPool_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')

		#Mid-Left branch
		self.conv_ML_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_ML_1')
		self.conv_ML_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[3],stride=2,padding='VALID',activation=activation,name=name+'conv_ML_2')

		#Mid-Right branch
		self.conv_MR_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_MR_1')
		self.conv_MR_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],stride=2,padding='VALID',activation=activation,name=name+'conv_MR_2')

		#Right branch
		self.conv_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_R_1')
		self.conv_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=activation,name=name+'conv_R_2')
		self.conv_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=activation,stride=2,padding='VALID',name=name+'conv_R_3')

		self.layers = [
			self.maxPool_L_1,
			self.conv_ML_1,self.conv_ML_2,
			self.conv_MR_1,self.conv_MR_2,
			self.conv_R_1,self.conv_R_2,self.conv_R_3,
		]

		self.named_params_dict = {}
		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())


	def forward(self,X,is_training=False):
		FX = X

		#left branch
		LX = self.maxPool_L_1.forward(FX)

		#Mid-Left branch
		MLX = self.conv_ML_1.forward(FX)
		MLX = self.conv_ML_2.forward(MLX)

		#Mid-Right branch
		MRX = self.conv_MR_1.forward(FX)
		MRX = self.conv_MR_2.forward(MRX)

		#Right branch
		RX = self.conv_R_1.forward(FX)
		RX = self.conv_R_2.forward(RX)
		RX = self.conv_R_3.forward(RX)

		#concate branches

		FX = tf.concat([LX,MLX,MRX,RX],axis=3)

		return FX

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params
	
	def get_params_dict(self):
		return self.named_params_dict

	def to_dict(self):
		return {'type':'Reduction_B_block',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}