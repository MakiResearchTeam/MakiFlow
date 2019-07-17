import tensorflow as tf
import numpy as np
from makiflow.layers import Layer,ConvLayer, ActivationLayer, MaxPoolLayer, BatchNormLayer
from makiflow.save_recover.activation_converter import ActivationConverter

# Reference: https://arxiv.org/pdf/1602.07261.pdf page 8, Figure 18

class ReductionB(Layer):

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='reduction_b'):
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
				Conv4,
			]
		
		Notes
		-----
			Suggest what size of input picture is (W x H). After the passage through this block it have size as (W1 x H1), where W1=(W-3)/2 + 1, H1=(H-3)/2 + 1
			Data flow scheme: 
			(left)       /-------------->MaxPool------->|
			(mid-left)  |/---->Conv1(1x1)---->Conv4---->|
			input------>|                               |+(concate)=final_output
			(mid-right) |\-->Conv1(1x1)->Conv2--------->|
			(right)      \-->Conv1(1x1)->Conv2->Conv3-->|
			Where four branches are concate together:
			final_output = in_f + out_f[3] + out_f[1] + out_f[2]

		"""
		assert(len(out_f) == 4)
		Layer.__init__(self)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation

		self.layers = []

		# Left branch
		self.maxPool_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		self.layers += [self.maxPool_L_1]

		# Mid-Left branch
		self.conv_ML_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_ML_1')
		self.batch_norm_ML_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm_ML_1')
		self.activ_ML_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv_ML_1,self.batch_norm_ML_1,self.activ_ML_1]

		self.conv_ML_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[3],stride=2,padding='VALID',activation=None,name=name+'conv_ML_2')
		self.batch_norm_ML_2 = BatchNormLayer(D=out_f[3],name=name+'_batch_norm_ML_2')
		self.activ_ML_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv_ML_2,self.batch_norm_ML_2,self.activ_ML_2]
		
		# Mid-Right branch
		self.conv_MR_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_MR_1')
		self.batch_norm_MR_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm_MR_1')
		self.activ_MR_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv_MR_1,self.batch_norm_MR_1,self.activ_MR_1]
		
		self.conv_MR_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],stride=2,padding='VALID',activation=None,name=name+'conv_MR_2')
		self.batch_norm_MR_2 = BatchNormLayer(D=out_f[1],name=name+'_batch_norm_MR_2')
		self.activ_MR_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv_MR_2,self.batch_norm_MR_2,self.activ_MR_2]
		
		# Right branch
		self.conv_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv_R_1')
		self.batch_norm_R_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm_R_1')
		self.activ_R_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv_R_1,self.batch_norm_R_1,self.activ_R_1]
		
		self.conv_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=None,name=name+'conv_R_2')
		self.batch_norm_R_2 = BatchNormLayer(D=out_f[1],name=name+'_batch_norm_R_2')
		self.activ_R_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv_R_2,self.batch_norm_R_2,self.activ_R_2]
		
		self.conv_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=None,stride=2,padding='VALID',name=name+'conv_R_3')
		self.batch_norm_R_3 = BatchNormLayer(D=out_f[2],name=name+'_batch_norm_R_3')
		self.activ_R_3 = ActivationLayer(activation=activation)
		self.layers += [self.conv_R_3,self.batch_norm_R_3,self.activ_R_3]
		
		self.named_params_dict = {}

		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())


	def forward(self,X,is_training=False):
		FX = X

		# Left branch
		LX = self.maxPool_L_1.forward(FX,is_training)

		# Mid-Left branch
		MLX = self.conv_ML_1.forward(FX,is_training)
		MLX = self.batch_norm_ML_1.forward(MLX,is_training)
		MLX = self.activ_ML_1.forward(MLX,is_training)

		MLX = self.conv_ML_2.forward(MLX,is_training)
		MLX = self.batch_norm_ML_2.forward(MLX,is_training)
		MLX = self.activ_ML_2.forward(MLX,is_training)

		# Mid-Right branch
		MRX = self.conv_MR_1.forward(FX,is_training)
		MRX = self.batch_norm_MR_1.forward(MRX,is_training)
		MRX = self.activ_MR_1.forward(MRX,is_training)
		
		MRX = self.conv_MR_2.forward(MRX,is_training)
		MRX = self.batch_norm_MR_2.forward(MRX,is_training)
		MRX = self.activ_MR_2.forward(MRX,is_training)

		# Right branch
		RX = self.conv_R_1.forward(FX,is_training)
		RX = self.batch_norm_R_1.forward(RX,is_training)
		RX = self.activ_R_1.forward(RX,is_training)

		RX = self.conv_R_2.forward(RX,is_training)
		RX = self.batch_norm_R_2.forward(RX,is_training)
		RX = self.activ_R_2.forward(RX,is_training)

		RX = self.conv_R_3.forward(RX,is_training)
		RX = self.batch_norm_R_3.forward(RX,is_training)
		RX = self.activ_R_3.forward(RX,is_training)

		# Concate branches

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
		return {
			'type':'ReductionBlockB',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}
	
