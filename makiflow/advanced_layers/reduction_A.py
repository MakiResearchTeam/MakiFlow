import tensorflow as tf
import numpy as np
from makiflow.layers import MakiLayer,ConvLayer, ActivationLayer, MaxPoolLayer, BatchNormLayer
from makiflow.save_recover.activation_converter import ActivationConverter

# Reference: https://arxiv.org/pdf/1602.07261.pdf page 4, Figure 7

class ReductionA(MakiLayer):

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='reductuon_a'):
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
			(left)       /------------->MaxPool-------->|
			(mid)input->|-------------->Conv4---------->|+(concate)=final_output
			(right)      \-->Conv1(1x1)->Conv2->Conv3-->|
			Where three branches are concate together:
			final_output = in_f + out_f[2] + out_f[3].
		"""
		assert(len(out_f) == 4)
		MakiLayer.__init__(self)
		self.name = name
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation

		self.layers = []	
		
		# Left branch
		self.maxPool1_L_1 = MaxPoolLayer(ksize=[1,3,3,1],strides=[1,2,2,1],padding='VALID')
		self.layers += [self.maxPool1_L_1]
		# Mid branch
		self.conv1_M_1 = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f[3],stride=2,padding='VALID',activation=None,name=name+'conv1_M_1')
		self.batch_norm1_M_1 = BatchNormLayer(D=out_f[3],name=name+'_batch_norm1_M_1')
		self.activ1_M_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_M_1,self.batch_norm1_M_1,self.activ1_M_1]

		# Right branch
		self.conv1_R_1 = ConvLayer(kw=1,kh=1,in_f=in_f,out_f=out_f[0],activation=None,name=name+'conv1_R_1')
		self.batch_norm1_R_1 = BatchNormLayer(D=out_f[0],name=name+'_batch_norm1_R_1')
		self.activ1_R_1 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_1,self.batch_norm1_R_1,self.activ1_R_1]

		self.conv1_R_2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=None,name=name+'conv1_R_2')
		self.batch_norm1_R_2 = BatchNormLayer(D=out_f[1],name=name+'_batch_norm1_R_2')
		self.activ1_R_2 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_2,self.batch_norm1_R_2,self.activ1_R_2]

		self.conv1_R_3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],stride=2,padding='VALID',activation=None,name=name+'conv1_R_3')
		self.batch_norm1_R_3 = BatchNormLayer(D=out_f[2],name=name+'_batch_norm1_R_3')
		self.activ1_R_3 = ActivationLayer(activation=activation)
		self.layers += [self.conv1_R_3,self.batch_norm1_R_3,self.activ1_R_3]

		self.named_params_dict = {}

		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())

	
	def forward(self,X,is_training=False):
		FX = X

		# Left branch
		LX = self.maxPool1_L_1.forward(FX,is_training)

		# Mid branch
		MX = self.conv1_M_1.forward(FX,is_training)
		MX = self.batch_norm1_M_1.forward(MX,is_training)
		MX = self.activ1_M_1.forward(MX,is_training)

		# Right branch
		RX = self.conv1_R_1.forward(FX,is_training)
		RX = self.batch_norm1_R_1.forward(RX,is_training)
		RX = self.activ1_R_1.forward(RX,is_training)
	
		RX = self.conv1_R_2.forward(RX,is_training)
		RX = self.batch_norm1_R_2.forward(RX,is_training)
		RX = self.activ1_R_2.forward(RX,is_training)

		RX = self.conv1_R_3.forward(RX,is_training)
		RX = self.batch_norm1_R_3.forward(RX,is_training)
		RX = self.activ1_R_3.forward(RX,is_training)

		# Concat branches
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
		return {
			'type':'ReductionBlockA',
			'params':{
				'name': self.name,
				'in_f': self.in_f,
				'out_f': self.out_f,
				'activation': ActivationConverter.activation_to_str(self.f),
			}
		}
	
