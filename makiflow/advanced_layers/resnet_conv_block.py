from makiflow.layers import Layer, ConvLayer, BatchNormLayer, ActivationLayer
from makiflow.save_recover.activation_converter import ActivationConverter
import tensorflow as tf

# Reference: https://www.udemy.com/advanced-computer-vision/ 

# ConvBlock 
class ResnetConvBlock(Layer):

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='convblock'):
		"""
		Parameters
		----------
		in_f : int
			Number of the input feature maps.
		out_f : list of int
			Numbers of the outputs feature maps of the convolutions in the block.
			out_f = [
				(main branch) Conv1,
				(main branch) Conv2,
				(skip and main branch) Conv3,
			]

		Notes
		-----
			Data flow scheme: 
			(skip connection)/------------------------->(Conv3)--------------------------->(BN)-------------------------->|
			input-----------|                                                                                             |->(summ)=final_output
			(main branch)    \---->(Conv1)--->(BN)--->(ActivLayer)---->(Conv2)--->(BN)-->(ActivLayer)--->(Conv3)-->(BN)-->|
			Where two branches are summ together:
			final_output = (skip connection) + (main branch).
		"""
		assert(len(out_f) == 3)
		Layer.__init__(self)
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		self.name = name
		
		# Main branch
		self.Conv1 = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f[0],activation=None,name=name+'_Conv1')
		self.Batch1 = BatchNormLayer(D=out_f[0],name=name+'_Batch1')
		self.Activ1 = ActivationLayer(activation)

		self.Conv2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=None,name=name+'_Conv2')
		self.Batch2 = BatchNormLayer(D=out_f[1],name=name+'_Batch2')
		self.Activ2 = ActivationLayer(activation)

		self.Conv3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=None,name=name+'_Conv3')
		self.Batch3 = BatchNormLayer(D=out_f[2],name=name+'_Batch3')

		# Skip branch
		self.Conv0 = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f[2],activation=None,name=name+'_Conv0')
		self.Batch0 = BatchNormLayer(D=out_f[2],name=name+'_Batch0')

		self.layers = [
			self.Conv0, self.Batch0,
			self.Conv1,self.Batch1,self.Activ1, self.Conv2,self.Batch2,self.Activ2, self.Conv3,self.Batch3,
		]

		self.named_params_dict = {}

		for layer in self.layers:
			self.named_params_dict.update(layer.get_params_dict())

		
	def forward(self,X,is_training=False):
		FX = X
		# Main branch
		FX = self.Conv1.forward(FX,is_training)
		FX = self.Batch1.forward(FX,is_training)
		FX = self.Activ1.forward(FX,is_training)
		
		FX = self.Conv2.forward(FX,is_training)
		FX = self.Batch2.forward(FX,is_training)
		FX = self.Activ2.forward(FX,is_training)
		
		FX = self.Conv3.forward(FX,is_training)
		FX = self.Batch3.forward(FX,is_training)
		FX = self.Activ1.forward(FX,is_training)

		# Skip branch
		SX = X
		SX = self.Conv0.forward(SX,is_training)
		SX = self.Batch0.forward(SX,is_training)
		
		return SX + FX 

	def get_params_dict(self):
		return self.named_params_dict

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params

	def to_dict(self):
		return {
			'type': 'ResnetConvBlock',
			'params':{
				'name': self.name,
				'in_f': self.in_f,
				'out_f': self.out_f,
				'activation': ActivationConverter.activation_to_str(self.f),
			}
		}
