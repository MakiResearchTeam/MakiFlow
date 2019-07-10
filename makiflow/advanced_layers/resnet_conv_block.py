from makiflow.layers import Layer, ConvLayer, BatchNormLayer, ActivationLayer
from makiflow.save_recover.activation_converter import ActivationConverter
import tensorflow as tf

# ConvBlock 
class ResnetConvBlock:

	def __init__(self,in_f,out_f,activation=tf.nn.relu,name='convblock'):
		"""
		Parameters
		----------
		in_f : int
			Number of the input feature maps relative to the first convolution in the block.
		out_f : list of int
			Number of the output feature maps of the convolution in the block.
		
		Notes
		-----
			out_f is list of int(out_f[2] is FM on third Conv layer) where out_f[2] == (FM on Conv in skip connection)
			Data flow scheme: 
			(skip connection)/------------------------->(Conv 3x3)----------------------------->(BN)--------------------------->|
			input-----------|                                                                                                   |->(summ)=final_output
			(main branch)    \-->(Conv3x3)->(BN)->(ActivLayer)-->(Conv3x3)->(BN)->(ActivLayer)-->(Conv3x3)->(BN)->(ActivLayer)->|
			Where two branches are summ together:
			final_output = (skip connection) + (main branch).
		"""
		assert(len(out_f) == 3)
		self.in_f = in_f
		self.out_f = out_f
		self.f = activation
		self.name = name
		
		#Main branch
		self.Conv1 = ConvLayer(kw=3,kh=3,in_f=in_f,out_f=out_f[0],activation=None,name=name+'_Conv1')
		self.Batch1 = BatchNormLayer(D=out_f[0],name=name+'_Batch1')
		self.Activ1 = ActivationLayer(activation)

		self.Conv2 = ConvLayer(kw=3,kh=3,in_f=out_f[0],out_f=out_f[1],activation=None,name=name+'_Conv2')
		self.Batch2 = BatchNormLayer(D=out_f[1],name=name+'_Batch2')
		self.Activ2 = ActivationLayer(activation)

		self.Conv3 = ConvLayer(kw=3,kh=3,in_f=out_f[1],out_f=out_f[2],activation=None,name=name+'_Conv3')
		self.Batch3 = BatchNormLayer(D=out_f[2],name=name+'_Batch3')

		#skip branch
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
		#main branch
		for layer in self.layers[2:]:
			FX = layer.forward(FX,is_training)
		#skip branch
		SX = X
		for layer in self.layers[:2]:
			SX = layer.forward(SX,is_training)
		
		return SX + FX 

	def get_params_dict(self):
		return self.named_params_dict

	def get_params(self):
		params = []
		for layer in self.layers:
			params += layer.get_params()
		return params

	def to_dict(self):
		return {'type':'ConvBlock_resnet',
			'params':{
				'name': self.name,
				'in_f':self.in_f,
				'out_f':self.out_f,
				'activation':ActivationConverter.activation_to_str(self.f),
			}
		}
