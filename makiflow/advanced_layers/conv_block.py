from makiflow.layers import MakiLayer

# Reference: https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf


class ConvBlock(MakiLayer):

	def __init__(self,skip_branch,main_branch,name='ConvBlock_'):
		"""
		Parameters
		----------
		main_branch : list
			List of different layers performed on the main branch.

		skip_branch : list
			List of different layers performed on the skip branch
			 
			WARNING! OUTPUT TENSOR OF MAIN BRANCH MUST BE THE SAME SHAPE AS THE SKIP BRANCH!
			
		"""
		MakiLayer.__init__(self)

		self.skip_branch = skip_branch
		self.main_branch = main_branch
		self.name = name

		self.named_params_dict = {}

		for layer in self.main_branch:
			self.named_params_dict.update(layer.get_params_dict())
		
		for layer in self.skip_branch:
			self.named_params_dict.update(layer.get_params_dict())
	
	
	def forward(self, X, is_training=False):
		FX = X
		# Main branch
		for layer in self.main_branch:
			FX = layer.forward(FX, is_training)

		SX = X
		# Skip branch
		for layer in self.skip_branch:
			SX = layer.forward(SX,is_training)

		return SX + FX


	def get_params(self):
		params = []

		for layer in self.main_branch:
			params += layer.get_params()
		
		for layer in self.skip_branch:
			params += layer.get_params()
			
		return params
	
	def get_params_dict(self):
		return self.named_params_dict


	def to_dict(self):
		desc =  {
			'type':'ConvBlock',
			'params': {
				'name': self.name,
				'main_branch': [],
				'skip_branch': [],
			}
		}

		for layer in self.main_branch:
			desc['params']['main_branch'].append(layer.to_dict())
		
		for layer in self.skip_branch:
			desc['params']['skip_branch'].append(layer.to_dict())

		return desc
