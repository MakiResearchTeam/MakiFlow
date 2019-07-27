from __future__ import absolute_import
from makiflow.beta_layers import MakiTensor, MakiOperation, InputLayer
import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from makiflow.utils import error_rate, sparse_cross_entropy
from copy import copy

from collections import ChainMap

EPSILON = np.float32(1e-37)

class Classificator:
	def __init__(self, input: InputLayer, output: MakiTensor, num_classes: int):
		self.X = input.get_data_tensor()
		self.batch_sz = input.get_shape()[0]
		used = {}
		# Contains pairs {layer: tensor}, where `tensor` is output tensor of `layer`
		output_tensors = {}
		def create_tensor(from_,is_training):
			if used.get(from_.get_name()) is None:
				layer = from_.get_parent_layer()
				used[layer.name] = True
				X = copy(from_.get_data_tensor())
				takes = []
				# Check if we at the beginning of the computational graph, i.e. InputLayer
				if from_.get_parent_tensor_names() is not None:
					for elem in from_.get_parent_tensors(): 
						takes += [create_tensor(elem,is_training)]
						
					X = layer.forward(takes[0] if len(takes) == 1 else takes,is_training)

				output_tensors[layer.name] = X
				return X
			else:
				return output_tensors[from_.get_name()]

		self.output_test = create_tensor(output,is_training=False) 
		del used, output_tensors
		self.output = output
		self.labels = tf.placeholder(tf.int32, shape=self.batch_sz)
		self.__collect_params_and_dict()
		# For learning
		self.names_trainble_variables = [name_var for name_var in self.all_named_params_dict]
		self.names_untrainble_variables = []
		self.Create_list_or_train_var()


	def __collect_params_and_dict(self):
		current_tensor = self.output
		self.all_named_params_dict = {}
		self.all_params = []

		self.all_params = current_tensor.get_parent_layer().get_params()
		# {LayerName: {params_name:values,...} }
		self.all_named_params_dict[current_tensor.get_parent_layer().name] = current_tensor.get_parent_layer().get_params_dict()

		for _,elem in (current_tensor.get_previous_tensors().items()):
			self.all_params += elem.get_parent_layer().get_params()
			self.all_named_params_dict[elem.get_parent_layer().name] = elem.get_parent_layer().get_params_dict()

	def Add_train_variables(self,layer_names:list):
		assert(self.session is not None)
		# Check untrainble list
		for name in layer_names:
			if name not in self.names_untrainble_variables:
				raise NameError(f'{name} layer do not exist in untrainble list')
		
		for name in layer_names:
			self.names_untrainble_variables.remove(str(name))
			self.names_trainble_variables.append(str(name))
		
		self.Create_list_or_train_var()
			
	
	def Remove_from_train_variables(self,layer_names:list):
		assert(self.session is not None)
		# Check trainble list
		for name in layer_names:
			if name not in self.names_trainble_variables:
				raise NameError(f'{name} layer do not exits in trainble list')

		for name in layer_names:
			self.names_trainble_variables.remove(str(name))
			self.names_untrainble_variables.append(str(name))

		self.Create_list_or_train_var()
		
	def Create_list_or_train_var(self):
		self.trainble_var = []
		for name_layer in self.names_trainble_variables:
			cur_layer = self.all_named_params_dict[name_layer]
			self.trainble_var += list(cur_layer.values())

	def load_weights(self, path,names_of_load_layer=None):
		"""
		This function uses default TensorFlow's way for restoring models - checkpoint files.
		:param path - full path+name of the model.
		Example: '/home/student401/my_model/model.ckpt'
		"""
		assert (self.session is not None)
		# Store which params will be loaded
		load_params_dict = {}

		if names_of_load_layer is None:
			load_params_dict = dict(ChainMap(*self.all_named_params_dict.values()))
		else:
			# Get named params in form of dict
			# Combine what before output and output tensor
			dict_of_all_tensors = dict(ChainMap(self.output.get_previous_tensors(),self.output.get_self_pair()))
			
			for name in names_of_load_layer:
				load_params_dict.update(dict_of_all_tensors[name].get_parent_layer().get_params_dict())

		saver = tf.train.Saver(load_params_dict)
		saver.restore(self.session, path)
		print('Model restored')

	def save_weights(self, path,names_of_save_layer=None):
		"""
		This function uses default TensorFlow's way for saving models - checkpoint files.
		:param path - full path+name of the model.
		Example: '/home/student401/my_model/model.ckpt'
		"""
		assert (self.session is not None)
		# Store which params will be loaded
		load_params_dict = {}

		if names_of_save_layer is None:
			load_params_dict = dict(ChainMap(*self.all_named_params_dict.values()))
		else:
			# Get named params in form of dict
			# Combine what before output and output tensor
			dict_of_all_tensors = dict(ChainMap(self.output.get_previous_tensors(),self.output.get_self_pair()))
			
			for name in names_of_save_layer:
				load_params_dict.update(dict_of_all_tensors[name].get_parent_layer().get_params_dict())

		saver = tf.train.Saver(load_params_dict)
		save_path = saver.save(self.session, path)
		print('Model saved to %s' % save_path)

	def set_session(self, session: tf.Session):
		self.session = session
		init_op = tf.variables_initializer(self.all_params)
		self.session.run(init_op)

	def evaluate(self, Xtest, Ytest):
		# TODO: n_batches is never used
		# Validating the network
		Xtest = Xtest.astype(np.float32)

		Yish_test = tf.nn.softmax(self.output_test)
		n_batches = Xtest.shape[0] // self.batch_sz

		# For train data
		test_cost = 0
		predictions = np.zeros(len(Xtest))
		for k in tqdm(range(n_batches)):
			# Test data
			Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
			Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
			Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
			test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
			predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

		error = error_rate(predictions, Ytest)
		test_cost = test_cost / (len(Xtest) // self.batch_sz)
		print('Accuracy:', 1 - error, 'Cost:', test_cost)

	def pure_fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1):
		"""
		Method for training the model. Works faster than `verbose_fit` method because
		it uses exponential decay in order to speed up training. It produces less accurate
		train error mesurement.

		Parameters
		----------
			Xtrain : numpy array
				Training images stacked into one big array with shape (num_images, image_w, image_h, image_depth).
			Ytrain : numpy array
				Training label for each image in `Xtrain` array with shape (num_images).
				IMPORTANT: ALL LABELS MUST BE NOT ONE-HOT ENCODED, USE SPARSE TRAINING DATA INSTEAD.
			Xtest : numpy array
				Same as `Xtrain` but for testing.
			Ytest : numpy array
				Same as `Ytrain` but for testing.
			optimizer : tensorflow optimizer
				Model uses tensorflow optimizers in order train itself.
			epochs : int
				Number of epochs.
			test_period : int
				Test begins each `test_period` epochs. You can set a larger number in order to
				speed up training.

		Returns
		-------
			python dictionary
				Dictionary with all testing data(train error, train cost, test error, test cost)
				for each test period.
		"""

		assert (optimizer is not None)
		assert (self.session is not None)
		# This is for correct working of tqdm loop. After KeyboardInterrupt it breaks and
		# starts to print progress bar each time it updates.
		# In order to avoid this problem we handle KeyboardInterrupt exception and close
		# the iterator tqdm iterates through manually. Yes, it's ugly, but necessary for
		# convinient working with MakiFlow in Jupyter Notebook. Sometimes it's helpful
		# even for console applications.
		iterator = None

		Xtrain = Xtrain.astype(np.float32)
		Xtest = Xtest.astype(np.float32)

		# For training	
		cost = (
			tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output.get_data_tensor(), labels=self.labels)),
			self.output.get_data_tensor())
		train_op = (cost, optimizer.minimize(cost[0],var_list=self.trainble_var) )
		# Initialize optimizer's variables
		self.session.run(tf.variables_initializer(optimizer.variables()))

		# For testing
		Yish_test = tf.nn.softmax(self.output_test)

		n_batches = Xtrain.shape[0] // self.batch_sz

		train_costs = []
		train_errors = []
		test_costs = []
		test_errors = []
		try:
			for i in range(epochs):
				Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
				train_cost = np.float32(0)
				train_error = np.float32(0)
				iterator = tqdm(range(n_batches))

				for j in iterator:
					Xbatch = Xtrain[j * self.batch_sz:(j + 1) * self.batch_sz]
					Ybatch = Ytrain[j * self.batch_sz:(j + 1) * self.batch_sz]
					(train_cost_batch, y_ish), _ = self.session.run(
						train_op,
						feed_dict={self.X: Xbatch, self.labels: Ybatch})
					# Use exponential decay for calculating loss and error
					train_cost = 0.99 * train_cost + 0.01 * train_cost_batch
					train_error_batch = error_rate(np.argmax(y_ish, axis=1), Ybatch)
					train_error = 0.99 * train_error + 0.01 * train_error_batch

				# Validating the network on test data
				if i % test_period == 0:
					# For test data
					test_cost = np.float32(0)
					test_predictions = np.zeros(len(Xtest))

					for k in range(len(Xtest) // self.batch_sz):
						# Test data
						Xtestbatch = Xtest[k * self.batch_sz:(k + 1) * self.batch_sz]
						Ytestbatch = Ytest[k * self.batch_sz:(k + 1) * self.batch_sz]
						Yish_test_done = self.session.run(Yish_test, feed_dict={self.X: Xtestbatch}) + EPSILON
						test_cost += sparse_cross_entropy(Yish_test_done, Ytestbatch)
						test_predictions[k * self.batch_sz:(k + 1) * self.batch_sz] = np.argmax(Yish_test_done, axis=1)

					# Collect and print data
					test_cost = test_cost / (len(Xtest) // self.batch_sz)
					test_error = error_rate(test_predictions, Ytest)
					test_errors.append(test_error)
					test_costs.append(test_cost)

					train_costs.append(train_cost)
					train_errors.append(train_error)

					print('Epoch:', i, 'Train accuracy: {:0.4f}'.format(1 - train_error),
						  'Train cost: {:0.4f}'.format(train_cost),
						  'Test accuracy: {:0.4f}'.format(1 - test_error), 'Test cost: {:0.4f}'.format(test_cost))
		except Exception as ex:
			print(ex)
		finally:
			if iterator is not None:
				iterator.close()
			return {'train costs': train_costs, 'train errors': train_errors,
					'test costs': test_costs, 'test errors': test_errors}


