# Copyright (C) 2020  Igor Kilbas, Danil Gribanov, Artem Mukhin
#
# This file is part of MakiFlow.
#
# MakiFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# MakiFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Foobar.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import absolute_import
from makiflow.layers import RNNBlock
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
import json



# IT WILL BE EITHER SEPARATE UTIL OR A CLASS METHOD LATER.
# FIX IT IN THE FUTURE.
def toSparse(texts, chars):
	"put ground truth texts into sparse tensor for ctc_loss"
	indices = []
	values = []
	shape = [len(texts), 0] # last entry must be max(labelList[i])

	# go over all texts
	for (batchElement, text) in enumerate(texts):
		# convert to string of label (i.e. class-ids)
		try:
			labelStr = [chars.index(c) for c in text]
		except Exception as ex:
			print(ex)
			print('problem text', text)

		# sparse tensor must have size of max. label-string
		if len(labelStr) > shape[1]:
			shape[1] = len(labelStr)
		# put each label into sparse tensor
		for (i, label) in enumerate(labelStr):
			indices.append([batchElement, i])
			values.append(label)

	return (indices, values, shape)



class DecoderType:
	BestPath = 0
	BeamSearch = 1
	WordBeamSearch = 2

class TextRecognizer:
    def __init__(self, cnn_layers, rnn_layers, input_shape, chars, max_seq_length, decoder_type=DecoderType.BeamSearch, name='MakiRecognizer'):
        self.name = str(name)
        self.batch_sz = input_shape[0]
        self.cnn_layers = cnn_layers
        self.rnn_layers = rnn_layers
        self.input_shape = input_shape
        self.input_image = tf.placeholder(tf.float32, shape=input_shape, name='image')
        self.chars = chars
        self.max_seq_length = max_seq_length
        self.decoder_type = decoder_type
        self.setup_cnn()
        self.setup_rnn()
        self.setupCTC()

        # COLLECT ALL THE PARAMS
        self.params = []
        self.named_params_dict = {}
        for layer in self.cnn_layers:
            self.params += layer.get_params()
            self.named_params_dict.update(layer.get_params_dict())
        
        for layer in self.rnn_layers:
            self.params += layer.get_params()
            self.named_params_dict.update(layer.get_params_dict())
        self.params += [self.kernel]
        self.named_params_dict.update({'kernel_projector': self.kernel})
              
    
    def set_session(self, session):
        assert(session is not None)
        self.session = session
        init_op = tf.variables_initializer(self.params)
        self.session.run(init_op)


    def save_weights(self, path):
        """
        This function uses default TensorFlow's way for saving models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        save_path = saver.save(self.session, path)
        print('Model saved to %s' % save_path)
    

    def load_weights(self, path):
        """
        This function uses default TensorFlow's way for restoring models - checkpoint files.
        :param path - full path+name of the model.
        Example: '/home/student401/my_model/model.ckpt'
        """
        assert (self.session is not None)
        saver = tf.train.Saver(self.named_params_dict)
        saver.restore(self.session, path)
        print('Model restored')


    def to_json(self, path):
        """
        Convert model's architecture to json file and save it.
        Parameters
        ----------
        path : string
            Path to file to save in. Example: `my_folder/my_model/maki_recognizer.json`
        """

        model_dict = {
            'name': self.name,
            'input_shape': self.input_shape,
            'chars': self.chars,
            'max_seq_length': self.max_seq_length,
            'decoder_type': self.decoder_type
        }
        cnn_layers_dict = {
            'cnn_layers': []
        }
        for layer in self.cnn_layers:
            cnn_layers_dict['cnn_layers'].append(layer.to_dict())
        
        rnn_layers_dict = {
            'rnn_layers': []
        }
        for layer in self.rnn_layers:
            rnn_layers_dict['rnn_layers'].append(layer.to_dict())

        model_dict.update(cnn_layers_dict)
        model_dict.update(rnn_layers_dict)
        model_json = json.dumps(model_dict, indent=1)
        json_file = open(path, mode='w')
        json_file.write(model_json)
        json_file.close()
        print("Model's architecture is saved to {}.".format(path))
      

    def setup_cnn(self):
        X = self.input_image
        for layer in self.cnn_layers:
            X = layer.forward(X)
        
        self.cnn_out = X
    

    def setup_rnn(self):
        rnn_input = tf.squeeze(self.cnn_out, axis=[2])
        self.rnn_block = RNNBlock(rnn_layers=self.rnn_layers, seq_length=self.cnn_out.shape[1], dynamic=True, bidirectional=True)
        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = self.rnn_block.forward(rnn_input)
        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)
        # Get the number of cells of the last RNN layer
        numHidden = self.rnn_layers[-1].num_cells
        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        self.kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.chars) + 1], stddev=0.1), dtype=tf.float32, name='kernel_projector')
        self.rnnOut3d = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=self.kernel, rate=1, padding='SAME'), axis=[2])


    def setupCTC(self):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        self.ctcIn3dTBC = tf.transpose(self.rnnOut3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]) , tf.placeholder(tf.int32, [None]), tf.placeholder(tf.int64, [2]))

        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [self.batch_sz])
        self.loss = tf.reduce_mean(tf.nn.ctc_loss(labels=self.gtTexts, inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, ctc_merge_repeated=True))

        if self.decoder_type == DecoderType.BestPath:
            self.decoder = tf.nn.ctc_greedy_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoder_type == DecoderType.BeamSearch:
            self.decoder = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=100, merge_repeated=False)
        else:
            print('This decoder type is not implemented yet. BeamSearch decorder will be used instead.')
            self.decoder_ = tf.nn.ctc_beam_search_decoder(inputs=self.ctcIn3dTBC, sequence_length=self.seqLen, beam_width=100, merge_repeated=False)


    def fit(self, Xtrain, Ytrain, Xtest, Ytest, optimizer, epochs=10, test_period=1):
        """
        Parameters
        ----------
        Xtrain : list
            List of training images.
        Ytrain : list
            List of ground truth training texts. All texts will be then converted to sparse tensors.
        Xtest : list
            List of testing images.
        Ytest : list
            List of ground truth testing texts.
        optimizer : tensorflow optimizer
            Used for minimizing the loss. You have configure it before passing in.
        epochs : int
            Number of epochs to do.
        test_period : int
            Each `test_period` epoch the test is performed.
        """
        try:
            train_op = optimizer.minimize(self.loss)
            # Initialize optimizer's variables
            self.session.run(tf.variables_initializer(optimizer.variables()))

            losses = []
            n_batches = len(Xtrain) // self.batch_sz
            for i in range(epochs):
                Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
                
                iterator = range(n_batches)
                batch_loss = 0
                for j in tqdm(iterator):
                    Xbatch = Xtrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    Ybatch = Ytrain[j*self.batch_sz:(j+1)*self.batch_sz]
                    Ybatch = toSparse(Ybatch, self.chars)
                    _, l = self.session.run(
                        [train_op, self.loss],
                        feed_dict={
                                self.gtTexts: Ybatch,
                                self.input_image: Xbatch,
                                self.seqLen: [self.max_seq_length]*self.batch_sz
                            }
                    )
                    batch_loss += l

                batch_loss /= n_batches
                losses.append(batch_loss)
                print('Epoch:', i, 'Loss: {:0.4f}'.format(batch_loss))
                if i % test_period == 0:
                    Xtest, Ytest = shuffle(Xtest, Ytest)
                    texts = self.infer_batch(Xtest[:self.batch_sz])
                    print('Recognizing test images...')
                    print('RECOGNIZED/GROUND TRUTH')
                    for recognized, gt in zip(texts, Ytest[:self.batch_sz]):
                        print(recognized, ':', gt)
                
        except Exception as ex:
            print(ex)
            iterator.close() 
        finally:
            return {'losses': losses}
    
    def decoder_output_to_text(self, ctcOutput):
        """
        Extract texts from output of the CTC decoder.
        Parameters
        ----------
        ctcOutput : list
            Decoded output of the CTC function.
        
        Returns
        -------
        list
            Contains texts for all the `ctcOutput`s.
        """

        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(self.batch_sz)]

        # ctc returns tuple, first element is SparseTensor 
        decoded=ctcOutput[0][0] 

        # go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx]
            batchElement = idx2d[0] # index according to [b,t]
            encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.chars[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    def infer_batch(self, imgs):
        """
        Feed a batch into the NN to recognize the texts.

        Parameters
        ----------
        imgs : list
            Images to recognize.
            WARNING! NUMBER OF THE IMAGES IN THE LIST MUST BE EQUAL THE BATCH SIZE!
        Returns
        -------
        list
            Contains recognized texts for all the given images.
        """
        
        # decode, optionally save RNN output
        feedDict = {self.input_image : imgs, self.seqLen : [self.max_seq_length] * self.batch_sz}
        evalRes = self.session.run([self.decoder, self.ctcIn3dTBC], feedDict)
        decoded = evalRes[0]
        texts = self.decoder_output_to_text(decoded)
        return texts