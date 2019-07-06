# Run this file if you wanna check whether MakiFlow CNN works
#import sys
#sys.path.append('T:/codeStuff/Jupyter/Network/fruits_360')
#from util import get_fruit360
import cv2

from makiflow.save_recover.builder import Builder
from makiflow.layers import ConvLayer, DenseLayer, MaxPoolLayer, FlattenLayer, AvgPoolLayer, DropoutLayer, BatchNormLayer
from makiflow.advanced_layers import ResnetIndentityBlock
from makiflow import ConvModel
import tensorflow as tf
import numpy as np
import tensorflow.keras
from tensorflow.keras.datasets import cifar10
from makiflow.advanced_layers.stem import StemBlock
from makiflow.advanced_layers.inception_resnet_A import Inception_A
from makiflow.advanced_layers.inception_resnet_B import Inception_B
from makiflow.advanced_layers.inception_resnet_C import Inception_C
from makiflow.advanced_layers.reduction_A import  Reduction_A
from makiflow.advanced_layers.reduction_B import  Reduction_B

def get_layers():
    layers = [#
            StemBlock(in_f=3,name='1'),#
            BatchNormLayer(D=384,name='3'),
            ConvLayer(kw=3, kh=3, in_f=384, out_f=550, name='10'),
            AvgPoolLayer(),#

            FlattenLayer(),
            DenseLayer(input_shape=550,output_shape=160,name='2'),
            BatchNormLayer(D=160,name='3'),
            DenseLayer(input_shape=160, output_shape=10, activation=None, name='predictions'),
        # The last layer always is gonna have no activation function! Just always pass None into 'activation' argument!
        ]
    return layers

def get_train_test_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    Xtrain /= 255
    Xtest /= 255

    #Xtrain_100 = []
    #Xtest_100 = []
    #for i in range(len(Xtrain)):
    #    Xtrain_100.append(cv2.resize(Xtrain[i], dsize=(100, 100)))
    #for i in range(len(Xtest)):
    #    Xtest_100.append(cv2.resize(Xtest[i], dsize=(100, 100)))

    Ytrain = Ytrain.reshape(len(Ytrain),)
    Ytest = Ytest.reshape(len(Ytest),)

    #return (np.array(Xtrain_100).astype(np.float32), Ytrain), (np.array(Xtest_100).astype(np.float32), Ytest)
    return (Xtrain, Ytrain), (Xtest, Ytest)

if __name__ == "__main__":
    layers = get_layers()
    model = ConvModel(layers=layers, input_shape=[64, 32, 32, 3], num_classes=10, name='My_MakiFlow_little_VGG')
    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    
    #(Xtrain, Ytrain), (Xtest, Ytest),ans = get_fruit360(count=20)
    
    epochs = 4
    lr = 0.01
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon,momentum=0.9)
    #info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)

    print('over! one')

    






