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
            Reduction_A(in_f=3,out_f=[32,32,64,64],name='1'),#131, 16x16
            BatchNormLayer(D=131,name='sas'),
            Reduction_B(in_f=131,out_f=[80,100,120,131],name='01'),#482 8x8
            BatchNormLayer(D=482,name='z'),
            Reduction_B(in_f=482,out_f=[200,260,360,482],name='71'),#4x4 1584 
            Reduction_B(in_f=1584,out_f=[300,360,450,600],name='81'),#2994 2x2
            AvgPoolLayer(),#

            FlattenLayer(),
            DenseLayer(input_shape=2994,output_shape=256,name='2'),
            BatchNormLayer(D=256,name='3'),
            DenseLayer(input_shape=256, output_shape=10, activation=None, name='predictions'),
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
    
    epochs = 1
    lr = 0.01
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon,momentum=0.9)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    model.evaluate(Xtest,Ytest)
    print('over! one')

    model.to_json('T:/download/trash/mod.json')
    model.save_weights('T:/download/trash/model.ckpt')

    print('saved!')

    new_model = Builder.convmodel_from_json('T:/download/trash/mod.json')
    new_model.set_session(session)
    new_model.load_weights('T:/download/trash/model.ckpt')


    new_model.evaluate(Xtest,Ytest)
    print('end!')


    






