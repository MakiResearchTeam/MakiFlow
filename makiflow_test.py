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
from makiflow.advanced_layers.convblock_resnet import ConvBlock_resnet

def get_layers():
    layers = [#
            ConvLayer(kw=3,kh=3,in_f=3,out_f=64,name='qe'),
            MaxPoolLayer(),#16
            MaxPoolLayer(),#8
            ConvLayer(kw=3,kh=3,in_f=64,out_f=128,name='qwfs'),
            ConvBlock_resnet(in_f=128,out_f=[64,128,256],name='rar'),
            MaxPoolLayer(),#4
            ConvLayer(kw=3,kh=3,in_f=256,out_f=512,name='xgrw'),
            MaxPoolLayer(),#2
            ConvBlock_resnet(in_f=512,out_f=[256,512,1024],name='rar'),
            AvgPoolLayer(),#1

            FlattenLayer(),
            DenseLayer(input_shape=1024,output_shape=64,name='dwad'),
            #DropoutLayer(0.7),
            #BatchNormLayer(D=64,name='dsaxz'),
            #DenseLayer(input_shape=64,output_shape=10,name='zxcgg'),
            #DropoutLayer(0.8),
            BatchNormLayer(D=64,name='dsaxz'),

            DenseLayer(input_shape=64, output_shape=10, activation=None, name='predictions'),
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
    model = ConvModel(layers=layers, input_shape=[32, 32, 32, 3], num_classes=10, name='My_MakiFlow_little_VGG')
    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    
    #(Xtrain, Ytrain), (Xtest, Ytest),ans = get_fruit360(count=20)
    
    epochs = 1
    lr = 0.01
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    
    in_mat = np.random.randn(1, 2, 2, 3).astype(np.float32)

    #temp = None
    #with session.as_default():
    #print(model.named_params_dict)
    #print(temp.shape)
    model.to_json('T:/download/shiru/mod.json')
    model.save_weights('T:/download/shiru/model.ckpt')
    #tf.reset_default_graph()
    print('saved!')
    
    new_model = Builder.convmodel_from_json('T:/download/shiru/mod.json')
    #new_ses = tf.Session()
    new_model.set_session(tf.Session())
    new_model.load_weights('T:/download/shiru/model.ckpt')

    model.evaluate(Xtest,Ytest)
    new_model.evaluate(Xtest,Ytest)
    
    #temp_one = None
    #with new_ses.as_default():
    print(model.predict(Xtrain[:32])[0])
    print(new_model.predict(Xtrain[:32])[0])
    print(len(model.named_params_dict))
    print(len(new_model.named_params_dict))
    
    
    #flag = True
    #for i in range(len(temp_one)):
    #    if temp_one[i] != temp[i]:
    #        flag = False
    #print(flag)
    print('end!')




    






