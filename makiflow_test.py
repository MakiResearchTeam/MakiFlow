# Run this file if you wanna check whether MakiFlow CNN works

from makiflow.layers import ConvLayer, DenseLayer, MaxPoolLayer, FlattenLayer
from makiflow.advanced_layers import ResnetIndentityBlock
from makiflow import ConvModel
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10

def get_layers():
    layers = [
            ConvLayer(kw=3, kh=3, in_f=3, out_f=64, name='input_layer'),
            ConvLayer(kw=3, kh=3, in_f=64, out_f=64, name=2),
            MaxPoolLayer(),
        
            ConvLayer(kw=3, kh=3, in_f=64, out_f=128, name=4),
            ResnetIndentityBlock(in_f1=128, out_f1=128, name=5),
            MaxPoolLayer(),
        
            ConvLayer(kw=3, kh=3, in_f=128, out_f=256, name=6),
            ResnetIndentityBlock(in_f1=256, out_f1=256, name=7),
            MaxPoolLayer(),
        
            ConvLayer(kw=3, kh=3, in_f=256, out_f=512, name=8),
            ResnetIndentityBlock(in_f1=512, out_f1=512, name=9),
            MaxPoolLayer(),
        
            FlattenLayer(),
            DenseLayer(input_shape=2048, output_shape=1024, name=12),
            DenseLayer(input_shape=1024, output_shape=1024, name=13),
            DenseLayer(input_shape=1024, output_shape=10, activation=None, name='out_put_layer')
        # The last layer always is gonna have no activation function! Just always pass None into 'activation' argument!
        ]
    return layers

def get_train_test_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    Xtrain /= 255
    Xtest /= 255

    Ytrain = Ytrain.reshape(len(Ytrain),)
    Ytest = Ytest.reshape(len(Ytest),)

    return (Xtrain, Ytrain), (Xtest, Ytest)


if __name__ == "__main__":
    layers = get_layers()
    model = ConvModel(layers=layers, input_shape=[64, 32, 32, 3], num_classes=10, name='My_MakiFlow_little_VGG')
    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    epochs = 2
    lr = 1e-4
    epsilon = 1e-6
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)

    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)






