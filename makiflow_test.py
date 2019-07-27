# Run this file if you wanna check whether MakiFlow CNN works

from makiflow.layers import ConvLayer, DenseLayer, MaxPoolLayer, FlattenLayer
from makiflow import ConvModel
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10
def get_layers():
    layers = [
        # 32x32
        ConvLayer(kw=3, kh=3, in_f=3, out_f=32, name='input'),
        MaxPoolLayer(),
        # 16x16
        ConvLayer(kw=3, kh=3, in_f=32, out_f=64, name='input'),
        MaxPoolLayer(),
        # 8x8
        ConvLayer(kw=3, kh=3, in_f=64, out_f=128, name='input'),
        MaxPoolLayer(),
        # 4x4
        ConvLayer(kw=3, kh=3, in_f=128, out_f=64, name='input'),
        MaxPoolLayer(),
        # 2x2
        ConvLayer(kw=3, kh=3, in_f=64, out_f=16, name='input'),
        MaxPoolLayer(),
        # 1x1
        FlattenLayer(),
        DenseLayer(in_d=16, out_d=32, name='1'),
        DenseLayer(in_d=32, out_d=10, activation=None, name='1'),
    ]
    return layers

def get_train_test_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    Xtrain = Xtrain.astype(np.float32) / 255
    Xtest = Xtest.astype(np.float32) / 255

    Ytrain = Ytrain.reshape(len(Ytrain),)
    Ytest = Ytest.reshape(len(Ytest),)

    return (Xtrain, Ytrain), (Xtest, Ytest)

if __name__ == "__main__":
    layers = get_layers()
    model = ConvModel(layers=layers, input_shape=[16, 32, 32, 3], num_classes=10, name='My_MakiFlow_little_VGG')
    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    
    epochs = 10
    lr = 1e-4
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)




    






