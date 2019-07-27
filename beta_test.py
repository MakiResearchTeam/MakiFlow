from makiflow.layers import InputLayer, DenseLayer, ConvLayer, MaxPoolLayer, FlattenLayer
from makiflow.models.classificator import Classificator
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10

def get_layers():

    in_x = InputLayer(input_shape=[16, 32, 32, 3], name='input')
    x = ConvLayer(kh=3, kw=3, in_f=3, out_f=16, name='conv1')(in_x)
    x = MaxPoolLayer(name='pool1')(x)
    # 16x16
    x = ConvLayer(kh=3, kw=3, in_f=16, out_f=32, name='conv2')(x)
    x = MaxPoolLayer(name='pool2')(x)
    # 8x8
    x = ConvLayer(kh=3, kw=3, in_f=32, out_f=64, name='conv3')(x)
    x = MaxPoolLayer(name='pool3')(x)
    # 4x4
    x = ConvLayer(kh=3, kw=3, in_f=64, out_f=32, name='conv4')(x)
    x = MaxPoolLayer(name='pool4')(x)
    # 2x2
    x = ConvLayer(kh=3, kw=3, in_f=32, out_f=16, name='conv5')(x)
    x = MaxPoolLayer(name='pool5')(x)
    # 1x1
    x = FlattenLayer(name='flatten')(x)
    x = DenseLayer(in_d=16, out_d=10, activation=None, name='dense1')(x)
    return in_x, x


def get_train_test_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = cifar10.load_data()

    Xtrain = Xtrain.astype(np.float32) / 255
    Xtest = Xtest.astype(np.float32) / 255

    Ytrain = Ytrain.reshape(len(Ytrain), )
    Ytest = Ytest.reshape(len(Ytest), )

    return (Xtrain, Ytrain), (Xtest, Ytest)


if __name__ == "__main__":
    in_x, out = get_layers()
    model = Classificator(input=in_x, output=out)


    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    epochs = 1
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    model.save_architecture('T:/download/shiru/shit.json')

