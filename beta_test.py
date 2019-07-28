from makiflow.layers import InputLayer, DenseLayer, ConvLayer, MaxPoolLayer, FlattenLayer, BatchNormLayer
from makiflow.models.classificator import Classificator
from makiflow.save_recover.builder import Builder
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10

def get_layers():

    in_x = InputLayer(input_shape=[64, 32, 32, 3], name='input')
    x = ConvLayer(kh=3, kw=3, in_f=3, out_f=16, name='conv1')(in_x)
    x = MaxPoolLayer(name='pool1')(x)
    # 16x16
    x = BatchNormLayer(D=16,name='batch1')(x)
    x = ConvLayer(kh=3, kw=3, in_f=16, out_f=32, name='conv2')(x)
    x = MaxPoolLayer(name='pool2')(x)
    # 8x8
    x = BatchNormLayer(D=32,name='batch2')(x)
    x = ConvLayer(kh=3, kw=3, in_f=32, out_f=64, name='conv3')(x)
    x = MaxPoolLayer(name='pool3')(x)
    # 4x4
    x = BatchNormLayer(D=64,name='batch3')(x)
    x = ConvLayer(kh=3, kw=3, in_f=64, out_f=32, name='conv4')(x)
    x = MaxPoolLayer(name='pool4')(x)
    # 2x2
    x = BatchNormLayer(D=32,name='batch4')(x)
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
    epochs = 5
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    model.save_architecture('T:/download/shiru/shit.json')
    model.save_weights(path='T:/download/shiru/')

    del model

    new_model = Builder.classificator_from_json('T:/download/shiru/shit.json')
    new_session = tf.Session()
    new_model.set_session(new_session)
    new_model.load_weights(path='T:/download/shiru/')
    #info = new_model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    new_model.evaluate(Xtest,Ytest,64)
    new_model.save_architecture('T:/download/shiru/new.json')
