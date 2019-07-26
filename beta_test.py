# Run this file if you wanna check whether MakiFlow CNN works

from makiflow.beta_layers import InputLayer, DenseLayer,SumLayer
from makiflow.classificator import Classificator
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

def get_layers():
    in_x = InputLayer(input_shape=[64,784])
    x = DenseLayer(input_shape=784, output_shape=500, name='dense1')(in_x)
    x = DenseLayer(input_shape=500, output_shape=200, name='dense2')(x)
    x = DenseLayer(input_shape=200, output_shape=10, activation=None, name='dense3')(x)
    return in_x, x


def get_train_test_data():
    (Xtrain, Ytrain), (Xtest, Ytest) = mnist.load_data()

    Xtrain = Xtrain.astype(np.float32).reshape(Xtrain.shape[0],28*28) / 255
    Xtest = Xtest.astype(np.float32).reshape(Xtest.shape[0],28*28) / 255

    Ytrain = Ytrain.reshape(len(Ytrain), )
    Ytest = Ytest.reshape(len(Ytest), )

    return (Xtrain, Ytrain), (Xtest, Ytest)


if __name__ == "__main__":
    in_x, out = get_layers()
    model = Classificator(input=in_x, output=out, num_classes=10)


    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    epochs = 4
    lr = 1e-4
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)

    model.save_weights('T:\download\shiru')

    new_session = tf.Session()
    new_model = Classificator(input=in_x,output=out,num_classes=10)
    new_model.set_session(new_session)
    new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense1','dense2'])
    new_model.evaluate(Xtest,Ytest)











