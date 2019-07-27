from makiflow.beta_layers import InputLayer, DenseMakiLayer,SumMakiLayer
from makiflow.classificator import Classificator
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

def get_layers():
    in_x = InputLayer(input_shape=[64,784])
    x = DenseMakiLayer(input_shape=784, output_shape=100, name='dense1')(in_x)
    x = DenseMakiLayer(input_shape=100, output_shape=10, activation=None, name='dense5')(x)
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
    model = Classificator(input=in_x, output=out)


    session = tf.Session()
    model.set_session(session)

    (Xtrain, Ytrain), (Xtest, Ytest) = get_train_test_data()
    epochs = 5
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)

    model.save_architecture('../beta_model.json')

    model.save_weights('../beta_model.ckpt')

    model.set_session(session)
    model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=1)
    model.load_weights('../beta_model.ckpt')
    model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=1)

