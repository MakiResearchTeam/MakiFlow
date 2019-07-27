from makiflow.beta_layers import InputLayer, DenseMakiLayer,SumMakiLayer
from makiflow.classificator import Classificator
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

def get_layers():
    in_x = InputLayer(input_shape=[64,784])
    x = DenseMakiLayer(input_shape=784, output_shape=5000, name='dense1')(in_x)
    c = DenseMakiLayer(input_shape=784, output_shape=5000, name='dense1_0')(in_x)
    z = DenseMakiLayer(input_shape=784, output_shape=1000, name='dense1_7')(in_x)
    z = DenseMakiLayer(input_shape=1000, output_shape=5000, name='dense1_1')(z)

    x = SumMakiLayer(name='summo')([x,c,z])
    x = DenseMakiLayer(input_shape=5000, output_shape=10, activation=None, name='dense5')(x)
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
    epochs = 2
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    model.set_layers_trainable([('dense5',False),('dense1_1',False),('dense1_0',False)])
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    print('\nadd back\n')
    model.set_layers_trainable([('dense5',True),('dense1_1',True),('dense1_0',True)])
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=1)
    #model.save_architecture('../beta_model.json')

    model.save_weights('T:\download\shiru',layer_names=['dense1','dense5','dense1_1','dense1_0'])

    new_model = Classificator(input=in_x, output=out)
    new_session = tf.Session()
    new_model.set_session(new_session)
    new_model.load_weights("T:\download\shiru",layer_names=['dense1','dense5','dense1_1','dense1_0'])
    new_model.evaluate(Xtest,Ytest)

    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    new_model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)

