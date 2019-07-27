# Run this file if you wanna check whether MakiFlow CNN works

from makiflow.beta_layers import InputLayer, DenseLayer,SumLayer
from makiflow.classificator import Classificator
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import mnist

def get_layers():
    in_x = InputLayer(input_shape=[64,784])
    x = DenseLayer(input_shape=784, output_shape=300, name='dense1')(in_x)
    x = DenseLayer(input_shape=300, output_shape=100, name='dense2')(x)
    x = DenseLayer(input_shape=100, output_shape=100, name='dense3')(x)
    x = DenseLayer(input_shape=100, output_shape=100, name='dense4')(x)
    x = DenseLayer(input_shape=100, output_shape=10, activation=None, name='dense5')(x)
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
    epochs = 5
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)

    model.save_weights('T:\download\shiru')

    new_session = tf.Session()
    new_model = Classificator(input=in_x,output=out,num_classes=10)
    new_model.set_session(new_session)
    #new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense1'])
    #new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense2'])
    #new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense3'])
    #new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense4'])
    #new_model.load_weights('T:\download\shiru',names_of_load_layer=['dense5'])

    print('\n remove 3 4 \n')
    new_model.Remove_from_train_variables(layer_names=['dense3','dense4','dense5'])

    new_model.pure_fit(Xtrain,Ytrain,Xtest,Ytest,optimizer= optimizer,epochs=epochs)

    new_model.evaluate(Xtest,Ytest)

    print('\n add 1 \n')
    new_model.Add_train_variables(layer_names=['dense3'])

    new_model.pure_fit(Xtrain,Ytrain,Xtest,Ytest,optimizer= optimizer,epochs=epochs)

    new_model.evaluate(Xtest,Ytest)

    print('\nremove 3 and 4\n')
    new_model.Remove_from_train_variables(layer_names=['dense1','dense2'])

    new_model.pure_fit(Xtrain,Ytrain,Xtest,Ytest,optimizer= optimizer,epochs=epochs)

    new_model.evaluate(Xtest,Ytest)












