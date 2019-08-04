from makiflow.layers import InputLayer, DenseLayer, ConvLayer, MaxPoolLayer, FlattenLayer, BatchNormLayer,SumLayer, ConcatLayer
from makiflow.models.classificator import Classificator
from makiflow.models.ssd import SSDModel, DetectorClassifier
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import cifar10


def get_ssd_outs():
    in_x = InputLayer(input_shape=[32, 12, 12, 3], name='Input')
    x = ConvLayer(kh=3, kw=3, in_f=3, out_f=16, name='conv1')(in_x)
    x = MaxPoolLayer(name='pool1')(x)
    # 6x6
    x = ConvLayer(kh=3, kw=3, in_f=16, out_f=16, name='conv2')(x)
    x = MaxPoolLayer(name='pool2')(x)
    # 3x3
    id1 = DetectorClassifier(
        x, kw=1, kh=1, in_f=16,
        num_classes=3, dboxes=[(1, 1)],
        name='id1'
    )

    return in_x, id1


def create_ssd_training_data():
    images = np.random.randn(128, 12, 12, 3)

    loc_mask = [0, 0, 0, 0, 0, 0, 1, 1, 1]
    loc_masks = np.array(loc_mask*128).reshape([128, 9])
    labels = np.ones((128, 9))
    gt_locs = np.ones((128, 9, 4))
    return images, loc_masks, labels, gt_locs


def test_ssd():
    in_x, id1 = get_ssd_outs()
    ssd = SSDModel([id1], input_s=in_x)
    ssd.set_session(tf.Session())
    images, loc_masks, labels, gt_locs = create_ssd_training_data()
    optimizer = tf.train.AdamOptimizer(learning_rate=10e-5)
    predictions = ssd.fit(images, loc_masks, labels, gt_locs, neg_samples_ratio=1.0, epochs=10, optimizer=optimizer,
                          loss_type='scan_loss')
    print(predictions)

def get_layers():
    in_x = InputLayer(input_shape=[64, 32,32,3], name='input')
    x = ConvLayer(kh=3, kw=3, in_f=3, out_f=16, name='conv1')(in_x)
    x = MaxPoolLayer(name='pool1')(x)
    # 16x16
    x = BatchNormLayer(D=16,name='batch1')(x)
    x = ConvLayer(kh=3, kw=3, in_f=16, out_f=32, name='conv2')(x)
    x = MaxPoolLayer(name='pool2')(x)
    # 8x8
    q = ConvLayer(kh=3, kw=3, in_f=32, out_f=31, name='conv3_0')(x)
    z = ConvLayer(kh=5, kw=5, in_f=32, out_f=57, name='conv3_1')(x)
    c = ConvLayer(kh=2, kw=2, in_f=32, out_f=43, name='conv3_2')(x)
    x = ConcatLayer(name='conc')([q,z,c])
    x = MaxPoolLayer(name='pool3')(x)
    # 4x4
    x = BatchNormLayer(D=131,name='batch3')(x)
    x = ConvLayer(kh=3, kw=3, in_f=131, out_f=128, name='conv4')(x)
    x = MaxPoolLayer(name='pool4')(x)
    # 2x2
    x = BatchNormLayer(D=128,name='batch4')(x)
    x = ConvLayer(kh=3, kw=3, in_f=128, out_f=64, name='conv5')(x)
    x = MaxPoolLayer(name='pool5')(x)
    # 1x1
    x = FlattenLayer(name='flatten')(x)
    x = DenseLayer(in_d=64,out_d=10,activation=None,name='outputsZ')(x)
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
    epochs = 2
    lr = 1e-3
    epsilon = 1e-8
    optimizer = tf.train.RMSPropOptimizer(learning_rate=lr, epsilon=epsilon)
    info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    model.save_architecture('T:/download/shiru/shit.json')
    model.save_weights(path='T:/download/shiru/')


    new_model = Builder.classificator_from_json('T:/download/shiru/shit.json')
    new_session = tf.Session()
    new_model.set_session(new_session)
    new_model.load_weights(path='T:/download/shiru/')
    new_model.evaluate(Xtest,Ytest,64)
    info = new_model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=epochs)
    new_model.save_architecture('T:/download/shiru/new.json')
