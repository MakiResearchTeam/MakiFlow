from __future__ import print_function

import time
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import backend as K
import numpy as np
import tensorflow as tf
import cv2


class FilterVisualizer(object):
    def __init__(self, conv_model):
        self.model = conv_model
        self.layers = conv_model.layers
    
    
    def __get_layer_output(layer_ind, filter_ind):
        loss = img
        for i in range(layer_ind):
            loss = layers[i].forward(loss)

        kernel = layers[layer_ind].params[0]
        loss = tf.nn.conv2d(loss, kernel, strides=[1, 1, 1, 1], padding='SAME')


    def __normalize(self, x):
        """utility function to normalize a tensor.
        # Arguments
            x: An input tensor.
        # Returns
            The normalized input tensor.
        """
        return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


    def __process_image(self, x, former):
        """utility function to convert a valid uint8 image back into a float array.
           Reverses `deprocess_image`.
        # Arguments
            x: A numpy-array, which could be used in e.g. imshow.
            former: The former numpy-array.
                    Need to determine the former mean and variance.
        # Returns
            A processed numpy-array representing the generated image.
        """
        if K.image_data_format() == 'channels_first':
            x = x.transpose((2, 0, 1))
        return (x / 255 - 0.5) * 4 * former.std() + former.mean()


    def __deprocess_image(self, x):
        """utility function to convert a float array into a valid uint8 image.
        # Arguments
            x: A numpy-array representing the generated image.
        # Returns
            A processed numpy-array, which could be used in e.g. imshow.
        """
        # normalize tensor: center on 0., ensure std is 0.25
        x -= x.mean()
        x /= (x.std() + K.epsilon())
        x *= 0.25

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        if K.image_data_format() == 'channels_first':
            x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x


    def __activation_loss(self, layer_ind, filter_ind, img):
        loss = img
        for i in range(layer_ind+1):
            loss = self.layers[i].forward(loss)

        return K.mean(loss[:, :, :, filter_ind])


    def __get_grads(self, loss, img):
        grads = K.gradients(loss, img)[0]
        return grads


    def generate_image(self, output_dim, layer_ind, filter_ind, 
                       epochs=25, upscaling_steps=9):
        """ 
        #Arguments:
        output_dim - tuple contains size of the output image. Example:  (512, 512)
        layer_ind - layer index which to take filter from.
        filter_ind - filter index which to visualize.
        """
        
        upscaling_factor = 1.2
        upscaling_steps = 20
        intermediate_dim = tuple(
                int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        input_img_data = np.random.random(
                        (1, intermediate_dim[0], intermediate_dim[1], 1)).astype(np.float32)
        input_img_data = (input_img_data - 0.5) * 20 + 128

        img = tf.placeholder(tf.float32, shape=[1, None, None, None])

        loss = self.__activation_loss(layer_ind, filter_ind, img)
        grads = K.gradients(loss, img)[0]
        grads = self.__normalize(grads)
        iterate = K.function([img], [loss, grads])


        for up in reversed(range(upscaling_steps)):
            # we run gradient ascent for e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value

            # Calulate upscaled dimension
            intermediate_dim = tuple(
                int(x / (upscaling_factor ** up)) for x in output_dim)
            # Upscale
            img = self.__deprocess_image(input_img_data[0])
            img = np.array(cv2.resize(img, intermediate_dim))
            img = img.reshape([*img.shape, 1])
            input_img_data = [self.__process_image(img, input_img_data[0])]

        finale_img = self.__deprocess_image(input_img_data[0])
        print(loss_value)
        return finale_img