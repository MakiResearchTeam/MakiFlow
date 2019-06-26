import numpy as np
import cv2
from tqdm import tqdm
import random 

class ImageAugmentator:
    def __init__(self):
        pass

    def set_parameters(self, change_scale=None, noise=None):
        """
        Parameters
        ----------
        change_scale : tuple of floats
            Range in which all the images will be randomly rescaled.
        noise : tuple of float and int
            First value in the tuple is scale of the noise, i.e. standart deviation; second
            value in the tuple is how many values in the interval (0, noise_scale) will be
            used for creation of new images.
        """
        self.change_scale = change_scale
        self.noise = noise

    def augment_data(self, Xtrain, Ytrain):
        """
        The method implies all the images have the same size.
        """
        new_Xtrain = []
        new_Ytrain = []
        mean_image = Xtrain[0].mean()
        to_right_side = False
        for image, label in tqdm(zip(Xtrain, Ytrain)):
            new_Xtrain.append(image)
            new_Ytrain.append(label)

            
            if self.change_scale is not None:
                old_size = image.shape
                new_size = (int(old_size[0]*self.change_scale[0]), int(old_size[1]*self.change_scale[1]))
                new_image = image.transpose([1, 0, 2])
                new_image = cv2.resize(new_image, new_size)
                new_image = new_image.transpose([1, 0])
                mask = np.ones(old_size) * mean_image
                if to_right_side:
                    mask[old_size[0]-new_size[0]:,old_size[1]-new_size[1]:, 0] = new_image
                    to_right_side = not to_right_side
                else:
                    mask[:new_size[0],:new_size[1], 0] = new_image
                    to_right_side = not to_right_side

                new_Xtrain.append(mask)
                new_Ytrain.append(label) 
            
            if self.noise is not None:
                first = True
                for scale in np.linspace(0, self.noise[0], self.noise[1]):
                    if first:
                        first = False
                        continue
                    new_image = self.add_noise(image, scale=scale)
                    new_Xtrain.append(new_image)
                    new_Ytrain.append(label)
        
        print('Old size:', len(Xtrain), 'New size:', len(new_Xtrain))
        return new_Xtrain, new_Ytrain

                
    
    def add_noise(self, image, scale=1):
        image_shape = image.shape
        noise = np.random.normal(size=image_shape, scale=scale).astype(np.float32)
        image += noise
        return image