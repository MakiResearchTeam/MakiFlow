from __future__ import absolute_import

# For saving train info
import pandas as pd
import tensorflow as tf

from makiflow.save_recover.builder import Builder


class ConvModelTrainer:
    """
    This class is made for training and testing multiple models.
    Trainer creates his own Sessions, be aware of that, it can cause RESOURCE_EXHAUSTED
    error if you already have a session on the same GPU.
    """

    def __init__(self, path_to_json, path_where_to_save):
        """
        :param path_to_json - path to JSON file where the model's architecture is stored.
        :param path_where_to_save - (IMPORTANT!) full path to the folder where trainer will save results
         (tests, trained models, etc)
        """
        self.path_to_json = path_to_json
        self.path_where_to_save = path_where_to_save

    def set_train_params(self, Xtrain, Ytrain, Xtest, Ytest, optimizer=None, epochs=1, test_period=1, model_count=1):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest = Xtest
        self.Ytest = Ytest
        self.optimizer = optimizer
        self.epochs = epochs
        self.test_period = test_period
        self.model_count = model_count

    def start_training(self):
        # TODO: info_frame is never used

        assert (self.Xtrain is not None)
        assert (self.Ytrain is not None)
        assert (self.Xtest is not None)
        assert (self.Ytest is not None)
        assert (self.optimizer is not None)
        assert (self.test_period is not None)
        assert (self.model_count is not None)
        for i in range(self.model_count):
            session = tf.Session()
            model = Builder.convmodel_from_json(self.path_to_json)
            model.name = model.name + str(i)
            model.set_session(session)
            train_info = model.pure_fit(self.Xtrain, self.Ytrain, self.Xtest, self.Ytest,
                                        optimizer=self.optimizer, epochs=self.epochs, test_period=self.test_period)
            print('Model {} is trained, saving weights and train info...'.format(model.name))
            model.save_weights(self.path_where_to_save + '/' + model.name + '.ckpt')
            info_frame = pd.DataFrame(train_info).to_csv(self.path_where_to_save + '/' + model.name + '.csv')
            session.close()
            tf.reset_default_graph()
            print('Success, start next training iteration.')
            

class SSDModelTrainer:
    def __init__(self, json_path_list, where_save_results):
        """
        Parameters
        ----------
        
        json_path_list : list
            List of strings represent paths to jsons contain SSDs' architectures.
        where_save_results : string
            String represent path to the folder where all the test info will be saved.
        """
        self.json_path_list = json_path_list
        self.where_save_results = where_save_results
        
        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
