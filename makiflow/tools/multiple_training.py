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
            String represent FULL path to the folder where all the test info will be saved.
        """
        self.__json_path_list = json_path_list
        self.__where_save_results = where_save_results
        
    def set_test_params(self, params_dict):
        self.__training_params = params_dict
    
    def set_optimizer(self, tf_optimizer):
        self.__tf_optimizer
        
    def set_train_dataset(self, train_images, train_masks, train_labels, train_locs):
        self.__train_images = train_images
        self__train_masks = train_masks
        self.__train_labels = train_labels
        self.__train_locs = train_locs
    
    def set_test_dataset(self, test_images, test_masks, test_labels, test_locs):
        self.__test_images = test_images
        self.__test_masks = test_masks
        self.__test_labels = test_labels
        self.__test_locs = test_locs
    
    def start_testing(self):
        for architeture in self.__json_path_list:
            test_number = 0
            for lr in self.__training_params['learning rates']:
                for batch_sz in self.__training_params['batch sizes']:
                    for loc_loss_weight in self.__training_params['loc loss weights']:
                        for neg_samples_ratio in self.__training_params['neg samples ratios']:
                            self.__sub_testing()
    
    def __sub_testing(self, architeture, test_id, lr, batch_sz, loc_loss_weight, neg_samples_ration):
        model = Builder.ssd_from_json(architeture, batch_sz)
        # Get data for easier access
        images = self.__train_images
        masks = self.__train_masks
        labels = self.__train_labels
        locs = self.__train_locs
        
        optimizer = self.__tf_optimizer(lr)
        epochs = self.__training_params['epochs']
        test_period = self.__training_params['test period']
        save_after_test = self.__training_params['save after test']  # Boolean
        test_on_train_data = self.__training_params['test on train data']  # Boolean
        
        training_iterations = epochs // test_period
        train_info = {}
        for i in range(training_iterations):
            # TRAINING PART
            sub_train_info = model.fit(images=images, 
                                          masks=masks, 
                                          labels=labels, 
                                          locs=locs, 
                                          loc_loss_weigth=loc_loss_weight, 
                                          neg_samples_ration=neg_samples_ration, 
                                          optimizer=optimizer, 
                                          epochs=test_period)
            if save_after_test:
                path = self.__where_save_results+model.name+'_test_'+str(test_)
                model.save_weights(path)
            
            # TESTING PART
            
            
            
            
                

        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
