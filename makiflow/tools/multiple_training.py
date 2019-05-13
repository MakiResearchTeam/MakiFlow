from __future__ import absolute_import

# For saving train info
import pandas as pd
# For creating test folders
import os
# For creating test configuration files
import json

import tensorflow as tf
from makiflow.save_recover.builder import Builder
from makiflow.ssd.tools.testing import SSDTester
from makiflow.ssd.tools.data_preparing import DataPreparator


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
            Example: 'my_folder/folder_for_tests/'
        """
        self.__json_path_list = json_path_list
        self.__where_save_results = where_save_results
        
    def set_test_params(self, params_dict, class_name_to_num):
        self.__training_params = params_dict
        self.__class_name_to_num = class_name_to_num
        
        self.__test_tester = SSDTester()
        self.__test_on_train_data = params_dict['test on train data']
        if self.__test_on_train_data:
            self.__train_tester = SSDTester()
    
    def set_optimizer(self, tf_optimizer):
        self.__tf_optimizer
        
    def set_train_dataset(self, path_to_train_images, train_masks, train_labels, train_locs, annotation_dict):
        self.__train_preparator = DataPreparator(annotation_dict, self.__class_name_to_num, path_to_train_images)
        self.__train_preparator.load_images()
        # 
        self.__train_images = train_images
        self__train_masks = train_masks
        self.__train_labels = train_labels
        self.__train_locs = train_locs
        self.__train_annotation_dict = annotation_dict
        
        if self.__test_on_train_data:
            self.__train_tester.prepare_ground_truth_labels(annotation_dict, self.__class_name_to_num)
    
    def set_test_dataset(self, test_images, annotation_dict):
        self.__test_images = test_images
        self.__test_annotation_dict = annotation_dict
        self.__test_tester.prepare_ground_truth_labels(annotation_dict, self.__class_name_to_num)
    
    def start_testing(self):
        for architeture in self.__json_path_list:
            test_number = 0
            prepare_training_data(architeture)
            for lr in self.__training_params['learning rates']:
                for batch_sz in self.__training_params['batch sizes']:
                    for loc_loss_weight in self.__training_params['loc loss weights']:
                        for neg_samples_ratio in self.__training_params['neg samples ratios']:
                            self.__sub_testing()
    
    def __sub_testing(self, architeture, test_id, lr, batch_sz, loc_loss_weight, neg_samples_ration):
        session = tf.Session()
        model = Builder.ssd_from_json(architeture, batch_sz)
        model.set_session(session)
        
        # Create folder where all the test data will be stored
        test_folder_path = self.__where_save_results + model.name + '_test_' + str(test_id) + '/'
        os.makedirs(test_folder_path)
        
        # Create json file with the configuration of the test
        config_dict = {
            'learning rate': lr,
            'batch size': batch_sz,
            'loc loss weights': loc_loss_weight,
            'neg samples ratios': neg_samples_ration
        }
        config_json = json.dumps(config_dict, indent=1)
        json_file = open(test_folder_path+'config.json', mode='w')
        json_file.write(config_json)
        json_file.close()
        
        # Get data for easier access
        # Training
        images = self.__train_images
        masks = self.__train_masks
        labels = self.__train_labels
        locs = self.__train_locs
        # Testing
        test_images = self.__test_images
        
        optimizer = self.__tf_optimizer(lr)
        epochs = self.__training_params['epochs']
        test_period = self.__training_params['test period']
        save_after_test = self.__training_params['save after test']  # Boolean
        test_on_train_data = self.__training_params['test on train data']  # Boolean
        # Get testing parameters
        conf_trashhold = self.__training_params['testing confidence trashhold']
        iou_trashhold = self.__training_params['testing iou']
        
        training_iterations = epochs // test_period
        
        # CREATE NECESSARY TABLES FOR STORING ALL THE DATA
        # Loss table
        pos_conf_losses = []
        neg_cong_losses = []
        loc_losses = []
        train_info = {
            'pos_conf_losses': pos_conf_losses,
            'neg_conf_losses': neg_cong_losses,
            'loc_losses': loc_losses
        }
        # Test precision table
        test_map_info = {
            'map': []
        }
        for class_name in self.__class_name_to_num:
            class_precision = {
                class_name: []
            }
            test_map_info.update(class_precision)
            
        # Train precision table
        if self.__test_on_train_data:
            train_map_info = {
                'map': []
            }
            for class_name in self.__class_name_to_num:
                class_precision = {
                    class_name: []
                }
                train_map_info.update(class_precision)
        
        
        
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
            # Collect data about losses
            pos_conf_losses += sub_train_info['pos conf losses']
            neg_cong_losses += sub_train_info['neg conf losses']
            loc_losses += sub_train_info['loc losses']
            
            # SAVING PART
            if save_after_test:
                path = self.__where_save_results+model.name+'_test_'+str(test_)
                model.save_weights(
                    test_folder_path+model.name+'_epoch_'+str(i)+'.ckpt')
            
            # TESTING PART
            # On test data
            metrics = self.__test_tester.mean_average_precision(
                ssd=model,
                images=test_images,
                conf_trashhold=conf_trashhold,
                iou_trashhold=iou_trashhold
            )
            test_map_info['map'].append(metrics[0])
            # Collect average precision values
            for i in len(metrics[1]):
                test_map_info[self.__class_name_to_num[i+1]].append(metrics[1][i][0])
            
            # On train data
            if self.__test_on_train_data:
                metrics = self.__test_tester.mean_average_precision(
                    ssd=model,
                    images=test_images,
                    conf_trashhold=conf_trashhold,
                    iou_trashhold=iou_trashhold
                )
                train_map_info['map'].append(metrics[0])
                # Collect average precision values
                for i in len(metrics[1]):
                    train_map_info[self.__class_name_to_num[i+1]].append(metrics[1][i][0])
                    
        # Free all taken resources
        session.close()
        tf.reset_default_graph()
                    
        # SAVE ALL THE COLLECTED DATA
        loss_df = pd.DataFrame(train_info)
        loss_df.to_csv(test_folder_path+'loss_info.csv')
        
        test_map_df = pd.DataFrame(test_map_info)
        test_map_df = pd.DataFrame(test_folder_path+'test_map_info.csv')
        
        if self.__test_on_train_data:
            train_map_df = pd.DataFrame(test_map_info)
            train_map_df = pd.DataFrame(test_folder_path+'train_map_info.csv')
        
        print('Test', test_id,'is finished.')
            
            
                

        
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
