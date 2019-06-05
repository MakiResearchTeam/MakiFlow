from __future__ import absolute_import

from makiflow.conv_model import ConvModel
from makiflow.save_recover.builder import Builder
from makiflow.tools.test_visualizer import TestVisualizer
import tensorflow as tf
# For saving train info
import pandas as pd
# For creating test folders
import os
# For creating test configuration files
import json


"""
EXAMPLE OF THE TEST PARAMETERS:
params = {
    'epochs': 50,
    'test period': 10,
    'save after test': True,
    'learning rates': [1e-3, 1e-4, 1e-5, 1e-6],
    'batch sizes': [32, 64, 96, 128],
}
"""

class ConvModelTester:
    def __init__(self, json_paths_list, where_to_save):
        """
        Parameters
        ----------
        
        json_paths_list : list
            List of strings represent paths to jsons contain ConvModels' architectures.
        where_to_save : string
            String represent FULL path to the folder where all the test info will be saved.
            Example: 'my_folder/folder_for_tests/'
        """
        self.__json_path_list = json_paths_list
        self.__where_to_save = where_to_save
        self.__best_params = {
            'architecture': None,
            'accuracy': 0.0,
            'test_id': None
        }
        
    def set_hyperparams(self, params_dict):
        self.__params = params_dict
        
        
    def set_train_test_data(self, Xtrain, Ytrain, Xtest, Ytest):
        self.__xtrain = Xtrain
        self.__ytrain = Ytrain
        self.__xtest = Xtest
        self.__ytest = Ytest
    
    
    def set_optimizer(self, optimizer):
        self.__optimizer = optimizer
        
    
    def begin(self):
        for architecture in self.__json_path_list:
            test_id = 0
            for lr in self.__params['learning rates']:
                for batch_sz in self.__params['batch sizes']:
                    self.__train_test_save(architecture, lr, batch_sz, test_id)
                    test_id += 1
        
        # Save info about best one
        best_json = json.dumps(self.__best_params, indent=1)
        json_file = open(self.__where_to_save+'best.json', mode='w')
        json_file.write(best_json)
        json_file.close()
    
    
    def __train_test_save(self, architecture, lr, batch_sz, test_id):
        model = Builder.convmodel_from_json(architecture, batch_sz)
        session = tf.Session()
        model.set_session(session)
        
        # Create folder where all the test data will be stored
        test_folder_path = self.__where_to_save + model.name + '_test_' + str(test_id) + '/'
        os.makedirs(test_folder_path)
        
        # Create json file with the configuration of the test
        config_dict = {
            'learning rate': lr,
            'batch size': batch_sz,
        }
        config_json = json.dumps(config_dict, indent=1)
        json_file = open(test_folder_path+'config.json', mode='w')
        json_file.write(config_json)
        json_file.close()
        
        # Get training and testing data for easier access
        Xtrain = self.__xtrain
        Ytrain = self.__ytrain
        Xtest = self.__xtest
        Ytest = self.__ytest
        
        # Get some params for easier access
        save_after_test = self.__params['save after test']
        optimizer = self.__optimizer(lr)
        
        # Create holders for test data
        train_costs = []
        train_errors = []
        test_costs = []
        test_errors = []
        
        test_period = self.__params['test period']
        train_operations = self.__params['epochs'] // test_period
        for i in range(train_operations):
            info = model.pure_fit(Xtrain, Ytrain, Xtest, Ytest, optimizer=optimizer, epochs=test_period)
            train_costs += info['train costs']
            train_errors += info['train errors']
            test_costs += info['test costs']
            test_errors += info['test errors']
            
            if save_after_test:
                model.save_weights(
                    test_folder_path+model.name+'_epoch_'+str((i+1)*test_period)+'.ckpt')
            
            # Check if the model better than previuos ones
            if (1 - test_errors[-1]) > self.__best_params['accuracy']:
                self.__best_params['architecture'] = model.name
                self.__best_params['accuracy'] = 1 - test_errors[-1]
                self.__best_params['test id'] = test_id
                
        # Free all taken resources
        session.close()
        tf.reset_default_graph()
                
        # Create graphs with cost and error values        
        TestVisualizer.plot_test_values([train_costs, test_costs], ['train cost', 'test cost'],
                                        x_label='epochs', y_label='cost',
                                        save_path=test_folder_path+'train_test_costs.png')
        TestVisualizer.plot_test_values([train_errors, test_errors], ['train errors', 'test error'],
                                        x_label='epochs', y_label='error',
                                        save_path=test_folder_path+'train_test_errors.png')
        
        test_data = {'train costs': train_costs, 'train errors': train_errors,
                'test costs': test_costs, 'test errors': test_errors}
        test_data_df = pd.DataFrame(test_data)
        test_data_df.to_csv(test_folder_path+'test_data.csv')
        print('Test', test_id, 'is finished.')
        
            
                
        