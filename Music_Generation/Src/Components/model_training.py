from Entity.entity_config import (ModelCallbacksConfig,
                                ModelTrainingConfig)

from keras.callbacks import ModelCheckpoint  
from Utils import load_object 
import os                              


class ModelTraining:

    def __init__(self,callbacks_config:ModelCallbacksConfig,
                training_config:ModelTrainingConfig):
        
        self.pickle_model_architecture_directory = training_config.pickle_model_architecture_directory
        self.save_best_model_path = callbacks_config.save_best_model_path
        self.epochs = training_config.epochs
        self.batch_size = training_config.batch_size
        self.validation_batch_size = training_config.validation_batch_size
        self.pickle_train_directory = training_config.pickle_train_directory
        # self.pickle_test_directory = training_config.pickle_test_directory
        self.pickle_validation_directory = training_config.pickle_validation_directory



    def Train(self):

        # pickle_model_architecture = load_object(os.path.join(self.pickle_model_architecture_directory,"model_architecture"))
        print("Model loaded")
        os.makedirs(self.save_best_model_path,exist_ok=True)

        best_model = os.path.join(self.save_best_model_path,'best_model.h5')
        model_checkpoint = ModelCheckpoint(save_best_only = True,filepath = best_model)

        training_input_data = load_object(os.path.join(self.pickle_train_directory,"train_input"))
        training_output_data = load_object(os.path.join(self.pickle_train_directory,"train_output"))
        print("train data loaded")
        
        validation_input_data = load_object(os.path.join(self.pickle_validation_directory,"validation_input"))
        validation_output_data = load_object(os.path.join(self.pickle_validation_directory,"validation_output"))
        
        print("validation data loaded")

        # self.model_hist = pickle_model_architecture.fit(x = training_input_data, y = training_output_data,
        #                             epochs = self.epochs,
        #                             batch_size = self.batch_size,
        #                             validation_data = (validation_input_data,validation_output_data),
        #                             validation_batch_size = self.validation_batch_size,
        #                             callbacks=[model_checkpoint])


                                    

