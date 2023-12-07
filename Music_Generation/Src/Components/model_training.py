from gc import callbacks
from Entity.entity_config import (ModelCallbacksConfig,
                                ModelTrainingConfig)

from keras.callbacks import ModelCheckpoint                                 


class ModelTraining:

    def __init__(self,callbacks_config:ModelCallbacksConfig,
                training_config:ModelTrainingConfig):

       self.callbacks_config = callbacks_config         
       self.training_config =training_config


    def Train(self,Model,training_data,validation_data):
        

        model_checkpoint = ModelCheckpoint(save_best_only = True,filepath = self.callbacks_config.save_best_model_path)
        
        self.model_hist = Model.fit(x = training_data[0], y = training_data[1],
                                    epochs = self.training_config.epochs,
                                    batch_size = self.training_config.batch_size,
                                    validation_data = (validation_data[0],validation_data[1]),
                                    validation_batch_size = self.training_config.validation_batch_size,
                                    callbacks=[model_checkpoint])


                                    

