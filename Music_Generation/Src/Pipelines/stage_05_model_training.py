import os
import sys

path = os.path.abspath("Music_Generation/Src")
sys.path.append(path)

from Configuration.Config import ConfigurationManager
from Components.model_training import ModelTraining
from Logger import logger

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:

    def __init__(self):
        pass
        # self.training_data = training_data
        # self.validation_data = validation_data
        # self.Model = Model
        

    def main(self):

        config = ConfigurationManager()
        model_callbacks_config = config.get_model_callbacks_config()
        model_training_config = config.get_model_training_config()

        model_training = ModelTraining(callbacks_config = model_callbacks_config,
                    training_config = model_training_config)

        # if self.Model==None:
        #     logger.info(f"Model is not provided!")
        #     return
 
        # if self.training_data==None:
        #     logger.info(f"Training data is not provided!")
        #     return

        # if self.validation_data==None:
        #     logger.info(f"Validation data is not provided!")
        #     return    

        model_training.Train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        