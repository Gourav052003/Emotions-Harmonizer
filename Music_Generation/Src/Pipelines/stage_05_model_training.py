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

    def start(self):

        config = ConfigurationManager()
        model_callbacks_config = config.get_model_callbacks_config()
        model_training_config = config.get_model_training_config()

        model_training = ModelTraining(callbacks_config = model_callbacks_config,
                    training_config = model_training_config)

        model_training.Train()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Model_training_pipeline = ModelTrainingPipeline()
        Model_training_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        