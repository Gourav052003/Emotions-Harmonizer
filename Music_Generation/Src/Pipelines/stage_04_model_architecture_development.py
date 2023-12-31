import os
import sys

path = os.path.abspath("Music_Generation/Src")
sys.path.append(path)

from Configuration.Config import ConfigurationManager
from Components.model_architecture import ModelArchitecture
from Logger import logger


STAGE_NAME = "Model Architecture Development"

class ModelArchitectureDevelopmentTrainingPipeline:

    def __init__(self):
        pass

    def start(self):
        config = ConfigurationManager()
        Model_architecture_config = config.get_model_architecture_path_config()
        model_architecture = ModelArchitecture(config = Model_architecture_config) 
        model_architecture.Encoder_Decoder_model()
        


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Model_architecture_developement_pipeline = ModelArchitectureDevelopmentTrainingPipeline()
        Model_architecture_developement_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e