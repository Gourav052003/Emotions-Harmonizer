import os
import sys

path = os.path.abspath("Music_Generation/Src")
sys.path.append(path)

from distutils.command.config import config
from pyexpat import model
from Configuration.Config import ConfigurationManager
from Components.model_testing import ModelTesting
from Logger import logger

STAGE_NAME = "Model Testing"

class ModelTestingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        model_testing_config = config.get_model_testing_config()
        model_testing = ModelTesting(config = model_testing_config)
        model_testing.predict()

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTestingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e           
