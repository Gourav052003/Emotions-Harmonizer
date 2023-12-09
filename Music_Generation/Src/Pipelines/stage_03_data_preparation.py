import os
import sys

path = os.path.abspath("Music_Generation/Src")
sys.path.append(path)

from Configuration.Config import ConfigurationManager
from Components.data_preparation import DataPreparation
from Logger import logger

STAGE_NAME = 'Data Preparation Stage'

class DataPreparationTrainingPipeline:

    def __init__(self):
        pass

    def main(self):

        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config = data_preparation_config)
        data_preparation.get_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

