from xmlrpc.client import boolean
from Configuration.Config import ConfigurationManager
from Components.data_preparation import DataPreparation
from Logger import logger

STAGE_NAME = 'Data Preparation Stage'

class DataPreparationTrainingPipeline:

    def __init__(self,Training:bool = False,Validation:bool = False,Testing:bool = False):
        self.Training = Training
        self.Validation = Validation
        self.Testing = Testing

    def main(self):

        config = ConfigurationManager()
        data_preparation_config = config.get_data_preparation_config()
        data_preparation = DataPreparation(config = data_preparation_config)
        self.train_data,self.validation_data,self.testing_data = data_preparation.get_data(self.Training,self.Validation,self.Testing)

        return self.train_data,self.validation_data,self.testing_data

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline(Testing=True)
        train_data,validation_data,testing_data = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

