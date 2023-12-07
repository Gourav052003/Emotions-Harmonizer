from Configuration.Config import ConfigurationManager
from Components.model_training import ModelTraining
from Logger import logger

STAGE_NAME = "Model Training"

class ModelTrainingPipeline:

    def __init__(self,Model= None,training_data=None,validation_data=None):
        self.training_data = training_data
        self.validation_data = validation_data
        self.Model = Model
        

    def main(self):

        config = ConfigurationManager()
        model_callbacks_config = config.get_model_callbacks_config()
        model_training_config = config.get_model_training_config()

        model_training = ModelTraining(callbacks_config = model_callbacks_config,
                    training_config = model_training_config)

        if self.Model==None:
            logger.info(f"Model is not provided!")
            return
 
        if self.training_data==None:
            logger.info(f"Training data is not provided!")
            return

        if self.validation_data==None:
            logger.info(f"Validation data is not provided!")
            return    

        model_training.Train(self.Model,self.training_data,self.validation_data)


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        