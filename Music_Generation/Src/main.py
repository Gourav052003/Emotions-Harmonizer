from Logger import logger
from Pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from Pipelines.stage_03_data_preparation import DataPreparationTrainingPipeline
from Pipelines.stage_04_model_architecture_development import ModelArchitectureDevelopmentTrainingPipeline
from Pipelines.stage_05_model_training import ModelTrainingPipeline
from Pipelines.stage_06_model_testing import ModelTestingPipeline

STAGE_NAME = "Data Ingestion stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 


STAGE_NAME = "Data Validation stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataValidationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


STAGE_NAME = "Data Preparation stage"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataPreparationTrainingPipeline(Validation=True,Testing=True)
        train_data,validation_data,testing_data = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Architecture Development"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelArchitectureDevelopmentTrainingPipeline()
        model = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

STAGE_NAME = "Model Training"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline(model,validation_data,testing_data)
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e  
      


STAGE_NAME = "Model Testing"

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTestingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        




