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
        Data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
        Data_ingestion_training_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e 


    STAGE_NAME = "Data Validation stage"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Data_ingestion_training_pipeline= DataValidationTrainingPipeline()
        Data_ingestion_training_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e


    STAGE_NAME = "Data Preparation stage"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Data_preparation_training_pipeline = DataPreparationTrainingPipeline()
        Data_preparation_training_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Model Architecture Development"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Model_architecture_developement_pipeline = ModelArchitectureDevelopmentTrainingPipeline()
        Model_architecture_developement_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e

    STAGE_NAME = "Model Training"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Model_training_pipeline = ModelTrainingPipeline()
        Model_training_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e  
      


    STAGE_NAME = "Model Testing"

    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        Model_testing_pipeline = ModelTestingPipeline()
        Model_testing_pipeline.start()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e        




