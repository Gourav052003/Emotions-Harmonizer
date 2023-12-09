from Constants import *
from Utils import read_yaml
from Entity.entity_config import (DataIngestionConfig,
                                DataValidationConfig,
                                DataPreparationConfig,
                                ModelArchitecturePathConfig,
                                ModelCallbacksConfig,
                                ModelTrainingConfig,
                                ModelTestingConfig)

from github import Github,ContentFile,Repository

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

    def get_data_ingestion_config(self)->DataIngestionConfig:

        github_object = Github()
        repository = github_object.get_repo(REPOSITORY.as_posix())
        
        data_ingestion_config = DataIngestionConfig(
            repository = repository,
            folder = FOLDER,
            save_folder=SAVE_FOLDER
            )

        return data_ingestion_config  


    def get_data_validation_config(self)->DataValidationConfig:
        
        config = self.config.Data_Validation
        params = self.params.Data_Validation

        data_validation_config = DataValidationConfig(

            source_music_directory=config.SOURCE_MUSIC_DIRECTORY,
            train_music_directory=config.TRAIN_MUSIC_DIRECTORY,
            validation_music_directory=config.VALIDATION_MUSIC_DIRECTORY,
            test_music_directory = config.TEST_MUSIC_DIRECTORY,
            test_size=params.TEST_SIZE,
            validation_size = params.VALIDATION_SIZE

        )

        return data_validation_config


    def get_data_preparation_config(self)->DataPreparationConfig:

        config = self.config.Data_Preparation
        params = self.params.Data_Preparation

        data_preparation_config = DataPreparationConfig(

            train_music_directory=config.TRAIN_MUSIC_DIRECTORY,
            validation_music_directory=config.VALIDATION_MUSIC_DIRECTORY,
            test_music_directory = config.TEST_MUSIC_DIRECTORY,
            pickle_train_directory = config.PICKLE_TRAIN_DIRECTORY,
            pickle_test_directory = config.PICKLE_TEST_DIRECTORY,
            pickle_validation_directory = config.PICKLE_VALIDATION_DIRECTORY,
            duration = params.DURATION,
            input_timestamp = params.INPUT_TIMESTAMP,
            output_timestamp = params.OUTPUT_TIMESTAMP,
            sample_rate = params.SAMPLE_RATE
        )    

        return data_preparation_config
    
    def get_model_architecture_path_config(self)->ModelArchitecturePathConfig:
        
        config = self.config.Model_architecture_development
        params = self.params.Model_config 

        model_architecture_path_config = ModelArchitecturePathConfig(
                pickle_model_architecture_directory = config.PICKLE_MODEL_ARCHITECTURE_DIRECTORY,
                sample_rate=params.SAMPLE_RATE,
                encoder_timestamp=params.ENCODER_TIMESTAMP,
                decoder_timestamp=params.DECODER_TIMESTAMP,
                learning_rate = params.LEARNING_RATE,
                loss = params.LOSS
        )

        return model_architecture_path_config

    def get_model_callbacks_config(self)->ModelCallbacksConfig:

        config = self.config.Model_Callbacks

        model_callbacks_config = ModelCallbacksConfig(

            save_best_model_path=config.SAVE_BEST_MODEL_PATH

        )

        return model_callbacks_config

    def get_model_training_config(self)->ModelTrainingConfig:

        config = self.config.Model_Training
        params = self.params.Model_Training

        model_training_config = ModelTrainingConfig(
            pickle_model_architecture_directory = config.PICKLE_MODEL_ARCHITECTURE_DIRECTORY,
            pickle_train_directory = config.PICKLE_TRAIN_DIRECTORY,
            pickle_test_directory = config.PICKLE_TEST_DIRECTORY,
            pickle_validation_directory = config.PICKLE_VALIDATION_DIRECTORY,
            epochs= params.EPOCHS,
            batch_size=params.BATCH_SIZE,
            validation_batch_size=params.VALIDATION_BATCH_SIZE
        )

        return model_training_config


    def get_model_testing_config(self)->ModelTrainingConfig:

        config = self.config.Model_Testing
        params = self.params.Model_Testing

        model_testing_config = ModelTestingConfig(best_model_path=config.BEST_MODEL_PATH,
                                                testing_file_path=config.TESTING_FILE_PATH,
                                                results_path = config.RESULTS_PATH,
                                                testing_duration=params.TESTING_DURATION,
                                                )

        return model_testing_config


