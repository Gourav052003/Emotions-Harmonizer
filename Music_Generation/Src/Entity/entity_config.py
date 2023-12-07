from dataclasses import dataclass
from pathlib import Path
from github import Repository


@dataclass(frozen=True)
class DataIngestionConfig:
    repository:Repository
    folder:Path
    save_folder:Path

@dataclass(frozen=True)
class DataValidationConfig:
    source_music_directory:Path
    train_music_directory:Path
    validation_music_directory:Path
    test_music_directory:Path  
    test_size:float
    validation_size:float  

@dataclass(frozen=True)
class DataPreparationConfig:
    train_music_directory:Path
    validation_music_directory:Path
    test_music_directory:Path 
    duration:float
    input_timestamp:float
    output_timestamp:float
    sample_rate:int

@dataclass(frozen=True)
class ModelArchitecturePathConfig():
    sample_rate:int
    encoder_timestamp:int
    decoder_timestamp:int
    learning_rate:float
    loss:str

@dataclass(frozen=True)
class ModelCallbacksConfig:
    save_best_model_path:Path

@dataclass(frozen=True)
class ModelTrainingConfig:
    epochs:int
    batch_size:int
    validation_batch_size:int

@dataclass(frozen = True)
class ModelTestingConfig:
    best_model_path:Path 
    testing_file_path:Path 
    testing_duration:int  


