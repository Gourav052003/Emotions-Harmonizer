from Entity.entity_config import DataValidationConfig
import os
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataValidation:

    def __init__(self,config:DataValidationConfig):
      
        self.test_size = config.test_size
        self.validation_size = config.validation_size
        self.source_music_directory = config.source_music_directory
        self.train_music_directory = config.train_music_directory
        self.validation_music_directory = config.validation_music_directory
        self.test_music_directory = config.test_music_directory
        
    def train_validation_test_splitting(self):

        music_files = np.array(os.listdir(self.source_music_directory))

        self.train,self.test = train_test_split(music_files,
                                        test_size = self.test_size,
                                        shuffle = True)

        self.train,self.validation = train_test_split(self.train,
                                            test_size = self.validation_size,    
                                            shuffle = True)

     
    def copy_to(self,filenames,destination_music_dir):

        for filename in filenames:

            source_music_path = Path(os.path.join(self.source_music_directory,filename))
            shutil.copy2(source_music_path,destination_music_dir)


    def execute_splitting(self):
        
        music_dir = [self.train_music_directory,
                    self.validation_music_directory,
                    self.test_music_directory]

        for m_dir in music_dir:

            os.makedirs(m_dir,exist_ok = True)

            for music_file in os.listdir(m_dir):
                if music_file != '.ipynb_checkpoints':
                    os.remove(Path(os.path.join(m_dir,music_file)))


        self.copy_to(self.train,self.train_music_directory)

        self.copy_to(self.validation,self.validation_music_directory)

        self.copy_to(self.test,self.test_music_directory)
    