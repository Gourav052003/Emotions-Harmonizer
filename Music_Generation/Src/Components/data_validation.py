from Entity.entity_config import DataValidationConfig
import os
import shutil
import numpy as np
from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split

class DataValidation:

    def __init__(self,config:DataValidationConfig):
        self.config = config
        self.music_files = np.array(os.listdir(config.source_music_directory))

    def train_validation_test_splitting(self):

        self.train,self.test = train_test_split(self.music_files,
                                        test_size = self.config.test_size,
                                        shuffle = True)

        self.train,self.validation = train_test_split(self.train,
                                            test_size = self.config.validation_size,    
                                            shuffle = True)

     
    def copy_to(self,filenames,destination_music_dir):

        for filename in filenames:

            source_music_path = Path(os.path.join(self.config.source_music_directory,filename))
            shutil.copy2(source_music_path,destination_music_dir)


    def execute_splitting(self):
        
        music_dir = [self.config.train_music_directory,
                    self.config.validation_music_directory,
                    self.config.test_music_directory]

        for m_dir in music_dir:

            os.makedirs(m_dir,exist_ok = True)

            for music_file in os.listdir(m_dir):
                if music_file != '.ipynb_checkpoints':
                    os.remove(Path(os.path.join(m_dir,music_file)))


        self.copy_to(self.train,self.config.train_music_directory)

        self.copy_to(self.validation,self.config.validation_music_directory)

        self.copy_to(self.test,self.config.test_music_directory)
    