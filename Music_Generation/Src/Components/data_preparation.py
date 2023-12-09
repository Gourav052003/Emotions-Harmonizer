from xmlrpc.client import Boolean
from Entity.entity_config import DataPreparationConfig
from tqdm import tqdm
from glob import glob
from pickle import dump
import librosa as lbrs
import numpy as np
import os


class DataPreparation:

    def __init__(self,config:DataPreparationConfig):
      
        self.duration = config.duration
        self.sample_rate = config.sample_rate
        self.input_timestamp = config.input_timestamp
        self.output_timestamp = config.output_timestamp
        self.train_music_directory = config.train_music_directory
        self.validation_music_directory = config.validation_music_directory
        self.test_music_directory = config.test_music_directory
        self.pickle_train_directory = config.pickle_train_directory
        self.pickle_test_directory = config.pickle_test_directory
        self.pickle_validation_directory = config.pickle_validation_directory


    def prepare_data(self,MUSIC_DIRECTORY):

        filenames = glob(MUSIC_DIRECTORY+'/*')

        features,sample_rate = lbrs.load(filenames[0],duration = self.duration)

        all_timestamp_data = []

        timestamp_intervals = list(range(0,len(features)+self.sample_rate,
                                self.sample_rate))

        for start_timestamp_no in tqdm(range(len(timestamp_intervals)-1)):

            timestamp_features = []

            for filename in filenames:

                features,sample_rate = lbrs.load(filename,duration = self.duration)

                one_second_features = features[timestamp_intervals[start_timestamp_no]:timestamp_intervals[start_timestamp_no+1]]

                timestamp_features.append(one_second_features)


            all_timestamp_data.append(timestamp_features)


        input_timestamp_data = []
        output_timestamp_data = []

        for timestamp_data in all_timestamp_data:

            if len(input_timestamp_data)<self.input_timestamp:
                input_timestamp_data.append(np.array(timestamp_data))

            elif len(output_timestamp_data)<self.output_timestamp:
                output_timestamp_data.append(np.array(timestamp_data))


        return input_timestamp_data,output_timestamp_data,sample_rate


    def get_data(self,Training:bool = False,Validation:bool = False,Testing:bool = False):

        if Training:
            train_input,train_output,train_sample_rate = self.prepare_data(self.train_music_directory)    
        
        if Validation:
            validation_input,validation_output,validation_sample_rate = self.prepare_data(self.validation_music_directory)    
        
        if Testing:
            test_input,test_output,test_sample_rate = self.prepare_data(self.test_music_directory) 

        os.makedirs(self.pickle_train_directory,exist_ok=True)
        os.makedirs(self.pickle_test_directory,exist_ok=True)
        os.makedirs(self.pickle_validation_directory,exist_ok=True)

        dump(train_input,os.path.join(self.pickle_train_directory,"train_input.pkl"))
        dump(train_input,os.path.join(self.pickle_train_directory,"train_output.pkl"))
        dump(train_sample_rate,os.path.join(self.pickle_train_directory,"train_sample_rate.pkl"))

        dump(test_input,os.path.join(self.pickle_test_directory,"test_input.pkl"))
        dump(test_input,os.path.join(self.pickle_test_directory,"test_output.pkl"))
        dump(test_sample_rate,os.path.join(self.pickle_test_directory,"test_sample_rate.pkl"))

        dump(validation_input,os.path.join(self.pickle_validation_directory,"validation_input.pkl"))
        dump(validation_input,os.path.join(self.pickle_validation_directory,"validation_output.pkl"))
        dump(validation_sample_rate,os.path.join(self.pickle_validation_directory,"validation_sample_rate.pkl"))
