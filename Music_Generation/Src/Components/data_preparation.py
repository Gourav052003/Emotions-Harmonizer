from xmlrpc.client import Boolean
from Entity.entity_config import DataPreparationConfig
from tqdm import tqdm
from glob import glob
import librosa as lbrs
import numpy as np


class DataPreparation:

    def __init__(self,config:DataPreparationConfig):
        self.config = config


    def prepare_data(self,MUSIC_DIRECTORY):


        # music_files_features = []

        filenames = glob(MUSIC_DIRECTORY+'/*')

        features,sample_rate = lbrs.load(filenames[0],duration = self.config.duration)

        all_timestamp_data = []

        timestamp_intervals = list(range(0,len(features)+self.config.sample_rate,
                                self.config.sample_rate))

        for start_timestamp_no in tqdm(range(len(timestamp_intervals)-1)):

            timestamp_features = []

            for filename in filenames:

                features,sample_rate = lbrs.load(filename,duration = self.config.duration)

                one_second_features = features[timestamp_intervals[start_timestamp_no]:timestamp_intervals[start_timestamp_no+1]]

                timestamp_features.append(one_second_features)


            all_timestamp_data.append(timestamp_features)


        input_timestamp_data = []
        output_timestamp_data = []

        for timestamp_data in all_timestamp_data:

            if len(input_timestamp_data)<self.config.input_timestamp:
                input_timestamp_data.append(np.array(timestamp_data))

            elif len(output_timestamp_data)<self.config.output_timestamp:
                output_timestamp_data.append(np.array(timestamp_data))


        return input_timestamp_data,output_timestamp_data,sample_rate


    def get_data(self,Training:bool = False,Validation:bool = False,Testing:bool = False):

        train_input,train_output,train_sample_rate = None,None,None
        validation_input,validation_output,validation_sample_rate = None,None,None
        test_input,test_output,test_sample_rate = None,None,None


        if Training:
            train_input,train_output,train_sample_rate = self.prepare_data(self.config.train_music_directory)    
        
        if Validation:
            validation_input,validation_output,validation_sample_rate = self.prepare_data(self.config.validation_music_directory)    
        
        if Testing:
            test_input,test_output,test_sample_rate = self.prepare_data(self.config.test_music_directory) 

        return (train_input, train_output,train_sample_rate),(validation_input,validation_output,validation_sample_rate) ,(test_input, test_output,test_sample_rate)       
