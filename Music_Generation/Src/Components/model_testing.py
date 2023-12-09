from Entity.entity_config import ModelTestingConfig
import librosa as lbrs
import numpy as np
import os
from IPython.display import Audio
from keras.models import load_model

class ModelTesting:

    def __init__(self,config : ModelTestingConfig):
        self.best_model_path = config.best_model_path
        self.file_path = config.testing_file_path
        self.testing_duration = config.testing_duration
        self.results_path = config.results_path

    def predict(self):

        file_data = []
    
        filename = self.file_path.split('/')[-1].split('.mp3')[0]
        results_file_directory = os.path.join(self.results_path, filename)
        os.makedirs(results_file_directory,exist_ok=True)

        ########################################################################################################

        features,sample_rate = lbrs.load(self.file_path)
        duration = lbrs.get_duration(y = features,sr=sample_rate)
        last_secs_features,sample_rate = lbrs.load(self.file_path,offset=duration-self.testing_duration)

        for start_timestamp in range(0,sample_rate*self.testing_duration,sample_rate):
            file_data.append(last_secs_features[start_timestamp:start_timestamp+sample_rate].reshape(1,sample_rate))
        
        best_model = load_model(self.best_model_path)
        prediction = best_model.predict(file_data)        
        prediction = np.array(prediction).flatten()

        extrapolation = np.concatenate(features,prediction)
    
        #########################################################################################################

        original_music_file = os.path.join(results_file_directory,'original.wav')
        tested_music_file = os.path.join(results_file_directory,'tested.wav')
        predicted_music_file = os.path.join(results_file_directory,'predicted.wav')
        extrapolated_music_file = os.path.join(results_file_directory,'extrapolated.wav')

        lbrs.output.write_wav(original_music_file, features, sample_rate, norm=False)
        lbrs.output.write_wav(tested_music_file, last_secs_features, sample_rate, norm=False)
        lbrs.output.write_wav(predicted_music_file, prediction, sample_rate, norm=False)
        lbrs.output.write_wav(extrapolated_music_file, extrapolation, sample_rate, norm=False)


        



