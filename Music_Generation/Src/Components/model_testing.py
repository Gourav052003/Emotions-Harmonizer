from Entity.entity_config import ModelTestingConfig
import librosa as lbrs
import numpy as np
from IPython.display import Audio
from keras.models import load_model

class ModelTesting:

    def __init__(self,config : ModelTestingConfig):
        self.config = config

    def predict(self):

        file_data = []

        features,sample_rate = lbrs.load(self.config.testing_file_path)
        duration = lbrs.get_duration(y = features,sr=sample_rate)
        last_20_secs_features,sample_rate = lbrs.load(self.config.testing_file_path,
                                                offset=duration-self.config.testing_duration)
        Audio(last_20_secs_features,rate = sample_rate)


        for start_timestamp in range(0,sample_rate*self.config.testing_duration,sample_rate):
            file_data.append(last_20_secs_features[start_timestamp:start_timestamp+sample_rate].reshape(1,sample_rate))
        
        best_model = load_model(self.config.best_model_path)
        prediction = best_model.predict(file_data)
        
        prediction = np.array(prediction).flatten()

        Audio(prediction,rate=sample_rate)



