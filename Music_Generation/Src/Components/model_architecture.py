from Entity.entity_config import ModelArchitecturePathConfig
from keras.layers import Input,Dense,LeakyReLU,BatchNormalization,Concatenate,Reshape,Flatten
from keras.optimizers import Adam
from keras import Model
from pickle import dump
import os

class ModelArchitecture:

    def __init__(self,config = ModelArchitecturePathConfig):
        self.pickle_model_architecture = config.pickle_model_architecture_directory
        self.encoder_timestamp = config.encoder_timestamp
        self.sample_rate = config.sample_rate
        self.decoder_timestamp = config.decoder_timestamp
        self.learning_rate = config.learning_rate
        self.loss = config.loss

    def hidden_layers(self,bn1,unit_no,architecture_type = None):
        du1 = Dense(64,name = f"{architecture_type}_dense_1_unit_{unit_no}")(bn1)
        lr1 = LeakyReLU(0.3,name = f"{architecture_type}_leakyRelu_1_unit_{unit_no}")(du1)
        bn1 = BatchNormalization(name = f"{architecture_type}_batch_norm_hidden_1_{unit_no}")(lr1)
     
        du2 = Dense(128,name = f"{architecture_type}_dense_2_unit_{unit_no}")(bn1)
        lr2 = LeakyReLU(0.3,name = f"{architecture_type}_leakyRelu_2_unit_{unit_no}")(du2)
        bn2 = BatchNormalization(name = f"{architecture_type}_batch_norm_hidden_2_{unit_no}")(lr2)

        du3 = Dense(256,name = f"{architecture_type}_dense_3_unit_{unit_no}")(bn2)
        lr3 = LeakyReLU(0.3,name = f"{architecture_type}_leakyRelu_3_unit_{unit_no}")(du3)
        bn3 = BatchNormalization(name = f"{architecture_type}_batch_norm_hidden_3_{unit_no}")(lr3)

        du4 = Dense(512,name = f"{architecture_type}_dense_4_unit_{unit_no}")(bn3)
        lr4 = LeakyReLU(0.3,name = f"{architecture_type}_leakyRelu_4_unit_{unit_no}")(du4)
        bn4 = BatchNormalization(name = f"{architecture_type}_batch_norm_hidden_4_{unit_no}")(lr4)

        return bn4

    def relation_neural_network(self,i=None,ou=None,inter_i=None,unit_no=None,architecture_type=None,sample_rate=22050,timestamp=20):

        if architecture_type == 'encoder':

            ou.append(i)
            c1 = Concatenate(name = f"{architecture_type}_rel_nn_concatenate_1_unit_{unit_no}")(ou)
            r1 = Reshape((sample_rate,timestamp))(c1)
            d1 = Dense(1,name = f"{architecture_type}_rel_nn_dense_1_unit_{unit_no}")(r1)
            l1 = LeakyReLU(alpha = 0.4)(d1)
            f1 = Flatten()(l1)
            return f1

        elif architecture_type=='inter':
            c1 = Concatenate(name = f"{architecture_type}_rel_nn_concatenate_1_unit_{unit_no}")(inter_i)
            r1 = Reshape((sample_rate,timestamp))(c1)
            d1 = Dense(1,name = f"{architecture_type}_rel_nn_dense_1_unit_{unit_no}")(r1)
            l1 = LeakyReLU(alpha = 0.4)(d1)
            f1 = Flatten()(l1)
            return f1

        else:

            ou.append(i)
            c1 = Concatenate(name = f"{architecture_type}_rel_nn_concatenate_1_unit_{unit_no}")(ou)
            r1 = Reshape((sample_rate,timestamp))(c1)
            d1 = Dense(1,name = f"{architecture_type}_rel_nn_dense_1_unit_{unit_no}")(r1)
            l1 = LeakyReLU(alpha = 0.4)(d1)
            f1 = Flatten()(l1)
            return f1
    

    def Encoder_Decoder_model(self):

        encoder_input_units = []
        encoder_output_units = []
        decoder_output_units = []

        i,ou = None,None

        for unit_no in range(self.encoder_timestamp):
            
            if unit_no == 0:
                i = Input((self.sample_rate),name = f"encoder_Input_1_unit_{unit_no}")
                r = self.relation_neural_network(i = i,ou = [i],inter_i = None,unit_no = unit_no,architecture_type="encoder",sample_rate = self.sample_rate,timestamp = 2)
                h = self.hidden_layers(r,unit_no,architecture_type = "encoder")
                d = Dense(self.sample_rate)(h)
                ou = LeakyReLU(alpha = 1)(d)

            else:
                i = Input((self.sample_rate),name = f"encoder_Input_1_unit_{unit_no}")
                r = self.relation_neural_network(i = i,ou = encoder_output_units.copy(),inter_i = None,unit_no = unit_no,architecture_type="encoder",sample_rate = self.sample_rate,timestamp = unit_no+1)
                h = self.hidden_layers(r,unit_no,architecture_type = "encoder")
                d = Dense(self.sample_rate)(h)
                ou = LeakyReLU(alpha = 1)(d)


            encoder_input_units.append(i)
            encoder_output_units.append(ou)
        

        r = self.relation_neural_network(inter_i = encoder_output_units.copy(),unit_no = 777,architecture_type="inter",sample_rate = self.sample_rate,timestamp = self.encoder_timestamp)
        h = self.hidden_layers(r,unit_no,architecture_type = "inter")
        d = Dense(self.sample_rate)(h)
        inter_i = LeakyReLU(alpha = 1)(d)


        for unit_no in range(self.decoder_timestamp):

            if unit_no == 0:
                r = self.relation_neural_network(i = inter_i,ou = [inter_i],inter_i = None,unit_no = unit_no,architecture_type="decoder",sample_rate = self.sample_rate,timestamp = 2)
                h = self.hidden_layers(r,unit_no,architecture_type = "decoder")
                d = Dense(self.sample_rate)(h)
                ou = LeakyReLU(alpha = 1)(d)

            else:
                r = self.relation_neural_network(i = inter_i,ou = decoder_output_units.copy(),inter_i = None,unit_no = unit_no,architecture_type="decoder",sample_rate = self.sample_rate,timestamp = unit_no+1)
                h = self.hidden_layers(r,unit_no,architecture_type = "decoder")
                d = Dense(self.sample_rate)(h)
                ou = LeakyReLU(alpha = 1)(d)

            decoder_output_units.append(ou)


        model = Model(inputs = encoder_input_units,outputs = decoder_output_units)
        model.compile(optimizer = Adam(learning_rate=self.learning_rate),loss=self.loss)
        model.summary()

        os.makedirs(self.pickle_model_architecture,exist_ok=True)
        dump(model,os.path.join(self.pickle_model_architecture,"model_architecture.pkl"))

        
        