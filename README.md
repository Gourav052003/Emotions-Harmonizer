# Harmonizing Emotions: A Musical Journey Through Innovative Extrapolation 
🎶🌌 In the symphony of life, 🎵 music serves as a powerful conductor of emotions, weaving intricate patterns that resonate within the depths of our souls. 

The quest for joy and the alleviation of melancholy often find solace in the artistry of musicians and composers. 🎻🎹

 Yet, in our digital age, where music is fragmented into bite-sized pieces, the challenge emerges: how can we transform fleeting moments of musical bliss into a sustained, transformative experience? 🤔🎶 This research endeavors to explore a novel solution to this conundrum, proposing an innovative idea that involves generating sequential "10-second" music chunks. 🔄🎼
 
  These musical fragments, when skillfully combined, have the potential to orchestrate a seamless composition of desired duration, thus unlocking a symphony of emotions. 🎶💖
  
   Delving into the intricacies of human psychology, the study challenges the conventional belief that repetitive exposure to a short musical piece can adequately replace the profound impact of a longer composition. 🧠🎵 
   
   By extrapolating the musical experience, we aim to bridge the gap between the ephemeral and the enduring, offering a unique pathway to emotional resonance. 🌈🎶 
   
   Join us on this melodic journey through the nitty-gritty details of our solution, where each note becomes a stepping stone towards a richer, more immersive musical experience. 🚀🎵
   
   As we navigate the intersection of innovation and emotion, the allure of extended musical euphoria awaits discovery, promising to revolutionize the way we perceive and experience the transformative power of music. 🌟🎶

# Problem Definition
To develop a Deep learning model for Remixed Music Extrapolation using ``LSTMs`` recurrent neural networks for harmonizing emotions is considered to be a difficult task because of training time it takes to train the ``LSTMs`` and On the top of it, Resource Exhaustion is another big issue in carrying out this task, because the architectural strategy we used need for Implementing this innovative extrapolation is not able to backed by ``LSTMs`` due to Resource Exhaustion Problem. With the aim of developing Music Extrapolation model and providing solutions to the problems, we need to think from the scratch and develop our own custom Encoder Decoder Model without using ``LSTMs``.

# Solution offered
Music Extrapolation Model is built using the Keras functional Model API which helps us to create a flexible neural architecture other than Sequential architecture with only one Input and one Output. Model Architecture we are developing is a variant of ``Sequence2Sequence`` Model i.e. ``Many2Many`` also know as ``Encoder-Decoder Model``.


![encoder-decoder](https://github.com/Gourav052003/Emotions-Harmonizer/assets/81559597/4365f717-09ed-4548-8267-dd5c8c884b4b)


This ``Many2Many Encoder-Decoder`` Model takes 20 inputs one at each timestamp ``t`` denoted by ``Xt`` and as an output it give ``10 outputs`` one at each timestamp ``t`` denoted by ``St`` on Decoder Side. Here one Input represent one vector of shape ``(1, 22050)`` which represent one second for sample rate of ``22050``. It means Encoder part takes ``20 seconds`` of Music Inputs using custom recurrent units for ``20 timestamps`` and encodes the information to get the context of ``20 Timestamps`` into single vector of shape ``(1, 22050)``. Then this vector is passed as input to the decoder for ``10 timestamps`` and decoder unlike encoder, gives ``10 outputs`` vectors each of shape ``(1, 22050)``. Each vector representing newly generated one second of music of sample rate ``22050``.


![Encoder](https://github.com/Gourav052003/Emotions-Harmonizer/assets/81559597/de2d6bef-b8f4-4c45-948b-f74150fd9334)


At the Deeper level of single ``Encoder recurrent unit Architecture``, it takes ``2 inputs`` one as music feature vector ``Xt`` other as a context vector ``Ct`` for time timestamp ``t``and Both Inputs are of same shape ``(1, 22050)``. Encoder Starts by appending Input vector ``Xt`` to Context vector ``Ct`` using append operation Denoted by ``A``. Here ``Ct`` context vector is again a list of output sequences ``St`` of Encoder Units from time stamp ``S0``   to ``St-1``. Here ``Ct = Xt`` for timestamp ``t = 0``. After the Appending Operation our results will be a list of Inputs and Context vectors like ``[ St-1, St-2, … , S1, S0, Xt ]``. Than Concatenation is performed to get one single numpy array ``[ St-1 St-2 … S1 S0 Xt ]`` of size ``(1,22050*t)`` where ``t`` is current timestamp. After Concatenation, the resultant array is  reshaped to ``(22050, t)`` numpy array. This Numpy array is passed to Dense layer with one neuron which give output of shape ``(22050, 1)`` and this output is passed for ``LeakyReLU`` activation layer with ``0.3`` as ``alpha value``. Then output is flattened to get array of shape ``(1, 22050)`` and this Flattened vector is passed to Hidden layers for further deeper level of processing to generate the output sequence ``St`` for an Encoder recurrent unit at timestamp ``t``. 


![Context](https://github.com/Gourav052003/Emotions-Harmonizer/assets/81559597/9454e5a2-1c9c-4240-a42b-4f6fdc2cb213)


All the Outputs ``S0`` to ``St``  from the Encoder is Passed to Context Block which encodes the information to get the context of all encoder timestamp together and passed to decoder for extrapolation of Music for next ``10 timestamps``. All Internal Working is Similar to Encoder unit, Only difference is that it takes only one Input as a list of sequences from  ``S0`` to ``St``, which produced by Encoder recurrent unit. 


![Decoder](https://github.com/Gourav052003/Emotions-Harmonizer/assets/81559597/eae9ce2a-501a-4532-beb7-41fffd74e4aa)


After Fetching all the context of Encoder Units using Context block into a Vector ``C`` with shape ``(1, 22050)``. It is passed to Decoder for every timestamp ``t``. Everything works same like Encoder; only difference is Decoder takes same input ``C`` which is context vector for every timestamp ``t``.


![Hiddden Network](https://github.com/Gourav052003/Emotions-Harmonizer/assets/81559597/280c6807-c906-4f3b-9bb5-278a235a7e4c)


Here in Encoder, Decoder and Context Unit there is one Hidden layer which plays an important role in learning patterns and understanding the context of Music Sequences. Hidden layers consists of four blocks, where each block consists of ``Dense`` layer, ``LeakyReLU`` layer for Activation and ``Batch Normalization``. Every block has varying Neurons for its ``Dense`` layer ranging from ``64`` to ``512``. In last of hidden layer we have ``Dense`` layer with ``22050`` Nodes with ``LeakyReLU`` as activation for generating Output

# Steps to Excecute the Implementation

1. Clone the ``Emotions-Harmonizer`` Repository
    ```
    git clone https://github.com/Gourav052003/Emotions-Harmonizer.git
    ```

2. Setting Up the Virtual Environment 
    ```
    virtualenv venv --python=python3.9
    ```

3. Installing all the Dependencies
    ```
    pip install -r requirements.txt
    ```

4. Start your Training and Testing Pipeline to build and test the model
    
    * Using DVC (Data Version Control)
    
    ```
    dvc init
    dvc repro
    ```

    * Using Python script
    ```
    cd .\Music_Generation\Src
    python main.py
    ```

# AWS-CI/CD-Deployment-with-Github-Actions

1. Login to AWS console.

2. Create IAM (Identity and Access Management) user for Deployment with specific access

    ``Description: About the deployment``

        * Build Docker Image of the source code
        * Push Docker Image to ECR (Elastic Container registry) to save your Docker Image in AWS
        * Launch EC2 (virtual Machine)
        * Pull Image from ECR in EC2
        * Launch Docker Image in EC2

    ``Policy:``

        * AmazonEC2ContainerRegistryFullAccess
        * AmazonEC2FullAccess

3. Create ECR repo to store/save Docker Image and save the URI

4. Create EC2 machine (Ubuntu) 

5. Open EC2 and Install Docker in EC2 Machine

    ``Optional steps``

        1. sudo apt-get update -y
        2. sudo apt-get upgrade
	
	``Required steps``

        curl -fsSL https://get.docker.com -o get-docker.sh

        sudo sh get-docker.sh

        sudo usermod -aG docker ubuntu

        newgrp docker

6. Configure EC2 as self-hosted runner
    ```
    setting>actions>runner>new self hosted runner> choose os> then run command one by one
    ```

7. Setup github secrets:

        AWS_ACCESS_KEY_ID=

        AWS_SECRET_ACCESS_KEY=

        AWS_REGION = us-east-1

        AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

        ECR_REPOSITORY_NAME = simple-app    