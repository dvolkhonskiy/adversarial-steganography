# Adversarial-Model-For-Steganography

Deep adversarial neural network for steganography. 
Code for paper https://arxiv.org/pdf/1703.05502.pdf --- Steganographic Generative Adversarial Networks. NIPS 2016 workshop on Adversarial Training


## Usage

To embed some text to images in folder 'folder' call

    $ python utils/apply_to_files.py


To evaluate (train, sample) SGAN call

    $ python main.py
    
To evaluate Steganalysis, call

    $ python main_steganalysis.py
    
