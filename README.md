# Visual-Question-Answering

This is RNN+CNN Visual Question Answering Model. It uses VGG16 for image feature extraction.
[VQA Dataset](http://visualqa.org/download.html) is used for training the model.

### Dependency

1. Keras version 2.0+
2. Tensorflow 1.2+
3. Spacy version 2.0+
    * To upgrade & install Glove Vectors
       * python -m spacy download en_vectors_web_lg
4. OpenCV 

### Usage
Download my pretrained model from [here](https://drive.google.com/drive/folders/1vlVDWGP_xwBaqZnFFTRwpSDriLxu-tHM?usp=sharing)

For running pretrained model in Google Colab [Click Here](https://colab.research.google.com/github/PratikSavla/Visual-Question-Answering/blob/master/VQA_Appplication.ipynb)

For training the model run:
```
$ python train.py
```

### Reference
```
https://github.com/VT-vision-lab/VQA_LSTM_CNN
```
