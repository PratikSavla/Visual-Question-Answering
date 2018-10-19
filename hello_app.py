import warnings
warnings.filterwarnings("ignore")
import os, argparse
import keras
import cv2, spacy, numpy as np
from keras.models import model_from_json
from keras.optimizers import SGD
from sklearn.externals import joblib
from keras import backend as K
from keras.utils.vis_utils import plot_model
K.set_image_data_format('channels_first')

VQA_model_file_name      = 'VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'VQA/FULL_labelencoder_trainval.pkl'

def get_image_model():
    from keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet')
    from keras.models import Model

    new_input = model.input
    hidden_layer = model.layers[-2].output

    model_new = Model(new_input, hidden_layer)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model_new.compile(optimizer=sgd, loss='categorical_crossentropy')
    return model_new

model_vgg = get_image_model()

def get_image_features(image_file_name):
    image_features = np.zeros((1, 4096))
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    image_features[0,:] = model_vgg.predict(im)[0]
    return image_features

def get_question_features(question):
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor

import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")

def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model

model_vqa = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
labelencoder = joblib.load(label_encoder_file_name)

import numpy as np
import cv2
from flask import request
from flask import jsonify
from flask import Flask

app = Flask(__name__)

@app.route('/hello', methods=['POST'])
def hello():
    message = request.get_json(force=True)
    #question = message['name']
    question = u'what is there in the picture?'
    import speech_recognition as sr
    r = sr.Recognizer()
    #try:
   # 	mic = sr.Microphone(device_index=2)
    #	with mic as source:
   # 		print("Ask the question:")                                                                                   
    #		audio = r.listen(source)
    #except sr.UnknownValueError:

    cap = cv2.VideoCapture(1)
    while(True):
    	ret, frame = cap.read()
    	#question = u'what is there in the picture?'
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    	cv2.imshow('frame',gray)
    	if cv2.waitKey(1) & 0xFF == ord('c'):
            
            image_features = np.zeros((1, 4096))
            im = cv2.resize(frame, (224, 224))
            im = im.transpose((2,0,1))
            im = np.expand_dims(im, axis=0)
            speak.Speak("I am ready for your question 3 2 1")
            with sr.Microphone() as source:                                                                       
                print("Ask the question:")                                                                                   
                audio = r.listen(source)
            try:
                print("You asked " + r.recognize_google(audio) + " ?")
                question = r.recognize_google(audio)+" ?"
            except sr.UnknownValueError:
                print("Could not understand the question")
            except sr.RequestError as e:
                print("Could not request results; {0}".format(e))
            image_features[0,:] = model_vgg.predict(im)[0]
            question_features = get_question_features(question)
            y_output = model_vqa.predict([question_features, image_features])
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            labelencoder = joblib.load(label_encoder_file_name)
            for label in reversed(np.argsort(y_output)[0,-5:]):
                ans = "According to me the answer is "+labelencoder.inverse_transform(label)
                speak.Speak(ans)
                break
            response = {'greeting':question+'\n'+ ans + '.'}
            break
    cv2.destroyAllWindows()
    return jsonify(response)

if __name__=='__main__':
   # manager.run()
   app.run(debug=True)
   #app.run(debug=True,host='192.168.0.11')
