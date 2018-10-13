
#get_ipython().run_line_magic('matplotlib', 'inline')
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
#K.set_image_dim_ordering('th')

import win32com.client as wincl
speak = wincl.Dispatch("SAPI.SpVoice")
# In[2]:

#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 4} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

# File paths for the model, all of these except the CNN Weights are 
# provided in the repo, See the models/CNN/README.md to download VGG weights
VQA_model_file_name      = 'VQA/VQA_MODEL.json'
VQA_weights_file_name   = 'VQA/VQA_MODEL_WEIGHTS.hdf5'
label_encoder_file_name  = 'VQA/FULL_labelencoder_trainval.pkl'


# In[3]:

print("VGG Model loading...")
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


# In[4]:


model_vgg = get_image_model()
#model_vgg.summary()

print("VGG Model loaded!")
# In[5]:


def get_image_features(image_file_name):
    ''' Runs the given image_file to VGG 16 model and returns the 
    weights (filters) as a 1, 4096 dimension vector '''
    image_features = np.zeros((1, 4096))
    # Magic_Number = 4096  > Comes from last layer of VGG Model

    # Since VGG was trained as a image of 224x224, every new image
    # is required to go through the same transformation
    im = cv2.resize(cv2.imread(image_file_name), (224, 224))
    im = im.transpose((2,0,1)) # convert the image to RGBA

    
    # this axis dimension is required because VGG was trained on a dimension
    # of 1, 3, 224, 224 (first axis is for the batch size
    # even though we are using only one image, we have to keep the dimensions consistent
    im = np.expand_dims(im, axis=0) 

    image_features[0,:] = model_vgg.predict(im)[0]
    return image_features


# In[6]:


def get_question_features(question):
    ''' For a given question, a unicode string, returns the time series vector
    with each word (token) transformed into a 300 dimension representation
    calculated using Glove Vector '''
    word_embeddings = spacy.load('en_vectors_web_lg')
    tokens = word_embeddings(question)
    question_tensor = np.zeros((1, 30, 300))
    for j in range(len(tokens)):
        question_tensor[0,j,:] = tokens[j].vector
    return question_tensor


# In[7]:


##If there is any error with "en_vectors_web_lg" run the following:
#!python -m spacy download en_vectors_web_lg


# In[8]:


def get_VQA_model(VQA_model_file_name, VQA_weights_file_name):
    vqa_model = model_from_json(open(VQA_model_file_name).read())
    vqa_model.load_weights(VQA_weights_file_name)
    vqa_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    return vqa_model


# In[9]:
print("VQA Model loading...")

model_vqa = get_VQA_model(VQA_model_file_name, VQA_weights_file_name)
#model_vqa.summary()
print("VQA Model Loaded!")
labelencoder = joblib.load(label_encoder_file_name)
# In[10]:

# from gtts import gTTS
# from pygame import mixer # Load the required library
# import os
# mixer.init()

import speech_recognition as sr  

# get audio from the microphone                                                                       
r = sr.Recognizer()                                                                                   
with sr.Microphone() as source:                                                                       
    print("Ask:")                                                                                   
    audio = r.listen(source)
try:
    print("You asked " + r.recognize_google(audio) + " ?")
    question = r.recognize_google(audio)+" ?"
except sr.UnknownValueError:
    print("Could not understand the question")
except sr.RequestError as e:
    print("Could not request results; {0}".format(e))


import numpy as np
import cv2

cap = cv2.VideoCapture(1)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #question = u'what is there in the picture?'
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if cv2.waitKey(1) & 0xFF == ord('c'):
    	image_features = np.zeros((1, 4096))
    	im = cv2.resize(frame, (224, 224))
    	im = im.transpose((2,0,1))
    	im = np.expand_dims(im, axis=0) 

    	image_features[0,:] = model_vgg.predict(im)[0]
    	question_features = get_question_features(question)
    	y_output = model_vqa.predict([question_features, image_features])
    	warnings.filterwarnings("ignore", category=DeprecationWarning)
    	labelencoder = joblib.load(label_encoder_file_name)
    	for label in reversed(np.argsort(y_output)[0,-5:]):
    		text = "According to me the answer is "+labelencoder.inverse_transform(label)
    		break
    	speak.Speak(text)
    	cv2.imshow(grey)
    	cv2.waitKey(0)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()