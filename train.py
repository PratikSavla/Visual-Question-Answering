from keras.models import Sequential
from keras.layers.core import Reshape, Activation, Dropout
from keras.layers import LSTM, Concatenate, Dense, Embedding
from keras.utils import np_utils

import h5py
import importlib
import numpy as np
import json

# defining model(s)
def model():
    # Image model
    model_image = Sequential()
    model_image.add(Reshape((4096,), input_shape=(4096,)))
    model_image.add(Dense(1024))
    model_image.add(Activation('tanh'))
    model_image.add(Dropout(0.5))


    # Language Model
    model_language = Sequential()
    model_language.add(Embedding(12603, 300, input_length=26))
    model_language.add(LSTM(512, return_sequences=True, input_shape=(26, 300)))
    model_language.add(LSTM(512, return_sequences=True))
    model_language.add(LSTM(512, return_sequences=False))
    model_language.add(Dense(1024))
    model_language.add(Activation('tanh'))
    model_language.add(Dropout(0.5))


    # combined model
    model = Sequential()
    model.add(Concatenate([model_language, model_image], axis=1))

    for i in range(3):
        model.add(Dense(1024))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

    model.add(Dense(1000))
    model.add(Activation('softmax'))

    return model

#finding most occuring word
def most_common(lst):
    return max(set(lst), key=lst.count)
    
#reading the training dataset
def get_train_data(img_norm = 0):

    dataset = {}
    train_data = {}
    
    print('loading json file...')
    with open('DATA/data_prepro.json') as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    print('loading image feature...')
    with h5py.File('DATA/data_img.h5','r') as hf:
        tem = hf.get('images_train')
        img_feature = np.array(tem)
        
    print('loading h5 file...')
    with h5py.File('DATA/data_prepro.h5','r') as hf:
        tem = hf.get('ques_train')
        train_data['question'] = np.array(tem)
        tem = hf.get('ques_length_train')
        train_data['length_q'] = np.array(tem)
        tem = hf.get('img_pos_train')
        train_data['img_list'] = np.array(tem)-1
        tem = hf.get('answers')
        train_data['answers'] = np.array(tem)-1

    print('Normalizing image feature')
    if img_norm:
        tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,2048)))

    return dataset, img_feature, train_data

#reading validation/test data
def get_data_test():
    dataset = {}
    test_data = {}
    
    print('loading json file...')
    with open('DATA/data_prepro.json') as data_file:
        data = json.load(data_file)
    for key in data.keys():
        dataset[key] = data[key]

    print('loading image feature...')
    with h5py.File('DATA/data_img.h5','r') as hf:
        tem = hf.get('images_test')
        img_feature = np.array(tem)
    
    print('loading h5 file...')
    with h5py.File('DATA/data_prepro.h5','r') as hf:
        tem = hf.get('ques_test')
        test_data['question'] = np.array(tem)
        tem = hf.get('ques_length_test')
        test_data['length_q'] = np.array(tem)
        tem = hf.get('img_pos_test')
        test_data['img_list'] = np.array(tem)-1
        tem = hf.get('question_id_test')
        test_data['ques_id'] = np.array(tem)
        
    tem = hf.get('MC_ans_test')
    test_data['MC_ans_test'] = np.array(tem)
    
    print('Normalizing image feature')
    if img_norm:
        tem =  np.sqrt(np.sum(np.multiply(img_feature, img_feature)))
        img_feature = np.divide(img_feature, np.tile(tem,(1,2048)))

    nb_data_test = len(test_data[u'question'])
    val_all_answers_dict = json.load(open('DATA/val_all_answers_dict.json'))
    val_answers = np.zeros(nb_data_test, dtype=np.int32)

    ans_to_ix = {v: k for k, v in dataset[u'ix_to_ans'].items()}
    count_of_not_found = 0
    for i in xrange(nb_data_test):
        qid = test_data[u'ques_id'][i]
        try : 
            val_ans_ix =int(ans_to_ix[most_common(val_all_answers_dict[str(qid)])]) -1
        except KeyError:
            count_of_not_found += 1
            val_ans_ix = 480
        val_answers[i] = val_ans_ix
    print("Beware: " + str(count_of_not_found) + " number of val answers are not really correct")

    return dataset, img_feature, test_data

#loading the training and testing data
dataset, train_img_feature, train_data = get_train_data()

train_X = [train_data[u'question'], train_img_feature]
train_Y = np_utils.to_categorical(train_data[u'answers'], 1000)

dataset, test_img_feature,  test_data, val_answers = get_data_test()

test_X = [test_data[u'question'], test_img_feature]
test_Y = np_utils.to_categorical(val_answers, args.nb_classes)

#loading the model
model = model()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

#training the model
model.fit(train_X, train_Y, batch_size = 128, nb_epoch=300, validation_data=(test_X, test_Y))

# evaluate the model
scores = model.evaluate(test_X, test_Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# saving model
model_json = model.to_json()
with open("VQA/VQA_MODEL.json", "w") as json_file:
    json_file.write(model_json)
# saving weights
model.save_weights("VQA/VQA_MODEL_WEIGHTS.hdf5")
print("Model Saved")
