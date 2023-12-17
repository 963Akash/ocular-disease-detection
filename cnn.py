import cv2                 
import numpy as np      
import pandas as pd   
import os                  
from random import shuffle 
from tqdm import tqdm
from tensorflow.python.framework import ops
# Visualize training history
from keras.models import Sequential
from keras.layers import Dense

TRAIN_DIR = 'E:\\Ocular Disease Recognition\\train'
TEST_DIR = 'E:\\Ocular Disease Recognition\\test'


IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'odir-{}-{}.model'.format(LR, '2conv-basic')

  
def label_img(img):
    df = pd.read_csv('df.csv')[['filename','target']].set_index('filename')
    return eval(df.loc[img][0])

def create_train_data():
    training_data = []Z
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        print('##############')
        print(label)
        path = os.path.join(TRAIN_DIR,img)
        print(path)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        #img = cv2.resize(img,None,fx=0.5,fy=0.5)
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('_')[0]
        print(img_num)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data

train_data = create_train_data()
# If you have already created the dataset:
#train_data = np.load('train_data.npy')


import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
#tf.reset_default_graph()
ops.reset_default_graph()

convnet =  input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 8, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

train = train_data[:-1]
test = train_data[:-10]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = [i[1] for i in train]
print(X.shape)
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
test_y = [i[1] for i in test]
print(test_x.shape)

history=model.fit({'input': X}, {'targets': Y},n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}),snapshot_step=20, show_metric=True, run_id=MODEL_NAME)
model.save(MODEL_NAME)
# h=history.history
# plt.plot(h['accuracy'])
# plt.plot(h['val_accuracy'], c="red")
# plt.title("acc vs v-acc")
# plt.show()
# plt.savefig("accuracy plot")
#
# plt.plot(h['loss'])
# plt.plot(h['val_loss'], c="red")
# plt.title("loss vs v-loss")
# plt.show()
# plt.savefig("loss plot")











        
