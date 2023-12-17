import shutil
import os
import sys
import json
import math
from PIL import Image
from flask import Flask, render_template, request
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from keras.models import load_model

import random

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        dirPath = "E:\\Ocular Disease Recognition\\static\\images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "\\" + fileName)
        fileName=request.form['filename']
        dst = "E:\\Ocular Disease Recognition\\static\\images"
        shutil.copy("E:\\Ocular Disease Recognition\\test\\"+fileName, dst)
        
        verify_dir = 'E:\\Ocular Disease Recognition\\static\\images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'odir-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in tqdm(os.listdir(verify_dir)):
                path = os.path.join(verify_dir, img)
                img_num = img.split('_')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

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


        accuracy=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))
            str_labels = ["Normal","Diabetes","Glaucoma","Cataract","Age related Macular Degeneration","Hypertension","Pathological Myopia","Other diseases/abnormalities"]
            str_label = str_labels[np.argmax(model_out)]
            print(f"The predicted image is {str_label} with a accuracy of {model_out[0]*100}")
            accuracy = f"The predicted image is {str_label} with a accuracy of {model_out[0]*100}"

        return render_template('index.html', cnn_status=str_label,cnn_accuracy=accuracy ,cnn_ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('index.html')
  
    
if __name__ == '__main__':
    app.run(debug=True)
