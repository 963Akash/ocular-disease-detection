import tkinter as tk
from tkinter.filedialog import askopenfilename
import shutil
import os
import sys
from PIL import Image, ImageTk

import serial
##import numpy as np


window = tk.Tk()
window.title("Dr.Diagnosis ")
window.geometry("500x510")
window.configure(background ="black")

title = tk.Label(text="Click below to choose picture for testing disease....", background = "red", fg="Brown", font=("", 15))
title.grid()

def analysis():
    import cv2  # working with, mainly resizing, images
    import numpy as np  # dealing with arrays
    import os  # dealing with directories
    from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
    from tqdm import \
        tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
    verify_dir = 'testpicture'
    IMG_SIZE = 50
    LR = 1e-3
    MODEL_NAME = 'Parkinson-{}-{}.model'.format(LR, '2conv-basic')

    def process_verify_data():
        verifying_data = []
        for img in tqdm(os.listdir(verify_dir)):
            path = os.path.join(verify_dir, img)
            img_num = img.split('.')[0]
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            verifying_data.append([np.array(img), img_num])
        np.save('verify_data.npy', verifying_data)
        return verifying_data

    verify_data = process_verify_data()
    #verify_data = np.load('verify_data.npy')

    import tflearn
    from tflearn.layers.conv import conv_2d, max_pool_2d
    from tflearn.layers.core import input_data, dropout, fully_connected
    from tflearn.layers.estimator import regression
    import tensorflow as tf
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

    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for num, data in enumerate(verify_data):

        img_num = data[1]
        img_data = data[0]

        y = fig.add_subplot(3, 4, num + 1)
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
        # model_out = model.predict([data])[0]
        model_out = model.predict([data])[0]
        print(model_out)
        print('model {}'.format(np.argmax(model_out)))

        status='Not Defied'
        if np.argmax(model_out) == 0:
            str_label = 'Mild'
        elif np.argmax(model_out) == 1:
            str_label = 'Moderate'
        elif np.argmax(model_out) == 2:
            str_label = 'NO_DR'
        elif np.argmax(model_out) == 3:
            str_label = 'proliferate'
        elif np.argmax(model_out) == 4:
            str_label = 'Severe'
 
         if str_label == 'Mild':
            status='Mild'
            

        elif str_label == 'Moderate':
            status='Moderate'
            

        elif str_label == 'NO_DR': 
            status='NO_DR'
            

        elif str_label == 'proliferate':
            status='proliferate'
            

        elif str_label == 'Severe':
            status='Severe'
            

        message = tk.Label(text='Status: '+status, background="lightgreen",
                           fg="Brown", font=("", 15))
        message.grid(column=0, row=3, padx=10, pady=10)
def openphoto():
    dirPath = "testpicture"
    fileList = os.listdir(dirPath)
    for fileName in fileList:
        os.remove(dirPath + "/" + fileName)
    
    fileName = askopenfilename(initialdir='E:\\diagnetic_Retinopathy\\test', title='Select image for analysis ',
                           filetypes=[('image files', '.png')])
    dst = "testpicture"
    print(fileName)
    print (os.path.split(fileName)[-1])
    shutil.copy(fileName, dst)
    load = Image.open(fileName)
    render = ImageTk.PhotoImage(load)
    img = tk.Label(image=render, height="250", width="500")
    img.image = render
    img.place(x=0, y=0)
    img.grid(column=0, row=1, padx=10, pady = 10)
    title.destroy()
    button1.destroy()
    button2 = tk.Button(text="Analyse Image", command=analysis)
    button2.grid(column=0, row=2, padx=10, pady = 10)
button1 = tk.Button(text="Get Photo", command = openphoto)
button1.grid(column=0, row=1, padx=10, pady = 10)

window.mainloop()



