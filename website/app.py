from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
#For API CALL ADDRESS SUBMIT
import urllib.request
import json
# Import google_streetview for the api module
import os
import google_streetview.api
import time
import glob
#import streetview
import itertools
from config import gkey
from my_functions import address_form, image_form


model = Sequential()

model.add(Conv2D(filters=4, kernel_size=2, padding='same',
                 activation='relu', input_shape=(400, 400, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=8, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.1))

model.add(Conv2D(filters=12, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))

model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('static/model/final_model.hdf5')


COUNT = 0
FORM_COUNT = 0
IS_ADDRESS = True
ln = ''

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route("/", methods=["GET", "POST"])
def main():

    global FORM_COUNT
    global COUNT
    global IS_ADDRESS

    if request.method == "POST":

        if "address" in request.form:

            submit_address = request.form["address"]

            # return model predictions from user address input
            data, FORM_COUNT = address_form(model, submit_address, FORM_COUNT)

        else:

            request_files = request.files['image']

            # return model predictions from user address input
            data, COUNT = image_form(model, request_files, COUNT)

            IS_ADDRESS = False

    else:

        data = {'Best_guess': '', 'Brick': '', 'Siding': '', 'Unknown': ''}
   
    return render_template('index.html', data=data)


###########displays image from address input##########
@app.route('/address_load_img')
def address_load_img():
    global FORM_COUNT
    global IS_ADDRESS

    if IS_ADDRESS == True:

        return send_from_directory("static/images/address_submit/", f"{FORM_COUNT-1}.jpg")
    
    elif IS_ADDRESS == False:

        return send_from_directory("static/images/upload_images/", f"{COUNT-1}.jpg")

# ###########displays image from uploaded image##########
# @app.route('/image_load_img')
# def image_load_img():
#     global COUNT
#     return send_from_directory('static/images/upload_images/', f"{COUNT-1}.jpg")

if __name__ == '__main__':
    app.run(debug=True)


