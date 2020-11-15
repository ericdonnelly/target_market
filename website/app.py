from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

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
app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

###########home page: model characteristics###########
@app.route('/')
def main():
    return render_template('index.html')

###########user-uploads-the-image page##########
@app.route('/uploadImage')
def uploadImage():
    return render_template('uploadImage.html')

###########prediction page##########
@app.route('/prediction', methods=['POST'])
def prediction():
    global COUNT
    # img = request.files['image']
    request.files['image'].save(f'static/{COUNT}.jpg')    
    image = plt.imread(f'static/{COUNT}.jpg')
    resized_image = resize(image, (400,400,3))
    data = {}
    predictions = model.predict(np.array([resized_image]))[0]
    predictions = list(predictions)
    best_guess_index = predictions.index(max(predictions))
    classifications = {0: 'Brick', 1: 'Siding', 2: 'Unknown'}
    best_guess_category = classifications[best_guess_index]
    data['Best_guess'] = f'The model has identified {best_guess_category}.'
    for i, prediction in enumerate(predictions):
        data[classifications[i]] = f'{classifications[i]}: {round(100*prediction,0)}%'
    COUNT += 1
    return render_template('prediction.html', data=data)

###########displays image##########
@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', f"{COUNT-1}.jpg")

if __name__ == '__main__':
    app.run(debug=True)



