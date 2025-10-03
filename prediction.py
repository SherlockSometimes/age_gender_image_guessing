#import the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os, random

warnings.filterwarnings('ignore')

# Tensorflow and Keras Modules
import tensorflow as tf
from keras.utils import load_img


BASE_DIR = 'guess/'

model = tf.keras.models.load_model('gender_age_detection_model.keras')
# print (model.summary ())

# change the label of the gender
gender_dict = {0:'Male', 1:'Female'}

#random sample of images
image_sample = random.sample(os.listdir(BASE_DIR), k=4)

# labels - age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []

#create a loop for all dataset images for validation
for filename in image_sample:
    image_path = os.path.join(BASE_DIR, filename)
    temp = filename.split('_')
    #get age from 0
    age = int(temp[0])
    #get gender from 1
    gender = int(temp[1])
    # #get race from 2
    # gender = int(temp[2])
    #append all
    image_paths.append(image_path)
    age_labels.append(age)
    gender_labels.append(gender)

def extract_features(images_):
    features = []
    for image in images_:
        # change to gray image
        img = load_img(image, color_mode='grayscale')
        #resize by 128*128
        img = img.resize((128, 128))
        img = np.array(img)
        features.append(img)
    features = np.array(features)
    # ignore this step if using RGB
    features = features.reshape(len(features), 128, 128, 1)
    return features

images = extract_features(image_paths)
images = images/255.0

for index,image in enumerate(images):
    # predict from model
    pred = model.predict(image.reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print(image_paths[index])
    print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
    print("Actual Gender:", gender_dict[gender_labels[index]], "Actual Age:", age_labels[index])
    print("##############################################")
