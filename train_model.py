#import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, warnings, random, time

warnings.filterwarnings('ignore')

# Tensorflow and Keras Modules
# import tensorflow as tf
from keras.utils import load_img
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D, Input

#directory
BASE_DIR = 'UTKFace/'

# labels - age, gender, ethnicity
image_paths = []
age_labels = []
gender_labels = []

# Get a list of all image files
image_set = os.listdir(BASE_DIR)

# Shuffle the set to avoid learning patterns in the dataset order
random.shuffle(image_set)

#create a loop for all dataset images
for filename in (os.listdir(BASE_DIR)):
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

# convert to dataframe
df = pd.DataFrame()
df['image'], df['age'], df['gender'] = image_paths, age_labels, gender_labels
df.head()

# change the label of the gender
gender_dict = {0:'Male', 1:'Female'}

#check the age distribution
sns.displot(df['age'])

# Count plot of gender 
sns.countplot(df['gender'])

def extract_features(images):
    ''' Extract images in a more easily processable format
    Convert images to grayscale and resize them to 128x128
    '''
    features = []
    for image in images:
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

X = extract_features(df['image'])

# make normalization 0-1
## Pixel color scale is 0-255, dividing makes it a 0-1 scale
X = X/255.0

y_gender = np.array(df['gender'])
y_age = np.array(df['age'])

#Create Model
input_shape = (128, 128, 1)
# declare input
inputs = Input((input_shape))
# Convolutional layers 1
conv_1 = Conv2D(32, kernel_size=(3, 3), activation='relu') (inputs)
maxp_1 = MaxPooling2D(pool_size=(2, 2)) (conv_1)
# Convolutional layers 2
conv_2 = Conv2D(64, kernel_size=(3, 3), activation='relu') (maxp_1)
maxp_2 = MaxPooling2D(pool_size=(2, 2)) (conv_2)
# Convolutional layers 3
conv_3 = Conv2D(128, kernel_size=(3, 3), activation='relu') (maxp_2)
maxp_3 = MaxPooling2D(pool_size=(2, 2)) (conv_3)
# Convolutional layers 4
conv_4 = Conv2D(256, kernel_size=(3, 3), activation='relu') (maxp_3)
maxp_4 = MaxPooling2D(pool_size=(2, 2)) (conv_4)


flatten = Flatten() (maxp_4)

# Fully connected layers
dense_1 = Dense(256, activation='relu') (flatten)
dense_2 = Dense(256, activation='relu') (flatten)

dropout_1 = Dropout(0.3) (dense_1)
dropout_2 = Dropout(0.3) (dense_2)

# two outputs
output_1 = Dense(1, activation='sigmoid', name='gender_out') (dropout_1)
output_2 = Dense(1, activation='relu', name='age_out') (dropout_2)

model = Model(inputs=[inputs], outputs=[output_1, output_2])

model.compile(loss=['binary_crossentropy', 'MAE'], optimizer='adam', metrics=['accuracy',['accuracy']])

from tensorflow.keras.utils import plot_model
plot_model(model)

epochs_val = 30

history = model.fit(x=X, y=[y_gender, y_age], batch_size=32, epochs=epochs_val, validation_split=0.2)
epoch_time = int(time.time())
model_filename = f"models/gender_age_detection_model_epochs_{epochs_val}_{epoch_time}.keras"
model.save(model_filename)
# model.save_weights("gender_age_detection_model.weights.h5")

# plot results for gender
acc = history.history['gender_out_accuracy']
val_acc = history.history['val_gender_out_accuracy']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Accuracy Graph')
plt.legend()
plt.figure()

#for loss function
loss = history.history['gender_out_loss']
val_loss = history.history['val_gender_out_loss']

# plt.plot(epochs, loss, 'b', label='Training Loss')
# plt.plot(epochs, val_loss, 'r', label='Validation Loss')
# plt.title('Loss Graph')
# plt.legend()
# plt.show()

#random number for get image
for index in range(6):
    idx = random.randint(1, y_age.size)
    print("Original Gender:", gender_dict[y_gender[idx]], "Original Age:", y_age[idx])
    # predict from model
    pred = model.predict(X[idx].reshape(1, 128, 128, 1))
    pred_gender = gender_dict[round(pred[0][0][0])]
    pred_age = round(pred[1][0][0])
    print("Predicted Gender:", pred_gender, "Predicted Age:", pred_age)
    print('############################################')
    # plt.axis('off')
    # plt.imshow(X[idx].reshape(128, 128), cmap='gray')