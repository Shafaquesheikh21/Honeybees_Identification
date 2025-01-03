#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


import os
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, Flatten, MaxPooling2D, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import optimizers, losses
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os.path
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import cv2


# In[ ]:


data = "/content/drive/MyDrive/New Dataset"


# In[ ]:


import os
import pandas as pd
from pathlib import Path

image_dir = Path(data)

# Adding patterns for '.png', '.PNG', '.jpf', and '.jpeg' extensions
filepaths = list(image_dir.glob(r'**/*.PNG')) + list(image_dir.glob(r'**/*.png')) + \
            list(image_dir.glob(r'**/*.jpf')) + list(image_dir.glob(r'**/*.jpeg'))

# Extracting labels
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# Converting to pandas Series
filepaths = pd.Series(filepaths, name='Filepath').astype(str)
labels = pd.Series(labels, name='Label')

# Creating DataFrame
image_df = pd.concat([filepaths, labels], axis=1)


# In[ ]:


filepaths


# In[ ]:


pathcerena = '/content/drive/MyDrive/New Dataset/Apis Cerena Indica'
pathdorsata = '/content/drive/MyDrive/New Dataset/Apis Dorsata'
pathflorea = '/content/drive/MyDrive/New Dataset/Apis Florea'
pathmellifera = '/content/drive/MyDrive/New Dataset/Apis Mellifera'
pathtrigona = '/content/drive/MyDrive/New Dataset/Trigona'
from pathlib import Path

data_dir_path = Path(pathcerena)
image_count = len(list(data_dir_path.glob('*.png'))) + len(list(data_dir_path.glob('*.jpg'))) + len(list(data_dir_path.glob('*.jpeg')))
print(f'Total number of images of the Apis Cerena Indica honeybee is: {image_count}')

data_dir_path = Path(pathdorsata)
image_count = len(list(data_dir_path.glob('*.png'))) + len(list(data_dir_path.glob('*.jpg'))) + len(list(data_dir_path.glob('*.jpeg')))
print(f'Total number of images of the Apis Dorsata honeybee is:: {image_count}')

data_dir_path = Path(pathflorea)
image_count = len(list(data_dir_path.glob('*.png'))) + len(list(data_dir_path.glob('*.jpg'))) + len(list(data_dir_path.glob('*.jpeg')))
print(f'Total number of images of the Apis Florea honeybee is:: {image_count}')

data_dir_path = Path(pathmellifera)
image_count = len(list(data_dir_path.glob('*.png'))) + len(list(data_dir_path.glob('*.jpg'))) + len(list(data_dir_path.glob('*.jpeg')))
print(f'Total number of images of the Apis Mellifera honeybee is:: {image_count}')

data_dir_path = Path(pathtrigona)
image_count = len(list(data_dir_path.glob('*.png'))) + len(list(data_dir_path.glob('*.jpg'))) + len(list(data_dir_path.glob('*.jpeg')))
print(f'Total number of images of the Trigona honeybee is:: {image_count}')


# In[ ]:


import os
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def display_images_from_path_in_row(path, class_name):
    files = os.listdir(path)
    images = [file for file in files if file.endswith(('.png', '.jpg', '.jpeg'))]
    num_images = min(len(images), 5)
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        img_path = os.path.join(path, images[i])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_images, i+1)
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle(class_name, fontsize=16)
    plt.show()

pathcerena = '/content/drive/MyDrive/New Dataset/Apis Cerena Indica'
pathdorsata = '/content/drive/MyDrive/New Dataset/Apis Dorsata'
pathflorea = '/content/drive/MyDrive/New Dataset/Apis Florea'
pathmellifera = '/content/drive/MyDrive/New Dataset/Apis Mellifera'
pathtrigona = '/content/drive/MyDrive/New Dataset/Trigona'

data_dir_paths = [pathcerena, pathdorsata, pathflorea, pathmellifera, pathtrigona]
class_names = ['Apis Cerena Indica', 'Apis Dorsata', 'Apis Florea', 'Apis Mellifera', 'Trigona']

for data_dir, class_name in zip(data_dir_paths, class_names):
    display_images_from_path_in_row(data_dir, class_name)


# In[ ]:


IMAGE_SHAPE = (224, 224)

def prepare_image(file):
    img = image.load_img(file, target_size=IMAGE_SHAPE)
    img_array = image.img_to_array(img)
    return tf.keras.applications.efficientnet.preprocess_input (img_array)


# In[ ]:


import glob
directories = os.listdir(data)

files = []
labels = []

for folder in directories:
    fileList = glob.glob(data + '/'+ folder + '/*')
    labels.extend([folder for l in fileList])
    files.extend(fileList)
len(files), len(labels)


# In[ ]:


selected_files = []
selected_labels = []

for file, label in zip(files, labels):
    if 'mask' not in file:
        selected_files.append(file)
        selected_labels.append(label)


len(selected_files), len(selected_labels)


# In[ ]:


images = {
    'image': [],
    'target': []
}

for i, (file, label) in enumerate(zip(selected_files, selected_labels)):
    images['image'].append(prepare_image(file))
    images['target'].append(label)

print('Finished.')


# In[ ]:


images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

images['target'] = le.fit_transform(images['target'])

classes = le.classes_
print(f'the target classes are: {classes}')


# In[ ]:


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.applications.vgg16 import VGG16

# Assuming 'images' is your DataFrame containing image paths and corresponding labels
# Assuming the column names in 'images' DataFrame are 'image' for image paths and 'target' for labels

# Convert image and target columns to numpy arrays
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])


# In[ ]:


# Convert image and target columns to numpy arrays
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])


# In[ ]:


# Initialize LabelEncoder
le = LabelEncoder()

# Encode the target labels
images['target'] = le.fit_transform(images['target'])


# In[ ]:


classes = le.classes_
print(f'The target classes are: {classes}')


# In[ ]:


# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images['image'], images['target'], test_size=0.10)


# In[ ]:


# Shape of the data
print(f"x_train shape: {x_train.shape}, x_test shape: {x_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")


# In[ ]:


# Assuming IMAGE_SHAPE is defined somewhere in your code
IMAGE_SHAPE = (224, 224)  # Example shape, replace with your actual image shape


# In[ ]:


# Load pre-trained VGG16 model
base_model = VGG16(
    include_top=False,
    weights='imagenet',
    input_shape=(*IMAGE_SHAPE, 3),
    classes=len(classes))  # Number of classes

# Freeze base model layers
base_model.trainable = False

# Define your custom head on top of the base model
x = base_model.output
x = layers.Conv2D(256, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D(keepdims=True)(x)
x = layers.Conv2D(128, 3, padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.GlobalAveragePooling2D(keepdims=True)(x)
x = layers.Flatten()(x)
x = layers.Dense(64)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(len(classes), activation='softmax')(x)  # Output layer with softmax activation

# Create the model
incept_model = keras.models.Model(inputs=base_model.input, outputs=x)

# Compile the model
incept_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
incept_model.summary()


# In[ ]:


history = incept_model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=32, epochs=20)


# In[ ]:


from tensorflow.keras.models import load_model
incept_model.save('/content/drive/MyDrive/Saved Models/honeybee_classifier_new.h5')


# In[ ]:


incept_model.evaluate(x=x_test, y = y_test, batch_size=32, verbose=1)


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis = 0))
    pred = classes[np.argmax(res)]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred}')


# In[ ]:


predict_image(data + '/Apis Cerena Indica/img10.jpg', 'Apis Cerena Indica')


# In[ ]:


def predict_image(img_path, label):
  """Predicts the class of an image and displays it with the true label.

  Args:
      img_path: Path to the image file.
      label: True label of the image (for informational purposes).

  Returns:
      None
  """
  img1 = prepare_image(img_path)
  res = incept_model.predict(np.expand_dims(img1, axis=0))
  pred_class = np.argmax(res)  # Get the index of the predicted class
  pred_label = classes[pred_class]  # Use the index to get the class name

  img = image.load_img(img_path)
  plt.imshow(np.array(img))
  plt.title(f'True: {label}\nPredicted: {pred_label}')
  plt.show()


# In[ ]:


predict_image(data + '/Apis Cerena Indica/img10.jpg', '0')


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Apis Cerena Indica')


# In[ ]:


def predict_image(img_path, label):
  """Predicts the class of an image and displays it with the true label.

  Args:
      img_path: Path to the image file.
      label: True label of the image (for informational purposes).

  Returns:
      None
  """
  img1 = prepare_image(img_path)
  res = incept_model.predict(np.expand_dims(img1, axis=0))
  pred_class = np.argmax(res)  # Get the index of the predicted class
  pred_label = classes[pred_class]  # Use the index to get the class name

  img = image.load_img(img_path)
  plt.imshow(np.array(img))
  plt.title(f'True: {label}\nPredicted: {pred_label}')
  plt.show()


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Apis Cerena Indica')


# In[ ]:


import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Assuming your data is prepared and stored in the 'images' dictionary

# Ensure 'images' has columns named 'image' (for image data) and 'target' (for labels)

# Convert image and target columns to NumPy arrays (if not already)
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])

# Initialize LabelEncoder
le = LabelEncoder()

# Encode the target labels (converting class names to numerical labels)
images['target'] = le.fit_transform(images['target'])

# Get the encoded classes for reference
classes = le.classes_
print(f'The target classes are (encoded): {classes}')  # Output: [0 1 2 3 4]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(images['image'], images['target'], test_size=0.10)

# ... rest of your model building and training code ...

def predict_image(img_path, label):
  """Predicts the class of an image and displays it with the true label.

  Args:
      img_path: Path to the image file.
      label: True label of the image (for informational purposes).

  Returns:
      None
  """
  img1 = prepare_image(img_path)
  res = incept_model.predict(np.expand_dims(img1, axis=0))
  pred_class = np.argmax(res)  # Get the index of the predicted class

  # Use the predicted class index to retrieve the corresponding class name
  pred_label = classes[pred_class]

  img = image.load_img(img_path)
  plt.imshow(np.array(img))
  plt.title(f'True: {label}\nPredicted: {pred_label}')
  plt.show()


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Trigona')


# In[ ]:


def predict_image(img_path, label):
  """Predicts the honeybee species from an image.

  Args:
      img_path: Path to the image file.
      label: True label of the image (for comparison).

  Returns:
      Predicted class name as a string.
  """
  img1 = prepare_image(img_path)
  res = incept_model.predict(np.expand_dims(img1, axis=0))
  pred = classes[np.argmax(res)]

  # Don't print here, return the predicted class
  return pred

# Example usage
predicted_class = predict_image(data + '/Trigona/img12.jpeg', 'Trigona')
print(f"Predicted Class: {predicted_class}")


# In[ ]:


def predict_image(img_path, label):
  """Predicts the honeybee species from an image and returns the class name.

  Args:
      img_path: Path to the image file.
      label: True label of the image (for comparison).

  Returns:
      Predicted class name as a string.
  """
  img1 = prepare_image(img_path)
  res = incept_model.predict(np.expand_dims(img1, axis=0))
  pred = classes[np.argmax(res)]

  # Access the corresponding class name using the predicted index
  predicted_class_name = pred

  # Don't print here, return the predicted class name
  return predicted_class_name

# Example usage
predicted_class = predict_image(data + '/Trigona/img12.jpeg', 'Trigona')
print(f"Predicted Class: {predicted_class}")


# In[ ]:


def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis = 0))
    pred = classes[np.argmax(res)]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred}')


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Trigona')


# In[ ]:


def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis=0))
    pred_class_index = np.argmax(res)
    pred_class_name = classes[pred_class_index]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred_class_name}')


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Trigona')


# In[ ]:


predict_image(data + '/Apis Florea/img102.jpg', 'Apis Florea')


# In[ ]:


from sklearn.preprocessing import LabelEncoder

# Assuming 'images' is your DataFrame containing image paths and corresponding labels
# Assuming the column names in 'images' DataFrame are 'image' for image paths and 'target' for labels

# Convert image and target columns to numpy arrays
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the target labels
images['target'] = le.fit_transform(images['target'])

# Store the class names in the 'classes' array
classes = le.classes_

print(f'The target classes are: {classes}')  # Verify if classes are correctly populated

def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis=0))
    pred_class_index = np.argmax(res)
    pred_class_name = classes[pred_class_index]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred_class_name}')

# Your remaining code...


# In[ ]:


# Assuming 'images' is your DataFrame containing image paths and corresponding labels
# Assuming the column names in 'images' DataFrame are 'image' for image paths and 'target' for labels

# Convert image and target columns to numpy arrays
images['image'] = np.array(images['image'])
images['target'] = np.array(images['target'])

# Initialize LabelEncoder
le = LabelEncoder()

# Fit and transform the target labels
encoded_labels = le.fit_transform(images['target'])

# Store the class names in the 'classes' array
classes = le.classes_

# Create a dictionary to map encoded labels to class names
label_to_classname = dict(zip(encoded_labels, classes))

print(f'The target classes are: {classes}')  # Verify if classes are correctly populated

def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis=0))
    pred_class_index = np.argmax(res)
    pred_class_name = label_to_classname[pred_class_index]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred_class_name}')

# Your remaining code...


# In[ ]:


classes = le.classes_
print(f'the target classes are: {classes}')


# In[ ]:


# Define a dictionary to map class numbers to species names
class_to_species = {
    0: 'Apis Cerena Indica',
    1: 'Apis Dorsata',
    2: 'Apis Florea',
    3: 'Apis Mellifera',
    4: 'Trigona'
}

# Modify the prediction function to use the dictionary
def predict_image(img_path, label):
    img1 = prepare_image(img_path)
    res = incept_model.predict(np.expand_dims(img1, axis=0))
    pred_class = np.argmax(res)
    pred_species = class_to_species[pred_class]

    img = image.load_img(img_path)
    plt.imshow(np.array(img))
    plt.title(f'True: {label}\nPredicted: {pred_species}')

# Example usage


# In[ ]:


predict_image(data + '/Trigona/img12.jpeg', 'Trigona')


# In[ ]:


predict_image(data + '/Apis Florea/img100.jpeg', 'Apis Florea')


# In[ ]:


predict_image(data + '/Apis Dorsata/img12.jpeg', 'Apis Dorsata')


# In[ ]:


predict_image(data + '/Apis Mellifera/img16.jpeg', 'Apis Mellifera')


# In[ ]:


import tensorflow as tf

# Replace with the actual path to your model file
model_path = '/content/drive/MyDrive/Saved Models/honeybee_classifier_new.h5'

try:
  model = tf.keras.models.load_model(model_path)
  print("Model loaded successfully!")
except Exception as e:
  print("Error loading model:", e)


# In[ ]:




