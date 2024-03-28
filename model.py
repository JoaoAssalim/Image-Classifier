import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

df = pd.read_csv("./objects_train.csv")
df.head()

image_paths = df.iloc[:, 0].tolist()
train_labels = df.iloc[:, 1].tolist()

train_images = []
for image_path in image_paths:
    image = load_img(image_path, target_size=(128, 128))
    image_array = img_to_array(image) / 255.0
    train_images.append(image_array)
train_images = np.array(train_images)

train_labels = np.array(train_labels)
if len(train_labels.shape) == 1:
    train_labels = to_categorical(train_labels)

num_classes = train_labels.shape[1]

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='sigmoid')
    ])
    return model

input_shape = (128, 128, 3)
learning_rate = 0.005

model = build_model(input_shape, num_classes)
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, batch_size=32)
model.save("classifier.h5")