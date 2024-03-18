import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

# Paths to image datasets
train_data_dir = './images/train/'
validation_data_dir = './images/validation/'

# Setup Image Augmentation parameters
train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255
)

# Import and Augment the images from dataset
train_data = train_datagenerator.flow_from_directory(
    directory=train_data_dir,
    color_mode="grayscale",
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)
validation_data = validation_datagenerator.flow_from_directory(
    directory=validation_data_dir,
    color_mode='grayscale',
    target_size=(48,48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

# Define Class labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

#-------------------------------------------------------------
# Define Model Architecture
#-------------------------------------------------------------

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.summary())

#-------------------------------------------------------------
# Define Training Parameters
#-------------------------------------------------------------

num_train_imgs = 0
for root, dirs, files in os.walk(train_data_dir):
	num_train_imgs += len(files)

num_test_imgs = 0
for root, dirs, files in os.walk(validation_data_dir):
	num_test_imgs += len(files)

# print(num_train_imgs)
# print(num_test_imgs)
epochs = 100

#-------------------------------------------------------------
# Begin Training
#-------------------------------------------------------------

history = model.fit(train_data, steps_per_epoch=num_train_imgs//32, epochs=epochs, validation_data=validation_data, validation_steps=num_test_imgs//32)

# Save trained model in .h5 file format
model.save('model_file.h5')