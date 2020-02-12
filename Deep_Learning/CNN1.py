"""

This script is an implementation of a Convolutional Neural Network with 
2 hidden layers. More layers can be easily added.

This network was written for cat/dog classification, data preprocessing not included here.

"""


from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# initialise the CNN
classifier = Sequential()

# Convolution
classifier.add(Convolution2D(filters = 32, kernel_size = [3,3], 
                             input_shape = (64,64,3), activation = 'relu'))


# Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# 2nd convolutional layer
classifier.add(Convolution2D(filters = 32, kernel_size = [3,3], activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compiling
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

# fitting CNN to images
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# basic data augmentation
test_datagen = ImageDataGenerator(rescale=1./255)

# define the training set
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

# define the test set
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

# train the CNN
classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)
