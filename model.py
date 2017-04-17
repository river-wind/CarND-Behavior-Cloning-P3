import cv2
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Dropout, Activation, Lambda,Cropping2D
import matplotlib.pyplot as plt

#Get the input data as entere into the .csv file
lines = []
with open('./data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        lines.append(line)

#load the images based on the entries read in from the .csv
#images = []
#measurements = []
#for line in lines:
#	source_path = line[0]
#	filename = source_path.split('\\')[-1]
#	current_path = './data/IMG/' + filename
#	image = cv2.imread(current_path)
#	images.append(image)
#	measurement = float(line[3])
#	measurements.append(measurement)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

#load the images into batches, including left, right and center cameras
def generator(lines, batch_size=32):
    while 1: # Loop forever, yield when needed
        shuffle(lines)
        batch_lines = lines
        correction = 0.1
        images = []
        angles = []
        for batch_line in batch_lines:
            
            #Adding the images from the three cameras and steering angles.
            name1 = 'data3\\IMG\\'+batch_line[0].split('\\')[-1]
            name2 = 'data3\\IMG\\'+batch_line[1].split('\\')[-1]
            name3 = 'data3\\IMG\\'+batch_line[2].split('\\')[-1]
			#add center images and angles to arrays
            center_image = cv2.imread(name1)
            center_angle = float(batch_line[3])
            images.append(center_image)
            angles.append(center_angle)
			
			#add left images and angles to arrays
            left_image = cv2.imread(name2)
            left_angle = center_angle + correction
            images.append(left_image)
            angles.append(left_angle)
    		#add right images and angles to arrays
            right_image = cv2.imread(name3)
            right_angle = center_angle - correction
            images.append(right_image)
            angles.append(right_angle)

			#break into image and steering angle batches
            for i in range(0, len(images), batch_size):
                image_batch = images[i:i+batch_size]
                angle_batch = angles[i:i+batch_size]
                images, angles = shuffle(images, angles)
                
                #Augment the images by horizontally flipping them
                augmented_images, augmented_measurements = [], []
                for image, measurement in zip(image_batch, angle_batch):
                    augmented_images.append(image)
                    augmented_measurements.append(measurement)
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(measurement*-1.0)

            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)  #shuffle the training data

#Training the data using the generator function.
train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)

#old method of building the train arrays
#X_train = np.array(images)
#y_train = np.array(measurements)


#build the model
model = Sequential()
#Cropping unwanted parts of the image (sky, hills and car hood).
model.add(Cropping2D(cropping=((70, 25), (1, 1)), input_shape=(160, 320, 3)))

#Preprocessing the data.
model.add(Lambda(lambda x: (x / 255.0) - 0.5))#, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))

#add convolution layers similar to the NVidia model
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(32, 5, 5, subsample=(2,2),activation="relu"))
model.add(Convolution2D(48, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(MaxPooling2D())  #removed maxpooling, as it was adding a significant lag time to training
model.add(Dropout(0.5))  # trying to avoid overfitting
#model.add(Flatten(input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(120))
#model.add(Activation('relu'))  #removed to speed up training a bit
model.add(Dense(5))
model.add(Activation('softmax'))  #left in one softmax from prior models
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#old method: fit()
#hist = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=4)

#new method, fit_generator.  Actually train the model with the training and validation data here.
hist = model.fit_generator(train_generator, samples_per_epoch= len(train_lines), validation_data=validation_generator, nb_val_samples=len(validation_lines), nb_epoch=4)

print(hist.history)

model.save('model.h5')

#print the keys contained in the history object.
print(hist.history.keys())

#plot the training and validation loss for each epoch.
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
