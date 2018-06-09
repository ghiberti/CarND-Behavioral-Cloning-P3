import csv
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.core import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Flatten, Dense, Dropout, Convolution2D, Cropping2D

def read_csv(path):
    """
    Reads driving_log.csv file while allowing approx. 10% of the examples
    with driving angle of 0.
    
    Arguments:
    path: path to directory that contains driving_log.csv
    
    returns read rows as a list of lines.
    
    """
    
    lines = []
    with open(path + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # Check if driving angle is 0
            if float(line[3]) == 0. :
                # Choose between 0 or 1 with 90% possibility given to 0.
                if np.random.choice([0,1], p=[0.9, 0.1]):
                    lines.append(line)
            else:
                lines.append(line)
            
    return lines    

    

def generator(data, batch_size=32):
    """
    Generates batches of images and measurments for training
    including adding flipped image, measurement pairs in order to
    reduce possible bias in training set. Images are also converted
    to RGB during processing.
    
    Arguments:
    data: zip object of image paths and measurements
    batch_size: required size of data batch. defaults to 32.
    
    returns features, labels as numpy arrays
    """
    # Unroll zip object to a list
    lines = list(data)
    
    lines_nr = len(lines)
    
    while 1:
        
        np.random.shuffle(lines)
        
        # Set step to batch_size/2 in order to account for
        # flipped image, measurement pairs
        for i in range(0, lines_nr, int(batch_size/2)):
            
            batch = lines[i:i+int(batch_size/2)]
            
            images = []
            measurements = []
                        
            for line in batch:
                    
                    image = cv2.imread(line[0])
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    flipped_img= cv2.flip(image, 1)
                    images.append(image)
                    images.append(flipped_img)

                    measurement = float(line[1])
                    flipped_measure = -measurement
                    
                    measurements.append(measurement)
                    measurements.append(flipped_measure)                    
            
            features = np.array(images)
            labels = np.array(measurements)
            
            yield features, labels 
            


def covnet():
    
    """
    Creates convolutional network model in Keras.
    
    returns model
    """
    
    model = Sequential()

    # Normalize image input
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    
    # Crops 50 px from top and 20px from bottom of the image
    # Outputs 90x320x3
    model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(3,160,320)))

    # Conv layer1, outputs 21x79x16
    model.add(Convolution2D(filters=16, kernel_size=7, strides = 4, padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    # Conv layer2, outputs 9x38x32
    model.add(Convolution2D(filters=32, kernel_size=5, strides = 2, padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    # Conv layer3, outputs 4x18x64
    model.add(Convolution2D(filters=64, kernel_size=3, strides = 2, padding='valid'))
    model.add(LeakyReLU(alpha=0.1))

    # Flattens to 4608
    model.add(Flatten())

    model.add(Dense(40))
    model.add(Dropout(0.5))
    model.add(Dense(20))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    
    return model


def main(csv_path):
    """
    Implements training pipeline for autonomous driving training.
    Saves the trained model to 'model.h5' file.
    
    Arguments:
    path: path to directory that contains driving_log.csv
    
    
    """
        # Read driving_log to numpy array
        csv_lines = np.array(read_csv(csv_path))
        
        # Split data into 80% training and 20% validation
        x_train, x_val, y_train , y_val = train_test_split(csv_lines[:,0], csv_lines[:,3], test_size=0.20, random_state=42) 
        
        # Match features and labels together for training and validation
        examples_train = zip(x_train, y_train)
        examples_val = zip(x_val, y_val)
        
        # Initialize generators for training and validation
        training_generator = generator(examples_train)
        validation_generator = generator(examples_val)
        
        model = covnet()

        model.compile(loss='mse', optimizer ='adam')
        
        # Train the model
        model.fit_generator(training_generator, steps_per_epoch = len(x_train) \
                            , epochs = 5, validation_data = validation_generator, validation_steps = len(x_val))

        model.save('model.h5')

if __name__ == "__main__":
    csv_path = sys.argv[1]
    main(csv_path)