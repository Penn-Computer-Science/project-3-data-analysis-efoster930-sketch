import seaborn as sns
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

#Prints out the versions of tf.
print(tf.__version__)

#Goes to Google to get the mnist dataframe.
mnist = tf.keras.datasets.mnist
#x_trains has images inside it. y_train has labels. It just builds it out and train the model. X=Y test has the same thing but it just shows the model working.
(x_train, y_train), (x_test,y_test) = mnist.load_data()
#Shows the shape of the training data.
sns.countplot(x=y_train)
#Shows the first image in the training data.
plt.show()

#Check to make sure that there NO values that are not a number (NaN).

print("Any NaN Training:",np.isnan(x_train).any())
print("Any NaN Testing:",np.isnan(x_test).any())

#tell the model what shape to expect.
input_shape = (28,28,1) 
#The 28 by 28 is the size of the image. The 1 represents that it is a grayscale image. If it were RGB, it would be 3.

#reshape the training and testing data to include the color channel.
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1], x_train.shape[2],1)
x_train = x_train/255.0 #normalize the data to be between 0 and 1

x_test = x_test.reshape(x_test.shape[0],x_test.shape[1], x_test.shape[2],1)
x_test = x_test/255.0 #normalize the data to be between 0 and 1

#convert our labels to be one-hot, not sparse.
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10) #one-hot encode the labels.
plt.imshow(x_train[100][:,:,0]) #Show an example image from MNIST. The [:,:,0] is to get rid of the color channel for displaying. The comma separates the dimensions.

#Show an example image from MNIST
plt.imshow(x_train[random.randint(0,59999)][:,:,0],cmap='gray')
plt.show()
batch_size = 128  #This is how many images we process at once.
num_classes = 10 #There are 10 digits (0-9)
epochs = 2 #How many times we go through the whole dataset.

#Build the model.... Finally...
model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(5,5),padding = 'same',activation = 'relu',input_shape=input_shape), #What it's basically doing is looking for edges in the image.
        tf.keras.layers.Conv2D(32,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape), #Another convolution layer to find more complex features.
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0,25),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape),
        tf.keras.layers.Conv2D(64,(3,3),padding = 'same',activation = 'relu',input_shape=input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

#The code forces the model to use a category.
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['acc']) #This code is setting up how the model learns.
history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, validation_data=(x_test, y_test)) #This code is training the model.

#plot out training and validation accuracy and loss
flg, ax = plt.subplots(2,1) #Create 2 plots, one on top of the other.
ax[0].plot(history.history['loss'], color = 'b', label="Training Loss")
ax[0].plot(history.history['val_loss'], color = 'r', label="Validation Loss")
legend = ax[0].legend(loc='best', shadow=True)
ax[0].set_title('Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
#The section of code top plotting the loss.

#The section of code below is plotting the accuracy.
ax[1].plot(history.history['acc'], color = 'b', label="Training Accuracy")
ax[1].plot(history.history['val_loss'], color = 'r', label="Validation Accuracy")
legend = ax[1].legend(loc='best', shadow=True)
ax[1].set_title('Accuracy')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')

plt.tight_layout() #Make sure everything fits without overlapping.
plt.show()