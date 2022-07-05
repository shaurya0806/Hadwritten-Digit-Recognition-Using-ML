#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf


# #Tensorflow already contain MNIST data set which can be loaded using Keras

# In[2]:


#Creating a variable mnist to have handwritten dataset images of
mnist = tf.keras.datasets.mnist


# In[3]:


#unpacking the dataset into train and test dataset
(x_train, y_train),(x_test,y_test) = mnist.load_data()


# In[4]:


x_train.shape


# In[5]:


#To plot the image
import matplotlib.pyplot as plt


# In[6]:


plt.imshow(x_train[0])


# In[7]:


plt.show()


# In[8]:


#To change the image configuration to binary
plt.imshow(x_train[0], cmap = plt.cm.binary)


# #Checking the values of each pixels
# #Before Normalisation

# In[9]:


#Just by printing
print(x_train[0]) #Brfore Normalization
#From Output we will be seeing that pic shown is inverted to existing data, due to binary


# #Normalizing the data | Pre-processing step

# In[10]:


x_train = tf.keras.utils.normalize (x_train, axis = 1)
x_test = tf.keras.utils.normalize (x_test, axis = 1)


# In[11]:


plt.imshow(x_train[0], cmap = plt.cm.binary)


# In[12]:


#After Normalization
print(x_train[0])


# In[13]:


#To check the label inside the network
print(y_train[0])


# #Resizing image to make it suitable for apply Convolution Operation

# In[14]:


import numpy as np
IMG_SIZE=28 #defined a variable
x_trainr= np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) #INcresing one dimesion for kernal or filter operation
x_testr = np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
print("Training Sample dimension",x_trainr.shape)
print("Test Sample Dimension",x_testr.shape)


# #Creating a Deep Learning Network

# In[15]:


from tensorflow.keras.models import Sequential #For different layers
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D


# In[16]:


#Creating a neural network 
model = Sequential() #Defining new variable and to add the layers sequentially

#first Convolution Layer (28-3+1=26x26)
model.add(Conv2D(64,(3,3),input_shape = x_trainr.shape[1:])) #only for 1st convolution layer to mention input layer size 64 kernel and 3x3 size
model.add(Activation("relu"))## activation function to make it non-linear, if <0 remove
model.add(MaxPooling2D(pool_size=(2,2))) ##Maxpooling 

#2nd Convolution Layer
model.add(Conv2D(64,(3,3))) #2nd covolution layer
model.add(Activation("relu"))## activation function 
model.add(MaxPooling2D(pool_size=(2,2))) ##Maxpooling

#3rd Convolution Layer
model.add(Conv2D(64,(3,3))) #3rd covolution layer
model.add(Activation("relu"))## activation function 
model.add(MaxPooling2D(pool_size=(2,2))) ##Maxpooling

#Fully connected Layer 1
model.add (Flatten()) #flatten the data from 2D to 1D
model.add (Dense(64))
model.add(Activation("relu"))

#Fully connected Layer 2
model.add (Dense(32))
model.add(Activation("relu"))

#Last Fully Connected Layer , output must be equal to number of classes, 10(0-9)
model.add(Dense(10))
model.add(Activation('softmax')) #activation function is changed to Softmax(Class probabilities)


# In[17]:


model.summary()


# In[18]:


print("Total Training Samples = ",len(x_trainr))


# In[19]:


model.compile(loss ="sparse_categorical_crossentropy", metrics=['accuracy'])


# In[20]:


model.fit(x_trainr,y_train,epochs=5, validation_split = 0.3)##Training the model


# #If val_accuracy < accuracy then it goes under overfitting

# In[21]:


#Evaluating on testing data set MNIST
test_loss,test_acc = model.evaluate(x_testr,y_test)
print("Test loss on 10,000 test samples", test_loss)
print("Validation accuracy on 10,000 test samples",test_acc)


# In[22]:


prediction = model.predict([x_testr])


# In[23]:


#these predictions are based on one hot encoding so these are only arrays, containing softmax predictionf
print(prediction) 


# In[24]:


#to get the maximum value index we are using numpy
print(np.argmax(prediction[0]))


# In[25]:


#To check the above output as true or false
plt.imshow(x_test[0])


# In[26]:


#Again checking the prediction 
print(np.argmax(prediction[126]))


# In[27]:


#To check the above output as true or false
plt.imshow(x_test[126])


# # Now to check by drawing by ourself, we will have to import opencv

# In[28]:


#Did pip install opencv-python
import cv2


# In[29]:


img = cv2.imread('eight.png')


# In[30]:


plt.imshow(img)


# In[31]:


img.shape


# In[32]:


#converting it into grey image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[33]:


#Reduced the size
gray.shape


# In[34]:


#new variable "resized"
resized = cv2.resize(gray, (28,28), interpolation = cv2.INTER_AREA)


# In[35]:


#Here we can see that the size thas baan reduced to 28,28
resized.shape


# In[36]:


#new variable newing for normalizing the image in 0 to 1 scaling
newing = tf.keras.utils.normalize(resized, axis=1)


# In[37]:


newing = np.array(newing).reshape(-1,IMG_SIZE,IMG_SIZE,1) #kernal operation of Convolution layer


# In[38]:


newing.shape


# In[39]:


#Now running the png file for prediction in the model and we are using the same variable as above for prediction
prediction = model.predict(newing)


# In[40]:


#Here we will see the output of the image
print(np.argmax(prediction))


# In[ ]:




