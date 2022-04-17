#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# In[3]:


mnist = tf.keras.datasets.mnist


# In[4]:


(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[5]:


x_train.shape #every image has 28 into 28 size.


# In[6]:


x_test.shape 


# In[7]:


y_train.shape


# In[8]:


import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show()


# In[9]:


plt.imshow(x_train[0], cmap=plt.cm.binary)


# In[10]:


print(x_train[0])  #we're printing the vaues before normalization


# In[11]:


plt.imshow(x_train[1], cmap=plt.cm.binary) #black and white are reversed in the image because of our binary command


# In[12]:


print(x_train[1])


# In[13]:


#now we'll normalize the data 
#for gray image the values ar ebetween 0 to 255
x_train=tf.keras.utils.normalize(x_train, axis=1) #axis can be both 0 or 1
x_test=tf.keras.utils.normalize(x_test,axis=1)
plt.imshow(x_train[0], cmap=plt.cm.binary)


# In[14]:


print(x_train[0])


# In[15]:


print(y_train[0]) #y_train represents the label


# In[16]:


import numpy as np
IMG_SIZE=28
x_trainr=np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) #we're just increasing one dimention for kernal=filter convolution
x_testr=np.array(x_test).reshape(-1,IMG_SIZE, IMG_SIZE,1)
print("training set dimentions",x_trainr.shape)
print("testing set dimentions",x_testr.shape)


# In[17]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


# In[18]:


#creating a neural network now
#each filter having 3*3 size is a neuron
model = Sequential()

##First Convolution layer 0 1 2 3 (60000, 28, 28, 1) 28-3+1=26X26
model.add(Conv2D(64,(3,3),input_shape=x_trainr.shape[1:])) ##onl for first convolution layer to mention input layer size
#1 convolution layer having 64 different filters. Each filter has 3X3 size.
model.add(Activation("relu"))   ##Activation function to make it non linear, it will drop the values that are <0 and move the values >0 to he second layer.
model.add(MaxPooling2D(pool_size=(2,2)))  ##MaxPooling [it will get single maximum value of 2X2 matrix and rest it'll remove]


##Second Convolutin Layer 
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


##Third Convolution layer 
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))


##Fully Connected layer #1  20X20=400 [each 400 neurons will be connected to each of the 64 in the dense layer]
#before using fully connected layer we need to flatten
model.add(Flatten()) #it converts 2D to 1D
model.add(Dense(64))
model.add(Activation("relu"))

#Fully Connected Layer #2
model.add(Dense(32))
model.add(Activation("relu"))


#Last fully connected layer, output must be equal to no of classes,10,0-9
model.add(Dense(10)) ##this last layer must be equal to 10
model.add(Activation('softmax')) ##activation function is changed to softmax(class probabilities)

##if we had binary classification i.e. only two classes,one neuron in dense layer, activation fn:sigmoid




# In[19]:


model.summary()


# In[20]:


print("Total training samples=",len(x_trainr))


# In[21]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=['accuracy'])


# In[22]:


model.fit(x_trainr,y_train,epochs=5,validation_split=0.3) ##training my model


# In[23]:


##evaluating on test data set
test_loss,test_acc=model.evaluate(x_testr,y_test)
print("test loss on 10,000 test samples", test_loss)
print("Validation Accuracy on 10,000 test samples", test_acc)


# In[24]:


predictions=model.predict([x_testr])


# In[25]:


print(predictions)


# In[26]:


print(np.argmax(predictions[0]))


# In[27]:


plt.imshow(x_test[0])


# In[28]:


print(np.argmax(predictions[15]))


# In[29]:


plt.imshow(x_test[15])


# In[30]:


import numpy as np


# In[31]:


import cv2


# In[46]:


img=cv2.imread('Desktop/Nine.png')


# In[47]:


plt.imshow(img)


# In[48]:


img.shape


# In[49]:


gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[50]:


gray.shape


# In[51]:


resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)


# In[52]:


resized.shape


# In[53]:


newimg=tf.keras.utils.normalize(resized,axis=1)


# In[54]:


newimg=np.array(newimg).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# In[55]:


newimg.shape


# In[56]:


predictions=model.predict(newimg)


# In[57]:


print(np.argmax(predictions))


# In[61]:


converter=tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()

with open("model.tflite",'wb') as f:
 f.write(tflite_model)


# In[ ]:




