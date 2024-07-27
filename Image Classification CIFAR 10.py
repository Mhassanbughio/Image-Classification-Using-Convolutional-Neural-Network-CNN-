#!/usr/bin/env python
# coding: utf-8

# # ***Image Classification:***

# In[3]:


pip install tensorflowImage Classification:


# ***In this notebook, we will classify small images cifar10 dataset from tensorflow keras datasets. 
# There are total 10 classes as shown below. We will use CNN for classification***

# In[41]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np


# ***Load the dataset***

# In[6]:


(X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape


# # Data augmentation and preprocessing 

# In[9]:


X_train.shape


# In[11]:


y_train.shape


# In[12]:


y_train[:6]


# ***y_train is a 2D array, for our classification having 1D array is good enough. so we will convert this to now 1D array***
# 

# In[14]:


y_train=y_train.reshape(-1,)
y_train[:6]


# In[16]:


y_test=y_test.reshape(-1,)
y_test[:4]


# In[17]:


classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]


# In[18]:


def plot_sample(X, y, index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[19]:


plot_sample(X_train, y_train, 0)


# In[20]:


plot_sample(X_train,y_train,1)


# In[21]:


plot_sample(X_train,y_train,5)


# ***Normalize the images to a number from 0 to 1. Image has 3 channels (R,G,B) and each value in the channel can range from 0 to 255. Hence to normalize in 0-->1 range, we need to divide it by 255***

# # Normalizing the training data

# X_train = X_train / 255.0
# X_test = X_test / 255.0

# # Build simple artificial neural network for image classification

# In[25]:


ann = models.Sequential([
        layers.Flatten(input_shape=(32,32,3)),
        layers.Dense(3000, activation='relu'),
        layers.Dense(1000, activation='relu'),
        layers.Dense(10, activation='softmax')    
    ])

ann.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

ann.fit(X_train, y_train, epochs=5)


# #You can see that at the end of 5 epochs, accuracy is at around 49%

# # Model evaluation

# In[30]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# # ***Now let us build a convolutional neural network to train our images***

# In[31]:


cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])


# In[32]:


cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[33]:


cnn.fit(X_train, y_train, epochs=6)


# ***With CNN, at the end 5 epochs, accuracy was at around 70% which is a significant improvement over ANN. CNN's are best for image classification and gives superb accuracy. Also computation is much less compared to simple ANN as maxpooling reduces the image dimensions while still preserving the features***

# In[34]:


cnn.evaluate(X_test,y_test)


# # Model evaluation

# In[42]:


from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))


# In[35]:


y_pred = cnn.predict(X_test)
y_pred[:5]


# In[36]:


y_classes = [np.argmax(element) for element in y_pred]
y_classes[:5]


# In[37]:


y_test[:5]


# In[38]:


plot_sample(X_test, y_test,3)


# In[39]:


classes[y_classes[3]]


# In[40]:


classes[y_classes[3]]


# In[ ]:




