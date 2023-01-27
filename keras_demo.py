#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from plotly.offline import iplot
import cufflinks as cf


# In[2]:


df = pd.read_csv('data.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


y = [1 if x == 'M' else 0 for x in df['diagnosis']]


# In[6]:


X = df.drop(['diagnosis', 'Unnamed: 32'], axis = 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[7]:


X.shape


# In[8]:


def nrl_ntwk(df, check, learn_rate = 0.01, ptnc = 5, epoc = 20): # create a neural network function to call on different data
    df = np.array(df)    # accepts df for x and check for y
    check = np.array(check)
    x_train , x_test , y_train , y_test =train_test_split(df,
                            check,test_size =0.3, random_state=38)
    model = Sequential()
    model.add(Dense(units=256, activation='relu')) # add layers to neural network
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy",
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learn_rate))
            # adjusted learning rate. Trial and error to get value
    early_stop = EarlyStopping(monitor='val_loss',
        mode='min',
        verbose=1,
        patience=ptnc)
    model.fit(x=x_train,
        y=y_train,
        epochs=epoc,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[early_stop]
        )
    predictions = (model.predict(x_test)>0.5).astype("int32")
    cl = (classification_report(y_test,predictions, output_dict = True))
    print(classification_report(y_test,predictions)) # print results of neural network
    print(confusion_matrix(y_test,predictions))
    return cl # return output dictionary of classification report for numbers


# In[9]:


nrl_ntwk(X, y, learn_rate=0.001, ptnc=25, epoc=50)


# In[10]:


df = df.drop(['id', 'diagnosis', 'Unnamed: 32'], axis=1)


# In[11]:


cf.go_offline()


# In[12]:


df.iplot(kind = 'box')


# In[ ]:




