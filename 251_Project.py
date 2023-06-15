#!/usr/bin/env python
# coding: utf-8

# In[138]:


import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.layers.experimental.preprocessing as AugLayers
from tensorflow.keras import regularizers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[148]:


df_train = pd.read_csv("sign_mnist_train.csv")
df_test = pd.read_csv('sign_mnist_test.csv')
df_train.head()
from sklearn.model_selection import train_test_split


# In[149]:


x_train = df_train.iloc[:,1:785]
y_train = df_train.iloc[:,0:1]
x_test = df_test.iloc[:,1:785]
y_test = df_test.iloc[:,0:1]
x_train = x_train/255
x_test = x_test/255
mean = np.mean(x_train, axis=0)
sd = np.std(x_train, axis = 0)
x_train = (x_train - mean)/sd
x_test = (x_test - mean)/sd
# x_train1, x_val, y_train1, y_val = train_test_split(x_train, y_train, test_size = 0.2, shuffle = True, random_state = 0)


# In[145]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.countplot(x="label",data=df_train);


# In[155]:


from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
pca = PCA(n_components = 0.95)
pca.fit(x_train)
x_train_pca = pca.transform(x_train)
x_train_pca = pd.DataFrame(x_train_pca)
x_train_pca.head()
# 115 components 


# In[143]:


pca.explained_variance_ratio_


# In[165]:


from sklearn.linear_model import LogisticRegression
import time


# In[166]:


runtime = []
accuracy = []
for c in [0.001,0.01,0.1,1,10,100,1000,10000]:
    t0 = time.time()
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 10000, C = c)
    model.fit(x_train, y_train.values.ravel())
    y_pred = model.predict(x_test)
    accuracy.append(model.score(x_test, y_test))
    end_time = time.time()
    total_time = end_time - t0
    runtime.append(total_time)


# In[179]:


errors = [1-acc for acc in accuracy]
plt.plot([-3,-2,-1,0,1,2,3,4],errors,color='royalblue',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.xlabel('log C')
plt.ylabel('Test Error')
plt.title('Test Errors vs log C')
plt.savefig('LR_testerror_vs_logC.png',dpi=300)


# In[181]:


plt.plot([-3,-2,-1,0,1,2,3,4],runtime,color='royalblue',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.xlabel('log C')
plt.ylabel('Time taken')
plt.title('Time taken vs log C')
plt.savefig('LR_time_vs_logC.png',dpi=300)


# In[175]:


# confusion matrix
from sklearn.metrics import confusion_matrix
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter = 10000, C = 0.01)
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)
score = model.score(x_test, y_test)
cm = confusion_matrix(y_test, y_pred)


# In[177]:


plt.figure(figsize=(23,23))
sns.heatmap(cm, annot=True, fmt=".1f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.savefig('LR_conf_matrix.png',dpi=300)


# In[219]:


# Getting misclassified image labels
index = 0
misclassifiedIndexes = []
for label, predict in zip(y_test.values.ravel(), y_pred):
    if label != predict: 
        misclassifiedIndexes.append(index)
    index = index + 1


# In[221]:


# Showing the misclassified images
print(len(misclassifiedIndexes))
print(len(y_pred))
print(len(y_test))
plt.figure(figsize=(20,4))
for plotIndex, badIndex in enumerate(misclassifiedIndexes[0:5]):
    plt.subplot(1, 5, plotIndex + 1)
    plt.imshow(np.reshape(x_test.to_numpy()[badIndex], (28,28)), cmap=plt.cm.gray)
    plt.title('Predicted: {}, Actual: {}'.format(y_pred[badIndex], y_test.to_numpy()[badIndex]), fontsize = 15)
plt.savefig('LR_misclassified.png',dpi=300)


# In[126]:


t0 = time.time()
model = LogisticRegression(penalty = 'l2',multi_class='ovr', solver='newton-cg', max_iter = 10000, C = 1e5)
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)
end_time = time.time()
total_time = end_time - t0


# In[127]:


total_time


# In[128]:


model.score(x_test, y_test)


# In[222]:


# SVM
from sklearn.svm import SVC
c = []
for i in range(-4, 6):
    #print(i)
    c.append(pow(10, i))
error = []
accuracy = []
total_time = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i, decision_function_shape='ovr', kernel="poly", degree=3, gamma="auto")
    # Fit classifier
    SVM.fit(x_train, y_train.values.ravel())
    # Predict labels according
    y_pred = SVM.predict(x_test)
    # Print accuracy on test data and labels
    end_time = time.time()
    total_time.append(end_time - start_time)
    accuracy.append(SVM.score(x_test, y_test))
    


# In[223]:


total_time


# In[224]:


accuracy


# In[225]:


from sklearn.svm import SVC
import time
c = []
for i in range(-4, 6):
    #print(i)
    c.append(pow(2, i))
error = []
accuracy_lin = []
total_time_lin = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i, kernel='linear')
    # Fit classifier
    SVM.fit(x_train, y_train.values.ravel())
    # Predict labels according
    y_pred = SVM.predict(x_test)
    # Print accuracy on test data and labels
    end_time = time.time()
    total_time_lin.append(end_time - start_time)
    accuracy_lin.append(SVM.score(x_test, y_test))


# In[ ]:





# In[226]:


errors = [1-acc for acc in accuracy]
plt.plot([-4,-3,-2,-1,0,1,2,3,4,5],errors,color='royalblue',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.xlabel('log C')
plt.ylabel('Test Error')
plt.title('Test Errors vs log C')
plt.savefig('SVMlinear_testerror_vs_logC.png',dpi=300)


# In[234]:


from sklearn.neighbors import NearestNeighbors
import random
subset = []
for i in range(100):
    number = random.randint(1, 27455)
    if number not in subset:
        subset.append(number)

print(subset)
print(len(subset))
n = len(subset)
X_sample_100 = x_train.to_numpy()[subset]
Y_sample_100 = y_train.to_numpy()[subset]
sigma = 0
for (index, value) in enumerate(X_sample_100):
    label = Y_sample_100[index]
    X = x_train.to_numpy()[y_train.values.ravel() == label]
    nbrs = NearestNeighbors(n_neighbors=7).fit(X) # value of  k = 7
    dist, nb = nbrs.kneighbors([value])
    sigma += dist[0, 7-1]
sigma = sigma/n
gamma=1/(2*sigma**2)
print(sigma)


# In[ ]:


c = []
for i in range(-4, 6):
    #print(i)
    c.append(pow(2, i))
error = []
accuracy_gauss = []
total_time_gauss = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i,decision_function_shape='ovr',kernel='rbf',gamma=gamma)
    # Fit classifier
    SVM.fit(x_train, y_train.values.ravel())
    # Predict labels according
    y_pred = SVM.predict(x_test)
    end_time = time.time()
    total_time_gauss.append(end_time - start_time)
    # Print accuracy on test data and labels
    accuracy_gauss.append(SVM.score(y_pred, y_test))


# In[241]:


error_poly=[0.5348577802565533, 0.43070273284997207, 0.33853876185164533, 0.2714723926380368, 0.21862799776910202,
 0.19534300055772447, 0.18042387060791965, 0.1809815950920245, 0.1798661461238148, 0.17861126603457889]
error_lin = [1-acc for acc in accuracy_lin]
error_guass = [1-acc for acc in accuracy_guass]
powers = [-4,-3,-2,-1,0,1,2,3,4,5]
plt.plot(powers,error_poly,color='royalblue',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.plot(powers,error_lin,color='mediumvioletred',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
#plt.plot(powers,error_guass,color='firebrick',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.xlabel('log2 C')
plt.ylabel('Test Error')
plt.title('Test Errors vs log2 C (exponent of 2)')
plt.legend(['Polynomial SVM, degree 3','Linear SVM','Gaussian SVM'])
plt.savefig('SVMdiff_testerror_vs_logC.png',dpi=300)


# In[ ]:


from sklearn.svm import SVC
import time
c = []
for i in range(-4, 6):
    #print(i)
    c.append(pow(2, i))
error = []
accuracy_lin2 = []
total_time_lin2 = []
for i in c:
    start_time = time.time()
    SVM = SVC(C = i, kernel='linear', decision_function_shape='ovr')
    # Fit classifier
    SVM.fit(x_train, y_train.values.ravel())
    # Predict labels according
    y_pred = SVM.predict(x_test)
    # Print accuracy on test data and labels
    end_time = time.time()
    total_time_lin2.append(end_time - start_time)
    accuracy_lin2.append(SVM.score(x_test, y_test))


# In[243]:


powers = [-4,-3,-2,-1,0,1,2,3,4,5]
plt.plot(powers,total_time,color='royalblue',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.plot(powers,total_time_lin,color='mediumvioletred',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
#plt.plot(powers,error_guass,color='firebrick',linestyle='dashed',marker='o',linewidth=2,markerfacecolor='none',markersize=5)
plt.xlabel('log2 C')
plt.ylabel('Time Taken')
plt.title('Time Taken vs log2 C (exponent of 2)')
plt.legend(['Polynomial SVM, degree 3','Linear SVM','Gaussian SVM'])
plt.savefig('SVMdiff_runtimes_vs_logC.png',dpi=300)

