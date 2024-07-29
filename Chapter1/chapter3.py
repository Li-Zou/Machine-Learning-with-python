'''
algorithm: Adaline
other: features are standardized
stochatic gradient descent
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import random
#read the data
df = pd.read_csv('/Users/li/Desktop/python/ML_with_python/iris.data', header=None)
#Data preprocessing
o1=df.iloc[:,4].unique()#there are three categories
df.loc[df.iloc[:,4]=='Iris-setosa',4]=0
df.loc[df.iloc[:,4]=='Iris-versicolor',4]=1
df.loc[df.iloc[:,4]=='Iris-virginica',4]=2
X=np.array(df.iloc[:,[2,3]])
y=np.array(df.iloc[:,4]).reshape((-1,1))
X=(X-X.mean(axis=0))/X.std(axis=0)#standardize the feature data

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

[(y==0).sum()/len(y),(y==1).sum()/len(y),(y==2).sum()/len(y)]

[(y_train==0).sum()/len(y_train),(y_train==1).sum()/len(y_train),(y_train==2).sum()/len(y_train)]

[(y_test==0).sum()/len(y_test),(y_test==1).sum()/len(y_test),(y_test==2).sum()/len(y_test)]

y1=np.squeeze(y,axis=1)
y1=y1.astype(int)
np.bincount(y1)/len(y1)

y1=np.squeeze(y_train,axis=1)
y1=y1.astype(int)
np.bincount(y1)/len(y1)

y1=np.squeeze(y_test,axis=1)
y1=y1.astype(int)
np.bincount(y1)/len(y1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)#get mean and std value
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

from sklearn.linear_model import Perceptron
ppn=Perceptron(max_iter=40,eta0=0.1,random_state=1)
ppn.fit(X_train_std,y_train)
y_hat=ppn.predict(X_test_std)
print('misclassified sample rate:' (y_hat!=y).sum()/len(y))












