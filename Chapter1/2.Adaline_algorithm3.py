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
df=df[df.iloc[:,4]!='Iris-virginica']
df.loc[df.iloc[:,4]=='Iris-setosa',4]=1
df.loc[df.iloc[:,4]=='Iris-versicolor',4]=-1

X=np.array(df.iloc[:,[0,2]])
y=np.array(df.iloc[:,4]).reshape((-1,1))
X=(X-X.mean(axis=0))/X.std(axis=0)#standardize the feature data

#build the algorithm
def Adaline(X,y,c1,c2): ##X and y are numpy matrix, y includs binary label
#page 57 in the book: Python Machine learning, Second edition
    L1,L2=np.shape(X)
    X=np.hstack((np.ones((L1,1)),X))
    w=np.random.random_sample(size = L2+1).reshape((1,-1))
    iteration_number=0
    error=[]
    X0=np.hstack((X,y))
    while iteration_number<200:
        np.random.shuffle(X0) 
        XX,y=X0[:,:-1],X0[:,-1].reshape((-1,1))
        eta=c1/(iteration_number+c2)
        s=0
        for i in range(L1):
            y_hat=(XX[i,:]*w).sum()
            s=s+0.5*(((y[i]-y_hat)**2)[0])#error
            w1=eta*((y[i]-y_hat)[0])*XX[i,:]
            w=w+w1
        error.append(s)
        iteration_number=iteration_number+1  
        if iteration_number>1 and abs(error[-1]-error[-2])<0.001:
            break
    return w,error #w is numpy matrix 
w,error=Adaline(X,y,1,100) 
w=w[0].copy() 

#plot the figure
fig, axes = plt.subplots(nrows=2, ncols=2) 
plt.subplot(1, 2, 1)
xx=X[:,0]
xx=np.arange(min(xx),max(xx)+0.1,0.1)
plt.plot(xx,-(w[0]+w[1]*xx)/w[2],'k:',label='boundary')
plt.fill_between(xx,X[:,1].min(),-(w[0]+w[1]*xx)/w[2],alpha=0.5)
plt.fill_between(xx,-(w[0]+w[1]*xx)/w[2],X[:,1].max(),alpha=0.5)

r=y==1
r=r.reshape((-1,1))
r1=np.squeeze(r, axis=1)
X1=X[r1,:]
plt.scatter(X1[:,0],X1[:,1],s=100,label='Iris-setosa')

r=y==-1
r=r.reshape((-1,1))
r1=np.squeeze(r, axis=1)
X1=X[r1,:]
plt.scatter(X1[:,0],X1[:,1],s=100,label='Iris-versicolor')
plt.xlabel('standardized sepal length (cm)',fontdict={'fontname':'Comic Sans MS','fontsize':10})
plt.ylabel('standardized petal length (cm)',fontdict={'fontname':'Comic Sans MS','fontsize':10})
plt.legend(fontsize="7",loc='center right')
plt.autoscale(enable=True, axis='both', tight=True)#axis tight

plt.subplot(1, 2, 2) 
plt.plot(error)
plt.xlabel('iteration steps',fontdict={'fontname':'Comic Sans MS','fontsize':10})
plt.ylabel('error',fontdict={'fontname':'Comic Sans MS','fontsize':10})
fig.tight_layout()
fig.suptitle('Adaline algorithm',y=1.05)
plt.show()