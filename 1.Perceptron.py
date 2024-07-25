import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv(r'F:\all\2024~1\E427~1\_A8D2~1\_A8D2~1\_7CB2~1\2PYTHO~1.MAC\PYTHON~1\PYTHON~1\code\ch02\iris.data', header=None)
o1=df.iloc[:,4].unique()#there are three categories
df=df[df.iloc[:,4]!='Iris-virginica']
df.loc[df.iloc[:,4]=='Iris-setosa',4]=1
df.loc[df.iloc[:,4]=='Iris-versicolor',4]=-1

X=np.array(df.iloc[:,[0,2]])
y=np.array(df.iloc[:,4]).reshape((-1,1))

def perceptron(X,y): ##X and y are numpy matrix, y includs binary label
#page 40 in the book: Python Machine learning, Second edition
    L1,L2=np.shape(X)
    X=np.hstack((np.zeros((L1,1)),X))
    w=np.zeros((1,L2+1))*0.1
    learning_rate=0.1
    y_hat=np.zeros((L1,1))
    iteration_number=0
    while iteration_number<100:
        w0=w.copy()
        for i in range(L1):#sample
            if np.matmul(X[i,:],w.T)[0]>=0:
                y_hat[i]=1
            else:
                y_hat[i]=-1
            w=w+learning_rate*(y[i]-y_hat[i])*X[i,:]
            #the weight is updated incrementally after each sample
        iteration_number=iteration_number+1
        if sum((w[0]-w0[0])**2)<0.001:
            break
    return w #w is numpy matrix 
w=perceptron(X,y) 
w=w[0].copy()    
  
plt.figure() 
df1=df[df.iloc[:,4]==1]
xx=df.iloc[:,0]
xx=np.arange(min(xx),max(xx)+0.1,0.1)
plt.plot(xx,-(w[0]+w[1]*xx)/w[2],'k:',label='boundary')
plt.fill_between(xx,-(w[0]+w[1]*xx)/w[2],alpha=0.5)
plt.fill_between(xx,-(w[0]+w[1]*xx)/w[2],df.iloc[:,2].max(),alpha=0.5)
plt.scatter(df1.iloc[:,0],df1.iloc[:,2],s=100,label='Iris-setosa')
df1=df[df.iloc[:,4]==-1]
plt.scatter(df1.iloc[:,0],df1.iloc[:,2],s=100,label='Iris-versicolor')
plt.xlabel('sepal length (cm)',fontdict={'fontname':'Comic Sans MS','fontsize':20})
plt.ylabel('petal length (cm)',fontdict={'fontname':'Comic Sans MS','fontsize':20})
plt.legend()
plt.autoscale(enable=True, axis='both', tight=True)#axis tight
plt.savefig(r'F:\all\时光2024\要读的书籍\沈度学习四大名著书籍+源码\沈度学习四大名著书籍+源码\深度学习四大名著书籍+源码\2. Python Machine Learning. Machine Learning and Deep Learning with Python, scikit-learn and TensorFlow\my_note\1.perceptron.png',dpi=500)



    