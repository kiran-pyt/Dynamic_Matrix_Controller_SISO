#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import control
from control.matlab import *
from cvxopt import matrix
from cvxopt import solvers
import numpy.matlib

K=2.5
T=20
num=np.array([K]) 
den=np.array([T,1])
H=control.tf(num,den)
print(H)

n=25
h=5
t,y=control.step_response(H,np.arange(0,n*h,h))
plt.plot(t,y)

##Controller Parameters
ySP=1          # Setpoint
m=4            # Control horizon
p=10           # Prediction horizon
Q=1            # Output weight
R=0.04         # Input weight
duMax=0.05     # Input rate constraint
duMin=-duMax
uMax=0.5
uMin=-uMax     # Input constraint
S=y
Su_col=S[1:p+1]

Im_col=np.ones(shape=(m,1));
big_Su=np.empty(shape=(p, 0))
big_Su=np.column_stack([big_Su,Su_col])
bigIm=np.empty(shape=(m, 0))
bigIm=np.column_stack([bigIm,Im_col])


for i in range(1,m,1):
    
    Su_col=Su_col[:-1]
    Su_col=(np.insert(Su_col, 0, 0))
    big_Su=np.column_stack([big_Su,Su_col])
    Im_col=Im_col[:-1]
    Im_col=np.insert(Im_col, 0, 0)
    bigIm=np.column_stack([bigIm,Im_col]) 


big_Su
bigIm
#For Objective 
bigR=(np.ones(shape=(p,1)))*ySP
GammaY=np.diag((np.ones(shape=(p)))*Q)
GammaU=np.diag((np.ones(shape=(m)))*R)
Im=np.eye(m)

Hess=matrix(((np.dot((np.dot(np.transpose(big_Su),GammaY)),big_Su))+GammaU),tc='d')
C_LHS=matrix(np.vstack((Im,-Im,bigIm,-bigIm)),tc='d')

uPrev=0.0
Yk0=np.zeros(shape=(n,1))

maxTime=50
Y_SAVE=np.zeros(shape=(maxTime+1,1))
T_SAVE=np.zeros(shape=(maxTime+1,1))
U_SAVE=np.zeros(shape=(maxTime+1,1))
Y=np.zeros(shape=(maxTime+1,1))


X=range(1,maxTime,1)
for i in (X):
    time=(i-1)*h    # Current time
    print(time)
    
    # Calculate error
    if (i==1):
        YHAT=Yk0
        err=0
    else:
        err=0       # Replace this "y_plant-YHAT(1)"
    predErr=YHAT[1:p+1]-bigR
    grad=matrix(np.dot(np.dot(np.transpose(big_Su),GammaY),predErr))
    grad1=np.dot(np.dot(np.transpose(big_Su),GammaY),predErr)
    cRHS=matrix(np.vstack((np.matlib.repmat(duMax, m, 1),np.matlib.repmat(-duMin, m, 1))),tc='d')
    cRHS=matrix(np.vstack((cRHS,np.matlib.repmat((uMax-uPrev), m, 1))),tc='d')
    cRHS=matrix(np.vstack((cRHS,np.matlib.repmat(-(uMin-uPrev), m, 1))),tc='d')
    big_du=solvers.qp(Hess,grad,C_LHS,cRHS)
    du_sol=big_du['x']
    du_opt=du_sol[0]
    uk=uPrev+du_opt
    U_SAVE[i]=uk
    uPrev=uk
    YHAT=np.add(np.vstack((YHAT[1:],YHAT[-1])),(((np.dot(S,du_opt))).reshape(n,1)))
    yk=YHAT[1]
    Y_SAVE[i]=yk
    Y[i+1]=yk
    T_SAVE[i]=i*h
plt.plot(T_SAVE[1:maxTime],Y_SAVE[1:maxTime])
plt.xlabel('Time (Sec)')
plt.ylabel('Output m^3/s')

#plt.step(T_SAVE[1:maxTime],U_SAVE[1:maxTime])



plt.show()





