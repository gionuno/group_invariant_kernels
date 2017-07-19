#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:52:45 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

import scipy.io as scio;
import matplotlib.image as img;
import matplotlib.pyplot as plt;

from gikernel import *;

a = scio.loadmat("mnist_all.mat");
d = (28,28);

D = 10;
I = 50;
E_train = np.zeros((D*I,28*28));
T_train = 1e-8*np.ones((D*I,10));

E_test = np.zeros((2*D*I,28*28));
T_test = 1e-8*np.ones((2*D*I,10));
for i in range(I):
    for b in range(D):
        E_train[D*i+b] =  a['train'+str(b)][i,:]/255.0;
        E_train[D*i+b] -= np.mean(E_train[D*i+b]);
        E_train[D*i+b] /= np.linalg.norm(E_train[D*i+b]);
        T_train[D*i+b][b] = 1.0;
        T_train[D*i+b] /= np.sum(T_train[D*i+b]);

for i in range(2*I):
    for b in range(D):
        E_test[D*i+b]    = a['test'+str(b)][i,:]/255.0;
        E_test[D*i+b] -= np.mean(E_test[D*i+b]);
        E_test[D*i+b] /= np.linalg.norm(E_test[D*i+b]);
        T_test[D*i+b][b] = 1.0;
        T_test[D*i+b] /= np.sum(T_test[D*i+b]);

r = rd.randint(E_test.shape[0]);
x = 1.0*E_test[r];
c = 1.0*T_test[r];

N = 12;
M = 30;

T = 1.0*E_train[rd.permutation(np.arange(E_train.shape[0]))[:M]];
D = np.arange(1,21,2);
D = np.pi*np.r_[-D[::-1],D]/180.0;
P = np.arange(-1,2);
B = generate_bank(T,d,D,P,1+e);

S = np.linspace(-1-2*e,1+2*e,2*N+1);
phi  = [];
for t in range(B.shape[0]):
    y = np.dot(B[t],x.reshape(x.size));
    aux = [];
    for n in range(S.size):
        aux.append(1.0*(y <= S[n]));
    phi.append(np.array(aux));
phi  = np.array(phi);

f,axarr = plt.subplots(1,2);
axarr[0].imshow(x.reshape(d),cmap='gray');
axarr[0].set_xticklabels([]);
axarr[0].set_yticklabels([]);
axarr[0].grid(False)
axarr[1].imshow(np.mean(phi,axis=0).T,cmap='gray');
axarr[1].set_xticklabels([]);
axarr[1].set_yticklabels([]);
axarr[1].grid(False)

f,axarr = plt.subplots(M,2);
for i in range(M):
    axarr[i,0].imshow(B[i][0].reshape(d),cmap='gray');
    axarr[i,0].set_xticklabels([]);
    axarr[i,0].set_yticklabels([]);
    axarr[i,0].grid(False)
    aux = np.mean(phi[i],axis=1)/(D.size+P.size**2);
    axarr[i,1].bar(S,aux);
    axarr[i,1].set_xticklabels([]);
    axarr[i,1].set_yticklabels([]);
    axarr[i,1].grid(False)
plt.show()

f,axarr = plt.subplots(3,10)
for k in range(30):
    axarr[k/10,k%10].imshow(B[0][k].reshape(d),cmap='gray');
    axarr[k/10,k%10].set_xticklabels([]);
    axarr[k/10,k%10].set_yticklabels([]);
    axarr[k/10,k%10].grid(False)
plt.show();