#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:52:45 2017

@author: quien
"""

import glob;

import numpy as np;
import numpy.random as rd;

import matplotlib.image as img;
import matplotlib.pyplot as plt;

from gikernel import *;

dirnames = glob.glob("train_set/*/");

X = [];
C = [];
d = None;

e = 0.1;

c = 0;
for dir_ in dirnames:
    for name in glob.glob(dir_+"*.pgm"):
        a = img.imread(name)/255.0;
        d = a.shape;
        a -= 0.5;
        a /= np.linalg.norm(a);
        X.append(a);
        C.append(c);
    c += 1;
X = np.array(X);
C = np.array(C);

r = rd.randint(X.shape[0]/2);
x = 1.0*X[r];
c = 1*C[r];

N = 10;
M = 25;

T = 1.0*X[rd.permutation(np.arange(X.shape[0]))[:M]];
D = np.arange(1,11);
D = np.pi*np.r_[-D[::-1],D]/180.0;
P = np.arange(-1,2);
B = generate_bank(T,D,P);
S = np.linspace(-1+e,1+e,2*N+1);

phi = [];
for t in range(B.shape[0]):
    y = np.dot(B[t],x.reshape(x.size));
    aux = [];
    for n in range(S.size):
        aux.append(1.0*(y <= S[n]));
    phi.append(np.array(aux));
phi = np.array(phi);

f,axarr = plt.subplots(1,2);
axarr[0].imshow(x,cmap='gray');
axarr[0].set_xticklabels([]);
axarr[0].set_yticklabels([]);
axarr[0].grid(False)
axarr[1].imshow(np.mean(phi,axis=0),cmap='gray');
axarr[1].set_xticklabels([]);
axarr[1].set_yticklabels([]);
axarr[1].grid(False)

f,axarr = plt.subplots(min(10,M),3);
for i in range(min(10,M)):
    axarr[i,0].imshow(T[i],cmap='gray');
    axarr[i,0].set_xticklabels([]);
    axarr[i,0].set_yticklabels([]);
    axarr[i,0].grid(False)
    aux = np.sum(phi[i],axis=0);
    axarr[i,1].bar(np.arange(aux.size),aux);
    axarr[i,1].set_xticklabels([]);
    axarr[i,1].set_yticklabels([]);
    aux = np.sum(phi[i],axis=1);
    axarr[i,2].bar(np.arange(aux.size),aux);
    axarr[i,2].set_xticklabels([]);
    axarr[i,2].set_yticklabels([]);
plt.show()

Y = 2*C-1;
P = np.zeros((X.shape[0],B.shape[0]*(2*N+1)));
for i in range(X.shape[0]):
    P[i] = kernel(X[i],B,-1.0+e,1.0+e,N);

L = 1e-4;
W = np.linalg.solve(np.dot(P.T,P/X.shape[0])+L*np.eye(P.shape[1]),np.dot(P.T,Y/X.shape[0]));

