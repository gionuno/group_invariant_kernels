#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 16:15:40 2017

@author: quien
"""

import numpy as np;
import numpy.random as rd;

def rotate(x,a):
    y = np.zeros(x.shape);
    ca = np.cos(a);
    sa = np.sin(a);
    mx = x.shape[0]/2;
    my = x.shape[1]/2;
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            u = (x.shape[0]+ca*(i-mx)-sa*(j-my)+mx);
            v = (x.shape[1]+sa*(i-mx)+ca*(j-my)+my);   
            
            sl = int(np.floor(u));
            tl = int(np.floor(v));
            
            a = u-sl;
            b = v-tl;
            
            sh = (sl + 1 + x.shape[0])%x.shape[0];
            th = (tl + 1 + x.shape[1])%x.shape[1];
            sl = (sl + x.shape[0])%x.shape[0];
            tl = (tl + x.shape[1])%x.shape[1];
            
            y[i,j] = a*b*x[sh,th]+(1-a)*b*x[sl,th]+a*(1-b)*x[sh,tl]+(1-a)*(1-b)*x[sl,tl];
            
    return y;

def translate(x,t):
    y = np.zeros(x.shape);
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            ii = (x.shape[0]+i+t[0])%x.shape[0];
            jj = (x.shape[1]+j+t[1])%x.shape[1];            
            y[i,j] = x[ii,jj];
    return y;

def generate_bank(T,d,D,P,s):
    B = [];
    for tt in range(T.shape[0]):
        tta = 1.*T[tt];
        t = (0.99*tta + 0.01*rd.randn(tta.size));
        while np.linalg.norm(t) > s:            
            t = (0.99*tta + 0.01*rd.randn(tta.size));
        b = [t];
        for u in range(len(P)):
            for v in range(len(P)):
                aux = translate(t.reshape(d),[P[u],P[v]]);
                aux /= np.linalg.norm(aux);
                b.append(aux.reshape(t.size));
        for e in D:
            aux = rotate(t.reshape(d),e);
            aux /= np.linalg.norm(aux);
            b.append(aux.reshape(t.size));
        B.append(np.array(b));
    return np.array(B);

def kernel(x,B,l,h,N):
    sL = np.linspace(l,h,2*N+1);
    phi_x = np.zeros(B.shape[0]*(2*N+1));
    for j in range(B.shape[0]):
        y = np.dot(B[j],x.reshape(x.size,1));
        for k in range(2*N+1):
            phi_x[(2*N+1)*j+k] = np.sum(y<=sL[k]);
    return phi_x;