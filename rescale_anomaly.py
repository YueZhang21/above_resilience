#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 14:54:20 2021

@author: yuezhang
"""
import numpy as np

# cli(50,50,20*23) is the climate matrix for the entire B grid
# evi(600,600,20*23) is the evi matrix for the entire B grid
def rescale_anomaly(cli):
    c = np.zeros((cli.shape[0],cli.shape[1],cli.shape[2]))
    for row in np.arange(cli.shape[0]):
        for col in np.arange(cli.shape[1]):
            tcli = cli[row,col,:]
            tcli = tcli.reshape((20,23))
            c[row,col,:] = ((tcli - np.nanmean(tcli,axis=0))).reshape((20*23))
 
    z = c/np.nanstd(c,axis=2)[:,:,None]                 
            
    return z  

     
        
        
    
