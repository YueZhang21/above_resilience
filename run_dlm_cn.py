#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 09:19:38 2022

@author: yuezhang
"""
import numpy as np
import statsmodels.api as sm
from itertools import product
from datetime import datetime
import os
import pandas as pd
import numpy.matlib
from rescale_anomaly import rescale_anomaly
import scipy.linalg
from scipy.stats import t as tdstr
from scipy.stats import norm
from scipy.interpolate import interp1d
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from dlm_functions import forwardFilteringM, Model

print(datetime.now())
#%% Determine the name of the Bgrid
arrayid = int(os.environ['SLURM_ARRAY_TASK_ID'])
gridlist = list(pd.read_csv('Bgridlist.csv',header=None)[0])
gname3 = gridlist[arrayid]
gname2 = gname3[:2]+gname3[3:6]+gname3[7:]

#%% Generate  combinations
vlist1 = ['Tmax','Tmin','Tave','DD_0','DD5','NFFD','Eref',[]]
vlist2 = ['CMD','RH','VPD','PPT','PAS',[]]

vlist = list(product(vlist1,vlist2))
comblist0 = [];comblist1 = [];comblist2 = []

for ii,comb in enumerate(vlist):
    nne = 0
    comb_ne = [] # non-empty variable
    for vid,vc in enumerate(comb):
        if len(vc)!=0:
            nne = nne + 1 # the number of non-empty variable
            comb_ne.append(vc)
    if nne == 2:
        comblist2.append(comb_ne)
    elif nne == 1:
        comblist1.append(comb_ne)
    elif nne == 0:
        comblist0.append(comb_ne)
    
comblist = comblist0 + comblist1 + comblist2 

#%% 
iddir = '/fs/ess/PAS2094/ABoVE3/Multilinear_t/'
freq_all = np.zeros(len(comblist))
for gid in np.arange(len(gridlist)):
    g3 = gridlist[gid]
    g2 = g3[:2]+g3[3:6]+g3[7:]
    freq = np.load(iddir+g2+'/'+g2+'_frequency005'+'.npy')
    freq_all = np.column_stack((freq_all,freq))
  
freq_all = np.sum(freq_all, axis = 1)
id_cn = np.argmax(freq_all)
print(comblist[id_cn])


#%% Prepare X and Y for dlm
# Load climate variables, non-data value is nan
evidir = '/fs/ess/PAS2094/ABoVE3/EVI_NPY_prcs/EVI_NPY_prcs/'
clmdir = '/fs/ess/PAS2094/ABoVE3/ClimateNAEVI/'
Tmax = np.load(clmdir+gname2+'/'+'Tmax'+'/'+gname2+'_EVI_'+'Tmax'+'.npy')
Tmin = np.load(clmdir+gname2+'/'+'Tmin'+'/'+gname2+'_EVI_'+'Tmin'+'.npy')
Tave = np.load(clmdir+gname2+'/'+'Tave'+'/'+gname2+'_EVI_'+'Tave'+'.npy')
PPT = np.load(clmdir+gname2+'/'+'PPT'+'/'+gname2+'_EVI_'+'PPT'+'.npy')
DD_0 = np.load(clmdir+gname2+'/'+'DD_0'+'/'+gname2+'_EVI_'+'DD_0'+'.npy')
DD5 = np.load(clmdir+gname2+'/'+'DD5'+'/'+gname2+'_EVI_'+'DD5'+'.npy')
NFFD = np.load(clmdir+gname2+'/'+'NFFD'+'/'+gname2+'_EVI_'+'NFFD'+'.npy')
PAS = np.load(clmdir+gname2+'/'+'PAS'+'/'+gname2+'_EVI_'+'PAS'+'.npy')
Eref = np.load(clmdir+gname2+'/'+'Eref'+'/'+gname2+'_EVI_'+'Eref'+'.npy')
CMD = np.load(clmdir+gname2+'/'+'CMD'+'/'+gname2+'_EVI_'+'CMD'+'.npy')
RH = np.load(clmdir+gname2+'/'+'RH'+'/'+gname2+'_EVI_'+'RH'+'.npy')
VPD = np.load(clmdir+gname2+'/'+'VPD'+'/'+gname2+'_EVI_'+'VPD'+'.npy')

# for each 16-day period, remove seasonality by subtracting 16-day mean
# and then rescale by dividing standard devatiation of the time series

Tmax = rescale_anomaly(Tmax); Tmin = rescale_anomaly(Tmin)
Tave = rescale_anomaly(Tave); PPT = rescale_anomaly(PPT)
DD_0 = rescale_anomaly(DD_0); DD5 = rescale_anomaly(DD5)
NFFD = rescale_anomaly(NFFD); PAS = rescale_anomaly(PAS)
Eref = rescale_anomaly(Eref); CMD = rescale_anomaly(CMD)
RH = rescale_anomaly(RH); VPD = rescale_anomaly(VPD)

# load EVI
EVI = np.load(evidir+'EVI_prcs'+'_'+gname2+'.npy')
EVI = EVI.astype(float)
EVI[EVI==-3000] = np.nan
EVI = EVI/10000

# load idx_10 to filter out nan pixels in calculation
idx_10 = np.load(iddir+gname2+'/'+gname2+'_idx_10'+'.npy')
idx_bst = idx_10[:,:,0]

#%% Loop over all pixels
# Tmax [50,50,20*23], EVI[600,600,20*23]
# every climate pixel corresponds to 12x12 EVI pixels
npix1 = 600
ncevi = 12 
npix2 = 60
outdir = '/fs/ess/PAS2094/ABoVE3/DLM_common/'
# the maximum number of climate variables is 2
# and the maximum first dimenson of sm is 9 
# (0 local mean; 1 local mean; 2 autocorrelation; 
#  3 climate1; 4 climate2; 
#  5 seasonal; 6 seasonal; 7 seasonal; 8 seasonal)
listc = comblist[id_cn]
for rr in np.arange(npix1//npix2):
    for cc in np.arange(npix1//npix2):
        sC = np.zeros((npix2,npix2,9,EVI.shape[2]),dtype=np.float32) + np.nan
        sm = np.zeros((npix2,npix2,9,EVI.shape[2]),dtype=np.float32) + np.nan
        snu = np.zeros((npix2,npix2,EVI.shape[2]),dtype=np.float32) + np.nan
        slik = np.zeros((npix2,npix2,EVI.shape[2]),dtype=np.float32) + np.nan
        for rpix in np.arange(npix2):
            for cpix in np.arange(npix2):
                if np.isnan(idx_bst[rr*npix2+rpix,cc*npix2+cpix]) == 0:
                    pevi = EVI[rr*npix2+rpix,cc*npix2+cpix,:] # time series of evi of this pixel
                    nonnan_idx = np.argwhere(np.isfinite(pevi))[0]
                    N = pevi[nonnan_idx[0]:]
                    Y = N[1:]-np.nanmean(N)
                    X = N[:-1]-np.nanmean(N)
                    
                    if len(listc) == 0:
                        X = np.column_stack([X]) 
                    else:
                        for kk, varc in enumerate(listc):
                            BCLI = eval(varc)
                            zi = BCLI[(rr*npix2+rpix)//ncevi,(cc*npix2+cpix)//ncevi,:]
                            zi = zi[(zi.shape[0] - N.shape[0]):]
                            zi = zi[1:]
                            X = np.column_stack([X,zi]) 
                            
                    rseas = [1,2]
                    delta = 0.98
                    M = Model(Y,X,rseas,delta)
                    FF = forwardFilteringM(M)
                    psC = np.zeros((FF.get('sC').shape[0],FF.get('sC').shape[2]))
                    for jj in np.arange(FF.get('sC').shape[2]):
                        psC[:,jj] = np.diag(FF.get('sC')[:,:,jj])
                    
                    psm = FF.get('sm')
                    psnu = FF.get('snu')
                    pslik = FF.get('slik')
                    sC[rpix, cpix, 0:psm.shape[0], nonnan_idx[0]:] = psC
                    sm[rpix, cpix, 0:psm.shape[0], nonnan_idx[0]:] = psm
                    snu[rpix, cpix, nonnan_idx[0]:] = psnu
                    slik[rpix, cpix, nonnan_idx[0]:] = pslik
                    
        sm_name = outdir+gname2+'/'+gname2+'_r'+str(rr).zfill(2)+'c'+str(cc).zfill(2)+'_sm_cn.npy'
        sC_name = outdir+gname2+'/'+gname2+'_r'+str(rr).zfill(2)+'c'+str(cc).zfill(2)+'_sC_cn.npy'
        snu_name = outdir+gname2+'/'+gname2+'_r'+str(rr).zfill(2)+'c'+str(cc).zfill(2)+'_snu_cn.npy'
        slik_name = outdir+gname2+'/'+gname2+'_r'+str(rr).zfill(2)+'c'+str(cc).zfill(2)+'_slik_cn.npy'
        np.save(sm_name,sm) 
        np.save(sC_name, sC)
        np.save(snu_name,snu)
        np.save(slik_name, slik)

print(datetime.now()) 
        
            

