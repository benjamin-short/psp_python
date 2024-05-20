#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:43:33 2022

@author: besh2109
"""

import os
import pyspedas.psp as psp
import pyspedas as pys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import pytplot as pyt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from datetime import date
from matplotlib import ticker
from statistics import median
from mpl_toolkits.axes_grid1 import make_axes_locatable
import gc
import math
import random
import cartopy.crs as ccrs
import pyspedas.stereo as ste
from matplotlib import gridspec
import time
import pickle as pkl
from streamtracer import StreamTracer, VectorGrid

from findpeaks import findpeaks
from scipy.interpolate import UnivariateSpline
from scipy.optimize import curve_fit

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst
# from .davidtensor import david_rot_mat
# from .davidtensor import david_anis
# from .davidtensor import steven_anis

from .davidtensor import david_rot_mat
from .davidtensor import david_anis
from .davidtensor import steven_anis

from numpy.linalg import inv

import matplotlib.patches as mpatch

import re
import pandas as pd

import glob

from lmfit.models import SkewedGaussianModel

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

Rs_to_AU = 0.00465047 # 1 Rs = 0.00465047 AU

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

sweap_id = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']

jsoc_email = os.environ['JSOC_EMAIL']

def calculate_quartiles(hist, bins):
    # Calculate cumulative distribution function (CDF)
    cdf = np.nancumsum(hist)
    # print(1)
    # Normalize CDF to range [0, 1]
    cdf_normalized = cdf / cdf[-1]
    # print(2)
    # Calculate quartiles
    q0 = np.interp(0, cdf_normalized, bins)
    q1 = np.interp(0.25, cdf_normalized, bins)
    q2 = np.interp(0.5, cdf_normalized, bins)
    q3 = np.interp(0.75, cdf_normalized, bins)
    q4 = np.interp(0.95, cdf_normalized, bins)
    # print(3)

    return q0, q1, q2, q3, q4

def sliding_median(array, window_size):
    # Pad the array to handle edge cases
    padded_array = np.pad(array, (window_size//2, window_size//2), mode='edge')

    # Create a 2D array of shape (len(array), window_size)
    rolling_window = np.lib.stride_tricks.sliding_window_view(padded_array, window_shape=(window_size,))

    # Compute the median along the window axis
    medians = np.nanmedian(rolling_window, axis=1)
    
    # Determine if the number of NaNs exceeds half the window size
    num_nans = np.sum(np.isnan(rolling_window), axis=1)
    too_many_nans = num_nans > 3*window_size / 4
    
    # Set medians to np.nan where there are too many NaNs
    medians[too_many_nans] = np.nan
    
    return medians

def group_zeros(array):
    
    # Example array of 1s and 0s
    # array = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    
    # Initialize variables to track group boundaries
    group_started = False
    start_index = None
    end_index = None
    
    # List to store the start and end indices of groups of 0s
    groups = []
    
    # Iterate through the array
    for i, value in enumerate(array):
        if value == 0:
            if not group_started:
                # Start of a new group
                start_index = i
                group_started = True
            end_index = i
        elif group_started:
            # End of the current group
            groups.append((start_index, end_index))
            group_started = False
    
    # If the last group continued to the end of the array, add it
    if group_started:
        groups.append((start_index, end_index))
    
    # Print the start and end indices of groups of 0s
    # for start, end in groups:
    #     print(f"Group of 0s: Start Index {start}, End Index {end}")
    
    return groups

def brazil_text():
    
    brazil_list = ['            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░▒▒░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░      ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒            ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░                    ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ▒▒██████▒▒      ░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ██████████████        ░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░      ▒▒██████████████████▓▓░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ▒▒██████████████████████▓▓░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░          ██████████████████████████▓▓░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒              ░░░░░░░░    ▒▒██████████████░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒                ██▓▓██████████▒▒░░░░░░██████████░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒                    ████████████████████▓▓░░░░▓▓████  ░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒░░                  ████████▓▓██████████████▓▓░░  ▓▓░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒                ████████████████████████████▒▒  ░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒            ▓▓████████████████████████████▒▒░░░░░░░░░░░░  ▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒            ████████████████████████▓▓▓▓░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒        ▒▒████████████████▓▓██████░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░      ▓▓████████████████▓▓██░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░    ░░██████████████░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░  ▒▒  ░░░░░░░░  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░░░░░░░  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒░░░░░░▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ',\
                   '            ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒          ']
                                                                                                          
    return brazil_list
                                                                                                          
def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

def delete_contour(CS,r):
    for level in CS.collections:
        
        # breakpoint()
        
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            # go in reversed order due to deletions!
    
            # include test for "smallness" of your choice here:
            # I'm using a simple estimation for the diameter based on the
            #    x and y diameter...
            verts = path.vertices # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
            
            
            # help(path)
            # breakpoint()
    
            if diameter>5: # threshold to be refined for your actual dimensions!
                del(level.get_paths()[kp])  # no remove() for Path objects:(
                             
def t_r_plot(enc='all',enc_radius=67,atype='v'): 
    
    #SOON TO COME: 'estrahl'
    
    ptypes = ['ptemp','panis','ptpar','ptper','pdens']
    etypes = ['etemp','eanis','edrift']
    qtntypes = ['qtndens','qtntemp']
    vtypes = ['v']
    
    
    #----------------------Organizing CSVs for in Data-------------------------#
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    # csv_names = os.listdir(csv_path)
    # csv_names.remove('.DS_Store')
    
    csv_names = glob.glob(csv_path+'*_raw.csv')
    
    enc_tag = []
    for i in csv_names:
        
        a = re.findall(r'\d+',i[25:])
        enc_tag.append(int(a[0]))
    
    csv_names = sort_list(csv_names,enc_tag)
    enc_tag = sort_list(enc_tag,enc_tag)
    
    # filelist = []
    # for j in csv_names:
    #     j = csv_path+j
    #     filelist.append(j)
    
    filelist = csv_names
    
    # breakpoint()
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,19))
        
    if enc == 'no 13':
        enc = list(range(1,13))
        
    if enc == 'no 1':
        enc = list(range(2,14)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        
    if type(enc) is int:
        enc = [enc]
        
    # breakpoint()
    
    enc_lst_str = enc[0]

    filelist_new = [filelist[i-enc_lst_str] for i in enc]
    
    #--------------------------Reading in Quiescent Data---------------------------#
    
    df_list = [pd.read_csv(file) for file in filelist_new]
    
    df = pd.concat(df_list)
    
    q_starts = df['start dates'].to_numpy()
    q_ends = df['end dates'].to_numpy()
    q_duras = df['duration'].to_numpy()
    
    q_start_flt = pys.time_float(q_starts)
    q_end_flt = pys.time_float(q_ends)
    
    #--------------------------Start importing PSP Data-----------------------------#
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    shape_arr = []
    datafull = np.array([])
    Rfull = np.array([])
    Rtfull = np.array([])
    Rt_AU = np.array([])
    timefull = np.array([])
    
    for i in enc:
        
        # print(i)
        
        enc_ind = (i-1)
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        # if i == 7:
        #     enc_str ='2021-01-13/04:48:00'
        #     enc_end ='2021-01-18/00:00:00'
        #     enc_ends[0] = pys.time_float('2021-01-13/04:48:00')
        #     enc_ends[1] = pys.time_float('2021-01-18/00:00:00')
            
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        # if i in [1]: #choose encounters for which to use SPC instead of SPAN. In this case, the first orbit family: Enc 1-3

        #     psp.spc(trange=[t0,tf], level='L3')
        #     vel_data = pyt.get_data('vp_fit_RTN')
        
        if atype in ptypes:
            
            
            # if i in [1]: #choose encounters for which to use SPC instead of SPAN. In this case, the first orbit family: Enc 1-3

            #     psp.spc(trange=[t0,tf], level='L3')
            #     vel_data = pyt.get_data('vp_fit_RTN')
            
            psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
            # vel_data = pyt.get_data('VEL_RTN_SUN')
            
            if atype in ['panis','ptpar','ptper']:
                tens_data = pyt.get_data('psp_spi_T_TENSOR_INST')
                
                tens_time_arr = tens_data[0]
                tens_data_arr = tens_data[1]
                
                Txx = tens_data_arr[:,0]
                Tyy = tens_data_arr[:,1]
                Tzz = tens_data_arr[:,2]
                Txy = tens_data_arr[:,3]
                Txz = tens_data_arr[:,4]
                Tyz = tens_data_arr[:,5]
                
                tens = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                
                mag_data = pyt.get_data('psp_spi_MAGF_INST')
                
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]
                
                Bx = mag_data_arr[:,0]
                By = mag_data_arr[:,1]
                Bz = mag_data_arr[:,2]
                
                t_per = []
                t_par = []
                anis_me = []
                anis_st = []
                for m in range(len(Bx)):
                    Bx_tmp = Bx[m]
                    By_tmp = By[m]
                    Bz_tmp = Bz[m]
                    tens_tmp = tens[:,:,m]
                    rot_mat_tmp = david_rot_mat([Bx_tmp,By_tmp,Bz_tmp])
                    rot_inv_tmp = inv(rot_mat_tmp)
                    
                    tmp_tens = np.matmul(rot_mat_tmp,tens_tmp) #first calculation of T' = R T R^-1
                    tens_fin = np.matmul(tmp_tens,rot_inv_tmp)
                    
                    # breakpoint()
                    
                    T_perp_tmp = np.array(tens_fin[1,1]+tens_fin[2,2])/2
                    T_par_tmp = np.array(tens_fin[0,0])
                    
                    
                    anis_me_tmp = T_perp_tmp/T_par_tmp 
                    # anis_da = david_anis([Bx_tmp,By_tmp,Bz_tmp],tens_tmp) #steven and I agree on the number, David does not.
                    anis_st_tmp = steven_anis([Bx_tmp,By_tmp,Bz_tmp],tens_tmp)
                    
                    
                    t_per.append(T_perp_tmp)
                    t_par.append(T_par_tmp)
                    anis_me.append(anis_me_tmp)
                    anis_st.append(anis_st_tmp)
                    
                    
                t_per = np.array(t_per)
                t_par = np.array(t_par)
                anis_me = np.array(anis_me)
                anis_st = np.array(anis_st)
                
                if atype == 'ptper':
                    time_arr = tens_time_arr
                    data_arr = t_per
                elif atype == 'ptpar':
                    time_arr = tens_time_arr
                    data_arr = t_par
                else:
                    time_arr = tens_time_arr
                    data_arr = anis_me
                # data_arr = anis_st
                
            elif atype=='ptemp':
                # breakpoint()
                temp_data = pyt.get_data('psp_spi_TEMP')
                
                time_arr = temp_data[0]
                data_arr = temp_data[1]
                
            else:
                dens_data = pyt.get_data('psp_spi_DENS')
                
                time_arr = dens_data[0]
                data_arr = dens_data[1]
            
        elif atype in etypes:
            # psp.spe(trange=[t0,tf],level='L3',datatype='spi_sf00')
            
            temp_path = '/Users/besh2109/Desktop/psp_electrons/'
            temp_file = 'coret_e1toe8.tplot'
            
            drift_path = '/Users/besh2109/Desktop/psp_electrons/'
            drift_file = 'coredrift_e1toe8.tplot'
            
            pyt.tplot_restore(temp_path+temp_file)
            pyt.tplot_restore(drift_path+drift_file)
            
            tper_data = pyt.get_data('coretperp')
            tpar_data = pyt.get_data('coretpar')
            drift_data = pyt.get_data('coredrift')
     
            tper_time_arr = tper_data[0]
            tper_data_arr = tper_data[1]
            
            tpar_time_arr = tpar_data[0]
            tpar_data_arr = tpar_data[1]
            
            drift_time_arr = drift_data[0]
            drift_data_arr = drift_data[1]
            
            if atype == 'eanis':
                time_arr = tper_time_arr
                data_arr = tper_data_arr/tpar_data_arr
            elif atype == 'edrift':
                time_arr = drift_time_arr
                data_arr = drift_data_arr
            else:
                time_arr = tper_time_arr
                data_arr = 2*tper_data_arr+tpar_data_arr
                
        
        elif atype in qtntypes:
            psp.fields(trange=[t0,tf],datatype='sqtn_rfs_V1V2',level='l3',last_version=True)
            
            dens_data = pyt.get_data('electron_density')
            temp_data = pyt.get_data('electron_core_temperature')
            
            dens_time_arr = dens_data[0]
            dens_data_arr = dens_data[1]
            
            temp_time_arr = temp_data[0]
            temp_data_arr = temp_data[1]
            
            if atype == 'qtndens':
                time_arr = dens_time_arr
                data_arr = dens_data_arr
            else:
                time_arr = temp_time_arr
                data_arr = temp_data_arr
        
        else:
            
            if i in [1]:

                psp.spc(trange=[t0,tf], level='L3',username=sweap_id,password=sweap_pass,last_version=True)
                vel_data = pyt.get_data('psp_spc_vp_fit_RTN')
                
                vel_time_arr = vel_data[0]
                vel_data_arr = vel_data[1]
                
                v_r_data = vel_data_arr[:,0]
                v_t_data = vel_data_arr[:,1]
                v_n_data = vel_data_arr[:,2]
                
                V_mag = np.sqrt(v_r_data**2+v_t_data**2+v_n_data**2)
                
                spc_v_clean = sliding_median(V_mag,275) #about one minute long windows at max cadence
    
                interpolating_func = interp1d(vel_time_arr, spc_v_clean, kind='linear', fill_value='extrapolate')
            
                # Create a new array of time values with fixed cadence
                spc_time_down = np.arange(vel_time_arr[0], vel_time_arr[-1], 6)
            
                # Interpolate arr_two to the new time values
                spc_v_down = interpolating_func(spc_time_down)
                
                time_arr = spc_time_down
                data_arr = spc_v_down
                
            else:
                
                psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
                vel_data_spi = pyt.get_data('psp_spi_VEL_RTN_SUN')
                vel_time_arr_spi = vel_data_spi[0]
                vel_data_arr_spi = vel_data_spi[1]
                
                # spi_vr = vel_data_arr_spi[:,0]
                
                spi_r_data = vel_data_arr_spi[:,0]
                spi_t_data = vel_data_arr_spi[:,1]
                spi_n_data = vel_data_arr_spi[:,2]
                
                V_mag_spi = np.sqrt(spi_r_data**2+spi_t_data**2+spi_n_data**2)
                
                psp.spc(trange=[t0,tf], level='L3',username=sweap_id,password=sweap_pass)
                vel_data_spc = pyt.get_data('psp_spc_vp_fit_RTN')
                vel_time_arr_spc = vel_data_spc[0]
                vel_data_arr_spc = vel_data_spc[1]
                
                # spc_vr = vel_data_arr_spc[:,0]
                
                spc_r_data = vel_data_arr_spc[:,0]
                spc_t_data = vel_data_arr_spc[:,1]
                spc_n_data = vel_data_arr_spc[:,2]
                
                V_mag_spc = np.sqrt(spc_r_data**2+spc_t_data**2+spc_n_data**2)
            
                spc_v_clean = sliding_median(V_mag_spc,275) #about one minute long windows at max cadence
    
                interpolating_func = interp1d(vel_time_arr_spc, spc_v_clean, kind='linear', fill_value='extrapolate')
            
                # Create a new array of time values with fixed cadence
                spc_time_down = np.arange(vel_time_arr_spc[0], vel_time_arr_spc[-1], 6)
            
                # Interpolate arr_two to the new time values
                spc_v_down = interpolating_func(spc_time_down)
        
                #--------------------------------------------------------------------#
                
                span_check_savename = 'SPAN_SPC_QTN_flags'+'_enc_'+str(i)+'.cdf'
                span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
                
                pyt.tplot_restore(span_check_savepath+span_check_savename)
            
                SPAN_QTN_flag = pyt.get_data('SPAN_qual_flag')
                SPAN_flag_time = SPAN_QTN_flag[0]
                SPAN_flag = SPAN_QTN_flag[1]
                
                SPC_QTN_flag = pyt.get_data('SPC_qual_flag')
                SPC_flag_time = SPC_QTN_flag[0]
                SPC_flag = SPC_QTN_flag[1]
                
                flag_zero_group = group_zeros(SPAN_flag)
                
                bad_span_times = []
                vel_time_construct = np.array([],dtype=float)
                vel_mag_data_construct = np.array([],dtype=float)
                
                # breakpoint()
                bins = np.array([])
                for i in range(len(flag_zero_group)):
                    
                    t0s = SPAN_flag_time[flag_zero_group[i][0]]
                    tfs = SPAN_flag_time[flag_zero_group[i][1]]
                    # edge_times.append((t0,tf))
                    bins = np.append(bins,np.array([t0s,tfs]))
                    # bad_span_times.append((t0s,tfs))
                good_bad = 0 #loop through times when quality flag is good and when its bad. start with bad.
                # bad = 0
                for i in range(1,len(bins)):
                    

                    if i==1:
                        t0i = 10
                        tfi = bins[i]
                    elif i==len(bins)-1:
                        # tf0 = bins[i]
                        tfi = np.inf  
                    else:
                        t0i = bins[i-1]
                        tfi = bins[i]
                        
                    if good_bad == 0:
                        
                        spc_where = np.where((spc_time_down>t0i)&(spc_time_down<tfi))
                        spc_where = spc_where[0]
                        
                        vel_time_construct = np.append(vel_time_construct,spc_time_down[spc_where])
                        vel_mag_data_construct = np.append(vel_mag_data_construct,spc_v_down[spc_where])
                        
                        good_bad = 1 #alternate between good and bad times. Should start with bad.
                        # breakpoint()
                    elif good_bad ==1:
                        
                        spi_where = np.where((vel_time_arr_spi>t0i)&(vel_time_arr_spi<tfi))
                        spi_where = spi_where[0]
                        
                        vel_time_construct = np.append(vel_time_construct,vel_time_arr_spi[spi_where])
                        vel_mag_data_construct = np.append(vel_mag_data_construct,V_mag_spi[spi_where])
                        
                        good_bad = 0
                        
                
                time_arr = vel_time_construct
                data_arr = vel_mag_data_construct
            
            # # breakpoint()
            # if i in [1]: #choose encounters for which to use SPC instead of SPAN. In this case, the first orbit family: Enc 1-3

            #     psp.spc(trange=[t0,tf], level='L3',last_version=True,username=sweap_id,password=sweap_pass)
            #     vel_data = pyt.get_data('psp_spc_vp_fit_RTN')
            
            # else:
            #     psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',last_version=True,username=sweap_id,password=sweap_pass)
            #     vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
            
            
            # vel_time_arr = vel_data[0]
            # vel_data_arr = vel_data[1]
            
            # vr = vel_data_arr[:,0]
            # vt = vel_data_arr[:,1]
            # vn = vel_data_arr[:,2]
            
            # V = np.sqrt(vr**2+vt**2+vn**2)
            
            # time_arr = vel_time_arr
            # data_arr = V
            
        
        # vr = vel_data_arr[:,0]
        # vt = vel_data_arr[:,1]
        # vn = vel_data_arr[:,2]
        
        # V = np.sqrt(vr**2+vt**2+vn**2)
        
        # T = data_arr
        
        shape_arr.append(time_arr.shape)
        
        psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True,username=fields_id,password=fields_pass)
        pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
        
        
        pos_time_arr = pos_data[0]
        pos_data_arr = pos_data[1]
        
        # pos_time_arr = pos_time_arr[0:len(pos_time_arr)]
        
        x = pos_data_arr[:,0]
        y = pos_data_arr[:,1]
        z = pos_data_arr[:,2]
        
        R = (np.sqrt(x**2+y**2+z**2)-Rs)/Rs #parker spiral source height
        R_true = np.sqrt(x**2+y**2+z**2)/Rs #height from center of sun
        

        R_xp = pos_time_arr
        R_x = time_arr
        R_i = np.interp(R_x,R_xp,R)
        

        Rt_xp = pos_time_arr
        R_x = time_arr
        Rt_i = np.interp(R_x,Rt_xp,R_true)
        
        Rtfull = np.append(Rtfull,Rt_i)
        
        Rt_AU = np.append(Rt_AU,Rt_i*Rs_to_AU)

        datafull = np.append(datafull,data_arr)
        Rfull = np.append(Rfull,R_i)
        timefull = np.append(timefull,time_arr)
        
        
        # if 'dens' in atype:
        #     datafull = datafull*(Rt_AU**2)
        
        # breakpoint()
        
    # print(shape_arr)
    where_ap = np.array([],dtype='int64')
    # R_averages = []
    for k in range(len(q_starts)):
        ap = np.where((timefull>q_start_flt[k])&(timefull<=q_end_flt[k]))
        ap = ap[0]
        where_ap = np.append(where_ap,ap)
        # R_tmp = Rfull[ap]
        # R_avg = np.nanmean(R_tmp)
        # R_averages.append(R_avg)
        
    R_to_hist = np.delete(Rfull,where_ap)
    data_to_hist = np.delete(datafull,where_ap)
    # breakpoint()
    if atype in ptypes:

        if atype == 'panis': #stop laughing
            part_title = 'Proton Temperature Anisotropy'r' ($\frac{T\perp}{T_{||}}$)'
            drange = (0,4)
            
        elif atype == 'ptper':
            part_title = 'Proton Temperature Perpendicular (eV)'
            drange = (0,400)
        elif atype == 'ptpar':
            part_title = 'Proton Temperature Parallel (eV)'
            drange = (0,400)
        elif atype == 'ptemp':
            part_title = 'Proton Temperature (eV)'
            drange = (0,400)
        else:
            part_title = 'Proton Density'
            drange = (0,800)
        
        title_lab = 'Ti'
        
    elif atype in etypes:
        
        if atype == 'eanis':
            part_title = 'Electron Temperature Anisotropy'r' ($\frac{T\perp}{T_{||}}$)'
            drange = (0.5,1.25)
            
        elif atype == 'edrift':
            part_title = 'Electron Core Drift (km/s)'
            drange = (0,100)
        else:
            part_title = 'Electron Temperature (eV)'
            drange = (50,250)
            
        title_lab = 'Te'
    
    
    elif atype in qtntypes:
        if atype == 'qtndens':
            part_title = 'QTN Electron Density'
            drange = (0,800)
        else:
            part_title = 'QTN Electron Core Temperature (eV)'
            drange = (50,250)
    
        title_lab = 'QTN'
    
    else:
        
        part_title = 'Plasma Bulk Velocity Magnitude (km/s)'
        # drange = (50,1000)
        drange = (50,900)
        title_lab = '|Vsw|'
        
        
    if len(enc)>1:
        title = title_lab+' vs R, Enc '+str(enc[0])+' thru '+str(enc[-1])
        enc_lab=''
    else:
        title = title_lab+' vs R, Enc '+str(enc[0])
        enc_lab = ' '+str(enc[0])
    
    breakpoint()
    
    #----------------------------save data vs r-----------------------------#
    
    savepath = '/Users/besh2109/Desktop/Temperature Products/Ions vs R v0/'
    savename = 'psp_swp_T_vs_R.cdf'
    
    tplot_radial_full = Rfull
    
    tplot_span_temp_full = datafull
    
    tplot_radial_q = Rfull[where_ap]
    
    tplot_span_temp_q = datafull[where_ap]

    pyt.store_data("SPAN_temp_full", data={'x':tplot_radial_full, 'y':tplot_span_temp_full})
    pyt.store_data("SPAN_temp_q", data={'x':tplot_radial_q, 'y':tplot_span_temp_q})

    cdf_var_list = ["SPAN_temp_full","SPAN_temp_q"]
    
    pyt.tplot_save(cdf_var_list,savepath+savename) #saves the quality flags to a .cdf file
    
    #--------------------------------------#
    
    
    fig = plt.figure(figsize=(20,9)) 
    axs = fig.add_subplot(111)
    bincount = 25
    
    #--------------histogram for non quiescent regions----------------------#
    
    hist2d,xedge,yedge = np.histogram2d(datafull,Rfull, bins=bincount,range=[drange,[10, enc_radius]])
    # hist2d,xedge,yedge = np.histogram2d(R_to_hist,data_to_hist, bins=40,range=[[10, 50], drange])
    
    hist2d[hist2d<60]=np.nan
    
    histo = np.transpose(hist2d)
    
    xcenters = (xedge[:-1] + xedge[1:]) / 2
    ycenters = (yedge[:-1] + yedge[1:]) / 2
    
    q0_arr = np.array([])
    q1_arr = np.array([])
    q2_arr = np.array([])
    q3_arr = np.array([])
    q4_arr = np.array([])
    for ksj in range(len(histo)):
        
        q0,q1,q2,q3,q4 = calculate_quartiles(histo[ksj,:],xcenters)
        # plt.plot([q1,q2,q3])
        # plt.show()
        
        q0_arr = np.append(q0_arr,q0)
        q1_arr = np.append(q1_arr,q1)
        q2_arr = np.append(q2_arr,q2)
        q3_arr = np.append(q3_arr,q3)
        q4_arr = np.append(q4_arr,q4)
    
    # axs.fill_between(ycenters,q0_arr,q1_arr,color='tab:blue',alpha=0.5,zorder=3)
    o1 = axs.fill_between(ycenters,q1_arr,q3_arr,color='tab:blue',label='Non-Quiescent Velocity Span',alpha=0.85,zorder=3)
    # axs.fill_between(ycenters,q3_arr,q4_arr,color='tab:blue',alpha=0.5,zorder=3)
    # axs.plot(ycenters,q0_arr,color='grey',zorder=4)
    # axs.plot(ycenters,q1_arr,color='black',zorder=4,linewidth=0.75)
    # axs.plot(ycenters,q3_arr,color='black',zorder=4,linewidth=0.75)
    
    o3 = axs.plot(ycenters,q1_arr,color='black',zorder=4,linewidth=1.25, label='1st & 3rd Quartiles')
    o4 = axs.plot(ycenters,q2_arr,color='black',linestyle='dotted',zorder=4,linewidth=2,label='Velocity Median')
    axs.plot(ycenters,q3_arr,color='black',zorder=4,linewidth=1.25)
    
    adiabatic = 1500*ycenters**(-4/3) 
    
    axs.plot(ycenters,adiabatic,color='red',linestyle='dashed',linewidth=2,zorder=5)
    # axs.plot(ycenters,q4_arr,color='grey',zorder=4)
    
    #--------------histogram for quiescent regions----------------------#
    
    hist2d,xedge,yedge = np.histogram2d(datafull[where_ap],Rfull[where_ap], bins=bincount,range=[drange,[10, enc_radius]])
    # hist2d,xedge,yedge = np.histogram2d(R_to_hist,data_to_hist, bins=40,range=[[10, 50], drange])
    
    hist2d[hist2d<60]=np.nan
    
    histo = np.transpose(hist2d)
    
    xcenters = (xedge[:-1] + xedge[1:]) / 2
    ycenters = (yedge[:-1] + yedge[1:]) / 2
    
    q0_arr = np.array([])
    q1_arr = np.array([])
    q2_arr = np.array([])
    q3_arr = np.array([])
    q4_arr = np.array([])
    for ksj in range(len(histo)):
        
        q0,q1,q2,q3,q4 = calculate_quartiles(histo[ksj,:],xcenters)
        # plt.plot([q1,q2,q3])
        # plt.show()
        
        q0_arr = np.append(q0_arr,q0)
        q1_arr = np.append(q1_arr,q1)
        q2_arr = np.append(q2_arr,q2)
        q3_arr = np.append(q3_arr,q3)
        q4_arr = np.append(q4_arr,q4)

    # axs.fill_between(ycenters,q0_arr,q1_arr,color='tab:orange',alpha=0.5,zorder=4)
    o2 = axs.fill_between(ycenters,q1_arr,q3_arr,color='tab:orange',label='Quiescent Velocity Span',alpha=0.85,zorder=4)
    # axs.fill_between(ycenters,q3_arr,q4_arr,color='tab:orange',alpha=0.5,zorder=3)
    # axs.plot(ycenters,q0_arr,color='grey',zorder=5)
    axs.plot(ycenters,q1_arr,color='black',zorder=5,linewidth=1.25)
    axs.plot(ycenters,q2_arr,color='black',linestyle='dotted',zorder=5,linewidth=2)
    axs.plot(ycenters,q3_arr,color='black',zorder=5,linewidth=1.25)
    # axs.plot(ycenters,q4_arr,color='grey',zorder=5)
    
    #--------------finish plotting---------------#

    # X, Y = np.meshgrid(ycenters, xcenters)
    
    # axs.scatter(Rfull,datafull,s=1,color='steelblue',marker=",",label='Full Mission Data',zorder=1,alpha=0.25)
    # axs.scatter(Rfull[where_ap],datafull[where_ap],s=1,color='sandybrown',marker=",",label='Quiescent Regions',zorder=2,alpha=0.25)
    
    # axs.set_title(title,fontsize=36)
    axs.set_title('$V_{sw}$ vs R for Encounters 1 through 18',fontsize=36)
    # axs.set_ylabel(part_title,fontsize=24)
    # axs.set_ylabel('Solar Wind Bulk Velocity (km/s)',fontsize=24)
    axs.set_ylabel('Solar Wind Proton Core Temperature (eV)',fontsize=24)
    axs.set_xlabel('Radial Position of PSP ($R_{\odot}$)',fontsize=24)
    axs.tick_params(axis='both', which='major', labelsize=24)
    # if 'dens' not in atype:
    # axs.set_ylim(drange)
    # axs.set_ylim(125,425)
    axs.set_ylim(0,120)
    # axs.set_xlim(15,55)
    axs.set_xlim(10,65)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,3,2,1]
    lgnd = axs.legend([handles[idx] for idx in order],[labels[idx] for idx in order],fontsize=16,loc='upper right',markerscale=10)
    
    # lgnd = axs.legend(handles=[o1,o2,o3,o4],fontsize=16,loc='upper right',markerscale=10)
    # lgnd.legend_handles[0].set_sizes = [10]
    # lgnd.legend_handles[1].set_sizes = [10]
    # lgnd.legend_handles[0]._legmarker.set_markersize(10)
    
    plt.show()
    
def t_anis_beta(enc='all',enc_radius=40):
    
    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    mp = 1.6726 * 1e-27 #kg
    #kb = 1.380649*10e-23 #boltzmann constant, J/K

    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    if enc == 'no 13':
        enc = list(range(2,13))
        enc.append(14)
        enc.append(15)
        title_mod = 'for All Encounters, no 1 or 13'
        
    if enc == 'no 1':
        enc = list(range(2,15)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    if type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    # filelist_new = [filelist[i-1] for i in enc]
    
    # #--------------------------Reading in Quiescent Data---------------------------#
    
    # df_list = [pd.read_csv(file) for file in filelist_new]
    
    # df = pd.concat(df_list)
    
    # q_starts = df['start dates'].to_numpy()
    # q_ends = df['end dates'].to_numpy()
    # q_duras = df['duration'].to_numpy()
    
    # q_start_flt = np.array(pys.time_float(q_starts))
    # q_end_flt = np.array(pys.time_float(q_ends))
    
    # # long_check = np.where(q_duras > 3600)
    # # q_start_flt = q_start_flt[long_check]
    # # q_end_flt = q_end_flt[long_check]

    
    #--------------------------Start importing PSP Data-----------------------------#
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    anis_ben_full = np.array([])
    q_anis_ben_full = np.array([])
    non_q_anis_ben_full = np.array([])
    
    beta_par_full = np.array([])
    q_beta_par_full = np.array([])
    non_q_beta_par_full = np.array([])
    
    
    for i in enc:
        
        # print(i)
        
        enc_ind = (i-1)
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        # if i == 7:
        #     enc_str ='2021-01-13/04:48:00'
        #     enc_end ='2021-01-18/00:00:00'
        #     enc_ends[0] = pys.time_float('2021-01-13/04:48:00')
        #     enc_ends[1] = pys.time_float('2021-01-18/00:00:00')
            
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        if type(enc_radius) == int or type(enc_radius) == float:
            
            R_where = np.where(R<enc_radius)
            R_where = R_where[0]
            
            radius_mod = '<'+str(enc_radius)
        
        elif type(enc_radius) == list and len(enc_radius)!=1: #broken for now. Effectively still only uses the outer bound.
            R_where = np.where((R>enc_radius[0]) & (R<enc_radius[1]))
            R_where = R_where[0]
            radius_mod = str(enc_radius[0])+' - '+str(enc_radius[1])
            
        elif type(enc_radius) == list and len(enc_radius)==1:
            R_where = np.where(R<enc_radius[0])
            R_where = R_where[0]
            radius_mod = '<'+str(enc_radius[0])
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',last_version=True,username=sweap_id,password=sweap_pass)
        
        # temp_data = pyt.get_data('psp_spi_TEMP')
        # temp_time_arr = temp_data[0]
        # temp_data_arr = temp_data[1]
        
        # breakpoint()
        
        pos_data = pyt.get_data('psp_spi_SUN_DIST')
        pos_time_arr = pos_data[0]
        pos_data_arr = pos_data[1]
        
        pos_data_arr = pos_data_arr/Rs # convert to solar radii
        
        dens_data = pyt.get_data('psp_spi_DENS')
        dens_time_arr = dens_data[0]
        dens_data_arr = dens_data[1]
        
        density = dens_data_arr*1e6 #1/m^3 rather than 1/cm^3
        
        mag_data = pyt.get_data('psp_spi_MAGF_INST')
        mag_time_arr = mag_data[0]
        mag_data_arr = mag_data[1]
        
        Bx = mag_data_arr[:,0]
        By = mag_data_arr[:,1]
        Bz = mag_data_arr[:,2]
        
        B = np.sqrt(Bx**2+By**2+Bz**2)*1e-9 #convert to T instead of nT
        
        tens_data = pyt.get_data('psp_spi_T_TENSOR_INST')
        tens_time_arr = tens_data[0]
        tens_data_arr = tens_data[1]
        
        
        Txx = tens_data_arr[:,0]
        Tyy = tens_data_arr[:,1]
        Tzz = tens_data_arr[:,2]
        Txy = tens_data_arr[:,3]
        Txz = tens_data_arr[:,4]
        Tyz = tens_data_arr[:,5]
        
        tensor = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])

        tens_fa = []
        anis_steve = []
        anis_ben = []
        beta_par = []
        beta_perp = []
        alf_vel = []
        # beta_tot = []
        for j in range(len(Bx)):
            rot_mat = david_rot_mat([Bx[j],By[j],Bz[j]])

            rot_inv = inv(rot_mat)
            tmp_tens = np.matmul(rot_mat,tensor[:,:,j]) #first calculation of T' = R T R^-1
            tens_fin = np.matmul(tmp_tens,rot_inv)
            
            tens_fa.append(tens_fin) #temperature tensor rotated to FA coords.
            
            T_perp_tmp = np.array(tens_fin[1,1]+tens_fin[2,2])/2
            T_par_tmp = np.array(tens_fin[0,0])
            T_anis = T_perp_tmp/T_par_tmp
            
            # temp_J = temp_data_arr*eVtoJ
            temp_par_J = T_par_tmp*eVtoJ
            temp_perp_J = T_perp_tmp*eVtoJ
            
            beta_par_tmp = 2*mu*density[j]*temp_par_J/B[j]**2
            beta_perp_tmp = 2*mu*density[j]*temp_perp_J/B[j]**2
            
            alf_vel_tmp = B[j]/np.sqrt(mu*mp*density[j]) #m/s
            alf_vel_tmp = alf_vel_tmp*1e-3 #km/s
            alf_vel.append(alf_vel_tmp)
            
            # beta_tot_tmp =2*mu*density[j]*temp_J/b_mag[j]**2
            
            beta_par.append(beta_par_tmp)
            beta_perp.append(beta_perp_tmp)
            # beta_tot.append(beta_tot_tmp)
            
            anis_ben.append(T_anis)
       
            # anis_da = david_anis([Bx,By,Bz],tens_val_tmp)
            
            anis_da = steven_anis([Bx[j],By[j],Bz[j]],tensor[:,:,j])
            anis_steve.append(anis_da)
            
        
        anis_time = tens_time_arr
        anis_steve = np.array(anis_steve)
        anis_ben = np.array(anis_ben)
        beta_par = np.array(beta_par)
        beta_perp = np.array(beta_perp)
        alf_vel = np.array(alf_vel)
        
        anis_save_path = '/Users/besh2109/Desktop/Temperature Products/Ions/'
        anis_save_name = 'enc_'+str(i)+'_ion_thermal_products.cdf'

        pyt.store_data("position_Rs",data={'x':anis_time, 'y':pos_data_arr})

        pyt.store_data("T_anisotropy_ben", data={'x':anis_time, 'y':anis_ben})
        pyt.store_data("T_anisotropy_steve", data={'x':anis_time, 'y':anis_steve})
        # pyt.tplot_save('T_anisotropy',anis_save_path+anis_save_name)
        
        pyt.store_data("beta_par", data={'x':anis_time, 'y':beta_par})
        pyt.store_data("beta_perp", data={'x':anis_time, 'y':beta_perp})
        # pyt.tplot_save('beta_par',beta_save_path+beta_save_name)
        
        pyt.store_data("alfven_vel",data={'x':anis_time,'y':alf_vel})
        
        # pyt.store_data("psp_spi_VEL_RTN_SUN",data={'x':vel_time_arr,'y':vel_data_arr})
        
        cdf_var_list = ["position_Rs","T_anisotropy_ben","T_anisotropy_steve","beta_par","beta_perp","alfven_vel","psp_spi_VEL_RTN_SUN"]
        
        pyt.tplot_save(cdf_var_list,anis_save_path+anis_save_name) #saves the temperature anisotropy and plasma betas to a .cdf file

def brazil(enc='no 1',enc_radius=45,plot='all'):
    print()
    print("YOU ARE GOING TO BRAZIL!")
    print()
    
    for meme in brazil_text():
        print(meme)
        time.sleep(0.025)
    print()
    
    time.sleep(0.75)

    fp = findpeaks()

    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    #kb = 1.380649*10e-23 #boltzmann constant, J/K
    
    #----------------------Organizing CSVs for in Data-------------------------#
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    # csv_names = os.listdir(csv_path)
    # csv_names.remove('.DS_Store')
    
    csv_names = glob.glob(csv_path+'*_raw.csv')
    
    enc_tag = []
    for i in csv_names:
        
        a = re.findall(r'\d+',i[25:])
        enc_tag.append(int(a[0]))
    
    csv_names = sort_list(csv_names,enc_tag)
    enc_tag = sort_list(enc_tag,enc_tag)
    
    # filelist = []
    # for j in csv_names:
    #     j = csv_path+j
    #     filelist.append(j)
    
    filelist = csv_names
    
    # breakpoint()
        
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    if enc == 'no 13':
        enc = list(range(2,13))
        enc.append(14)
        enc.append(15)
        enc.append(16)
        title_mod = 'for All Encounters, no 1 or 13'
        
    if enc == 'no 1':
        enc = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    if type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    # breakpoint()
    
    filelist_new = [filelist[i-1] for i in enc]
    
    #--------------------------Reading in Quiescent Data---------------------------#
    
    df_list = [pd.read_csv(file) for file in filelist_new]
    
    df = pd.concat(df_list)
    
    q_starts = df['start dates'].to_numpy()
    q_ends = df['end dates'].to_numpy()
    q_duras = df['duration'].to_numpy()
    
    q_start_flt = np.array(pys.time_float(q_starts))
    q_end_flt = np.array(pys.time_float(q_ends))
    
    # long_check = np.where(q_duras > 3600)
    # q_start_flt = q_start_flt[long_check]
    # q_end_flt = q_end_flt[long_check]

    
    #--------------------------Start importing PSP Data-----------------------------#
    
    # hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    # pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    # hpos = pyt.get_data('position')
    
    # hpos_time_arr = hpos[0]
    # hpos_data_arr = hpos[1]

    pos_full = np.array([])
    q_pos_full = np.array([])
    non_q_pos_full = np.array([])
    
    time_full = np.array([])
    q_time_full = np.array([])
    non_q_time_full = np.array([])
    
    anis_ben_full = np.array([])
    q_anis_ben_full = np.array([])
    non_q_anis_ben_full = np.array([])
    
    beta_par_full = np.array([])
    q_beta_par_full = np.array([])
    non_q_beta_par_full = np.array([])
    
    alfven_full = np.array([])
    q_alfven_full = np.array([])
    non_q_alfven_full = np.array([])
    
    alfvenicity_full = np.array([])
    q_alfvenicity_full = np.array([])
    non_q_alfvenicity_full = np.array([])
    
    for i in enc:
        
        # print(i)
        
        enc_ind = (i-1)
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        radius_mod = '<'+str(enc_radius)
        
        
        anis_save_path = '/Users/besh2109/Desktop/Temperature Products/Ions v1/'
        anis_save_name = 'enc_'+str(i)+'_ion_thermal_products.cdf'
        
        
        pyt.tplot_restore(anis_save_path+anis_save_name)
        
        pos_data = pyt.get_data('position_Rs')
        pos_time_arr = pos_data[0]
        pos_data_arr = pos_data[1]
        
        
        anis_data = pyt.get_data('T_anisotropy_ben')
        anis_time_arr = anis_data[0]
        anis_data_arr = anis_data[1]

        
        beta_par_data = pyt.get_data('beta_par')
        beta_par_time_arr = beta_par_data[0]
        beta_par_data_arr = beta_par_data[1]
        
        alfven_data = pyt.get_data('alfven_vel')
        alfven_time_arr = alfven_data[0]
        alfven_vel_arr = alfven_data[1]
        
        vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
        vel_time_arr = vel_data[0]
        vel_data_arr = vel_data[1]
        
        vr = vel_data_arr[:,0]
        vt = vel_data_arr[:,1]
        vn = vel_data_arr[:,2]
        
        V = np.sqrt(vr**2+vt**2+vn**2)
        
        # breakpoint()
        
        pos_where = np.where(pos_data_arr<enc_radius) #find where in the data is below the set radial cutoff
        pos_where = pos_where[0]
        
        anis_time = anis_time_arr[pos_where]
        position = pos_data_arr[pos_where]
        anis_ben = anis_data_arr[pos_where]
        beta_par = beta_par_data_arr[pos_where]
        alfven = alfven_vel_arr[pos_where]
        vel_mag = V[pos_where]
        
        icme_start = pys.time_float('2022-09-06/15:25')
        icme_end = pys.time_float('2022-09-07/12:25')
        
        # icme_where = np.where(((anis_time>icme_start)&(anis_time<icme_end)))
        icme_where = np.where(~((anis_time>icme_start)&(anis_time<icme_end)))
        icme_where = icme_where[0]
        icme_where_check = len(icme_where)
        
        if icme_where_check > 0:
            
            anis_time = anis_time[icme_where]
            position = position[icme_where]
            anis_ben = anis_ben[icme_where]
            beta_par = beta_par[icme_where]
            alfven = alfven[icme_where]
            vel_mag = vel_mag[icme_where]
        
        
        
        alfvenicity = vel_mag/alfven

        q_time = np.array([],dtype=float)
        q_pos = np.array([],dtype=float)
        q_anis_ben = np.array([],dtype=float)
        q_beta_par = np.array([],dtype=float)
        q_alfven = np.array([],dtype=float)
        q_alfvenicity = np.array([],dtype=float)
        
        non_q_where = []
        for k in range(len(q_start_flt)):
            
            q_where = np.where((anis_time>q_start_flt[k])&(anis_time<q_end_flt[k]))
            q_where = q_where[0]
            
            
            """!!!"""
            non_q_where.append(q_where)
            """!!!"""
            
            if len(q_where) != 0:
                
                q_time = np.append(q_time,anis_time[q_where])
                q_pos = np.append(q_pos,position[q_where])
                q_anis_ben = np.append(q_anis_ben,anis_ben[q_where])
                q_beta_par = np.append(q_beta_par,beta_par[q_where])
                q_alfven = np.append(q_alfven,alfven[q_where])
                q_alfvenicity = np.append(q_alfvenicity,alfvenicity[q_where])
        
        non_where = np.where(~np.in1d(anis_time, q_time))
        non_q_time = anis_time[non_where[0]]
        non_q_pos = position[non_where[0]]
        non_q_anis_ben = anis_ben[non_where[0]]
        non_q_beta_par = beta_par[non_where[0]]
        non_q_alfven = alfven[non_where[0]]
        non_q_alfvenicity = alfvenicity[non_where[0]]
        
        time_full = np.append(time_full,anis_time)
        q_time_full = np.append(q_time_full,q_time)
        non_q_time_full = np.append(non_q_time_full,non_q_time)
            
        pos_full = np.append(pos_full,position)
        q_pos_full = np.append(q_pos_full,q_pos)
        non_q_pos_full = np.append(non_q_pos_full,non_q_pos)
        
        anis_ben_full = np.append(anis_ben_full,anis_ben)
        q_anis_ben_full = np.append(q_anis_ben_full,q_anis_ben)
        non_q_anis_ben_full = np.append(non_q_anis_ben_full,non_q_anis_ben)
        
        beta_par_full = np.append(beta_par_full,beta_par)
        q_beta_par_full = np.append(q_beta_par_full,q_beta_par)
        non_q_beta_par_full = np.append(non_q_beta_par_full,non_q_beta_par)
        
        alfven_full = np.append(alfven_full,alfven)
        q_alfven_full = np.append(q_alfven_full,q_alfven)
        non_q_alfven_full = np.append(non_q_alfven_full,non_q_alfven)
        
        alfvenicity_full = np.append(alfvenicity_full,alfvenicity)
        q_alfvenicity_full = np.append(q_alfvenicity_full,q_alfvenicity)
        non_q_alfvenicity_full = np.append(non_q_alfvenicity_full,non_q_alfvenicity)
        
        
        if plot in ['both','encs']:
        
            fig = plt.figure(figsize=(20,10))
            
            y_datas = [q_anis_ben,non_q_anis_ben]
            x_datas = [q_beta_par,non_q_beta_par]
            labels = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
            titles = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
            # cmaps = ['autumn','winter']
            # colors = ['tab:orange','tab:blue']
            cmaps = ['autumn','autumn']
            colors = ['tab:blue','tab:blue']
            
            inst_x = np.logspace(np.log10(0.0001), np.log10(50.0),200)
            
            a1 = 0.65
            b1 = 0.4
            beta01 = -0.0004
            inst_anis_ion_cyc = (1 + a1/(inst_x-beta01)**b1) #Ion Cyclotron Instability, Hellinger 2006
            
            a2 = 0.77
            b2 = 0.76
            beta02 = -0.016
            inst_anis_mirror = (1 + a2/(inst_x-beta02)**b2) #Mirror Instability, Hellinger 2006
            
            a3 = -0.47
            b3 = 0.53
            beta03 = 0.59
            inst_anis_par_firehose = (1 + a3/(inst_x-beta03)**b3) #Oblique Firehose Instability, Hellinger 2006
             
            a4 = -1.4
            b4 = 1
            beta04 = -0.11
            inst_anis_obl_firehose = (1 + a4/(inst_x-beta04)**b4) #Oblique Firehose Instability, Hellinger 2006
            
            fig.suptitle("Brazil Plots for Encounter "+str(i)+" ("+radius_mod+" $R_{\odot}$)",fontsize=36)
            
            for ii in range(2):
            
                axs = fig.add_subplot(1,2,ii+1)
            
                y_space = np.logspace(np.log10(0.1), np.log10(10.0), 70)
                x_space = np.logspace(np.log10(0.0001), np.log10(30.0), 70)
                hist2d,xedge,yedge = np.histogram2d(x_datas[ii],y_datas[ii], bins=(x_space,y_space))
                # hist2d,xedge,yedge = np.histogram2d(q_beta_par,q_anis_ben, bins=(x_space,y_space))
                # hist2d,xedge,yedge = np.histogram2d(beta_par,anis_ben, bins=(x_space,y_space))
                histo = np.transpose(hist2d)
                
                xcenters = (xedge[:-1] + xedge[1:]) / 2
                ycenters = (yedge[:-1] + yedge[1:]) / 2
        
                r_contour = axs.contour(xcenters,ycenters,histo, cmap=cmaps[ii], vmin=20, levels=10,zorder=2)
                # cb = fig.colorbar(r_contour)
                # fig1.colorbar(CS)
        
                delete_contour(r_contour,1)
        
                axs.scatter(x_datas[ii],y_datas[ii],marker=',',color=colors[ii],label=labels[ii],s=1)
                
                axs.plot(inst_x,inst_anis_ion_cyc,linestyle='-',zorder=3,color='black',label='Ion Cyclotron Instability')
                # axs.plot(inst_x,inst_anis_mirror,linestyle='-',zorder=4,color='red',label='Mirror Instability')
                axs.plot(inst_x,inst_anis_par_firehose,linestyle='-',zorder=5,color='lime',label='Parallel Firehose')
                # print('yeet')
                # axs.plot(inst_x,inst_anis_obl_firehose,linestyle='-',zorder=6,color='firebrick',label='Oblique Firehose')
                
                axs.set_title(titles[ii],fontsize=32)
                axs.set_ylabel('Temperature Anisotropy',fontsize=32)
                axs.set_xlabel('Plasma Beta Parallel',fontsize=32)
                axs.tick_params(axis='both', which='major', labelsize=32)
                axs.set_xlim(0.0001,50)
                axs.set_ylim(0.1,10)
                
    
                axs.set_xscale("log")
                axs.set_yscale("log")
                axs.grid()
                
                axs.tick_params(width=3, length=7)
                if ii ==1:
                    # axs.sharex()
                    # print('pizza')
                    # axs.tick_params('y', labelbottom=False)
                    axs.set_yticks([])
                    axs.set_ylabel('')
                
            
            plt.subplots_adjust(wspace=0.01, hspace=0)
            # leg = axs.legend(fontsize=20,loc='upper right')
            plt.tight_layout()
            plt.show()
            
            plt.clf()
            plt.cla()
            plt.close('all')
            plt.close(fig)
        
    # breakpoint()
        
    q_peaks = []
    non_q_peaks = []
    x_errs_q = []
    y_errs_q = []
    x_errs_non_q = []
    y_errs_non_q = []
    
    if plot in ['both','all']:
        
        fig = plt.figure(figsize=(10,10))
        # fig = plt.figure(figsize=(20,10))
        
        y_datas = [q_anis_ben_full,non_q_anis_ben_full]
        x_datas = [q_beta_par_full,non_q_beta_par_full]
        c_datas = [q_alfvenicity_full,non_q_alfvenicity_full]
        labels = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
        titles = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
        # cmaps = ['autumn','winter']
        # colors = ['tab:orange','tab:blue']
        cmaps = ['autumn','autumn']
        colors = ['tab:blue','tab:blue']
        
        
        inst_x = np.logspace(np.log10(0.0001), np.log10(50.0),200)
        
        a1 = 0.65
        b1 = 0.4
        beta01 = -0.0004
        inst_anis_ion_cyc = (1 + a1/(inst_x-beta01)**b1) #Ion Cyclotron Instability Hellinger 2006
        
        a2 = 0.77
        b2 = 0.76
        beta02 = -0.016
        inst_anis_mirror = (1 + a2/(inst_x-beta02)**b2) #Mirror Instability Hellinger 2006
        
        a3 = -0.47
        b3 = 0.53
        beta03 = 0.59
        inst_anis_par_firehose = (1 + a3/(inst_x-beta03)**b3) #Oblique Firehose Instability Hellinger 2006
         
        a4 = -1.4
        b4 = 1
        beta04 = -0.11
        inst_anis_obl_firehose = (1 + a4/(inst_x-beta04)**b4) #Oblique Firehose Instability Hellinger 2006
        
        
        # fig.suptitle("Brazil Plots "+title_mod+" ("+radius_mod+" Rs)",fontsize=36)
        fig.suptitle("$\mathcal{R}$ vs $β_{\parallel p}$ for ICME wake",fontsize=36)
        
        # breakpoint()
        # gs=gridspec.GridSpec(1,3, width_ratios=[4,4,0.2])
        gs=gridspec.GridSpec(1,2, width_ratios=[4,0.2])
        # for ii in range(2):
        for ii in range(1):
        
            axs = fig.add_subplot(gs[ii])
        
            y_space = np.logspace(np.log10(0.1), np.log10(10.0), 28)
            x_space = np.logspace(np.log10(0.0001), np.log10(30.0), 28)
            hist2d,xedge,yedge = np.histogram2d(x_datas[ii],y_datas[ii], bins=(x_space,y_space))
            histo = np.transpose(hist2d)
            
            xcenters = (xedge[:-1] + xedge[1:]) / 2
            ycenters = (yedge[:-1] + yedge[1:]) / 2
            
            #----------check where peak is-----------#

            peaks = fp.fit(histo)
            peak_array = np.array(peaks['persistence'])
            main_peak = peak_array[0]
            main_peak_x = main_peak[0] #x-coordinate of the main peak
            main_peak_y = main_peak[1] #y-coordinate of the main peak
  
            main_peak_x_hist = histo[main_peak_y,:] #nets you the 1d histogram along the x axis at peak y-position
            main_peak_y_hist = histo[:,main_peak_x] #nets you the 1d histogram along the y axis at peak x-position
            
            
            spline_x = UnivariateSpline(xcenters, main_peak_x_hist-np.max(main_peak_x_hist)/2, s=0) #find FWHM of x 
            spline_y = UnivariateSpline(ycenters, main_peak_y_hist-np.max(main_peak_y_hist)/2, s=0) #   and y coordinates
            
            r1x, r2x = spline_x.roots()[:2]
            r1y, r2y = spline_y.roots()[:2]
            
            if ii == 0:
                q_peaks.append([xcenters[main_peak_x],ycenters[main_peak_y]])
                x_errs_q.append([r1x,r2x])
                y_errs_q.append([r1y,r2y])
                # print([xcenters[main_peak_x],ycenters[main_peak_y]])
            else:
                non_q_peaks.append([xcenters[main_peak_x],ycenters[main_peak_y]])
                x_errs_non_q.append([r1x,r2x])
                y_errs_non_q.append([r1y,r2y])
            
            #--------------------------------------#
            
            # breakpoint()
            
            
    
            r_contour = axs.contour(xcenters,ycenters,histo, cmap=cmaps[ii], vmin=20, levels=10,zorder=2)
            # cb = fig.colorbar(r_contour)
            # fig1.colorbar(CS)
    
            delete_contour(r_contour,1)
    
            sct = axs.scatter(x_datas[ii],y_datas[ii],marker=',',c=c_datas[ii],s=1,zorder=1,cmap='rainbow',vmax=6)
            # axs.scatter(x_datas[ii],y_datas[ii],marker=',',color=colors[ii],s=1,zorder=1)
            
            axs.plot(inst_x,inst_anis_ion_cyc,linestyle='-',zorder=3,color='black',label='Ion Cyclotron Instability')
            # axs.plot(inst_x,inst_anis_mirror,linestyle='-',zorder=4,color='red',label='Mirror Instability')
            axs.plot(inst_x,inst_anis_par_firehose,linestyle='-',zorder=5,color='brown',label='Parallel Firehose')
            # axs.plot(inst_x,inst_anis_obl_firehose,linestyle='-',zorder=6,color='firebrick',label='Oblique Firehose')
  
            # axs.axvline(x = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
            # axs.axhline(y = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
            # axs.scatter(q_beta_par_full,q_anis_ben_full,marker=',',color='tab:orange',label='Quiescent Solar Wind',s=1)
            
            # axs.set_title(titles[ii],fontsize=32)
            axs.set_ylabel('Temperature Anisotropy ($\mathcal{R}$)',fontsize=32)
            axs.set_xlabel('Plasma Beta Parallel ($β_{\parallel p}$)',fontsize=32)
            axs.tick_params(axis='both', which='major', labelsize=32)
            if ii ==0:
                axs.tick_params(axis='both', which='minor', size=6)
            
            # axs.set_xlim(0.0001,50)
            # axs.set_ylim(0.1,10)
            
            axs.set_xlim(0.002,30)
            axs.set_ylim(0.08,12)
                
            axs.set_xscale("log")
            axs.set_yscale("log")
            axs.grid()
            
            axs.tick_params(width=3, length=7)
            if ii ==1:
                for tick in axs.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
                axs.set_yticklabels([])
                axs.set_ylabel('')
            
        # ax3 = plt.subplot(gs[2])
        ax3 = plt.subplot(gs[1])
        
        cb = plt.colorbar(sct,cax=ax3)
        cb.set_label("Alfvén Mach Number  ($V_{sw}/V_{Alf}$)",fontsize=24)
        for t in cb.ax.get_yticklabels():
             t.set_fontsize(24)
        # plt.colorbar()
        plt.subplots_adjust(wspace=0.01, hspace=0)
        leg = axs.legend(fontsize=16,loc='upper right',markerscale=5)
        
        plt.show()
        
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.close(fig)
        
        
        
    if plot in ['both','radial']:
        

        
        radial_bins = [[8,15],[15,25],[25,35],[35,45]]
        # radial_bins = [[35,45],[25,35],[15,25],[8,15]]
        n_rads = len(radial_bins)
        # radial_bins = [[13,20],[20,30],[30,40]]
        r_labs_in = ['8','15','25','35']
        r_labs_out = ['15','25','35','45']
        
        r_labs_in.reverse()
        r_labs_out.reverse()
        # breakpoint()
        
        lab_pos = [0.21,0.4,0.6,0.78]
        title_pos= [0.32, 2*0.35]
        fig = plt.figure(figsize=(10,20))
        
        
        gs=gridspec.GridSpec(4,3, width_ratios=[5,5,0.2])
        
        for j in range(len(radial_bins)): 
        
            bins = radial_bins[j]
            
            tplot_savename = 'Brazil_data_'+str(bins[0])+'_to_'+str(bins[1])+'_Rs.cdf'
            tplot_savepath = '/Users/besh2109/Desktop/BrazilPlots/'   
            
            q_rad_where = np.where((q_pos_full>bins[0])&(q_pos_full<bins[1]))
            q_rad_where = q_rad_where[0]
            
            non_q_rad_where = np.where((non_q_pos_full>bins[0])&(non_q_pos_full<bins[1]))
            non_q_rad_where = non_q_rad_where[0]
            
            title_mod = "for radial distances "
            radial_mod = "of "+str(bins[0])+" through "+str(bins[1])
        
            q_times_rad_bins = q_time_full[q_rad_where]
            non_q_times_rad_bins = non_q_time_full[non_q_rad_where]
        
            q_anis_rad_bins = q_anis_ben_full[q_rad_where]
            non_q_anis_rad_bins = non_q_anis_ben_full[non_q_rad_where]
            
            q_beta_rad_bins = q_beta_par_full[q_rad_where]
            non_q_beta_rad_bins = non_q_beta_par_full[non_q_rad_where]
            
            q_alfvenicity_rad_bins = q_alfvenicity_full[q_rad_where]
            non_q_alfvenicity_rad_bins = non_q_alfvenicity_full[non_q_rad_where]
            
            y_datas = [q_anis_rad_bins,non_q_anis_rad_bins]
            x_datas = [q_beta_rad_bins,non_q_beta_rad_bins]
            c_datas = [q_alfvenicity_rad_bins,non_q_alfvenicity_rad_bins]
            
            pyt.store_data("q_region_times",data={'x':q_times_rad_bins, 'y':q_times_rad_bins})
            
            pyt.store_data("non_q_region_times",data={'x':non_q_times_rad_bins, 'y':non_q_times_rad_bins})
            
            pyt.store_data("q_region_brazil", data={'x':q_beta_rad_bins, 'y':q_anis_rad_bins})

            pyt.store_data("non_q_region_brazil",data={'x':non_q_beta_rad_bins, 'y':non_q_anis_rad_bins})
            
            pyt.store_data("q_color_scale",data={'x':q_alfvenicity_rad_bins, 'y':q_alfvenicity_rad_bins})

            pyt.store_data("non_q_color_scale",data={'x':non_q_alfvenicity_rad_bins, 'y':non_q_alfvenicity_rad_bins}) #stores last random bin set
            
            cdf_var_list = ["q_region_times","non_q_region_times","q_region_brazil","non_q_region_brazil","q_color_scale","non_q_color_scale"]
            
            pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename)
            
            labels = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
            titles = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
            # cmaps = ['autumn','winter']
            # colors = ['tab:orange','tab:blue']
            cmaps = ['autumn','autumn']
            colors = ['tab:blue','tab:blue']
            
            inst_x = np.logspace(np.log10(0.0001), np.log10(50.0),200)
            
            a1 = 0.65
            b1 = 0.4
            beta01 = -0.0004
            inst_anis_ion_cyc = (1 + a1/(inst_x-beta01)**b1) #Ion Cyclotron Instability, Hellinger 2006
            
            a2 = 0.77
            b2 = 0.76
            beta02 = -0.016
            # beta02 = 0
            inst_anis_mirror = (1 + a2/(inst_x-beta02)**b2) #Mirror Instability, Hellinger 2006
            
            a3 = -0.47
            b3 = 0.53
            beta03 = 0.59
            inst_anis_par_firehose = (1 + a3/(inst_x-beta03)**b3) #Oblique Firehose Instability, Hellinger 2006
             
            a4 = -1.4
            b4 = 1
            beta04 = -0.11
            inst_anis_obl_firehose = (1 + a4/(inst_x-beta04)**b4) #Oblique Firehose Instability, Hellinger 2006
            
            for ii in range(2):
            
                axs = fig.add_subplot(gs[3*j+ii])
                
                # print(3*j+ii)
            
                y_space = np.logspace(np.log10(0.1), np.log10(10.0), 30)
                x_space = np.logspace(np.log10(0.0001), np.log10(30.0), 30)
                hist2d,xedge,yedge = np.histogram2d(x_datas[ii],y_datas[ii], bins=(x_space,y_space))
                # hist2d,xedge,yedge = np.histogram2d(q_beta_par,q_anis_ben, bins=(x_space,y_space))
                # hist2d,xedge,yedge = np.histogram2d(beta_par,anis_ben, bins=(x_space,y_space))
                histo = np.transpose(hist2d)
                
                xcenters = (xedge[:-1] + xedge[1:]) / 2
                ycenters = (yedge[:-1] + yedge[1:]) / 2
        
                r_contour = axs.contour(xcenters,ycenters,histo, cmap=cmaps[ii], vmin=20, levels=6,zorder=2)
                # cb = fig.colorbar(r_contour)
                # fig1.colorbar(CS)
        
                delete_contour(r_contour,1)
                sct = axs.scatter(x_datas[ii],y_datas[ii],marker=',',c=c_datas[ii],s=1,zorder=1,cmap='rainbow',vmax=6)
                # axs.scatter(x_datas[ii],y_datas[ii],marker=',',color=colors[ii],s=1,zorder=1)
                
                axs.plot(inst_x,inst_anis_ion_cyc,linestyle='-',zorder=3,color='black',label='Ion Cyclotron Instability')
                axs.plot(inst_x,inst_anis_mirror,linestyle='-',zorder=4,color='blue',label='Mirror Instability')
                axs.plot(inst_x,inst_anis_par_firehose,linestyle='-',zorder=5,color='lime',label='Parallel Firehose')
                # axs.plot(inst_x,inst_anis_obl_firehose,linestyle='-',zorder=6,color='firebrick',label='Oblique Firehose')
      
                # axs.axvline(x = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
                # axs.axhline(y = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
                # axs.scatter(q_beta_par_full,q_anis_ben_full,marker=',',color='tab:orange',label='Quiescent Solar Wind',s=1)
                
                # axs.set_title(titles[ii],fontsize=32)
                # axs.set_ylabel('Temperature Anisotropy',fontsize=32)
                # axs.set_xlabel('Plasma Beta Parallel',fontsize=32)

                axs.tick_params(axis='both', which='major', labelsize=24)
                axs.set_xlim(0.002,30)
                axs.set_ylim(0.08,12)

                axs.set_xscale("log")
                axs.set_yscale("log")
                axs.grid()
                
                axs.tick_params(width=3, length=7)
                
                if j == 0:
                    fig.text(title_pos[ii],0.888, (labels[ii]), ha="center", va="center",fontsize=22)
                    fig.text(0.5,0.91,'Brazil Plots vs $R_{\odot}$',ha="center", va="center",fontsize=28)
                    fig.text(0.5,0.09,'Plasma Beta Parallel',ha="center", va="center",fontsize=22)
                    fig.text(0.01,0.5,'Ion Temperature Anisotropy',ha="center", va="center", rotation=90,fontsize=24)
                
                if ii ==1:
                    for tick in axs.yaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)
                    axs.set_yticklabels([])
                    axs.set_ylabel('')
                    
                if j!=n_rads-1:
                    for tick in axs.xaxis.get_major_ticks():
                        tick.tick1line.set_visible(False)
                        tick.tick2line.set_visible(False)
                        tick.label1.set_visible(False)
                        tick.label2.set_visible(False)
                    axs.set_xticklabels([])
                    axs.set_xlabel('')
                
            
            fig.text(-0.1,lab_pos[j], (r_labs_out[j]+" $R_{\odot}$ \n - \n "+r_labs_in[j]+" $R_{\odot}$"), ha="center", va="center", rotation=12,fontsize=24)
            
            ax3 = plt.subplot(gs[3*j+ii+1])
            
            cb = plt.colorbar(sct,cax=ax3)
            cb.set_label("Alfvenicity  $V_{sw}/V_{Alf}$",fontsize=20)
            for t in cb.ax.get_yticklabels():
                  t.set_fontsize(20)
                
            
        plt.subplots_adjust(wspace=0.02, hspace=0.01)
        leg = axs.legend(fontsize=16,loc='upper right',markerscale=5)
        
        plt.show()
        
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.close(fig)
            
            
        """
        if plotstyle == 'radial':
            
            for k in range(n_types):
                # print(k)
                
                # q_data_full = data_types[k]
                # non_q_data_full = data_types[k]
                
                q_data_single = q_data_full[:,k]
                non_q_data_single = non_q_data_full[:,k]
                
                n_bin_single = n_bins_list[k]
                drange_single = drange_list[k]
                
                for j in range(n_rads):
                    
                    rad_bins = radial_bins[j]
                    
                    q_rad_where = np.where((q_pos_full>rad_bins[0])&(q_pos_full<rad_bins[1]))
                    q_rad_where = q_rad_where[0]
                    
                    non_q_rad_where = np.where((non_q_pos_full>rad_bins[0])&(non_q_pos_full<rad_bins[1]))
                    non_q_rad_where = non_q_rad_where[0]
                    
                    q_ave_where = np.where((q_pos_average_full>rad_bins[0])&(q_pos_average_full<rad_bins[1]))
                    q_ave_where = q_ave_where[0]
                    
                    q_range_num = len(q_ave_where)
                    
                    title_mod = "for radial distances of "+str(rad_bins[0])+" - "+str(rad_bins[1])
                    
                    q_data_rad_bins = q_data_single[q_rad_where]
                    non_q_data_rad_bins = non_q_data_single[non_q_rad_where]
                    
                    # breakpoint()
                    
                    q_mean = np.nanmean(q_data_rad_bins)
                    non_q_mean = np.nanmean(non_q_data_rad_bins)
                    
                    q_med = np.nanmedian(q_data_rad_bins)
                    non_q_med = np.nanmedian(non_q_data_rad_bins)
                    
                    
                    x_datas = [non_q_data_rad_bins,q_data_rad_bins]
            
                    labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
                    titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
            
                    cmaps = ['autumn','autumn']
                    colors = ['tab:blue','tab:orange']
                    
            
                    
                    # axs = fig.add_subplot(1,1,1)
                    # print(j+k+1)
                    index = n_types*j+1+k
                    # if k ==0:
                    axs = fig.add_subplot(n_rads,n_types,index)
                    # axs = fig.add_subplot(n_rads,1,j+1)
                    
                    bins = np.linspace(drange_single[0],drange_single[1], n_bins)
                    hist1, _ = np.histogram(x_datas[0], bins=bins)
                    hist2, _ = np.histogram(x_datas[1], bins=bins)
                    
                    hist1_norm = hist1 / np.sum(hist1)
                    hist2_norm = hist2 / np.sum(hist2)
                    
                    # Calculate error bars for each bin (normalized)
                    error1 = np.sqrt(hist1) / np.sum(hist1)
                    error2 = np.sqrt(hist2) / np.sum(hist2)
                    
                    # Plot histograms with error bars
                    axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
                    axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
                    axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
                    axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
                    
                    # axs.set_ylabel("Normalized Counts",fontsize=32)
                    # axs.set_xlabel(datatype,fontsize=32)
                    axs.tick_params(axis='x', which='major', labelsize=24)
                    
                    if j!=n_rads-1:
                        # axs.tick_params('x', labelbottom=False)
                        axs.set_xticks([])
                        
                    # axs.tick_params('y', labelbottom=False)
                    axs.set_yticks([])
                    # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm),"Number of Quiescent Regions: "+str(q_range_num),fontsize=32)
                    # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*2.4,"Quiescent Mean: "+str(q_mean),fontsize=32)
                    # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*3.4,"Non-Quiescent Mean: "+str(non_q_mean),fontsize=32)
                    # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*4.4,"Quiescent Median: "+str(q_med),fontsize=32)
                    # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*5.4,"Non-Quiescent Median: "+str(non_q_med),fontsize=32)
                    
                    # plt.subplots_adjust(wspace=0.01, hspace=0)
                    
                    if index==1:
                        leg = axs.legend(fontsize=20,loc='upper right',markerscale=5)
                    plt.tight_layout()
                    
                fig.text(-0.035,0.25*k+0.14, (r_labs_out[k]+" Rs \n - \n "+r_labs_in[k]+" Rs"), ha="center", va="center", rotation=12,fontsize=32)
                fig.text(0.33*k+0.17,1.015, data_types[k], ha="center", va="center",fontsize=32)
                # plt.show()
            
            fig.text(-0.035,0.25*3+0.14, "45 Rs \n - \n 35 Rs", ha="center", va="center", rotation=12,fontsize=32)
            fig.text(-0.09,0.525, "Radial Distance", ha="center", va="center", rotation=90,fontsize=38)
            fig.text(0.5,1.05, "Quiescent Region Property Histograms", ha="center", va="center",fontsize=32)
            plt.subplots_adjust(wspace=0.02, hspace=0.02)
            
            if save:
                save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/Plots/Histograms/'+file_mod+'/'
                save_name = 'quiescent_'+dtype+'_histogram_'+str(rad_bins[0])+"_to_"+str(rad_bins[1])+'_Rs.png'
                
                if not os.path.exists(save_path):
                    print('should work')
                    os.makedirs(save_path)
                
                plt.savefig(save_path+save_name)
            
            else:
                plt.show()
                
            plt.clf()
            plt.cla()
            plt.close('all')
            plt.close(fig)
        
        """

def quiescent_histograms(enc='all',enc_radius=45,plot='all',dtype='ion_anis',comp='magnitude',save=False,plotstyle='radial',t0='2019-04-05',tf='2019-04-06',tmp_check=True, span_cut=True):

    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    #kb = 1.380649*10e-23 #boltzmann constant, J/K
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    if enc == 'no 13':
        enc = list(range(2,13))
        enc.extend(range(14,17))
        title_mod = 'for All Encounters, no 1 or 13'
        
    if enc == 'no 1':
        enc = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    if type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    #----------------------Organizing CSVs for in Data-------------------------#
    
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    
    #------------------cut out bad SPAN times------------------#
    
    # span_time = []
    # span_flag = []
    # span_flag_av = []
    # span_flag_ratio = []
    # span_rad = []
    good_span_times = []
    
    for pepe in range(1,17):
    
        span_check_savename = 'SPAN_ion_fov_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/FOV flags/'
        
        pyt.tplot_restore(span_check_savepath+span_check_savename)
    
        # fov_flag_exact = pyt.get_data('phi_fov')
        # fov_time = fov_flag_exact[0]
        # fov_flag = fov_flag_exact[1]
        
        fov_flag_average = pyt.get_data('phi_fov_average')
        fov_av_time = fov_flag_average[0]
        fov_flag_av = fov_flag_average[1]
        
        # fov_flag_ratio = pyt.get_data('phi_fov_ratio')
        # fov_av_time = fov_flag_ratio[0]
        # fov_ratio_av = fov_flag_ratio[1]
        
        # rad = pyt.get_data('psp_radial_dist_Rs')
        # rad_time = rad[0]
        # rad_Rs = rad[1]
        
        # span_time.append(fov_time)
        # span_flag.append(fov_flag)
        # span_flag_av.append(fov_flag_av)
        # span_flag_ratio.append(fov_ratio_av)
        # span_rad.append(rad_Rs)
        
        span_qtn_check_savename = 'SPAN_SPC_QTN_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_qtn_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
        
        pyt.tplot_restore(span_qtn_check_savepath+span_qtn_check_savename)
        
        qtn_flag_data = pyt.get_data('SPAN_qual_flag')
        qtn_flag_time = qtn_flag_data[0]
        qtn_flag = qtn_flag_data[1]
        
        # FOV_where = np.where(fov_flag_av==0)
        # FOV_where = FOV_where[0]
        
        # good_fov = qtn_flag_time[FOV_where]
        
        # qtn_in_where = np.where(np.isin(qtn_flag_time,good_fov))
        # qtn_in_where = qtn_in_where[0]
        
        # add one to places where FOV is good to make sure that the score for good FOV is 2.
        # qtn_flag[qtn_in_where]+=1 
        
        # print(pepe," : ",qtn_flag.max()," : ", fov_flag_av.max())
        
        # if pepe > 2:
            # breakpoint()
        
        good_where = qtn_flag<2
        
        #a little confusing but I'm grouping the zeros so I actually want to invert my bool array to have good times be 0s
        # good_flip = 1-good_where
        
        edge_indices = group_zeros(good_where)
        
        # edge_times = []
        for i in edge_indices:
            t0s = qtn_flag_time[i[0]]
            tfs = qtn_flag_time[i[1]]
            
            # edge_times.append((t0,tf))
            
            good_span_times.append((t0s,tfs))
        
        
        pyt.del_data()
    # breakpoint()
    #--------------------------Start importing PSP Data-----------------------------#
    
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    # if dtype in ['ion_anis','ion_beta_par','ion_beta_perp']:
    #     dtype_list = np.array(['ion_anis','ion_beta_par','ion_beta_perp'])
    #     datatypes = np.array(['Ion Temperature Anisotropy','Ion Beta Parallel','Ion Beta Perpendicular'])
    #     dtypewhere = np.where(dtype_list==dtype)
    #     dtypewhere = dtypewhere[0]
    #     datatype = datatypes[dtypewhere][0]
    #     dcheck = 'ion_thermal'
        
    # elif dtype in ['ion_dens','bulk_vel','ion_temp']:
    #     dtype_list = np.array(['ion_dens','bulk_vel','ion_temp'])
    #     datatypes = np.array(['Ion Density','Ion Bulk Velocity','Ion Temperature'])
    #     dtypewhere = np.where(dtype_list==dtype)
    #     dtypewhere = dtypewhere[0]
    #     # breakpoint()
    #     datatype = datatypes[dtypewhere][0]
    #     dcheck = 'ion_bulk'
        
    # else:
    #     try:
    #         raise NameError(dtype+" is not a valid data keyword.")
         
    #     except NameError:
    #         print(dtype+" is not a valid keyword.") 
    #         print("Valid keys are 'ion_anis','ion_beta_par','ion_beta_perp',")
    #         print("'ion_dens','bulk_vel', and 'ion_temp'.")
    #         print()
    #         raise
    
    dcheck = 'ion_thermal'
    # dcheck = 'ion_bulk'

    time_full = np.array([])
    q_time_full = np.array([])
    non_q_time_full = np.array([])
    
    time_dif_full = np.array([])
    q_time_dif_full = np.array([])
    non_q_time_dif_full = np.array([])
    
    # time_full = []
    # q_time_full = []
    # non_q_time_full = []
    
    pos_full = np.array([])
    q_pos_full = np.array([])
    non_q_pos_full = np.array([])
    
    # data_full = np.array([])
    # q_data_full = np.array([])
    # non_q_data_full = np.array([])
    
    data_full = []
    q_data_full = []
    non_q_data_full = []
    
    q_pos_average_full = np.array([],dtype=float)
    
    in_out_full = np.array([])
    q_in_out_full = np.array([])
    non_q_in_out_full = np.array([])
    
    alfven_full = np.array([])
    q_alfven_full = np.array([])
    non_q_alfven_full = np.array([])
    
    q_start_full = np.array([])
    q_end_full = np.array([])
    
    q_counts = 0 #count number o quiescent regions to be sure we're getting all of them in.
    
    for i in enc:
        
        csv_name = 'enc_'+str(i)+'_regions_raw.csv'
        
        q_df = pd.read_csv(csv_path+csv_name) #quiescent region Pandas dataframe
        
        q_starts = q_df['start dates'].to_numpy()
        q_ends = q_df['end dates'].to_numpy()
        q_duras = q_df['duration'].to_numpy()
        
        q_start_flt = np.array(pys.time_float(q_starts))
        q_end_flt = np.array(pys.time_float(q_ends))
        
        q_counts+=len(q_starts) #count number of q regions

        enc_ind = (i-1)
        
        perihelion_flt = per_flt[enc_ind] #float date of perihelion
        perihelion = pys.time_string(perihelion_flt) #string date of perihelion
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        # radius_mod = '<'+str(enc_radius)
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        # if not tmp_check:
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        if type(dtype)==str:
            dtype = [dtype]
        
        dlist = []
        drange_list = []
        n_bins_list = []
        data_types = []
        units = []
        
        for ii in range(len(dtype)):
            
            if dcheck == 'ion_thermal':
            
                anis_save_path = '/Users/besh2109/Desktop/Temperature Products/Ions v1/'
                anis_save_name = 'enc_'+str(i)+'_ion_thermal_products.cdf'
                
                pyt.tplot_restore(anis_save_path+anis_save_name)
            
                pos_data = pyt.get_data('position_Rs')
                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
            
                anis_data = pyt.get_data('T_anisotropy_ben')
                anis_time_arr = anis_data[0]
                anis_data_arr = anis_data[1]
                
                beta_par_data = pyt.get_data('beta_par')
                beta_par_time_arr = beta_par_data[0]
                beta_par_data_arr = beta_par_data[1]
                
                alfven_data = pyt.get_data('alfven_vel')
                alfven_time_arr = alfven_data[0]
                alfven_vel_arr = alfven_data[1]
                
                if dtype[ii] == 'ion_anis':
                    time_arr = anis_time_arr
                    data_arr = anis_data_arr
                    drange = (0,5)
                    n_bins = 50
                    file_mod = 'Ion_Anisotropy'
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)

                    data_types.append('Anisotropy')
                    units.append('Tperp/Tpar')
                    
                elif dtype[ii] == 'ion_beta_par':
                    time_arr = beta_par_time_arr
                    data_arr = beta_par_data_arr
                    drange = (0,1.5)
                    n_bins = 100
                    file_mod = 'Ion_Beta_Parallel'
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)
                    data_types.append('Beta Par')
                    units.append('Beta Par')
                    
                    
                
                elif dtype[ii] == 'alf_vel':
                    time_arr = alfven_time_arr
                    data_arr = alfven_vel_arr
                    drange = (10,1500)
                    n_bins = 100
                    file_mod = 'Alfvén_Vel'
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)
                    data_types.append('Alfvén Velocity')
                    units.append('km/s')

                # elif dtype == 'ion_beta_perp':
                #     time_arr = temp_time_arr
                #     data_arr = temp_data_arr
                # breakpoint()
            
            if dcheck == 'ion_bulk':
                
                anis_save_path = '/Users/besh2109/Desktop/Temperature Products/Ions/'
                anis_save_name = 'enc_'+str(i)+'_ion_thermal_products.cdf'
                
                # pyt.tplot_restore(anis_save_path+anis_save_name)
                
                # alf_data = pyt.get_data('alfven_vel')
                # alf_time_arr = alf_data[0] 
                # alf_vel_arr = alf_data[1]
                
                if i == 15 and pys.time_float(tf) > pys.time_float('2023-03-23'):
                    tf = '2023-03-23'
                
                if ii==0:
                    psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',last_version=True,username=sweap_id,password=sweap_pass)
            
                    pos_data = pyt.get_data('psp_spi_SUN_DIST')
                    pos_time_arr = pos_data[0]
                    pos_data_arr = pos_data[1]/Rs #convert to Rs
                    
                    vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
                    # vel_data = pyt.get_data('psp_spi_VEL_SC')
                    vel_time_arr = vel_data[0]
                    vel_data_arr = vel_data[1] #remember that this is a vector, 3 components
                    
                    dens_data = pyt.get_data('psp_spi_DENS')
                    dens_time_arr = dens_data[0]
                    dens_data_arr = dens_data[1]
                    
                    temp_data = pyt.get_data('psp_spi_TEMP')
                    temp_time_arr = temp_data[0]
                    temp_data_arr = temp_data[1]
                    
                    mag_data = pyt.get_data('psp_spi_MAGF_SC')
                    mag_time_arr = mag_data[0] 
                    mag_data_arr = mag_data[1] #remember that this is a vector
                
                # breakpoint()
                
                if dtype[ii] == 'ion_dens':
                    time_arr = dens_time_arr
                    data_arr = dens_data_arr
                    drange = (0,3900)
                    n_bins = 40
                    file_mod = 'Ion_Density'
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)
                    data_types.append('Density')
                    units.append('cm$^{-3}$')
                
                elif dtype[ii] == 'bulk_vel':
                    time_arr = vel_time_arr
                    
                    if comp == 'magnitude': #will change to allow for all components soon
                        vr = vel_data_arr[:,0]
                        vt = vel_data_arr[:,1]
                        vn = vel_data_arr[:,2]
                        
                        data_arr = np.sqrt(vr**2+vt**2+vn**2)
                    
                    # breakpoint()
                    drange = (0,900)
                    n_bins = 40
                    file_mod = 'Ion_Bulk_Velocity'
                    # data_arr = vel_data_arr
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)
                    
                    data_types.append('Velocity')
                    units.append('km/s')
                    
                    
                elif dtype[ii] == 'ion_temp':
                    time_arr = temp_time_arr
                    data_arr = temp_data_arr
                    drange = (0,250)
                    n_bins = 40
                    file_mod = 'Ion_Core_Temperature'
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)
                    
                    data_types.append('Temperature')
                    units.append('eV')
                    
                if dtype[ii] == 'mag_mag':
                    time_arr = mag_time_arr
                    
                    if comp == 'magnitude': #will change to allow for all components soon
                        br = mag_data_arr[:,0]
                        bt = mag_data_arr[:,1]
                        bn = mag_data_arr[:,2]
                        
                        data_arr = np.sqrt(br**2+bt**2+bn**2)
                        
                    drange = (0,1000)
                    n_bins = 45
                    file_mod = 'B_Field_Mag'
                    
                    data_types.append('|B| Magnitude')
                    units.append('nT')
                    
                    n_bins_list.append(n_bins)
                    drange_list.append(drange)
                    dlist.append(data_arr)

        
        pos_where = np.where(pos_data_arr<enc_radius)
        pos_where = pos_where[0]
        
        position = pos_data_arr[pos_where]
        time = time_arr[pos_where]
        
        # breakpoint()
        time_dif = np.diff(time)
        time_dif = np.append(time_dif,time_dif[-1]) #tack on another element to make it work
        data = data_arr[pos_where]
        
        # breakpoint()
        
        #-------------get rid of enc 13 ICME---------------#
        
        icme_start = pys.time_float('2022-09-06/15:25')
        icme_end = pys.time_float('2022-09-07/12:25')
        
        # icme_where = np.where(((anis_time>icme_start)&(anis_time<icme_end)))
        icme_where = np.where(~((time>icme_start)&(time<icme_end)))
        icme_where = icme_where[0]
        icme_where_check = len(icme_where)
        
        if icme_where_check > 0:
            
            time = time[icme_where]
            time_dif = time_dif[icme_where]
            position = position[icme_where]
            data = data[icme_where]
            
            
        #-----------------------------------------------#
        
        # breakpoint()
        
        data_large_arr = np.transpose(np.array(dlist))
        data = data_large_arr[pos_where]
        
        # alfven_vel = alf_vel_arr[pos_where]
        
        in_out_arr = np.zeros(pos_where.shape) #create array of 0s and 1s to say that PSP is heading inbound (0) or outbound (1)
        in_out_where = np.where(time>perihelion_flt) #check where time within Encounter is greater than perihelion
        in_out_where = in_out_where[0]
        
        in_out_arr[in_out_where] = 1
        
        # q_pos = np.array([],dtype=float)
        # q_time = np.array([],dtype=float)
        # q_data = np.array([],dtype=float)
        # q_pos_average = np.array([],dtype=float)
        # q_alfven = np.array([],dtype=float)
        # q_in_out = np.array([],dtype=float)
        
        q_pos = np.empty((0,),dtype=float)
        # q_time = np.empty((0,),dtype=float)
        q_time = []
        # q_data = np.empty((0,len(dtype)),dtype=float)
        q_data = []
        q_dif = []
        
        q_pos_average = np.empty((0,),dtype=float)
        q_alfven = np.empty((0,),dtype=float)
        q_in_out = np.empty((0,),dtype=float)
        
        q_starts_full_tmp = np.empty((0,),dtype=float)
        q_ends_full_tmp = np.empty((0,),dtype=float)
        
        # breakpoint()

        non_q_where = []
        q_num = len(q_start_flt)
        enc_q_count=0
        
        for k in range(q_num):
            
            q_where = np.where((time>q_start_flt[k])&(time<q_end_flt[k]))
            q_where = q_where[0]
            
            """!!!"""
            non_q_where.append(q_where)
            """!!!"""
            
            if len(q_where) != 0:
                
                # q_time = np.append(q_time,time[q_where])
                q_time.append(time[q_where])
                q_pos = np.append(q_pos,position[q_where])
                q_dif.append(time_dif[q_where])
                # q_data = np.append(q_data,data[q_where],axis=0)
                q_data.append(data[q_where])
                q_pos_average = np.append(q_pos_average,np.mean(position[q_where]))
                q_in_out = np.append(q_in_out,in_out_arr[q_where])
                # q_alfven = np.append(q_alfven,alfven_vel[q_where])
                q_starts_full_tmp = np.append(q_starts_full_tmp,q_start_flt[k])
                q_ends_full_tmp = np.append(q_ends_full_tmp,q_end_flt[k])
                
            if len(q_where) == 0:
                enc_q_count+=1
        
        q_time = np.concatenate(q_time,axis=0)
        q_data = np.concatenate(q_data,axis=0)
        q_time_dif = np.concatenate(q_dif,axis=0)
        
        non_where = np.where(~np.in1d(time, q_time))
        non_q_time = time[non_where[0]]
        non_q_time_dif = time_dif[non_where[0]]
        non_q_pos = position[non_where[0]]
        non_q_data = data[non_where[0]]
        non_q_in_out = in_out_arr[non_where[0]]
        # non_q_alfven = alfven_vel[non_where[0]]
        


        time_full = np.append(time_full,time)
        q_time_full = np.append(q_time_full,q_time)
        non_q_time_full = np.append(non_q_time_full,non_q_time)
        
        time_dif_full = np.append(time_dif_full,time_dif)
        q_time_dif_full = np.append(q_time_dif_full,q_time_dif)
        non_q_time_dif_full = np.append(non_q_time_dif_full,non_q_time_dif)
        
            
        # time_full.append(time)
        # q_time_full.append(q_time)
        # non_q_time_full.append(non_q_time)
        
        pos_full = np.append(pos_full,position)
        q_pos_full = np.append(q_pos_full,q_pos)
        non_q_pos_full = np.append(non_q_pos_full,non_q_pos)
        
        # data_full = np.append(data_full,data)
        # q_data_full = np.append(q_data_full,q_data)
        # non_q_data_full = np.append(non_q_data_full,non_q_data)
        
        # data_full = np.concatenate([data_full,data],axis=1)
        # q_data_full = np.concatenate([q_data_full,q_data],axis=1)
        # non_q_data_full = np.concatenate([non_q_data_full,non_q_data],axis=1)
        
        data_full.append(data)
        q_data_full.append(q_data)
        non_q_data_full.append(non_q_data)
        
        q_pos_average_full = np.append(q_pos_average_full,q_pos_average)
        
        q_start_full = np.append(q_start_full,q_starts_full_tmp)
        q_end_full = np.append(q_end_full,q_ends_full_tmp)
        
        in_out_full = np.append(in_out_full,in_out_arr)
        q_in_out_full = np.append(q_in_out_full,q_in_out)
        non_q_in_out_full = np.append(non_q_in_out_full,non_q_in_out)
        
        # alfven_full = np.append(alfven_full,alfven_vel)
        # q_alfven_full = np.append(q_alfven_full,q_alfven)
        # non_q_alfven_full = np.append(non_q_alfven_full,non_q_alfven)
        
    # breakpoint()
    # time_full = np.concatenate(time_full, axis=0)
    # q_time_full = np.concatenate(q_time_full,axis=0)
    # non_q_time_full = np.concatenate(non_q_time_full,axis=0)
    
    data_full = np.concatenate(data_full,axis=0)
    q_data_full = np.concatenate(q_data_full,axis=0)
    non_q_data_full = np.concatenate(non_q_data_full,axis=0)
    
    # breakpoint()
    
    #-----------------------------histogram plots----------------------------#
    
    # radial_bins = [[8,15],[15,25],[25,35],[35,45],[8,40]]
    radial_bins = [[8,15],[15,25],[25,35],[35,45],[45,55],[55,65],[65,75]]
    
    
    # radial_bins = [[35,45],[25,35],[15,25],[8,15],[8,40]]
    
    # n_rads = len(radial_bins)-1
    n_rads = 3
    # n_rads = len(radial_bins)
    # n_rads = 1
    
    n_types = len(dtype)
    
    # breakpoint()
    
    # radial_bins = [[1,15],[15,35]]
    # fig = plt.figure(figsize=(10,25))
    # fig = plt.figure(figsize=(20,15))
    fig = plt.figure(figsize=(20,15))
    
    # fig.suptitle(datatype+" Histogram "+title_mod+" Rs",fontsize=36)
    # r_labs_in = ['13.3','15','25','35']
    # r_labs_out = ['15','25','35','45']
    
    r_labs_in = ['13.3','15','25','35','45','55','65']
    r_labs_out = ['15','25','35','45','55','65','75']
    
    r_labs_in = r_labs_in[0:n_rads]
    r_labs_out = r_labs_out[0:n_rads]
    
    r_labs_in.reverse()
    r_labs_out.reverse()
    
    # breakpoint()
    
    if plotstyle == 'radial':
        
        for k in range(n_types):
            # print(k)
            
            # q_data_full = data_types[k]
            # non_q_data_full = data_types[k]
            d_label = data_types[k]
            
            # breakpoint()
            
            q_time_single = q_time_full[:]
            non_q_time_single = non_q_time_full[:]
            
            q_dif_single = q_time_dif_full[:]
            non_q_dif_single = non_q_time_dif_full[:]
            
            q_data_single = q_data_full[:,k]
            non_q_data_single = non_q_data_full[:,k]
            
            n_bin_single = n_bins_list[k]
            drange_single = drange_list[k]
            
            for j in range(n_rads):
                
                rad_bins = radial_bins[j]
                
                q_rad_where = np.where((q_pos_full>rad_bins[0])&(q_pos_full<rad_bins[1]))
                q_rad_where = q_rad_where[0]
                
                non_q_rad_where = np.where((non_q_pos_full>rad_bins[0])&(non_q_pos_full<rad_bins[1]))
                non_q_rad_where = non_q_rad_where[0]
                
                q_ave_where = np.where((q_pos_average_full>rad_bins[0])&(q_pos_average_full<rad_bins[1]))
                q_ave_where = q_ave_where[0]
                
                q_range_num = len(q_ave_where)
                
                title_mod = "for radial distances of "+str(rad_bins[0])+" - "+str(rad_bins[1])
                
                q_times_rad_bins = q_time_single[q_rad_where]
                non_q_times_rad_bins = non_q_time_single[non_q_rad_where]
                
                q_time_dif_rad_bins = q_dif_single[q_rad_where]
                non_q_time_dif_rad_bins = non_q_dif_single[non_q_rad_where]
                
                q_data_rad_bins = q_data_single[q_rad_where]
                non_q_data_rad_bins = non_q_data_single[non_q_rad_where]
                
                q_rad_starts = q_start_full[q_ave_where]
                q_rad_ends = q_end_full[q_ave_where]
                
                # breakpoint()\
                q_inf_where = np.where(q_data_rad_bins != np.inf)
                non_q_inf_where = np.where(non_q_data_rad_bins != np.inf)
                
                q_mean = np.nanmean(q_data_rad_bins[q_inf_where]) 
                # q_mean = np.nanmedian(q_data_rad_bins[q_inf_where])
                q_std = np.nanstd(q_data_rad_bins[q_inf_where])
                non_q_mean = np.nanmean(non_q_data_rad_bins[non_q_inf_where])
                # non_q_mean = np.nanmedian(non_q_data_rad_bins[non_q_inf_where])
                non_q_std = np.nanstd(non_q_data_rad_bins[non_q_inf_where])
                
                # breakpoint()
                # if rad_bins[0] == 8:
                print()
                print("Number of Quiescent Regions: "+str(q_range_num))
                print('Radial Bin: '+str(rad_bins[0])+'-'+str(rad_bins[1]))
                print('Quiescent Duration: '+str(np.sum(q_time_dif_rad_bins)/3600.))
                print(d_label+' Q Average: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)))
                print(d_label+' Non-Q Average: '+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)))
                print()
                
                q_med = np.nanmedian(q_data_rad_bins)
                non_q_med = np.nanmedian(non_q_data_rad_bins)
                
                if span_cut and d_label != '|B| Magnitude':
                    
                    q_data_rad_bins_tmp = np.array([])
                    non_q_data_rad_bins_tmp = np.array([])
                    q_time_dif_rad_bins_tmp = np.array([])

                    # breakpoint()
                    q_counts_new = 0
                    for memes in good_span_times:
                        t0 = memes[0]
                        tf = memes[1]
                        
                        q_span_where = np.where((q_times_rad_bins>t0)&(q_times_rad_bins<tf))
                        q_span_where = q_span_where[0]
                        
                        # print(q_span_where)
                        
                        non_q_span_where = np.where((non_q_times_rad_bins>t0)&(non_q_times_rad_bins<tf))
                        non_q_span_where = non_q_span_where[0]
                        
                        if len(q_span_where) != 0:
                        
                            q_data_rad_bins_tmp = np.append(q_data_rad_bins_tmp,q_data_rad_bins[q_span_where])
                            q_time_dif_rad_bins_tmp = np.append(q_time_dif_rad_bins_tmp,q_time_dif_rad_bins[q_span_where])
                   
                        if len(non_q_span_where) != 0:
                            
                            non_q_data_rad_bins_tmp = np.append(non_q_data_rad_bins_tmp,non_q_data_rad_bins[non_q_span_where])
                            
                        
                        # breakpoint()
                        for jj in range(len(q_rad_starts)):
                            
                            if ((q_rad_starts[jj]>t0)&(q_rad_starts[jj]<tf)) or ((q_rad_ends[jj]>t0)&(q_rad_ends[jj]<tf)):
                                
                                if q_counts_new == len(q_rad_starts): 
                                    pass #just means that all quiescent regions are contributing after cutting for SPAN.
                                         #Passing here avoids double counting as quiescent regions can span *ba dn tsst* two good portions of SPAN but be chopped in the middle.
                                else:
                                    q_counts_new+=1
                                
                    
                        
                    q_data_rad_bins = q_data_rad_bins_tmp
                    non_q_data_rad_bins = non_q_data_rad_bins_tmp
                    q_range_num = q_counts_new
                    q_time_dif_rad_bins = q_time_dif_rad_bins_tmp
                
                
                q_inf_where = np.where(q_data_rad_bins != np.inf)
                non_q_inf_where = np.where(non_q_data_rad_bins != np.inf)
                
                q_mean = np.nanmean(q_data_rad_bins[q_inf_where])
                q_std = np.nanstd(q_data_rad_bins[q_inf_where])
                non_q_mean = np.nanmean(non_q_data_rad_bins[non_q_inf_where])
                non_q_std = np.nanstd(non_q_data_rad_bins[non_q_inf_where])
                
                # breakpoint()
                
                # if rad_bins[0] == 8:
                    
                print("Number of Quiescent Regions: "+str(q_range_num))
                print('Radial Bin: '+str(rad_bins[0])+'-'+str(rad_bins[1]))
                print('Quiescent Duration: '+str(np.sum(q_time_dif_rad_bins)/3600.))
                print(d_label+' Q Average: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)))
                print(d_label+' Non-Q Average: '+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)))
                print()
                
                q_med = np.nanmedian(q_data_rad_bins)
                non_q_med = np.nanmedian(non_q_data_rad_bins)
                
                
                # breakpoint()
                x_datas = [non_q_data_rad_bins,q_data_rad_bins]
        
                labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
                titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
        
                cmaps = ['autumn','autumn']
                colors = ['tab:blue','tab:orange']
                
        
                
                # axs = fig.add_subplot(1,1,1)
                # print(j+k+1)
                index = n_types*j+1+k
                # if k ==0:
                axs = fig.add_subplot(n_rads,n_types,index)
                # axs = fig.add_subplot(n_rads,1,j+1)
                
                bins = np.linspace(drange_single[0],drange_single[1], n_bins_list[k])
                hist1, _ = np.histogram(x_datas[0], bins=bins)
                hist2, _ = np.histogram(x_datas[1], bins=bins)
                
                hist1_norm = hist1 / np.sum(hist1)
                hist2_norm = hist2 / np.sum(hist2)
                
                # Calculate error bars for each bin (normalized)
                error1 = np.sqrt(hist1) / np.sum(hist1)
                error2 = np.sqrt(hist2) / np.sum(hist2)
                
                # Plot histograms with error bars
                axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
                axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
                axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
                axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
                
                # axs.set_ylabel("Normalized Counts",fontsize=32)
                # axs.set_xlabel(datatype,fontsize=32)
                axs.tick_params(axis='x', which='major', labelsize=24)
                
                axs.set_xlim(drange_single)
                # axs.set_xlim((0,175))
                
                if j!=n_rads-1:
                    # axs.tick_params('x', labelbottom=False)
                    axs.set_xticks([])
                    
                # axs.tick_params('y', labelbottom=False)
                axs.set_yticks([])
                
                if j==0:
                    # x_pos = 0.1
                    x_pos = 0.4
                else:
                    x_pos = 0.4
                
                # axs.text(x_pos,0.65,"Number of Quiescent Regions: "+str(q_range_num),fontsize=32,transform=axs.transAxes)
                # axs.text(x_pos,0.55,"Quiescent Mean: "+str(round(q_mean,2))+' ± '+str(round(q_std,2)),fontsize=32,transform=axs.transAxes)
                # axs.text(x_pos,0.45,"Non-Quiescent Mean: "+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)),fontsize=32,transform=axs.transAxes)
                # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*4.4,"Quiescent Median: "+str(q_med),fontsize=32)
                # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*5.4,"Non-Quiescent Median: "+str(non_q_med),fontsize=32)
                # 
                # plt.subplots_adjust(wspace=0.01, hspace=0)
                
                if index==1:
                    leg = axs.legend(fontsize=20,loc='upper right',markerscale=5)
                plt.tight_layout()
                
                # fig.text(-0.035,0.25*k+0.14, (r_labs_out[j]+" Rs \n - \n "+r_labs_in[j]+" Rs"), ha="center", va="center", rotation=12,fontsize=32)
                
                fig.text(-0.04,(1/n_rads)*j+0.14, (r_labs_out[j]+" $R_{\odot}$ \n - \n "+r_labs_in[j]+" $R_{\odot}$"), ha="center", va="center", rotation=12,fontsize=32)
                

            
            fig.text((1/n_types)*k+0.13,1.015, data_types[k], ha="center", va="center",fontsize=32)
            fig.text((1/n_types)*k+0.13,-0.015, units[k], ha="center", va="center",fontsize=32)
            
            # fig.text((1/n_types)*k+0.5,1.015, data_types[k], ha="center", va="center",fontsize=32)
            # fig.text((1/n_types)*k+0.5,-0.015, units[k], ha="center", va="center",fontsize=32)
            # plt.show()
        
        # fig.text(-0.035,0.25*3+0.14, "45 Rs \n - \n 35 Rs", ha="center", va="center", rotation=12,fontsize=32)
        fig.text(-0.1,0.525, "Radial Distance", ha="center", va="center", rotation=90,fontsize=38)
        fig.text(0.5,1.05, "Quiescent Region Property Histograms", ha="center", va="center",fontsize=32)
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        
        if save:
            save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/Plots/Histograms/'+file_mod+'/'
            save_name = 'quiescent_'+dtype+'_histogram_'+str(rad_bins[0])+"_to_"+str(rad_bins[1])+'_Rs.png'
            
            if not os.path.exists(save_path):
                print('should work')
                os.makedirs(save_path)
            
            plt.savefig(save_path+save_name)
        
        else:
            plt.show()
            
        plt.clf()
        plt.cla()
        plt.close('all')
        plt.close(fig)
        
        # breakpoint()
    
    elif plotstyle == 'in/out':
        for j in range(len(radial_bins)):
        # for j in range(1):
            
            rad_bins = radial_bins[j]
            
            q_rad_where = np.where((q_pos_full>rad_bins[0])&(q_pos_full<rad_bins[1]))
            q_rad_where = q_rad_where[0]
            
            non_q_rad_where = np.where((non_q_pos_full>rad_bins[0])&(non_q_pos_full<rad_bins[1]))
            non_q_rad_where = non_q_rad_where[0]
            
            q_ave_where = np.where((q_pos_average_full>rad_bins[0])&(q_pos_average_full<rad_bins[1]))
            q_ave_where = q_ave_where[0]
            
            q_range_num = len(q_ave_where)
            
            rad_where = np.where((pos_full>rad_bins[0])&(pos_full<rad_bins[1]))
            rad_where = rad_where[0]
            
            title_mod = "for radial distances of "+str(rad_bins[0])+" - "+str(rad_bins[1])
            
            #----------------------sorting data by inbound and outbound---------------------------#
            
            # q_data_rad_bins = q_data_full[q_rad_where]
            # non_q_data_rad_bins = non_q_data_full[non_q_rad_where]
            # q_in_out_rad_bins = q_in_out_full[q_rad_where]
            # non_q_in_out_rad_bins = non_q_in_out_full[non_q_rad_where]
            
            # q_in_where = np.where(q_in_out_rad_bins==0)
            # q_in_where = q_in_where[0]
            # q_out_where = np.where(q_in_out_rad_bins==1)
            # q_out_where = q_out_where[0]
            
            # non_q_in_where = np.where(non_q_in_out_rad_bins==0)
            # non_q_in_where = non_q_in_where[0]
            # non_q_out_where = np.where(non_q_in_out_rad_bins==1)
            # non_q_out_where = non_q_out_where[0]
            
            # in_datas = [non_q_data_rad_bins[non_q_in_where],q_data_rad_bins[q_in_where]]
            # out_datas = [non_q_data_rad_bins[non_q_out_where],q_data_rad_bins[q_out_where]]
            
            data_rad_bins = data_full[rad_where]
            in_out_rad_bins = in_out_full[rad_where]
            
            in_where =  np.where(in_out_rad_bins==0)
            in_where = in_where[0]
            out_where =  np.where(in_out_rad_bins==1)
            out_where = out_where[0]
            
            in_datas = [data_rad_bins[in_where],data_rad_bins[out_where]]
            
            # breakpoint()
            
            in_mean = np.nanmean(data_rad_bins[in_where])
            out_mean = np.nanmean(data_rad_bins[out_where])
            tot_mean = np.nanmean([in_mean,out_mean])
            
    
            labels = ['Inbound','Outbound']
            # labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
    
            cmaps = ['autumn','autumn']
            colors = ['tab:blue','tab:orange']
            
            #----------------------create plots---------------------------#
            
            fig = plt.figure(figsize=(20,15))
            
            # fig.suptitle(datatype+" Histogram "+title_mod+" Rs",fontsize=36)
            
            # axs1 = fig.add_subplot(2,1,1)
            axs1 = fig.add_subplot(1,1,1)
            
            bins = np.linspace(drange[0],drange[1], n_bins)
            hist1, _ = np.histogram(in_datas[0], bins=bins)
            hist2, _ = np.histogram(in_datas[1], bins=bins)
            
            hist1_norm = hist1 / np.sum(hist1)
            hist2_norm = hist2 / np.sum(hist2)
            
            # Calculate error bars for each bin (normalized)
            error1 = np.sqrt(hist1) / np.sum(hist1)
            error2 = np.sqrt(hist2) / np.sum(hist2)
            
            # Plot histograms with error bars
            axs1.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
            axs1.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
            axs1.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
            axs1.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
            
            axs1.set_ylabel("Normalized Counts",fontsize=32)
            # axs1.set_xlabel(datatype,fontsize=32)
            axs1.tick_params(axis='both', which='major', labelsize=32)
            axs1.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm),'Total Mean: '+str(tot_mean),fontsize=22)
            
            # axs1.set_title('Inbound',fontsize=32)
            leg = axs1.legend(fontsize=20,loc='upper right',markerscale=5)
            
            # axs2 = fig.add_subplot(2,1,2)
            
            # bins = np.linspace(drange[0],drange[1], n_bins)
            # hist1, _ = np.histogram(out_datas[0], bins=bins)
            # hist2, _ = np.histogram(out_datas[1], bins=bins)
            
            # hist1_norm = hist1 / np.sum(hist1)
            # hist2_norm = hist2 / np.sum(hist2)
            
            # # Calculate error bars for each bin (normalized)
            # error1 = np.sqrt(hist1) / np.sum(hist1)
            # error2 = np.sqrt(hist2) / np.sum(hist2)
            
            # # Plot histograms with error bars
            # axs2.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
            # axs2.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
            # axs2.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
            # axs2.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
            
            # axs2.set_ylabel("Normalized Counts",fontsize=32)
            # axs2.set_xlabel(datatype,fontsize=32)
            # axs2.tick_params(axis='both', which='major', labelsize=32)
            # axs2.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm),"Number of Quiescent Regions: "+str(q_range_num),fontsize=32)
            # axs2.set_title('Outbound',fontsize=32)
            
            print('Inbound Mean: '+str(in_mean),'Outbound Mean: '+str(out_mean),'Total Mean: '+str(tot_mean))
            
            plt.subplots_adjust(wspace=0.01, hspace=0)
            plt.tight_layout()
            
            # plt.show()
    
            if save:
                save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/Plots/Histograms/'+file_mod+'/in_out/'
                save_name = 'quiescent_'+dtype+'_histogram_'+str(rad_bins[0])+"_to_"+str(rad_bins[1])+'_Rs_in_out.png'
                
                if not os.path.exists(save_path):
                    print('should work')
                    os.makedirs(save_path)
                
                plt.savefig(save_path+save_name)
            
            else:
                plt.show()
            
            plt.clf()
            plt.cla()
            plt.close('all')
            plt.close(fig)
        
def dura_r_plot(enc='all',enc_radius=67):
    
        
    #----------------------Organizing CSVs for in Data-------------------------#
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    csv_names = os.listdir(csv_path)
    csv_names.remove('.DS_Store')
    
    enc_tag = []
    for i in csv_names:
        
        a = re.findall(r'\d+',i)
        enc_tag.append(int(a[0]))
    
    csv_names = sort_list(csv_names,enc_tag)
    enc_tag = sort_list(enc_tag,enc_tag)
    
    filelist = []
    for j in csv_names:
        j = csv_path+j
        filelist.append(j)
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,14))
        
    if enc == 'no 13':
        enc = list(range(1,13))
        
    if enc == 'no 1':
        enc = list(range(2,14)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        
    if type(enc) is int:
        enc = [enc]
    
    filelist_new = [filelist[i-1] for i in enc]
    
    #--------------------------Reading in Quiescent Data---------------------------#
    
    df_list = [pd.read_csv(file) for file in filelist_new]
    
    df = pd.concat(df_list)
    
    q_starts = df['start dates'].to_numpy()
    q_ends = df['end dates'].to_numpy()
    q_duras = df['duration'].to_numpy()
    
    q_start_flt = pys.time_float(q_starts)
    q_end_flt = pys.time_float(q_ends)
    
    #--------------------------Start importing PSP Data-----------------------------#
    
    hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    shape_arr = []
    datafull = np.array([])
    Rfull = np.array([])
    timefull = np.array([])
    
    for i in enc:
        
        # print(i)
        
        enc_ind = (i-1)
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True)
        pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
        
        
        pos_time_arr = pos_data[0]
        pos_data_arr = pos_data[1]
        
        # pos_time_arr = pos_time_arr[0:len(pos_time_arr)]
        
        x = pos_data_arr[:,0]
        y = pos_data_arr[:,1]
        z = pos_data_arr[:,2]
        
        R = (np.sqrt(x**2+y**2+z**2)-Rs)/Rs #parker spiral source height
        R_true = np.sqrt(x**2+y**2+z**2)/Rs #height from center of sun
        

        Rfull = np.append(Rfull,R)
        timefull = np.append(timefull,pos_time_arr)
        
        # breakpoint()
        
    # print(shape_arr)
    where_ap = np.array([],dtype='int64')
    R_averages = []
    for k in range(len(q_starts)):
        ap = np.where((timefull>q_start_flt[k])&(timefull<=q_end_flt[k]))
        ap = ap[0]
        where_ap = np.append(where_ap,ap)
        R_tmp = Rfull[ap]
        R_avg = np.nanmean(R_tmp)
        R_averages.append(R_avg)
        
    breakpoint()
    
def brazil_analysis(span_cut=True, bin_num=28): #made a seperate plotting routine to make analysis of the brazil plots easier without needing to change the larger brazil script

    fp = findpeaks()

    radial_bins = [[8,15],[15,25],[25,35],[35,45]]
    # radial_bins = [[8,15],[15,25],[25,35]]
    # radial_bins = [[8,15],[15,20],[20,25],[25,30],[30,35]]

    n_rads = len(radial_bins)
    r_labs_in = ['13.3','15','25','35']
    r_labs_out = ['15','25','35','45']
    
    # r_labs_in = ['13.3','15','25']
    # r_labs_out = ['15','25','35']
    
    # r_labs_in = ['8','15','20','25','30']
    # r_labs_out = ['15','20','25','30','35']
    
    r_labs_in.reverse()
    r_labs_out.reverse()
    
    lab_pos = [0.21,0.4,0.6,0.78]
    # lab_pos = [0.24,0.5,0.76]
    title_pos= [0.32, 2*0.35]
    fig = plt.figure(figsize=(10,15))
    
    gs=gridspec.GridSpec(n_rads,3, width_ratios=[5,5,0.2])
    
    # breakpoint()1
    
    #------------------cut out bad SPAN times------------------#
    
    # span_time = []
    # span_flag = []
    # span_flag_av = []
    # span_flag_ratio = []
    # span_rad = []
    good_span_times = []
    
    for pepe in range(1,17):
    
        span_check_savename = 'SPAN_ion_fov_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/FOV flags/'
        
        pyt.tplot_restore(span_check_savepath+span_check_savename)
    
        # fov_flag_exact = pyt.get_data('phi_fov')
        # fov_time = fov_flag_exact[0]
        # fov_flag = fov_flag_exact[1]
        
        fov_flag_average = pyt.get_data('phi_fov_average')
        fov_av_time = fov_flag_average[0]
        fov_flag_av = fov_flag_average[1]
        
        # fov_flag_ratio = pyt.get_data('phi_fov_ratio')
        # fov_av_time = fov_flag_ratio[0]
        # fov_ratio_av = fov_flag_ratio[1]
        
        # rad = pyt.get_data('psp_radial_dist_Rs')
        # rad_time = rad[0]
        # rad_Rs = rad[1]
        
        # span_time.append(fov_time)
        # span_flag.append(fov_flag)
        # span_flag_av.append(fov_flag_av)
        # span_flag_ratio.append(fov_ratio_av)
        # span_rad.append(rad_Rs)
        
        span_qtn_check_savename = 'SPAN_SPC_QTN_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_qtn_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
        
        pyt.tplot_restore(span_qtn_check_savepath+span_qtn_check_savename)
        
        qtn_flag_data = pyt.get_data('SPAN_qual_flag')
        qtn_flag_time = qtn_flag_data[0]
        qtn_flag = qtn_flag_data[1]
        
        # FOV_where = np.where(fov_flag_av==0)
        # FOV_where = FOV_where[0]
        
        # good_fov = qtn_flag_time[FOV_where]
        
        # qtn_in_where = np.where(np.isin(qtn_flag_time,good_fov))
        # qtn_in_where = qtn_in_where[0]
        
        # add one to places where FOV is good to make sure that the score for good FOV is 2.
        # qtn_flag[qtn_in_where]+=1 
        
        # print(pepe," : ",qtn_flag.max()," : ", fov_flag_av.max())
        
        # if pepe > 2:
            # breakpoint()
        
        good_where = qtn_flag<2
        
        #a little confusing but I'm grouping the zeros so I actually want to invert my bool array to have good times be 0s
        # good_flip = 1-good_where
        
        edge_indices = group_zeros(good_where)
        
        # edge_times = []
        for i in edge_indices:
            t0s = qtn_flag_time[i[0]]
            tfs = qtn_flag_time[i[1]]
            
            # edge_times.append((t0,tf))
            
            good_span_times.append((t0s,tfs))
        
        
        pyt.del_data()
        

    #----------------------------------------------------------#
    
    
    q_peaks = []
    non_q_peaks = []
    x_errs_q = []
    y_errs_q = []
    x_errs_non_q = []
    y_errs_non_q = []
    
    
    for j in range(len(radial_bins)): 
    
        bins = radial_bins[j]
        
        tplot_savename = 'Brazil_data_'+str(bins[0])+'_to_'+str(bins[1])+'_Rs.cdf'
        tplot_savepath = '/Users/besh2109/Desktop/BrazilPlots/'   
        
        pyt.tplot_restore(tplot_savepath+tplot_savename)

        q_brazil = pyt.get_data("q_region_times")
        q_times_rad_bins = q_brazil[0]
        
        non_q_brazil = pyt.get_data("non_q_region_times")
        non_q_times_rad_bins = non_q_brazil[0]

        q_brazil = pyt.get_data("q_region_brazil")
        q_beta_rad_bins = q_brazil[0]
        q_anis_rad_bins = q_brazil[1]
        
        non_q_brazil = pyt.get_data("non_q_region_brazil")
        non_q_beta_rad_bins = non_q_brazil[0]
        non_q_anis_rad_bins = non_q_brazil[1]
        
        q_color = pyt.get_data("q_color_scale")
        q_alfvenicity_rad_bins = q_color[0]
        
        non_q_color = pyt.get_data("non_q_color_scale")
        non_q_alfvenicity_rad_bins = non_q_color[0]
        
        
        if span_cut:
            
            q_anis_rad_bins_tmp = np.array([])
            q_beta_rad_bins_tmp = np.array([])
            q_alfvenicity_rad_bins_tmp = np.array([])
            
            non_q_anis_rad_bins_tmp = np.array([])
            non_q_beta_rad_bins_tmp = np.array([])
            non_q_alfvenicity_rad_bins_tmp = np.array([])
            
            for memes in good_span_times:
                t0 = memes[0]
                tf = memes[1]
                
                q_span_where = np.where((q_times_rad_bins>t0)&(q_times_rad_bins<tf))
                q_span_where = q_span_where[0]
                
                non_q_span_where = np.where((non_q_times_rad_bins>t0)&(non_q_times_rad_bins<tf))
                non_q_span_where = non_q_span_where[0]
                
                if len(q_span_where) != 0:
                
                    q_anis_rad_bins_tmp = np.append(q_anis_rad_bins_tmp,q_anis_rad_bins[q_span_where])
                    q_beta_rad_bins_tmp = np.append(q_beta_rad_bins_tmp,q_beta_rad_bins[q_span_where])
                    q_alfvenicity_rad_bins_tmp = np.append(q_alfvenicity_rad_bins_tmp,q_alfvenicity_rad_bins[q_span_where])
           
                if len(non_q_span_where) != 0:
                    
                    non_q_anis_rad_bins_tmp = np.append(non_q_anis_rad_bins_tmp,non_q_anis_rad_bins[non_q_span_where])
                    non_q_beta_rad_bins_tmp = np.append(non_q_beta_rad_bins_tmp,non_q_beta_rad_bins[non_q_span_where])
                    non_q_alfvenicity_rad_bins_tmp = np.append(non_q_alfvenicity_rad_bins_tmp,non_q_alfvenicity_rad_bins[non_q_span_where])
                
            q_anis_rad_bins = q_anis_rad_bins_tmp
            q_beta_rad_bins = q_beta_rad_bins_tmp
            q_alfvenicity_rad_bins = q_alfvenicity_rad_bins_tmp
            
            non_q_anis_rad_bins = non_q_anis_rad_bins_tmp
            non_q_beta_rad_bins = non_q_beta_rad_bins_tmp
            non_q_alfvenicity_rad_bins = non_q_alfvenicity_rad_bins_tmp
        
        
        y_datas = [q_anis_rad_bins,non_q_anis_rad_bins]
        x_datas = [q_beta_rad_bins,non_q_beta_rad_bins]
        c_datas = [q_alfvenicity_rad_bins,non_q_alfvenicity_rad_bins]
        
        labels = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']
        titles = ['Quiscent Solar Wind','Non-Quiescent Solar Wind']

        cmaps = ['autumn','autumn']
        colors = ['tab:blue','tab:blue']
        
        inst_x = np.logspace(np.log10(0.0001), np.log10(50.0),200)
        
        a1 = 0.65
        b1 = 0.4
        beta01 = -0.0004
        inst_anis_ion_cyc = (1 + a1/(inst_x-beta01)**b1) #Ion Cyclotron Instability, Hellinger 2006
        
        a2 = 0.77
        b2 = 0.76
        beta02 = -0.016
        # beta02 = 0
        inst_anis_mirror = (1 + a2/(inst_x-beta02)**b2) #Mirror Instability, Hellinger 2006
        
        a3 = -0.47
        b3 = 0.53
        beta03 = 0.59
        inst_anis_par_firehose = (1 + a3/(inst_x-beta03)**b3) #Oblique Firehose Instability, Hellinger 2006
         
        a4 = -1.4
        b4 = 1
        beta04 = -0.11
        inst_anis_obl_firehose = (1 + a4/(inst_x-beta04)**b4) #Oblique Firehose Instability, Hellinger 2006

        
        for ii in range(2):
        
            axs = fig.add_subplot(gs[3*j+ii])
            
            # print(3*j+ii)
        
            y_space = np.logspace(np.log10(0.1), np.log10(10.0), bin_num)
            # x_space = np.logspace(np.log10(0.0001), np.log10(30.0), bin_num)
            x_space = np.logspace(np.log10(0.0001), np.log10(30.0), bin_num)
            hist2d,xedge,yedge = np.histogram2d(x_datas[ii],y_datas[ii], bins=(x_space,y_space))
            
            # hist2d,xedge,yedge = np.histogram2d(q_beta_par,q_anis_ben, bins=(x_space,y_space))
            # hist2d,xedge,yedge = np.histogram2d(beta_par,anis_ben, bins=(x_space,y_space))
            # breakpoint()
            
            histo = np.transpose(hist2d)
            
            # breakpoint()
            peaks = fp.fit(histo)
            peak_array = np.array(peaks['persistence'])
            main_peak = peak_array[0]
            main_peak_x = main_peak[0] #x-coordinate of the main peak
            main_peak_y = main_peak[1] #y-coordinate of the main peak
  
            main_peak_x_hist = histo[main_peak_y,:] #nets you the 1d histogram along the x axis at peak y-position
            main_peak_y_hist = histo[:,main_peak_x] #nets you the 1d histogram along the y axis at peak x-position
            
            # breakpoint()
            
            xcenters = (xedge[:-1] + xedge[1:]) / 2
            ycenters = (yedge[:-1] + yedge[1:]) / 2
            
            spline_x = UnivariateSpline(xcenters, main_peak_x_hist-np.max(main_peak_x_hist)/2, s=0) #find FWHM of x 
            spline_y = UnivariateSpline(ycenters, main_peak_y_hist-np.max(main_peak_y_hist)/2, s=0) #   and y coordinates
            
            r1x, r2x = spline_x.roots()[:2]
            r1y, r2y = spline_y.roots()[:2]
            
            # breakpoint()
            
            if ii == 0:
                q_peaks.append([xcenters[main_peak_x],ycenters[main_peak_y]])
                x_errs_q.append([r1x,r2x])
                y_errs_q.append([r1y,r2y])
                # print([xcenters[main_peak_x],ycenters[main_peak_y]])
            else:
                non_q_peaks.append([xcenters[main_peak_x],ycenters[main_peak_y]])
                x_errs_non_q.append([r1x,r2x])
                y_errs_non_q.append([r1y,r2y])
    
            r_contour = axs.contour(xcenters,ycenters,histo, cmap=cmaps[ii], vmin=20, levels=6,zorder=2)
            # cb = fig.colorbar(r_contour)
            # fig1.colorbar(CS)
    
            delete_contour(r_contour,1)
            sct = axs.scatter(x_datas[ii],y_datas[ii],marker=',',c=c_datas[ii],s=1,zorder=1,cmap='rainbow',vmax=6)
            # axs.scatter(x_datas[ii],y_datas[ii],marker=',',color=colors[ii],s=1,zorder=1)
            
            axs.plot(inst_x,inst_anis_ion_cyc,linestyle='-',zorder=3,color='black',label='Ion Cyclotron Instability')
            axs.plot(inst_x,inst_anis_mirror,linestyle='-',zorder=4,color='blue',label='Mirror Instability')
            axs.plot(inst_x,inst_anis_par_firehose,linestyle='-',zorder=5,color='lime',label='Parallel Firehose')
            # axs.plot(inst_x,inst_anis_obl_firehose,linestyle='-',zorder=6,color='firebrick',label='Oblique Firehose')
  
            # axs.axvline(x = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
            # axs.axhline(y = 1, color = 'grey', linestyle = 'dashed', label = 'axvline - full height')
            # axs.scatter(q_beta_par_full,q_anis_ben_full,marker=',',color='tab:orange',label='Quiescent Solar Wind',s=1)
            
            # axs.set_title(titles[ii],fontsize=32)
            # axs.set_ylabel('Temperature Anisotropy',fontsize=32)
            # axs.set_xlabel('Plasma Beta Parallel',fontsize=32)

            axs.tick_params(axis='both', which='major', labelsize=24)
            axs.set_xlim(0.0008,30)
            axs.set_ylim(0.08,12)

            axs.set_xscale("log")
            axs.set_yscale("log")
            axs.grid()
            
            axs.tick_params(width=3, length=7)
            
            if j == 0:
                fig.text(title_pos[ii],0.888, (labels[ii]), ha="center", va="center",fontsize=22)
                # fig.text(0.5,0.91,'Brazil Plots vs Rs, Bins: '+str(bin_num),ha="center", va="center",fontsize=28)
                fig.text(0.5,0.91,'Brazil Plots vs Spacecraft Position in $R_{\odot}$',ha="center", va="center",fontsize=28)
                fig.text(0.5,0.08,'Plasma Beta Parallel ($β_{\parallel p}$)',ha="center", va="center",fontsize=22)
                fig.text(0.01,0.5,'Proton Temperature Anisotropy ($\mathcal{R}$)',ha="center", va="center", rotation=90,fontsize=24)
                
            
            if ii ==1:
                for tick in axs.yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
                axs.set_yticklabels([])
                axs.set_ylabel('')
                
            if j!=n_rads-1:
                for tick in axs.xaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
                axs.set_xticklabels([])
                axs.set_xlabel('')
            
        
        fig.text(-0.1,lab_pos[j], (r_labs_out[j]+" $R_{\odot}$ \n - \n "+r_labs_in[j]+" $R_{\odot}$"), ha="center", va="center", rotation=12,fontsize=24)
        
        if j==1:
        
            
            cbar_ax = fig.add_axes([0.90, 0.15, 0.012, 0.7])
            # fig.colorbar(im, cax=cbar_ax)
        
            # ax3 = plt.subplot(gs[3*j+ii+1])
            
            cb = fig.colorbar(sct,cax=cbar_ax)
            cb.set_label("Alfvén Mach Number $V_{sw}/V_{Alf}$",fontsize=20)
            for t in cb.ax.get_yticklabels():
                  t.set_fontsize(20)
            
        
    plt.subplots_adjust(wspace=0.02, hspace=0.01)
    leg = axs.legend(fontsize=16,loc='upper right',markerscale=5)
    
    plt.show()
    
    plt.clf()
    plt.cla()
    plt.close('all')
    plt.close(fig)
    
    q = np.array(q_peaks)
    non_q = np.array(non_q_peaks)
    
    q_x_errs = np.array(x_errs_q)
    q_y_errs = np.array(y_errs_q)
    
    non_q_x_errs = np.array(x_errs_non_q)
    non_q_y_errs = np.array(y_errs_non_q)
    
    q_x_err_mins = []
    q_x_err_maxs = []
    for i in range(len(q[:,0])):
        errs = abs(q_x_errs[i,:] - q[i,0])
        q_x_err_mins.append(errs[0])
        q_x_err_maxs.append(errs[1])
        
    q_x_err = [q_x_err_mins,q_x_err_maxs]
        
    q_y_err_mins = []
    q_y_err_maxs = []
    for i in range(len(q[:,1])):
        errs = abs(q_y_errs[i,:] - q[i,1])
        q_y_err_mins.append(errs[0])
        q_y_err_maxs.append(errs[1])
        
    q_y_err = [q_y_err_mins,q_y_err_maxs]
        
    non_q_x_err_mins = []
    non_q_x_err_maxs = []
    for i in range(len(non_q[:,0])):
        errs = abs(non_q_x_errs[i,:] - non_q[i,0])
        non_q_x_err_mins.append(errs[0])
        non_q_x_err_maxs.append(errs[1])
        
    non_q_x_err = [non_q_x_err_mins,non_q_x_err_maxs]
        
    non_q_y_err_mins = []
    non_q_y_err_maxs = []
    for i in range(len(non_q[:,1])):
        errs = abs(non_q_y_errs[i,:] - non_q[i,1])
        non_q_y_err_mins.append(errs[0])
        non_q_y_err_maxs.append(errs[1])
        
    non_q_y_err = [non_q_y_err_mins,non_q_y_err_maxs]
    
    # breakpoint()
    
    fig = plt.figure(figsize=(15,15))
    
    axs = fig.add_subplot(111)
    
    axs.plot(inst_x,inst_anis_ion_cyc,linestyle='-',zorder=3,color='black',label='Ion Cyclotron Instability')
    axs.plot(inst_x,inst_anis_mirror,linestyle='-',zorder=4,color='blue',label='Mirror Instability')
    axs.plot(inst_x,inst_anis_par_firehose,linestyle='-',zorder=5,color='lime',label='Parallel Firehose')
    
    axs.plot(q[:,0],q[:,1],marker='^',zorder=6,color='tab:orange',label='Q region peak')
    axs.plot(non_q[:,0],non_q[:,1],marker='s',zorder=6,color='tab:blue',label='Non Q peak')
    
    axs.errorbar(q[:,0],q[:,1],xerr=q_x_err,yerr=q_y_err,capsize=3,color='orange')
    axs.errorbar(non_q[:,0],non_q[:,1],xerr=non_q_x_err,yerr=non_q_y_err,capsize=3,color='deepskyblue')
    
    # breakpoint()
    
    #--plot Enc 13 ICME wake--#
    
    wake_peak = [0.0021394602383593133, 0.5073189320733875]
    wake_x_errs =[[0.001612548024810383], [0.004481322296380754]]
    wake_y_errs = [[0.3780150197545924], [0.6378177701428402]]
    
    axs.plot(wake_peak[0],wake_peak[1],marker='8',zorder=6,color='tab:purple',label='Enc 13 ICME Wake')
    axs.errorbar(wake_peak[0],wake_peak[1],xerr=wake_x_errs,yerr=wake_y_errs,capsize=3,color='purple')
    
    #-------------------------#
    
    
    axs.tick_params(axis='both', which='major', labelsize=24)
    axs.set_xlim(0.0008,30)
    axs.set_ylim(0.08,12)

    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.grid()
    
    # axs.set_title('Evolution of Solar Wind vs Distance, Bins: '+str(bin_num),fontsize=32)
    axs.set_title('Evolution of Solar Wind vs Spacecraft Position in $R_{\odot}$',fontsize=32)
    axs.set_ylabel('Temperature Anisotropy ($\mathcal{R}$)',fontsize=26)
    axs.set_xlabel('Plasma Beta Parallel ($β_{\parallel p}$)',fontsize=26)
    
    axs.tick_params(width=3, length=7)
    
    plt.subplots_adjust(wspace=0.02, hspace=0.01)
    leg = axs.legend(fontsize=16,loc='upper right',markerscale=3)
    
    plt.show()
    
def delb_b(enc='all',enc_radius=65,save=False):
    
    print('calculating ∂B/B')
    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    #kb = 1.380649*10e-23 #boltzmann constant, J/K
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,15))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(1,13))
        enc.extend(range(14,17))
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc = list(range(2,15)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
        
    elif type(enc) is str and len(enc) == 1:
        enc = int(enc)
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    #----------------------Organizing CSVs for in Data-------------------------#
    
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    
    #--------------------------Start importing PSP Data-----------------------------#
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    duration_list = []
    
    non_q_dB_B = []
    non_q_t_scale = []
    
    q_dB_B = []
    q_t_scale = []
    
    q_rs = []
    nq_rs = []
    
    
    for i in enc:
        
        csv_name = 'enc_'+str(i)+'_regions_raw.csv'
        
        q_df = pd.read_csv(csv_path+csv_name) #quiescent region Pandas dataframe
        
        q_starts = q_df['start dates'].to_numpy()
        q_ends = q_df['end dates'].to_numpy()
        q_duras = q_df['duration'].to_numpy()
        
        q_start_flt = np.array(pys.time_float(q_starts))
        q_end_flt = np.array(pys.time_float(q_ends))
        
        # q_counts+=len(q_starts) #count number of q regions

        enc_ind = (i-1)
        
        perihelion_flt = per_flt[enc_ind] #float date of perihelion
        perihelion = pys.time_string(perihelion_flt) #string date of perihelion
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        # radius_mod = '<'+str(enc_radius)
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        # if not tmp_check:
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        
        pos_data = pyt.get_data('psp_spi_SUN_DIST')
        pos_time = pos_data[0]
        pos_rs = pos_data[1]/Rs
        
        psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN_4_Sa_per_Cyc',last_version=True,username=fields_id,password=fields_pass)
        # psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN',last_version=True,username=fields_id,password=fields_pass)
        mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
        # mag_data = pyt.get_data('psp_fld_l2_mag_RTN')
        
        mag_time_arr = mag_data[0]
        mag_data_arr = mag_data[1]
        
        Br = mag_data_arr[:,0]
        Bt = mag_data_arr[:,1]
        Bn = mag_data_arr[:,2]
        
        #-------------get rid of enc 13 ICME---------------#
        
        icme_start = pys.time_float('2022-09-06/15:25')
        icme_end = pys.time_float('2022-09-07/12:25')
        
        # icme_where = np.where(((mag_time_arr>icme_start)&(mag_time_arr<icme_end)))
        icme_where = np.where(~((mag_time_arr>icme_start)&(mag_time_arr<icme_end)))
        icme_where = icme_where[0]
        icme_where_check = len(icme_where)
        
        if icme_where_check > 0:
            
            mag_time_arr = mag_time_arr[icme_where]
            # position = position[icme_where]
            # data = data[icme_where]
            Br = Br[icme_where]
            Bt = Bt[icme_where]
            Bn = Bn[icme_where]
            
            
        #-----------------------------------------------#
        
        q_inds = []
        cut_times = []
        
        for j in range(len(q_starts)):
            
            q_time_where = np.where((mag_time_arr>q_start_flt[j]) & (mag_time_arr<q_end_flt[j]))
            q_inds.extend(q_time_where[0])
            q_time_tmp = mag_time_arr[q_time_where]
            cut_times.extend(q_time_tmp)
            q_Br_tmp = Br[q_time_where]
            q_Bt_tmp = Bt[q_time_where]
            q_Bn_tmp = Bn[q_time_where]
            
            q_δΒr = q_Br_tmp - np.nanmean(q_Br_tmp)
            q_δBt = q_Bt_tmp - np.nanmean(q_Bt_tmp)
            q_δBn = q_Bn_tmp - np.nanmean(q_Bn_tmp)
            
            q_δB = np.sqrt(np.nanmean(q_δΒr**2+q_δBt**2+q_δBn**2))
            q_B_mag = np.sqrt(q_Br_tmp**2+q_Bt_tmp**2+q_Bn_tmp**2)
            
            q_B_mag_av = np.nanmean(q_B_mag)
            
            q_dB_B.append(q_δB/q_B_mag_av)
            q_t_scale.append(q_duras[j])
            
            q_time_span_where = np.where((pos_time>q_start_flt[j]) & (pos_time<q_end_flt[j]))
            q_time_span_where = q_time_span_where[0]
            q_rs_pos = pos_rs[q_time_span_where]
            q_rs.append(np.nanmean(q_rs_pos))

        non_q_time_arr = np.delete(mag_time_arr,q_inds)
        non_q_Br = np.delete(Br,q_inds)
        non_q_Bt = np.delete(Bt,q_inds)
        non_q_Bn = np.delete(Bn,q_inds)
        
        
        n = 2
        iii = 0
        while iii<n:
            
            for j in range(len(q_starts)):
                
                incheck = [True]
                ii = 0
                while any(incheck): #checking for data that is not in the quiescent region list
                    random_index = np.random.choice(non_q_time_arr)
                    non_q_where = np.where((mag_time_arr>random_index) & (mag_time_arr<(random_index+q_duras[j])))
                    non_q_time_tmp = mag_time_arr[non_q_where]
                    incheck = np.isin(non_q_time_tmp,cut_times)
                    
                    non_q_span_where = np.where((pos_time>random_index) & (pos_time<(random_index+q_duras[j])))
                    non_q_span_time_tmp = pos_time[non_q_span_where]
                    
                    
                    if ii >= 15: #if the random check is taking too long, then just look for spots where data of the proper length can be found
                        zeros = np.zeros(mag_time_arr.shape)
                        index_arr = np.array(range(len(mag_time_arr)))
                        del_where = np.where(np.isin(mag_time_arr,cut_times))
                        del_inds = del_where[0]
                        # time_check = np.delete(mag_time_arr,del_inds)
                        # ind_check = np.delete(index_arr,del_inds)
                        zeros[del_where] = 1
                        groups = np.array(group_zeros(zeros))
                        times = mag_time_arr[groups]
                        
                        lengths = []
                        for ijk in mag_time_arr[groups]:
                            lengths.extend(np.diff(ijk))
                            
                        lengthwhere = np.where(lengths>q_duras[j])
                        lengthwhere = lengthwhere[0]
                        
                        group_tmp = groups[lengthwhere]
                        times_tmp = times[lengthwhere]
                        
                        random_index = np.random.choice(np.arange(len(group_tmp)))
                        time_index = times_tmp[random_index,0]
                        
                        non_q_where = np.where((mag_time_arr>time_index) & (mag_time_arr<(time_index+q_duras[j])))
                        non_q_time_tmp = mag_time_arr[non_q_where]
                        incheck = np.isin(non_q_time_tmp,cut_times)
                        
                        non_q_span_where = np.where((pos_time>time_index) & (pos_time<(time_index+q_duras[j])))
                        non_q_span_time_tmp = pos_time[non_q_span_where]
                        # breakpoint()
                        
                        
                        
                    ii+=1
                        
                        
                cut_times.extend(non_q_time_tmp)
                # print('done',ii)
                
                # non_q_time_tmp = mag_time_arr[non_q_time_where]
                non_q_Br_tmp = Br[non_q_where]
                non_q_Bt_tmp = Bt[non_q_where]
                non_q_Bn_tmp = Bn[non_q_where]
                
                non_q_δΒr = non_q_Br_tmp - np.nanmean(non_q_Br_tmp)
                non_q_δBt = non_q_Bt_tmp - np.nanmean(non_q_Bt_tmp)
                non_q_δBn = non_q_Bn_tmp - np.nanmean(non_q_Bn_tmp)
                
                non_q_δB = np.sqrt(np.nanmean(non_q_δΒr**2+non_q_δBt**2+non_q_δBn**2))
                non_q_B_mag = np.sqrt(non_q_Br_tmp**2+non_q_Bt_tmp**2+non_q_Bn_tmp**2)
                
                non_q_B_mag_av = np.nanmean(non_q_B_mag)
                
                non_q_dB_B.append(non_q_δB/non_q_B_mag_av)
                
                
                nq_rs_pos = pos_rs[non_q_span_where]
                nq_rs.append(np.nanmean(nq_rs_pos))
                # non_q_t_scale.append(q_duras[j])
            iii+=1
            
    x_datas = [non_q_dB_B,q_dB_B]
    r_datas = [nq_rs,q_rs]
    
    radial_bins = [[8,15],[15,25],[25,35],[35,45],[45,55],[55,65],[65,75]]

    # radial_bins = [[8,15],[15,25],[25,35],[35,45]]
    n_rads = len(radial_bins)

    r_labs_in = ['8','15','25','35']
    r_labs_out = ['15','25','35','45']
    
    r_labs_in.reverse()
    r_labs_out.reverse()

    labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
    titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']

    cmaps = ['autumn','autumn']
    colors = ['tab:blue','tab:orange']
    
    
    fig = plt.figure(figsize=(20,60))
    
    q_rs_arr = np.array(q_rs)
    nq_rs_arr = np.array(nq_rs)
    
    q_dB_arr = np.array(q_dB_B)
    nq_dB_arr = np.array(non_q_dB_B)
    
    for plot in range(n_rads):
        
        rad_bins = radial_bins[plot]
        axs = fig.add_subplot(n_rads,1,plot+1)
        
        axs.set_title("δB/|B| Histograms",fontsize=38)
        axs.set_ylabel('Counts',fontsize=38)
        axs.set_xlabel('δB/|B|',fontsize=38)
        
        q_rad_where = np.where((q_rs_arr>rad_bins[0])&(q_rs_arr<rad_bins[1]))
        q_rad_where = q_rad_where[0]
        non_q_rad_where = np.where((nq_rs_arr>rad_bins[0])&(nq_rs_arr<rad_bins[1]))
        non_q_rad_where = non_q_rad_where[0]
        
        q_data = q_dB_arr[q_rad_where]
        nq_data = nq_dB_arr[non_q_rad_where]
        
        bins = np.linspace(0,0.8,30)
        hist1, _ = np.histogram(nq_data, bins=bins)
        hist2, _ = np.histogram(q_data, bins=bins)
        
        error1 = np.sqrt(hist1) 
        error2 = np.sqrt(hist2) 
        
        # Plot histograms with error bars
        axs.bar(bins[:-1], hist1, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
        axs.bar(bins[:-1], hist2, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
        axs.errorbar(bins[:-1], hist1, yerr=error1, fmt='none', color='k', capsize=3)
        axs.errorbar(bins[:-1], hist2, yerr=error2, fmt='none', color='k', capsize=3)
        
        axs.tick_params(axis='both', which='major', labelsize=34)
        leg = axs.legend(fontsize=30,loc='upper right',markerscale=5)
        
    plt.show()
        
        # print(q_rad_where.shape)

    # fig = plt.figure(figsize=(30,15))
    # # axs = fig.add_subplot(1,1,1)
    # # print(j+k+1)
    # # index = n_types*j+1+k
    # # if k ==0:
    # axs = fig.add_subplot(111)
    # # axs = fig.add_subplot(n_rads,1,j+1)
    # axs.set_title("δB/|B| Histograms",fontsize=38)
    # axs.set_ylabel('Counts',fontsize=38)
    # axs.set_xlabel('δB/|B|',fontsize=38)
    
    # bins = np.linspace(0,0.8,30)
    # hist1, _ = np.histogram(x_datas[0], bins=bins)
    # hist2, _ = np.histogram(x_datas[1], bins=bins)
    
    # # hist1_norm = hist1 / np.sum(hist1)
    # # hist2_norm = hist2 / np.sum(hist2)
    
    # hist1_norm = hist1
    # hist2_norm = hist2
    
    # q_mean = np.nanmean(q_dB_B)
    # q_std = np.nanstd(q_dB_B)
    # q_median = np.nanmedian(q_dB_B)
    # non_q_mean = np.nanmean(non_q_dB_B)
    # non_q_std = np.nanstd(non_q_dB_B)
    # non_q_median = np.nanmedian(non_q_dB_B)

    
    # # Calculate error bars for each bin (normalized)
    # # error1 = np.sqrt(hist1) / np.sum(hist1)
    # # error2 = np.sqrt(hist2) / np.sum(hist2)
    
    # error1 = np.sqrt(hist1) 
    # error2 = np.sqrt(hist2) 
    
    # # Plot histograms with error bars
    # axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
    # axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
    # axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
    # axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
    
    # axs.tick_params(axis='both', which='major', labelsize=34)
    # leg = axs.legend(fontsize=30,loc='upper right',markerscale=5)
    
    # axs.text(0.4,75,'Quiescent Mean: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)),fontsize=30)
    # # axs.text(0.4,75,'Quiescent Median: '+str(round(q_median,3)),fontsize=22)
    # axs.text(0.4,70,'Non-Quiescent Mean: '+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,1)),fontsize=30)
    # # axs.text(0.4,65,'Non-Quiescent Median: '+str(round(non_q_median,3)),fontsize=22)
    
    # plt.show()
    
    breakpoint()
    
def alf_hist(span_cut=True,save=False):
    
    
    #------------------cut out bad SPAN times------------------#
    
    # span_time = []
    # span_flag = []
    # span_flag_av = []
    # span_flag_ratio = []
    # span_rad = []
    good_span_times = []
    
    for pepe in range(1,17):
    
        span_check_savename = 'SPAN_ion_fov_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/FOV flags/'
        
        pyt.tplot_restore(span_check_savepath+span_check_savename)
    
        # fov_flag_exact = pyt.get_data('phi_fov')
        # fov_time = fov_flag_exact[0]
        # fov_flag = fov_flag_exact[1]
        
        fov_flag_average = pyt.get_data('phi_fov_average')
        fov_av_time = fov_flag_average[0]
        fov_flag_av = fov_flag_average[1]
        
        # fov_flag_ratio = pyt.get_data('phi_fov_ratio')
        # fov_av_time = fov_flag_ratio[0]
        # fov_ratio_av = fov_flag_ratio[1]
        
        # rad = pyt.get_data('psp_radial_dist_Rs')
        # rad_time = rad[0]
        # rad_Rs = rad[1]
        
        # span_time.append(fov_time)
        # span_flag.append(fov_flag)
        # span_flag_av.append(fov_flag_av)
        # span_flag_ratio.append(fov_ratio_av)
        # span_rad.append(rad_Rs)
        
        span_qtn_check_savename = 'SPAN_SPC_QTN_flags_enc_'+str(pepe)+'.cdf'
        # span_check_savename = 'SPAN_ion_fov_flags_enc_15.cdf'
        span_qtn_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
        
        pyt.tplot_restore(span_qtn_check_savepath+span_qtn_check_savename)
        
        qtn_flag_data = pyt.get_data('SPAN_qual_flag')
        qtn_flag_time = qtn_flag_data[0]
        qtn_flag = qtn_flag_data[1]
        
        # FOV_where = np.where(fov_flag_av==0)
        # FOV_where = FOV_where[0]
        
        # good_fov = qtn_flag_time[FOV_where]
        
        # qtn_in_where = np.where(np.isin(qtn_flag_time,good_fov))
        # qtn_in_where = qtn_in_where[0]
        
        # add one to places where FOV is good to make sure that the score for good FOV is 2.
        # qtn_flag[qtn_in_where]+=1 
        
        # print(pepe," : ",qtn_flag.max()," : ", fov_flag_av.max())
        
        # if pepe > 2:
            # breakpoint()
        
        good_where = qtn_flag<2
        
        #a little confusing but I'm grouping the zeros so I actually want to invert my bool array to have good times be 0s
        # good_flip = 1-good_where
        
        edge_indices = group_zeros(good_where)
        
        # edge_times = []
        for i in edge_indices:
            t0s = qtn_flag_time[i[0]]
            tfs = qtn_flag_time[i[1]]
            
            # edge_times.append((t0,tf))
            
            good_span_times.append((t0s,tfs))
        
        
        pyt.del_data()
    
    
    fig = plt.figure(figsize=(15,15))
    # radial_bins = [[8,15],[15,25],[25,35],[35,45],[45,55],[55,65],[65,75]]
    radial_bins = [[8,15],[15,25],[25,35]]
    if span_cut:
        q_range_num = [42,55,31]
    else:
        q_range_num = [42,80,97]
    
    for i in range(len(radial_bins)): 
    
        bins = radial_bins[i]
        
        tplot_savename = 'Brazil_data_'+str(bins[0])+'_to_'+str(bins[1])+'_Rs.cdf'
        tplot_savepath = '/Users/besh2109/Desktop/BrazilPlots/'   
        
        pyt.tplot_restore(tplot_savepath+tplot_savename)

        q_brazil = pyt.get_data("q_region_times")
        q_times_rad_bins = q_brazil[0]
        
        non_q_brazil = pyt.get_data("non_q_region_times")
        non_q_times_rad_bins = non_q_brazil[0]

        # q_brazil = pyt.get_data("q_region_brazil")
        # q_beta_rad_bins = q_brazil[0]
        # q_anis_rad_bins = q_brazil[1]
        
        # non_q_brazil = pyt.get_data("non_q_region_brazil")
        # non_q_beta_rad_bins = non_q_brazil[0]
        # non_q_anis_rad_bins = non_q_brazil[1]
        
        q_color = pyt.get_data("q_color_scale")
        q_alfvenicity_rad_bins = q_color[0]
        
        non_q_color = pyt.get_data("non_q_color_scale")
        non_q_alfvenicity_rad_bins = non_q_color[0]
        
        if span_cut:
            
            q_times_tmp = np.array([])
            # q_anis_rad_bins_tmp = np.array([])
            # q_beta_rad_bins_tmp = np.array([])
            q_alfvenicity_rad_bins_tmp = np.array([])
            
            non_q_times_tmp = np.array([])
            # non_q_anis_rad_bins_tmp = np.array([])
            # non_q_beta_rad_bins_tmp = np.array([])
            non_q_alfvenicity_rad_bins_tmp = np.array([])
            
            for memes in good_span_times:
                t0 = memes[0]
                tf = memes[1]
                
                q_span_where = np.where((q_times_rad_bins>t0)&(q_times_rad_bins<tf))
                q_span_where = q_span_where[0]
                
                non_q_span_where = np.where((non_q_times_rad_bins>t0)&(non_q_times_rad_bins<tf))
                non_q_span_where = non_q_span_where[0]
                
                if len(q_span_where) != 0:
                
                    q_times_tmp = np.append(q_times_tmp,q_times_rad_bins)
                    # q_anis_rad_bins_tmp = np.append(q_anis_rad_bins_tmp,q_anis_rad_bins[q_span_where])
                    # q_beta_rad_bins_tmp = np.append(q_beta_rad_bins_tmp,q_beta_rad_bins[q_span_where])
                    q_alfvenicity_rad_bins_tmp = np.append(q_alfvenicity_rad_bins_tmp,q_alfvenicity_rad_bins[q_span_where])
           
                if len(non_q_span_where) != 0:
                    
                    non_q_times_tmp = np.append(non_q_times_tmp,non_q_times_rad_bins)
                    # non_q_anis_rad_bins_tmp = np.append(non_q_anis_rad_bins_tmp,non_q_anis_rad_bins[non_q_span_where])
                    # non_q_beta_rad_bins_tmp = np.append(non_q_beta_rad_bins_tmp,non_q_beta_rad_bins[non_q_span_where])
                    non_q_alfvenicity_rad_bins_tmp = np.append(non_q_alfvenicity_rad_bins_tmp,non_q_alfvenicity_rad_bins[non_q_span_where])
                
            q_times_rad_bins = q_times_tmp
            # q_anis_rad_bins = q_anis_rad_bins_tmp
            # q_beta_rad_bins = q_beta_rad_bins_tmp
            q_alfvenicity_rad_bins = q_alfvenicity_rad_bins_tmp
            
            non_q_times_rad_bins = non_q_times_tmp
            # non_q_anis_rad_bins = non_q_anis_rad_bins_tmp
            # non_q_beta_rad_bins = non_q_beta_rad_bins_tmp
            non_q_alfvenicity_rad_bins = non_q_alfvenicity_rad_bins_tmp

        # y_datas = [q_anis_rad_bins,non_q_anis_rad_bins]
        x_datas = [q_times_rad_bins,non_q_times_rad_bins]
        c_datas = [q_alfvenicity_rad_bins,non_q_alfvenicity_rad_bins]
    
        #-----------------------------histogram plots----------------------------#
        
        # radial_bins = [[8,15],[15,25],[25,35],[35,45],[8,40]]
        
        # radial_bins = [[35,45],[25,35],[15,25],[8,15],[8,40]]
        
        # n_rads = len(radial_bins)-1
        n_rads = 3
        # n_rads = len(radial_bins)
        
        n_types = 1
        
        # breakpoint()
        
        # radial_bins = [[1,15],[15,35]]
        # fig = plt.figure(figsize=(10,25))
        # fig = plt.figure(figsize=(20,15))
        # fig = plt.figure(figsize=(15,15))
        
        # fig.suptitle(datatype+" Histogram "+title_mod+" Rs",fontsize=36)
        # r_labs_in = ['13.3','15','25','35']
        # r_labs_out = ['15','25','35','45']
        
        r_labs_in = ['13.3','15','25','35','45','55','65']
        r_labs_out = ['15','25','35','45','55','65','75']
        
        r_labs_in = r_labs_in[0:n_rads]
        r_labs_out = r_labs_out[0:n_rads]
        
        r_labs_in.reverse()
        r_labs_out.reverse()
        
        # breakpoint()

    
    # for k in range(n_types):
        # print(k)
        
        # q_data_full = data_types[k]
        # non_q_data_full = data_types[k]
        d_label = 'Alfvén Mach Number'
        
        # breakpoint()
        
        q_time_single = q_times_rad_bins
        non_q_time_single = non_q_times_rad_bins
        
        q_data_single = q_alfvenicity_rad_bins
        non_q_data_single = non_q_alfvenicity_rad_bins
        
        n_bin_single = 60
        drange_single = (0,6)
        
        # for j in range(n_rads):
            
        rad_bins = radial_bins[i]
        
        # q_rad_where = np.where((q_pos_full>rad_bins[0])&(q_pos_full<rad_bins[1]))
        # q_rad_where = q_rad_where[0]
        
        # non_q_rad_where = np.where((non_q_pos_full>rad_bins[0])&(non_q_pos_full<rad_bins[1]))
        # non_q_rad_where = non_q_rad_where[0]
        
        # q_ave_where = np.where((q_pos_average_full>rad_bins[0])&(q_pos_average_full<rad_bins[1]))
        # q_ave_where = q_ave_where[0]
        
        # q_range_num = len(q_ave_where)
        
        title_mod = "for radial distances of "+str(rad_bins[0])+" - "+str(rad_bins[1])
        
        q_times_rad_bins = q_time_single
        non_q_times_rad_bins = non_q_time_single
        
        q_data_rad_bins = q_data_single
        non_q_data_rad_bins = non_q_data_single
        
        # q_rad_starts = q_start_full
        # q_rad_ends = q_end_full
        
        # breakpoint()
        
        q_mean = np.nanmean(q_data_rad_bins)
        q_std = np.nanstd(q_data_rad_bins)
        non_q_mean = np.nanmean(non_q_data_rad_bins)
        non_q_std = np.nanstd(non_q_data_rad_bins)
        
        # if rad_bins[0] == 8:
        print()
        print("Number of Quiescent Regions: "+str(q_range_num[i]))
        print('Radial Bin: '+str(rad_bins[0])+'-'+str(rad_bins[1]))
        print(d_label+' Q Average: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)))
        print(d_label+' Non-Q Average: '+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)))
        print()
        
        q_med = np.nanmedian(q_data_rad_bins)
        non_q_med = np.nanmedian(non_q_data_rad_bins)
        
        # if span_cut:
            
        #     q_data_rad_bins_tmp = np.array([])
        #     non_q_data_rad_bins_tmp = np.array([])

        #     # breakpoint()
        #     q_counts_new = 0
        #     for memes in good_span_times:
        #         t0 = memes[0]
        #         tf = memes[1]
                
        #         q_span_where = np.where((q_times_rad_bins>t0)&(q_times_rad_bins<tf))
        #         q_span_where = q_span_where[0]
                
        #         # print(q_span_where)
                
        #         non_q_span_where = np.where((non_q_times_rad_bins>t0)&(non_q_times_rad_bins<tf))
        #         non_q_span_where = non_q_span_where[0]
                
        #         if len(q_span_where) != 0:
                
        #             q_data_rad_bins_tmp = np.append(q_data_rad_bins_tmp,q_data_rad_bins[q_span_where])
           
        #         if len(non_q_span_where) != 0:
                    
        #             non_q_data_rad_bins_tmp = np.append(non_q_data_rad_bins_tmp,non_q_data_rad_bins[non_q_span_where])
                    
                
        #         # # breakpoint()
        #         # for jj in range(len(q_rad_starts)):
                    
        #         #     if ((q_rad_starts[jj]>t0)&(q_rad_starts[jj]<tf)) or ((q_rad_ends[jj]>t0)&(q_rad_ends[jj]<tf)):
                        
        #         #         if q_counts_new == len(q_rad_starts): 
        #         #             pass #just means that all quiescent regions are contributing after cutting for SPAN.
        #         #                  #Passing here avoids double counting as quiescent regions can span *ba dn tsst* two good portions of SPAN but be chopped in the middle.
        #         #         else:
        #         #             q_counts_new+=1
                        
            
                
        #     q_data_rad_bins = q_data_rad_bins_tmp
        #     non_q_data_rad_bins = non_q_data_rad_bins_tmp
        #     # q_range_num = q_counts_new
        
        
        # q_mean = np.nanmean(q_data_rad_bins)
        # q_std = np.nanstd(q_data_rad_bins)
        # non_q_mean = np.nanmean(non_q_data_rad_bins)
        # non_q_std = np.nanstd(non_q_data_rad_bins)
        
        # if rad_bins[0] == 8:
            
        # print("Number of Quiescent Regions: "+str(q_range_num))
        # print('Radial Bin: '+str(rad_bins[0])+'-'+str(rad_bins[1]))
        # print(d_label+' Q Average: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)))
        # print(d_label+' Non-Q Average: '+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)))
        # print()
        
        # q_med = np.nanmedian(q_data_rad_bins)
        # non_q_med = np.nanmedian(non_q_data_rad_bins)
        
        
        # breakpoint()
        x_datas = [non_q_data_rad_bins,q_data_rad_bins]

        labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
        titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']

        cmaps = ['autumn','autumn']
        colors = ['tab:blue','tab:orange']
        

        
        # axs = fig.add_subplot(1,1,1)
        # print(j+k+1)
        index = n_types*i+1
        # if k ==0:
        axs = fig.add_subplot(n_rads,n_types,index)
        # axs = fig.add_subplot(n_rads,1,j+1)
        
        bins = np.linspace(drange_single[0],drange_single[1], n_bin_single)
        hist1, _ = np.histogram(x_datas[0], bins=bins)
        hist2, _ = np.histogram(x_datas[1], bins=bins)
        
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        
        # Calculate error bars for each bin (normalized)
        error1 = np.sqrt(hist1) / np.sum(hist1)
        error2 = np.sqrt(hist2) / np.sum(hist2)
        
        # Plot histograms with error bars
        axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
        axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
        axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
        axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
        
        # axs.set_ylabel("Normalized Counts",fontsize=32)
        # axs.set_xlabel(datatype,fontsize=32)
        axs.tick_params(axis='x', which='major', labelsize=24)
        
        axs.set_xlim(drange_single)
        # axs.set_xlim((0,175))
        
        if i!=n_rads-1:
            # axs.tick_params('x', labelbottom=False)
            axs.set_xticks([])
            
        # axs.tick_params('y', labelbottom=False)
        axs.set_yticks([])
        
        if i==0:
            # x_pos = 0.1
            x_pos = 0.4
        else:
            x_pos = 0.4
        
        axs.text(x_pos,0.65,"Number of Quiescent Regions: "+str(q_range_num[i]),fontsize=32,transform=axs.transAxes)
        axs.text(x_pos,0.55,"Quiescent Mean: "+str(round(q_mean,2))+' ± '+str(round(q_std,2)),fontsize=32,transform=axs.transAxes)
        axs.text(x_pos,0.45,"Non-Quiescent Mean: "+str(round(non_q_mean,2))+' ± '+str(round(non_q_std,2)),fontsize=32,transform=axs.transAxes)
        # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*4.4,"Quiescent Median: "+str(q_med),fontsize=32)
        # axs.text(bins[int(n_bins/4)],np.max(hist1_norm)-np.mean(hist1_norm)*5.4,"Non-Quiescent Median: "+str(non_q_med),fontsize=32)
        # 
        # plt.subplots_adjust(wspace=0.01, hspace=0)
        
        if index==1:
            leg = axs.legend(fontsize=20,loc='upper right',markerscale=5)
        plt.tight_layout()
        
        # fig.text(-0.035,0.25*k+0.14, (r_labs_out[k]+" Rs \n - \n "+r_labs_in[k]+" Rs"), ha="center", va="center", rotation=12,fontsize=32)
        
        fig.text(-0.035,(1/n_rads)*i+0.14, (r_labs_out[i]+" $R_{\odot}$ \n - \n "+r_labs_in[i]+" $R_{\odot}$"), ha="center", va="center", rotation=12,fontsize=32)
        

    
    # fig.text((1/n_types)*k+0.13,1.015, data_types[k], ha="center", va="center",fontsize=32)
    # fig.text((1/n_types)*k+0.13,-0.015, units[k], ha="center", va="center",fontsize=32)
    
    fig.text((1/n_types)*0+0.5,1.015, 'Alfvén Mach Number', ha="center", va="center",fontsize=32)
    fig.text((1/n_types)*0+0.5,-0.015, '$V_{sw}/V_{Alf}$', ha="center", va="center",fontsize=32)
    # plt.show()

    # fig.text(-0.035,0.25*3+0.14, "45 Rs \n - \n 35 Rs", ha="center", va="center", rotation=12,fontsize=32)
    fig.text(-0.1,0.525, "Radial Distance", ha="center", va="center", rotation=90,fontsize=38)
    fig.text(0.5,1.05, "Quiescent Region Property Histograms", ha="center", va="center",fontsize=32)
    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    
    if save:
        # save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/Plots/Histograms/'+file_mod+'/'
        # save_name = 'quiescent_'+dtype+'_histogram_'+str(rad_bins[0])+"_to_"+str(rad_bins[1])+'_Rs.png'
        
        # if not os.path.exists(save_path):
        #     print('should work')
        #     os.makedirs(save_path)
        
        # plt.savefig(save_path+save_name)
        pass
    
    else:
        plt.show()
        
    plt.clf()
    plt.cla()
    plt.close('all')
    plt.close(fig)
    
def quiescent_volume(enc='all',enc_radius=65):
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        # enc = list(range(1,18))
        # enc_arr = enc_flt[0:17]
        
        enc = list(range(1,19))
        enc_arr = enc_flt[0:18]
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(2,13))
        enc.extend(range(14,17))
        # breakpoint()
        enc_arr = [enc_flt[i-1] for i in enc]
        # enc_arr = enc_flt[enc]
        title_mod = 'for All Encounters, no 1 or 13'    
    
    if enc == 'no 1':
        enc = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        enc_arr = enc_flt[1:16]
        title_mod = 'for All Encounters, no 1'
    
    elif type(enc)==int:
        enci = enc-1
        encs = [enci]
        enc_arr = [enc_flt[enci]]
    else:
        
        
        enc_arr = [enc_flt[i-1] for i in enc]

    
    #----------------------Organizing CSVs for in Data-------------------------#

    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    # csv_names = os.listdir(csv_path)
    # csv_names.remove('.DS_Store')
    
    csv_names = glob.glob(csv_path+'*_raw.csv')
    
    enc_tag = []
    for i in csv_names:
        
        a = re.findall(r'\d+',i[25:])
        enc_tag.append(int(a[0]))
    
    csv_names = sort_list(csv_names,enc_tag)
    enc_tag = sort_list(enc_tag,enc_tag)
    
    # filelist = []
    # for j in csv_names:
    #     j = csv_path+j
    #     filelist.append(j)
    
    filelist = csv_names
    
    #--------------------------Start importing PSP Data-----------------------------#
    
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    radial_bins = [[11.4,15],[15,25],[25,35],[35,45],[45,55],[55,65]]
    
    pos_time_full = np.array([])
    pos_full = np.array([])
    
    v_time_full = np.array([])
    v_mag_full = np.array([])
    
    step_time_full = np.array([])
    step_full = np.array([])
    
    dummy = 0
    for enc in enc_arr:
        
        # csv_name = 'enc_'+str(encs[dummy])+'_regions_raw.csv'
        
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        pyt.del_data()
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        
        pos_data = pyt.get_data('psp_spi_SUN_DIST')
        pos_time = pos_data[0]
        pos_rs = pos_data[1]/Rs
        
        vel_sc_data = pyt.get_data('psp_spi_VEL_SC')
        vel_sc_time = vel_sc_data[0]
        vel_sc = vel_sc_data[1]
        
        v_mag = np.sqrt(vel_sc[:,0]**2+vel_sc[:,1]**2+vel_sc[:,2]**2)
        
        #-------------integrate over encounter to find meters of plasma-------------#
        
        del_t = np.diff(vel_sc_time)
        del_t = np.append(del_t,del_t[-1]) #just add the last element on the end again
        
        distance_steps = v_mag*del_t #gives array of distance of plasma steps in km
        
        # total_km = np.nansum(distance_steps) #now we can sum over time steps for the whole encounter
        #                                   #and use the np.where function to sum over subsets of
        #                                   #encounters.
        
        
        pos_time_full = np.append(pos_time_full,pos_time)
        pos_full = np.append(pos_full,pos_rs)
        
        v_time_full = np.append(v_time_full,vel_sc_time)
        v_mag_full = np.append(v_mag_full,v_mag)
        
        step_time_full = np.append(step_time_full,vel_sc_time)
        step_full = np.append(step_full,distance_steps)
        
        # breakpoint()
        
        dummy+=1
        
    tot_array_list = []
    q_array_list = []
    
    for i in radial_bins:
        pos_where = np.where((pos_full>i[0])&(pos_full<i[1]))
        pos_where = pos_where[0]
        
        rad_dist_times = step_time_full[pos_where]
        rad_dist_steps = step_full[pos_where]
        
        q_data_tmp = np.array([])
        for j in filelist:
            
            q_df = pd.read_csv(j) #quiescent region Pandas dataframe
            
            q_starts = q_df['start dates'].to_numpy()
            q_ends = q_df['end dates'].to_numpy()
            q_duras = q_df['duration'].to_numpy()
            
            q_start_flt = np.array(pys.time_float(q_starts))
            q_end_flt = np.array(pys.time_float(q_ends))
            
            for k in range(len(q_start_flt)):
            
                q_where = np.where((rad_dist_times>q_start_flt[k])&(rad_dist_times<q_end_flt[k]))
                q_where = q_where[0]
                
                q_data_tmp = np.append(q_data_tmp,rad_dist_steps[q_where])
                
        q_array_list.append(q_data_tmp)
        tot_array_list.append(rad_dist_steps)
                
        
        # q_where = np.where()
        # quiescent_km = np.sum(distance_steps[q_where])
    
    q_km_list = []
    tot_km_list = []
    perc_list = []
    for i in range(len(q_array_list)):
        q_sum = np.nansum(q_array_list[i])
        tot_sum = np.nansum(tot_array_list[i])
        q_km_list.append(q_sum)
        tot_km_list.append(tot_sum)
        
        if q_sum == 0 and tot_sum == 0:
            perc_list.append(np.nan)
        else:
            perc_list.append(q_sum/tot_sum)
    
    # breakpoint()
    #------------------------create plots and fit decay-----------------------#
    
    fig = plt.figure(figsize=(20,15))
    
    axs = fig.add_subplot(111)
    
    x_values = np.mean(radial_bins,axis=1)
    y_values = np.array(perc_list)*100
    
    axs.plot(x_values,y_values,linewidth=3,color='tab:blue')
    axs.scatter(x_values,y_values,s=100,color='tab:blue')
    axs.set_title('Quiescent Region Radial Falloff',fontsize=30)
    axs.set_xlabel('Parker Solar Probe radial distance ($R_{\odot}$)',fontsize=28)
    axs.set_ylabel('Percent of Solar Wind that is Quiescent (%)',fontsize=28)
    axs.tick_params(axis='both', which='major', labelsize=28)
    
    def func(x, a, b):
        return a * np.exp(-(1/b) * x)
    
    def func2(x, a, b):
        return a * (1/x) + b
    
    popt, pcov = curve_fit(func, x_values, y_values)
    
    # axs.plot(x_values, func(x_values, *popt), 'g--',
    #      label='Exponential fit: $Ae^{-x/b}$, a=%5.3f, b=%5.3f $R_{\odot}$' % tuple(popt),linewidth=3)
    
    # x_values = np.linspace(5,215)
    
    axs.plot(x_values, func(x_values, *popt), 'g--',
         label='Exponential fit: $Ae^{-x/b}$, a=%5.1f, b=%5.1f $R_{\odot}$' % tuple(popt),linewidth=3)
    
    # popt, pcov = curve_fit(func2, x_values, y_values)
    
    # axs.plot(x_values, func2(x_values, *popt), 'r--',
    #      label='$1/R^2$ fit: a=%5.3f, b=%5.3f' % tuple(popt))
    
    plt.legend(fontsize=28)
    plt.show()
    # breakpoint()
    
def quiescent_list(lower=20,upper=30,nrand = None):
    
    #----------------------Organizing CSVs for in Data-------------------------#
    csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    csv_names = os.listdir(csv_path)
    # csv_names.remove('.DS_Store')
    csv_names = [s for s in csv_names if 'raw' in s]

    enc_tag = []
    for i in csv_names:
        
        # Find all integer numbers in the string using regular expression
        integers = re.findall(r'\d+', i)
        # Convert the extracted integers from strings to integers and append to the integers_list
        enc_tag.extend(map(int, integers))
    
    csv_names = sort_list(csv_names,enc_tag)
    enc_tag = sort_list(enc_tag,enc_tag)
    
    filelist = []
    for j in csv_names:
        j = csv_path+j
        filelist.append(j)
    
    q_start_list = np.array([])
    q_end_list = np.array([])
    q_duras_list = np.array([])
    
    for j in filelist:
        
        q_df = pd.read_csv(j) #quiescent region Pandas dataframe
        
        q_starts = q_df['start dates'].to_numpy()
        q_ends = q_df['end dates'].to_numpy()
        q_duras = q_df['duration'].to_numpy()
        
        q_start_flt = np.array(pys.time_float(q_starts))
        q_end_flt = np.array(pys.time_float(q_ends))
        
        q_start_list = np.append(q_start_list,q_starts)
        q_end_list = np.append(q_end_list,q_ends)
        q_duras_list = np.append(q_duras_list,q_duras)
        
    q_duras_list_minutes = q_duras_list/60
    
    where = np.where((q_duras_list_minutes>lower)&(q_duras_list_minutes<upper))
    where = where[0]
    
    final_starts = q_start_list[where]
    final_ends = q_end_list[where]
    ind_arr = np.array(range(len(final_starts)))
    # breakpoint()
    if nrand != None:
        # Combine the arrays column-wise
        choice = np.random.choice(ind_arr,nrand, replace=False)
        
        rand_starts = final_starts[choice]
        rand_ends = final_ends[choice]
        
        combined_array = np.column_stack((rand_starts, rand_ends))
        
        # Define the file name
        filepath = '/Users/besh2109/Desktop/'
        filename = 'rand_times.csv'
        
        # Write the arrays to a CSV file
        np.savetxt(filepath+filename, combined_array, delimiter=',',fmt='%s')
        
    else:
        # Combine the arrays column-wise
        combined_array = np.column_stack((final_starts, final_ends))
        
        # Define the file name
        filepath = '/Users/besh2109/Desktop/'
        filename = 'saved_times.csv'
        
        # Write the arrays to a CSV file
        np.savetxt(filepath+filename, combined_array, delimiter=',',fmt='%s')
    
    # breakpoint()
    
def temperature_analysis():
    
    tplot_path = '/Users/besh2109/Desktop/Temperature Products/Ions vs R v0/'
    tplot_name = 'psp_swp_T_vs_R.cdf'
    
    pyt.tplot_restore(tplot_path+tplot_name)
    
    non_q = pyt.get_data("SPAN_temp_full")
    non_q_rad = non_q[0]
    non_q_temp = non_q[1]
    
    q = pyt.get_data("SPAN_temp_q")
    q_rad = q[0]
    q_temp = q[1]
    
    # drange = [-20,150]
    drange = [-20,150] #testing with offset, the natural ranges happen to be pretty close,
    enc_radius = 65             # adding offset helps disambiguate the axes as I'm coding.

    bincount = 30
    
    #--------------histogram for non quiescent regions----------------------#
    
    # x_bins = np.logspace(np.log10(11.4),np.log10(enc_radius),25)
    # y_bins = np.logspace(np.log10(1),np.log10(150),25)
    
    hist2d,xedge_f,yedge_f = np.histogram2d(non_q_rad,non_q_temp, bins=[25,35],range=[[11.4, enc_radius],drange])
    # hist2d,xedge_f,yedge_f = np.histogram2d(non_q_rad,non_q_temp, bins=[x_bins,y_bins])
    
    histo=hist2d
    
    xcenters_f = (xedge_f[:-1] + xedge_f[1:]) / 2
    ycenters_f = (yedge_f[:-1] + yedge_f[1:]) / 2

    temp_fit_maxs = []
    
    norm_histo_full = np.zeros(histo.shape)
    for ksj in range(len(histo)):
        
        # bepo = np.nanmax(histo[ksj,:])
        bepo = np.nansum(histo[ksj,:])
        # print(bepo)
        if bepo == 0:
            norm_histo_full[ksj,:] = histo[ksj,:]
        else:
            norm_histo_full[ksj,:] = histo[ksj,:]/bepo
        
        hist_x = ycenters_f #the x axis IN THIS CASE is the y axis of the original 2d histogram 
        hist_y = norm_histo_full[ksj,:]

        model = SkewedGaussianModel()
        
        # set initial parameter values
        params = model.make_params(amplitude=10, center=20, sigma=20, gamma=5)
        
        # adjust parameters to best fit data.
        result = model.fit(hist_y, params, x=hist_x)
        
        # print(result.fit_report())
        # plt.plot(hist_x, hist_y, color='tab:blue')
        # # plt.plot(hist_x, result.init_fit,color='tab:orange')
        # plt.plot(hist_x, result.best_fit,color='tab:red')
        # plt.show()
        
        temp_fit_maxs.append(hist_x[np.argmax(result.best_fit)])
        # breakpoint()
    temp_fit_arr = np.array(temp_fit_maxs)
    
    #--------------histogram for quiescent regions----------------------#
    
    # x_bins = np.logspace(np.log10(11.4),np.log10(enc_radius),25)
    # y_bins = np.logspace(np.log10(1),np.log10(150),25)
    
    hist2d,xedge,yedge = np.histogram2d(q_rad,q_temp, bins=[25,35],range=[[11.4, enc_radius],drange])
    # hist2d,xedge,yedge = np.histogram2d(q_rad,q_temp, bins=[x_bins,y_bins])

    histo=hist2d
    
    xcenters_q = (xedge[:-1] + xedge[1:]) / 2
    ycenters_q = (yedge[:-1] + yedge[1:]) / 2
    
    q_temp_fit_maxs = []
    q_temp_x = []
    
    norm_histo_q = np.zeros(histo.shape)
    for ksj in range(len(histo)):
        
        bepo = np.nansum(histo[ksj,:])
        
        if bepo == 0:
            norm_histo_q[ksj,:] = histo[ksj,:]
        else:
            norm_histo_q[ksj,:] = histo[ksj,:]/bepo
        
        hist_x = ycenters_q #the x axis IN THIS CASE is the y axis of the original 2d histogram 
        hist_y = norm_histo_q[ksj,:]
        
        model = SkewedGaussianModel()
        
        # set initial parameter values
        params = model.make_params(amplitude=10, center=20, sigma=20, gamma=5)
        
        # adjust parameters to best fit data.
        result = model.fit(hist_y, params, x=hist_x)
        
        # print(result.fit_report())
        plt.plot(hist_x, hist_y, color='tab:blue')
        # plt.plot(hist_x, result.init_fit)
        plt.plot(hist_x, result.best_fit, color='tab:red')
        plt.show()
        
        q_temp_fit_maxs.append(hist_x[np.argmax(result.best_fit)])
        q_temp_x.append(xcenters_q[ksj])
        
    q_temp_fit_arr = np.array(q_temp_fit_maxs)
    q_temp_x_arr = np.array(q_temp_x)
    
    def Line(x,A,m):
        y = A*x**m
        return y

    nq_parameters, covariance = curve_fit(Line,xcenters_f,temp_fit_arr)
    nq_A = nq_parameters[0]
    nq_m = nq_parameters[1]
    
    nq_A_err = np.sqrt(covariance[0,0])
    nq_m_err = np.sqrt(covariance[1,1])

    nq_line = nq_A*xcenters_f**nq_m
    
    q_parameters, covariance = curve_fit(Line,xcenters_q[5:],q_temp_fit_arr[5:])
    q_A = q_parameters[0]
    q_m = q_parameters[1]
    
    q_A_err = np.sqrt(covariance[0,0])
    q_m_err = np.sqrt(covariance[1,1])
    
    q_line = q_A*xcenters_q**q_m
    
    #-------------------------------generate plot-----------------------------#
    
    
    fig = plt.figure(figsize=(20,9)) 
    axs = fig.add_subplot(111)

    axs.pcolormesh(xcenters_f,ycenters_f,norm_histo_full.T,cmap='Greys')
    
    # axs.plot(xcenters_f,temp_fit_arr, color='tab:blue',label='Non-quiescent temperature peaks',linewidth=2)
    # axs.plot(xcenters_q,q_temp_fit_arr, color='tab:orange',label='Quiescent temperature peaks',linewidth=2)
    
    
    axs.plot(xcenters_f,nq_line, color='tab:blue',label='Non-quiescent temperature fit, m='+str(round(nq_m,2))+
                                                                                                   '±'+str(round(nq_m_err,2)),linewidth=2)
    axs.plot(xcenters_q,q_line, color='tab:orange',label='Quiescent temperature fit, m='+str(round(q_m,2))+
                                                                                                   '±'+str(round(q_m_err,2)),linewidth=2)
    
    axs.set_title('$T_{p}$ vs R for Encounters 1 through 18',fontsize=36)
    # axs.set_ylabel(part_title,fontsize=24)
    # axs.set_ylabel('Solar Wind Bulk Velocity (km/s)',fontsize=24)
    axs.set_ylabel('Solar Wind Proton Core Temperature (eV)',fontsize=24)
    axs.set_xlabel('Radial Position of PSP ($R_{\odot}$)',fontsize=24)
    axs.tick_params(axis='both', which='major', labelsize=24)
    axs.tick_params(axis='x', which='minor', labelsize=20)
    # if 'dens' not in atype:
    # axs.set_ylim(drange)
    # axs.set_ylim(125,425)
    # axs.set_ylim(0,120)
    # axs.set_xlim(15,55)
    # axs.set_xlim(10,65)
    axs.set_yscale('log')
    axs.set_xscale('log')
    
    
    adiabatic = 1100*xcenters_f**(-4/3) 
    
    axs.plot(xcenters_f,adiabatic,color='red',linestyle='dashed',linewidth=2,zorder=5,label='Adiabatic Falloff m=-4/3')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    
    lgnd = plt.legend(fontsize=18)
    plt.show()
    
    
    breakpoint()