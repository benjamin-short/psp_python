#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:48:32 2025

@author: besh2109
"""

import os
import pyspedas.psp as psp
import pyspedas as pys
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const

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

from numpy.linalg import inv

import matplotlib.patches as mpatch

import re
import pandas as pd

import glob

import matplotlib as mpl

from lmfit.models import SkewedGaussianModel

import psp_regions.utils as utils

from scipy.stats import chisquare

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

# sweap_id = os.environ['PSP_SWEAP_ID']
# sweap_pass = os.environ['PSP_SWEAP_PW']

sweap_id = os.environ['PSP_SWEAP_ID_berk']
sweap_pass = os.environ['PSP_SWEAP_PW_berk']

jsoc_email = os.environ['JSOC_EMAIL']

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3 #solar radius in m
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

def delb_b(enc='all',enc_radius=65,save=False):
    
    print('calculating ∂B/B')
    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    #kb = 1.380649*10e-23 #boltzmann constant, J/K
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,19))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(1,13))
        enc.extend(range(14,19))
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
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20300101_090000_v43.cdf')
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
                        groups = np.array(utils.group_zeros(zeros))
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
    

    fig = plt.figure(figsize=(30,15))

    axs = fig.add_subplot(111)

    axs.set_title("δB/|B| Histograms",fontsize=38)
    axs.set_ylabel('Counts',fontsize=38)
    axs.set_xlabel('δB/|B|',fontsize=38)
    
    bins = np.linspace(0,0.8,30)
    hist1, _ = np.histogram(x_datas[0], bins=bins)
    hist2, _ = np.histogram(x_datas[1], bins=bins)
    
    # hist1_norm = hist1 / np.sum(hist1)
    # hist2_norm = hist2 / np.sum(hist2)
    
    hist1_norm = hist1
    hist2_norm = hist2
    
    q_mean = np.nanmean(q_dB_B)
    q_std = np.nanstd(q_dB_B)
    q_median = np.nanmedian(q_dB_B)
    non_q_mean = np.nanmean(non_q_dB_B)
    non_q_std = np.nanstd(non_q_dB_B)
    non_q_median = np.nanmedian(non_q_dB_B)

    
    # Calculate error bars for each bin (normalized)
    # error1 = np.sqrt(hist1) / np.sum(hist1)
    # error2 = np.sqrt(hist2) / np.sum(hist2)
    
    error1 = np.sqrt(hist1) 
    error2 = np.sqrt(hist2) 
    
    # Plot histograms with error bars
    axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
    axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
    axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
    axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
    
    axs.tick_params(axis='both', which='major', labelsize=34)
    leg = axs.legend(fontsize=30,loc='upper right',markerscale=5)
    
    axs.text(0.4,75,'Quiescent Mean: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)),fontsize=30)
    # axs.text(0.4,75,'Quiescent Median: '+str(round(q_median,3)),fontsize=22)
    axs.text(0.4,70,'Non-Quiescent Mean: '+str(round(non_q_mean,1))+' ± '+str(round(non_q_std,1)),fontsize=30)
    # axs.text(0.4,65,'Non-Quiescent Median: '+str(round(non_q_median,3)),fontsize=22)
    
    plt.show()

def norm_cross_heli(enc='all',trange=None,rlim=35,tau=30,overlap=0.5,tau_unit='min',increments=30,save=True, quiescent=False, general=True):
   
    
    if trange != None: #trange for if we want to look at individual segments of data. Leaving this as None will allow for statistical checks.
       
       t0 = trange[0]
       tf = trange[1]
       
       enc = utils.encounter_check(t0)
    
   #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,24))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(1,13))
        enc.extend(range(14,24))
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc = list(range(2,25)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
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
    
    #---------------------------Setting time units-----------------------------#
    if tau_unit == 's':
        unit = u.s
    elif tau_unit == 'min':
        unit = u.min
    elif tau_unit == 'hour':
        unit = u.hr
    elif tau_unit == 'day':
        unit = u.day
        
    else:
        print("Invalid tau_unit, using minutes.")
        unit = u.min
        
    tau = tau*unit #cross helicity scale in minutes
    
    #--------------------------------------------------------------------------#
    
    duration_list = []
    
    non_q_cross_heli = []
    non_q_t_scale = []
    
    q_cross_heli = []
    q_t_scale = []
    
    q_rs = []
    nq_rs = []
    
    
    for i in enc:
        
        
        if trange == None:
            t0, tf = utils.encounter_dates(i,rlim=rlim)
        else:
            pass
        
        print("Encounter ",i)
        print(t0,tf)
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True,time_clip=True)

        span_dens_data = pyt.get_data('psp_spi_DENS')
        dens_time = span_dens_data[0]
        density = span_dens_data[1] #number density in 1/cm^3
        
        pos_data = pyt.get_data('psp_spi_SUN_DIST')
        pos_time = pos_data[0]
        position = pos_data[1]
        
        # psp.spe(trange=[t0,tf],level='L3',datatype='spa_sf0_pad',username=sweap_id,password=sweap_pass,last_version=True)
        # psp.spe(trange=[t0,tf],level='L3',datatype='spa_sf0_pad',last_version=True)
        
        # breakpoint()
        
        Rs_km = const.R_sun.to(u.km).value
        pos_rs = position/Rs_km #convert to solar radii
        
        pos_where = np.where(position<25)
        pos_where = pos_where[0]
    
        rho = (const.m_p*density/(u.cm)**3).to(u.kg/u.m**3) #mass density in kg/m^3
        res = (np.median(np.diff(dens_time))*u.s).to(unit)
        
        window_len = int(np.round(tau/res).value)
        
        time_lim_max = np.max(dens_time)
        time_lim_min = np.min(dens_time)
        
        vel_data_spi = pyt.get_data('psp_spi_VEL_RTN_SUN')
        vel_time = vel_data_spi[0]
        vel_data = vel_data_spi[1]

        # vel_nan_where = np.where(~np.isnan(vel_data[:,0]))
        # vel_nan_where = vel_nan_where[0]
        
        # vel_time_clean = vel_time[vel_nan_where]
        # vel_data_clean = vel_data[vel_nan_where,:]
        
        Vr = vel_data[:,0]
        Vt = vel_data[:,1]
        Vn = vel_data[:,2]
        V_mag_spi = np.sqrt(Vr**2+Vt**2+Vn**2)
        
        Vr_down = Vr
        Vt_down = Vt
        Vn_down = Vn
        
        V_vec = np.array([Vr_down,Vt_down,Vn_down]).T #rebuild the full 3d vector for V, to be fed into the Elsasser variable calculation
        
        
        Vr_down_slide = utils.rolling_median_time(Vr_down,vel_time,30*60,unit='s')
        Vt_down_slide = utils.rolling_median_time(Vt_down,vel_time,30*60,unit='s')
        Vn_down_slide = utils.rolling_median_time(Vn_down,vel_time,30*60,unit='s')
        
        Vr_fluc = Vr_down - Vr_down_slide
        Vt_fluc = Vt_down - Vt_down_slide
        Vn_fluc = Vn_down - Vn_down_slide
        
        V_vec_fluc = np.array([Vr_fluc,Vt_fluc,Vn_fluc]).T #average over a 30 minute median
        
        # V_mag_slide = utils.rolling_median_time(V_mag_spi,vel_time,30*60,unit='s') #using 30 minute rolling windows, should output an array of same length as the input.
        
        # V_mag_slide = utils.sliding_average_time(V_mag_spi,vel_time,30*60, overlap_percentage=0,unit='s')
        

        # V_vec_fluc = (V_vec.T-B_mag_slide).T
        
        # breakpoint()
        
        
        
        
        
        psp.fields(trange=[t0,tf],datatype='mag_RTN_4_Sa_per_Cyc',level='l2',username=fields_id,password=fields_pass,last_version=True,time_clip=True)
        mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
        mag_time_arr = mag_data[0]
        mag_data_arr = mag_data[1]
        
        mag_where = np.where((mag_time_arr>time_lim_min)&(mag_time_arr<time_lim_max))
        mag_where = mag_where[0]
        
        mag_time = mag_time_arr[mag_where]
        mag_data = mag_data_arr[mag_where,:]
        
        mag_nan_where = np.where(~np.isnan(mag_data[:,0]))
        mag_nan_where = mag_nan_where[0]
        
        mag_time_clean = mag_time[mag_nan_where]
        mag_data_clean = mag_data[mag_nan_where,:]
        
        Br = mag_data_clean[:,0]
        Bt = mag_data_clean[:,1]
        Bn = mag_data_clean[:,2]
        B = np.sqrt(Br**2+Bt**2+Bn**2)
        
        # Downsample array2 to match time_arr1
        Br_down = np.interp(dens_time, mag_time_clean, Br)*u.nT
        Bt_down = np.interp(dens_time, mag_time_clean, Bt)*u.nT
        Bn_down = np.interp(dens_time, mag_time_clean, Bn)*u.nT
        
        B_vec = np.array([Br_down,Bt_down,Bn_down]).T #using the vector without changing to alfven units
        
        
        Br_down_slide = utils.rolling_median_time(Br_down,vel_time,30*60,unit='s')
        Bt_down_slide = utils.rolling_median_time(Bt_down,vel_time,30*60,unit='s')
        Bn_down_slide = utils.rolling_median_time(Bn_down,vel_time,30*60,unit='s')
        
        Br_fluc = Br_down - Br_down_slide
        Bt_fluc = Bt_down - Bt_down_slide
        Bn_fluc = Bn_down - Bn_down_slide
        
        B_vec_fluc = np.array([Br_fluc,Bt_fluc,Bn_fluc]).T
        
        B_mag_down = np.sqrt(Br_down**2+Bt_down**2+Bn_down**2)
        # B_vec_fluc = (B_vec.T/B_mag_down).T
        
        Br_slide = utils.rolling_median_time(Br_down,vel_time,30*60,unit='s') #using 30 minute windows.
        # B_mag_slide = utils.rolling_median(B_mag_down,500) #using 30 minute windows.
        
        # B_vec_fluc = (B_vec.T/B_mag_slide).T
        
        Br_sign = np.sign(Br_slide)
        Br_sign_for_z = np.array([Br_sign,Br_sign,Br_sign]).T #create a 3 component vector the same shape as 
        
        
        Br_Va_fluc = (Br_fluc.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s) #magnetic field strength in Alfven units
        Bt_Va_fluc = (Bt_fluc.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s)
        Bn_Va_fluc = (Bn_fluc.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s)
        B_Va_fluc = np.sqrt(Br_Va_fluc**2+Bt_Va_fluc**2+Bn_Va_fluc**2)
        
        B_Va_vec_fluc = np.array([Br_Va_fluc,Bt_Va_fluc,Bn_Va_fluc]).T #vector of the fluctuations
        
        # B_Va_mag_slide = utils.rolling_median(B_Va,500)

        # B_Va_vec_fluc = (B_Va_vec.T/B_Va_mag_slide).T #we need the fluctuations
        
        # breakpoint()
        
        
        #--------------- Elsasser Variables -----------------#
        
        # vdotb_raw = np.sum(V_vec*B_vec,axis=1)
        
        # z_plus,z_min = utils.calc_elsasser_variables(V_vec,B_Va_vec,Br_sign_for_z)
        
        
        vdotb_raw = np.sum(V_vec_fluc*B_vec_fluc,axis=1)
        
        z_plus,z_min = utils.calc_elsasser_variables(V_vec_fluc,B_Va_vec_fluc,Br_sign_for_z)
        
        z_plus_abs = np.sqrt(z_plus[:,0]**2+z_plus[:,1]**2+z_plus[:,2]**2)
        z_min_abs = np.sqrt(z_min[:,0]**2+z_min[:,1]**2+z_min[:,2]**2)
        
        z_plus_dot_z_min = np.sum(z_plus * z_min, axis=1) # needed for the residual energy. z+*z-
        
        E_plus = np.abs(z_plus_abs**2)
        E_min = np.abs(z_min_abs**2)
        
        cross_numerator = z_plus_abs**2-z_min_abs**2 #equivalent 
        cross_denom = z_plus_abs**2+z_min_abs**2
        
        cont_cross_helicity = -cross_numerator/cross_denom #continuous cross helicity, only to be used for diagnostics
        
        # rho_vb = (cross_numerator/4)/(V_mag_spi*B_Va) #chen shi's 2022 measure of alfvenicity, or rho = u*v/|u||v|, again continuous so we should only use this line for diagnostics
        
        V_fluc_norm = np.linalg.norm(V_vec_fluc, axis=1)
        B_fluc_norm = np.linalg.norm(B_vec_fluc, axis=1)
        B_Va_fluc_norm = np.linalg.norm(B_Va_vec_fluc,axis=1)
        
        rho_vb = vdotb_raw/(V_fluc_norm*B_fluc_norm) #chen shi's 2022 measure of alfvenicity, or rho = u*v/|u||v|, again continuous so we should only use this line for diagnostics

        # pyt.del_data()
        
        if trange != None:
            
            # Convert unix epoch seconds to datetime for nice plotting
            time_dt = pd.to_datetime(dens_time, unit='s')   # assuming trange is your time array in unix seconds
            
            z_plus_sqrd_av = utils.sliding_average_time(z_plus_abs**2,dens_time,30*60,overlap_percentage=0,unit='s')
            z_min_sqrd_av = utils.sliding_average_time(z_min_abs**2,dens_time,30*60,overlap_percentage=0,unit='s')
            
            V_mag_B_mag_av = utils.sliding_average_time(V_fluc_norm*B_fluc_norm,dens_time,30*60,overlap_percentage=0,unit='s')
            # B_mag_av, time_indices_B = utils.sliding_average_time(B_fluc_norm,dens_time,30*60,overlap_percentage=0,unit='s')
            B_Va_mag_av = utils.sliding_average_time(B_Va_fluc_norm,dens_time,30*60,overlap_percentage=0,unit='s')
            vdotb_av= utils.sliding_average_time(vdotb_raw,dens_time,30*60,overlap_percentage=0,unit='s')
            
            time_av = time_dt[z_plus_sqrd_av['center_indices']]
            
            cross_numerator_av = z_plus_sqrd_av['averages']-z_min_sqrd_av['averages'] #equivalent to 4(u*b)
            cross_denom_av = z_plus_sqrd_av['averages']+z_min_sqrd_av['averages']
            
            cross_heli_av = -cross_numerator_av/cross_denom_av
            
            rho_vb_av = vdotb_av['averages']/(V_mag_B_mag_av['averages'])
            
            
            #-----------------------create plots-----------------------------#
            
            
            # Create the figure with multiple subplots
            fig, axs = plt.subplots(3, 1, figsize=(18, 14), sharex=True)
            fig.suptitle("Our Measure vs Tamar's Measure of Alfvénicity", fontsize=16, fontweight='bold')
            
            # Plot 1: z+ and z- magnitudes
            axs[0].plot(time_dt, Br_down, label='Br', color='tab:blue', lw=1)
            axs[0].plot(time_dt, B_mag_down,label='|B|', color='black', lw=1)
            axs[0].set_ylabel(r'Magnetic Field Vector',fontsize=26)
            axs[0].legend(loc='upper right',fontsize=20)
            axs[0].grid(True, alpha=0.3)
            
            # Plot 3: Residual energy / cross helicity
            axs[1].plot(time_dt, cont_cross_helicity, 
                        label='Continuous Cross Helicity\n$(|z^+|^2 - |z^-|^2)/(|z^+|^2 + |z^-|^2)$', 
                        color='tab:purple', lw=1)
            axs[1].plot(time_av,cross_heli_av,label='30 minute average',color='tab:red',marker='^')
            
            axs[1].set_ylabel(r'Cross Helicity $\sigma_c$',fontsize=26)
            axs[1].set_ylim(0, 1.05)
            axs[1].legend(loc='lower right',fontsize=20)
            axs[1].grid(True, alpha=0.3)
            
            # Plot 4: Alfvénicity (Chen et al. 2022 style)
            axs[2].plot(time_dt, rho_vb, 
                        label=r'$\rho_{VB} = \frac{\vec{V}\cdot\vec{B}}{|V||B|}$', 
                        color='tab:orange', lw=1)
            axs[2].plot(time_av,rho_vb_av,label='30 minute average',color='tab:blue',marker='^')
            axs[2].set_ylabel(r'Alfvénicity $\rho_{VB}$',fontsize=26)
            axs[2].set_ylim(0, 1.05)
            axs[2].legend(loc='lower right',fontsize=20)
            axs[2].grid(True, alpha=0.3)
            
            # Final formatting
            axs[2].set_xlabel('Time',fontsize=26)
            for ax in axs:
                ax.tick_params(axis='x', rotation=30,labelsize=26)
            
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # adjust for suptitle
            plt.show()
            
                    
        breakpoint()

        
        if quiescent:
            #----------------------Organizing CSVs for in Data-------------------------#
            
            csv_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
            
            csv_name = 'enc_'+str(i)+'_regions_raw.csv'
            
            q_df = pd.read_csv(csv_path+csv_name) #quiescent region Pandas dataframe
            
            q_starts = q_df['start dates'].to_numpy()
            q_ends = q_df['end dates'].to_numpy()
            q_duras = q_df['duration'].to_numpy()*u.s
            
            q_duras_tau = q_duras.to(u.min)/tau #amount that each quiet time can be divided by tau
            q_duras_tau_frac = np.round(q_duras_tau.value,decimals=0) #integers of tau that can fit into the quiet time. Determines how many times a quiet time can be split up for analysis.
            
            q_starts_intau = np.array(q_starts[q_duras_tau_frac!=0]) 
            q_ends_intau = np.array(q_ends[q_duras_tau_frac!=0]) 
            q_dura_intau = np.array(q_duras[q_duras_tau_frac!=0]) 
            q_duras_tau_non_zero = np.array(q_duras_tau_frac[q_duras_tau_frac!=0])
            
            tau_sum = int(np.sum(q_duras_tau_non_zero))
            
            q_tau_starts_flt = []
            q_tau_ends_flt = []
            tau_sec = tau.to(u.s).value
            for j in range(len(q_starts_intau)):
                # print(int(q_duras_tau_non_zero[j]))
                q_start_tmp = pys.time_float(q_starts[j])
                k=1
                while k <= int(q_duras_tau_non_zero[j]):
                    # print(k)
                    q_end_tmp = q_start_tmp+tau_sec
                    
                    q_tau_starts_flt.append(q_start_tmp)
                    q_tau_ends_flt.append(q_end_tmp)
                    
                    q_start_tmp = q_end_tmp
                    
                    k+=1
                
            
            
            #BREAK QUIET TIMES INTO TAU SEGMENTS
            
            q_tau_start_flt = np.array(q_tau_starts_flt)
            q_tau_end_flt = np.array(q_tau_ends_flt)
            
            q_start_flt = np.array(pys.time_float(q_starts))
            q_end_flt = np.array(pys.time_float(q_ends))
            
            # q_counts+=len(q_starts) #count number of q regions

            enc_ind = (i-1)
            
            perihelion_flt = per_flt[enc_ind] #float date of perihelion
            perihelion = pys.time_string(perihelion_flt) #string date of perihelion
            
            enc_ends = enc_flt[enc_ind]
            
            enc_str = pys.time_string(enc_ends[0])
            enc_end = pys.time_string(enc_ends[1])
            
            
            #-------------get rid of enc 13 ICME---------------#
            
            icme_start = pys.time_float('2022-09-06/15:25')
            icme_end = pys.time_float('2022-09-07/12:25')
            
            # icme_where = np.where(((mag_time_arr>icme_start)&(mag_time_arr<icme_end)))
            icme_where = np.where(~((dens_time>icme_start)&(dens_time<icme_end)))
            icme_where = icme_where[0]
            icme_where_check = len(icme_where)
            
            if icme_where_check > 0:
                
                dens_time = dens_time[icme_where]
                # position = position[icme_where]
                # data = data[icme_where]
                Br_Va = Br_Va[icme_where]
                Bt_Va = Bt_Va[icme_where]
                Bn_Va = Bn_Va[icme_where]
                
                Vr_down = Vr_down[icme_where]
                Vt_down = Vt_down[icme_where]
                Vn_down = Vn_down[icme_where]
                
                
                
            #-----------------------------------------------#
            # breakpoint()
            
            q_inds = []
            cut_times = []
            for k in range(len(q_start_flt)): #quick for loop to cut out whole quiecent times
                
                q_time_where = np.where((dens_time>q_start_flt[j]) & (dens_time<q_end_flt[j]))
                q_inds.extend(q_time_where[0])
                q_time_tmp = dens_time[q_time_where]
                cut_times.extend(q_time_tmp)
                
            
            for j in range(len(q_tau_start_flt)): #second longer for loop to focus on tau interval cuts.
                
                q_time_where = np.where((dens_time>q_tau_start_flt[j]) & (dens_time<q_tau_end_flt[j]))
                # q_inds.extend(q_time_where[0])
                q_time_tmp = dens_time[q_time_where]
                
                
                q_Br_Va_tmp = Br_Va[q_time_where]
                q_Bt_Va_tmp = Bt_Va[q_time_where]
                q_Bn_Va_tmp = Bn_Va[q_time_where]
                
                q_dΒr_Va = q_Br_Va_tmp - np.nanmean(q_Br_Va_tmp)
                q_dBt_Va = q_Bt_Va_tmp - np.nanmean(q_Bt_Va_tmp)
                q_dBn_Va = q_Bn_Va_tmp - np.nanmean(q_Bn_Va_tmp)
                
                q_dB_Va = np.array([q_dΒr_Va,q_dBt_Va,q_dBn_Va])
                
                q_Vr_tmp = Vr_down[q_time_where]
                q_Vt_tmp = Vt_down[q_time_where]
                q_Vn_tmp = Vn_down[q_time_where]
                
                q_dVr = q_Vr_tmp - np.nanmean(q_Vr_tmp)
                q_dVt = q_Vt_tmp - np.nanmean(q_Vt_tmp)
                q_dVn = q_Vn_tmp - np.nanmean(q_Vn_tmp)
                
                q_dV = np.array([q_dVr,q_dVt,q_dVn])
                
                if q_dB_Va.size == 0:
                    q_cross_heli.append(np.nan)
                    
                else:
                
                    q_dVdB = np.nanmean(np.sum(q_dB_Va*q_dV,axis=0))
                    q_dV2 = np.nanmean(np.sum(q_dV*q_dV,axis=0))
                    q_dB2 = np.nanmean(np.sum(q_dB_Va*q_dB_Va,axis=0))
                    
                    q_cross_heli.append(2*q_dVdB/(q_dV2+q_dB2))
                    
                # q_t_scale.append(q_duras[j])
                q_t_scale.append(tau_sec)
                
                q_time_span_where = np.where((pos_time>q_tau_start_flt[j]) & (pos_time<q_tau_end_flt[j]))
                q_time_span_where = q_time_span_where[0]
                q_rs_pos = pos_rs[q_time_span_where]
                q_rs.append(np.nanmean(q_rs_pos))
                
            # breakpoint()
            non_q_time_arr = np.delete(dens_time,q_inds)
            non_q_Br = np.delete(Br,q_inds)
            non_q_Bt = np.delete(Bt,q_inds)
            non_q_Bn = np.delete(Bn,q_inds)
            
            
            n = 2
            iii = 0
            while iii<n:
                
                for j in range(len(q_tau_start_flt)):
                    
                    incheck = [True]
                    ii = 0
                    while any(incheck): #checking for data that is not in the quiescent region list
                        random_index = np.random.choice(non_q_time_arr)
                        # non_q_where = np.where((dens_time>random_index) & (dens_time<(random_index+q_duras[j])))
                        non_q_where = np.where((dens_time>random_index) & (dens_time<(random_index+tau_sec)))
                        non_q_time_tmp = dens_time[non_q_where]
                        incheck = np.isin(non_q_time_tmp,cut_times)
                        
                        # non_q_span_where = np.where((pos_time>random_index) & (pos_time<(random_index+q_duras[j])))
                        non_q_span_where = np.where((pos_time>random_index) & (pos_time<(random_index+tau_sec)))
                        non_q_span_time_tmp = pos_time[non_q_span_where]
                        
                        
                        if ii >= 15: #if the random check is taking too long, then just look for spots where data of the proper length can be found
                            zeros = np.zeros(dens_time.shape)
                            index_arr = np.array(range(len(dens_time)))
                            del_where = np.where(np.isin(dens_time,cut_times))
                            del_inds = del_where[0]
                            # time_check = np.delete(mag_time_arr,del_inds)
                            # ind_check = np.delete(index_arr,del_inds)
                            zeros[del_where] = 1
                            groups = np.array(utils.group_zeros(zeros))
                            times = dens_time[groups]
                            
                            lengths = []
                            for ijk in dens_time[groups]:
                                lengths.extend(np.diff(ijk))
                                
                            lengthwhere = np.where(lengths>q_duras[j])
                            lengthwhere = lengthwhere[0]
                            
                            group_tmp = groups[lengthwhere]
                            times_tmp = times[lengthwhere]
                            
                            random_index = np.random.choice(np.arange(len(group_tmp)))
                            time_index = times_tmp[random_index,0]
                            
                            non_q_where = np.where((dens_time>time_index) & (dens_time<(time_index+q_duras[j])))
                            non_q_time_tmp = dens_time[non_q_where]
                            incheck = np.isin(non_q_time_tmp,cut_times)
                            
                            non_q_span_where = np.where((pos_time>time_index) & (pos_time<(time_index+q_duras[j])))
                            non_q_span_time_tmp = pos_time[non_q_span_where]

                        ii+=1
                            
                            
                    cut_times.extend(non_q_time_tmp)
                    # print('done',ii)
                    
                    # non_q_time_tmp = mag_time_arr[non_q_time_where]
                    non_q_Br_Va_tmp = Br_Va[non_q_where]
                    non_q_Bt_Va_tmp = Bt_Va[non_q_where]
                    non_q_Bn_Va_tmp = Bn_Va[non_q_where]
                    
                    non_q_dBr_Va = non_q_Br_Va_tmp - np.nanmean(non_q_Br_Va_tmp)
                    non_q_dBt_Va = non_q_Bt_Va_tmp - np.nanmean(non_q_Bt_Va_tmp)
                    non_q_dBn_Va = non_q_Bn_Va_tmp - np.nanmean(non_q_Bn_Va_tmp)
                    
                    non_q_dB_Va = np.array([non_q_dBr_Va,non_q_dBt_Va,non_q_dBn_Va])
                    
                    
                    non_q_Vr_tmp = Vr_down[non_q_where]
                    non_q_Vn_tmp = Vn_down[non_q_where]
                    non_q_Vt_tmp = Vt_down[non_q_where]
                    
                    non_q_dVr = non_q_Vr_tmp - np.nanmean(non_q_Vr_tmp)
                    non_q_dVt = non_q_Vn_tmp - np.nanmean(non_q_Vn_tmp)
                    non_q_dVn = non_q_Vt_tmp - np.nanmean(non_q_Vt_tmp)
                    
                    non_q_dV = np.array([non_q_dVr,non_q_dVt,non_q_dVn])
                    
                    if non_q_dB_Va.size == 0:
                        non_q_cross_heli.append(np.nan)
                        print('ja')
                    else:
                    
                        # breakpoint()
                    
                        non_q_dVdB = np.nanmean(np.sum(non_q_dB_Va*non_q_dV,axis=0))
                        non_q_dV2 = np.nanmean(np.sum(non_q_dV*non_q_dV,axis=0))
                        non_q_dB2 = np.nanmean(np.sum(non_q_dB_Va*non_q_dB_Va,axis=0))
                        
                        non_q_cross_heli.append(2*non_q_dVdB/(non_q_dV2+non_q_dB2))
                                        
                    nq_rs_pos = pos_rs[non_q_span_where]
                    nq_rs.append(np.nanmean(nq_rs_pos))
                    # non_q_t_scale.append(q_duras[j])
                iii+=1
                    
            
        
        if general:
        
            Br_Va_av, Br_ind = utils.sliding_average(Br_Va.value,window_len,overlap)
            Bt_Va_av, Bt_ind = utils.sliding_average(Bt_Va.value,window_len,overlap)
            Bn_Va_av, Bn_ind = utils.sliding_average(Bn_Va.value,window_len,overlap)
            
            delBr = np.array([])
            delBt = np.array([])
            delBn = np.array([])
            for ind in range(len(Br_ind)-1):
                start = Br_ind[ind]
                stop = Br_ind[ind+1]
                
                delBr_i = Br_Va[start:stop].value-Br_Va_av[ind]
                delBr = np.append(delBr,delBr_i)       
                
                delBt_i = Bt_Va[start:stop].value-Bt_Va_av[ind]
                delBt = np.append(delBt,delBt_i) 
                
                delBn_i = Bn_Va[start:stop].value-Bn_Va_av[ind]
                delBn = np.append(delBn,delBn_i) 
                
                if ind == len(Br_ind)-2:
                    delBr_i = Br_Va[stop:].value-Br_Va_av[ind+1]
                    delBr = np.append(delBr,delBr_i)
                    
                    delBt_i = Bt_Va[stop:].value-Bt_Va_av[ind+1]
                    delBt = np.append(delBt,delBt_i) 
                    
                    delBn_i = Bn_Va[stop:].value-Bn_Va_av[ind+1]
                    delBn = np.append(delBn,delBn_i) 
            
            delB = np.array([delBr,delBt,delBn]).T
            
            
            Vr_av, Vr_ind = utils.sliding_average(Vr_down.value,window_len,overlap)
            Vt_av, Vt_ind = utils.sliding_average(Vt_down.value,window_len,overlap)
            Vn_av, Vn_ind = utils.sliding_average(Vn_down.value,window_len,overlap)
            
            delVr = np.array([])
            delVt = np.array([])
            delVn = np.array([])
            for ind in range(len(Vr_ind)-1):
                start = Vr_ind[ind]
                stop = Vr_ind[ind+1]
                
                delVr_i = Vr_down[start:stop].value-Vr_av[ind]
                delVr = np.append(delVr,delVr_i)       
                
                delVt_i = Vt_down[start:stop].value-Vt_av[ind]
                delVt = np.append(delVt,delVt_i) 
                
                delVn_i = Vn_down[start:stop].value-Vn_av[ind]
                delVn = np.append(delVn,delVn_i) 
                
                if ind == len(Vr_ind)-2:
                    delVr_i = Vr_down[stop:].value-Vr_av[ind+1]
                    delVr = np.append(delVr,delVr_i)
                    
                    delVt_i = Vt_down[stop:].value-Vt_av[ind+1]
                    delVt = np.append(delVt,delVt_i) 
                    
                    delVn_i = Vn_down[stop:].value-Vn_av[ind+1]
                    delVn = np.append(delVn,delVn_i) 
            
            delV = np.array([delVr,delVt,delVn]).T
            
            # breakpoint()
            
            delVdelB = np.sum(delV*delB, axis=1)
            delV_2 = np.sum(delV*delV, axis=1)
            delB_2 = np.sum(delB*delB, axis=1)
            
            cross_heli = 2*delVdelB/(delV_2+delB_2) #WTF???
            
            cross_heli_av, cross_heli_ind = utils.sliding_average(cross_heli,window_len,overlap)
            cross_heli_time = dens_time[Br_ind]
            
            fig = plt.figure(figsize=(30,15))
    
            axs = fig.add_subplot(111)
            
            axs.set_title("Normalized Cross Helicity, "+str(np.round(tau)),fontsize=38)
            axs.set_ylabel('$\sigma_{c}$',fontsize=38)
            
            # # Convert epoch timestamps to datetime
            # epoch_timestamps = np.array(beta_par_time)
            # datetime_series = pd.to_datetime(epoch_timestamps, unit='s')
            
            axs.plot(cross_heli_time,cross_heli_av)
            
            plt.show()
            
            
            if save:
                
                save_path = '/Users/besh2109/Desktop/Cross Heli/tplot_files/'
                save_name_cross = 'cross_heli_enc_'+str(i)+'.cdf'
                
                
                pyt.store_data("cont_cross_heli",data={'x':dens_time, 'y':cross_heli})
                
                pyt.store_data("norm_cross_heli",data={'x':cross_heli_time, 'y':cross_heli_av})
    
                
                cdf_var_list = ["cont_cross_heli","norm_cross_heli"]
                
                pyt.tplot_save(cdf_var_list,save_path+save_name_cross)
                
    breakpoint()


    fig, axs_arr = plt.subplots(3, 1, figsize=(25,30), sharex=True)
    
    for j, i in enumerate([35, 25, 15]):
        ax = axs_arr[j]  # Current axis for this subplot
        
        # x_datas = [np.abs(non_q_cross_heli),np.abs(q_cross_heli)]
        # x_datas = [non_q_cross_heli,q_cross_heli]
        r_datas = [nq_rs,q_rs]
        rs_lim=i
        
        nq_rs_arr = np.array(nq_rs)
        q_rs_arr = np.array(q_rs)
        
        q_rs_where = np.where(q_rs_arr<rs_lim)
        q_rs_where = q_rs_where[0]
        nq_rs_where = np.where(nq_rs_arr<rs_lim)
        nq_rs_where = nq_rs_where[0]
        
        x_datas = [np.array(non_q_cross_heli)[nq_rs_where],np.array(q_cross_heli)[q_rs_where]]
        # x_datas = [np.abs(np.array(non_q_cross_heli)[nq_rs_where]),np.abs(np.array(q_cross_heli)[q_rs_where])]
        num_q = np.sum(~np.isnan(np.array(q_cross_heli)[q_rs_where]))
        num_nq = np.sum(~np.isnan(np.array(non_q_cross_heli)[nq_rs_where]))
        
        print(num_q)
        print(num_nq)
    
        r_labs_in = ['8','15','25','35']
        r_labs_out = ['15','25','35','45']
        
        r_labs_in.reverse()
        r_labs_out.reverse()
    
        labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
        titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
    
        cmaps = ['autumn','autumn']
        colors = ['tab:blue','tab:orange']
        
        bins = np.linspace(-1,1,30)
        # bins = np.linspace(0,1,30)
        hist1, _ = np.histogram(x_datas[0], bins=bins)
        hist2, _ = np.histogram(x_datas[1], bins=bins)
        
        if j==0:
            hist2_0 = np.array(hist2)
        
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        
        # hist1_norm = hist1
        # hist2_norm = hist2
        
        q_mean = np.nanmean(q_cross_heli)
        q_std = np.nanstd(q_cross_heli)
        q_median = np.nanmedian(q_cross_heli)
        non_q_mean = np.nanmean(non_q_cross_heli)
        non_q_std = np.nanstd(non_q_cross_heli)
        non_q_median = np.nanmedian(non_q_cross_heli)
    
        
        # Calculate error bars for each bin (normalized)
        error1 = np.sqrt(hist1) / np.sum(hist1)
        error2 = np.sqrt(hist2) / np.sum(hist2)
        
        # error1 = np.sqrt(hist1)
        # error2 = np.sqrt(hist2)
        
        # Plot histograms with error bars
        ax.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
        ax.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
        ax.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
        ax.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
        
        ax.tick_params(axis='both', which='major', labelsize=34)
        
        if j==0:  # Top subplot
            ax.set_title("Normalized Cross Helicity Histograms",fontsize=38)
            leg = ax.legend(fontsize=30,loc='upper right',markerscale=5)
            
        ax.set_ylabel('Normalized Counts',fontsize=38)
        # ax.set_xlabel('|$\sigma_c$|',fontsize=38)
        if j==2:  # Bottom subplot
            ax.set_xlabel('$\sigma_c$',fontsize=38)
        
        ax.set_ylim(0,0.16)
        ax.text(-1.07,0.14,'Quiescent \u03c4: '+str(num_q),fontsize=35)
        ax.text(-1.07,0.125,'Non-quiescent \u03c4: '+str(num_nq),fontsize=35)
        ax.text(-1.07,0.1,'Radial Distance: <'+str(rs_lim)+' $R_\odot$',fontsize=35)
        # ax.annotate('Number of quiescent \u03c4: '+str(n_q), xy=(45, 750), xycoords='axes pixels', fontsize=45)
        # Calculate p-value for quiescent data vs uniform distribution
        num_bins = len(hist2)
        expected_rand = np.full(num_bins, num_q / num_bins)
        

        
        chi2_stat, p_value = chisquare(hist2,expected_rand)
        
        

        
        hist2_self_comp = np.append(hist2_0[:14],np.flip(hist2_0[:15]))
        
        scale_fact = np.sum(hist2_self_comp)/np.sum(hist2_0)
        expected_in_out = hist2_self_comp/scale_fact
        
        chi2_stat_0, p_value_0 = chisquare(hist2_0,expected_in_out)
        
        print(f'p-value (vs uniform): {p_value:.3e} at {rs_lim}')
        print(f'p-value (vs high stat dist): {p_value_0:.3e} at {rs_lim}')
        # ax.text(-1.07, 0.075, f'p-value (vs uniform): {p_value:.3e}', fontsize=35)
        
    fig.subplots_adjust(hspace=0.03)
    plt.show()
    
    
    
    fig, axs_arr = plt.subplots(1, 1, figsize=(25,10), sharex=True)
    
    ax = axs_arr  # Current axis for this subplot
    
    # Plot histograms with error bars
    ax.bar(bins[:-1], hist2, width=np.diff(bins), align='center', alpha=0.5, label='<15 $R_\odot$ distribution',edgecolor='black',color='tab:orange')
    ax.bar(bins[:-1], expected_rand, width=np.diff(bins), align='center', alpha=0.5, label='Null distribution',edgecolor='black',color='tab:green')
    ax.errorbar(bins[:-1], hist2, yerr=np.sqrt(hist2), fmt='none', color='k', capsize=3)
    # ax.errorbar(bins[:-1], expected_rand, yerr=np.sqrt(expected_rand), fmt='none', color='k', capsize=3)
    
    ax.text(-0.44, 15, f'p-value (vs null): {p_value:.3e}', fontsize=25)
    ax.set_ylabel('Counts',fontsize=38)
    ax.set_title("Normalized Cross Helicity Statistical Test",fontsize=38)
    ax.set_xlabel('$\sigma_c$',fontsize=38)
    ax.tick_params(axis='both', which='major', labelsize=34)
    
    leg = ax.legend(fontsize=30,loc='upper left',markerscale=5)
    
    plt.show()
    
    
    
    fig, axs_arr = plt.subplots(1, 1, figsize=(25,10), sharex=True)
    
    ax = axs_arr  # Current axis for this subplot
    
    # Plot histograms with error bars
    ax.bar(bins[:-1], hist2_0, width=np.diff(bins), align='center', alpha=0.5, label='<35 $R_\odot$ distribution',edgecolor='black',color='tab:orange')
    ax.bar(bins[:-1], expected_in_out, width=np.diff(bins), align='center', alpha=0.5, label='Mirrored distribution',edgecolor='black',color='tab:green')
    ax.errorbar(bins[:-1], hist2_0, yerr=np.sqrt(hist2_0), fmt='none', color='k', capsize=3)
    # ax.errorbar(bins[:-1], expected_in_out, yerr=np.sqrt(expected_in_out), fmt='none', color='k', capsize=3)
    
    ax.text(-0.44, 60, f'p-value (vs mirror): {p_value_0:.3e}', fontsize=25)
    ax.set_ylabel('Counts',fontsize=38)
    ax.set_title("Normalized Cross Helicity Statistical Test",fontsize=38)
    ax.set_xlabel('$\sigma_c$',fontsize=38)
    ax.tick_params(axis='both', which='major', labelsize=34)
    
    leg = ax.legend(fontsize=30,loc='upper left',markerscale=5)
    
    plt.show()
    # ax.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
    # ax.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
    