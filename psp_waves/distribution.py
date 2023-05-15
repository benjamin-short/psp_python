#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 16:48:41 2021

@author: besh2109
"""

import numpy as np
from numpy.linalg import inv
from scipy.io import readsav
from scipy import integrate
import pyspedas as pys
import pytplot as pyt
import os
import pandas as pd
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import cdflib

def distribution(trange=['2018-11-07/07:40:00','2018-11-07/08:20:00']):
    
    me = 9.10938356*1e-31 # kg
    mp = 1.672621898*1e-27 # kg
    
    evtoJ = 1.6022e-19 #eV to J conversion
    
    pys.psp.fields(trange=trange, datatype='dfb_ac_spec', level='l2')
    
    pys.psp.fields(trange=trange, datatype='mag_SC_4_Sa_per_Cyc',level='l2',last_version=True)
    
    mag_data = pyt.get_data('psp_fld_l2_mag_SC_4_Sa_per_Cyc')
    
    mag_time_arr = mag_data[0]
    mag_data_arr = mag_data[1]
    
    date_flt_str = pys.time_float(trange[0])
    date_flt_end = pys.time_float(trange[1])
    
    date_form_1 = pys.time_string(date_flt_str,fmt='/%Y/%m/')
    date_form_2 = pys.time_string(date_flt_str,fmt='%Y%m%d')
    
    rot_path = '/Users/besh2109/spedas_data/psp/data/sci/sweap/spi/L3/spi_sf00'+date_form_1
    rot_file = 'psp_swp_spi_sf00_L3_mom_INST_'+date_form_2+'_v02.cdf' 
                
    if os.path.isfile(rot_path+rot_file):
        pyt.cdf_to_tplot(rot_path+rot_file)
    else:
        pys.psp.spi(trange=trange, datatype='spi_sf00', level='L3')
        
    tens_data = pyt.get_data('T_TENSOR') #given in eV
                
    tens_time_arr = tens_data[0]
    tens_data_arr = tens_data[1]
    
    rot_cdf = cdflib.CDF(rot_path+rot_file)
    
    rot_mat_ins_sc = rot_cdf.varget('ROTMAT_SC_INST')
    rot_mat_ins_sc_inv = inv(rot_mat_ins_sc)
    
    dens_data = pyt.get_data('DENS')
    
    dens_time_arr = dens_data[0]
    dens_data_arr = dens_data[1]
    
    temp_path = '/Users/besh2109/Desktop/psp_core_temp/'
    temp_file = 'coret_e1toe8.tplot'
    
    drift_path = '/Users/besh2109/Desktop/psp_core_drift/'
    drift_file = 'coredrift_e1toe8.tplot'
    
    pyt.tplot_restore(temp_path+temp_file)
    pyt.tplot_restore(drift_path+drift_file)
    
    par_data_tmp = pyt.get_data('coretpar')
    per_data_tmp = pyt.get_data('coretperp')
    
    tpar_time_arr = par_data_tmp[0]
    tpar_data_arr = par_data_tmp[1]
    
    tper_time_arr = per_data_tmp[0]
    tper_data_arr = per_data_tmp[1]
    
    drift_data_tmp = pyt.get_data('coredrift')
    
    drift_time_arr = drift_data_tmp[0]
    drift_data_arr = drift_data_tmp[1]
    
    """ proton temp data """
    
    tens_where = np.where((tens_time_arr > date_flt_str) & (tens_time_arr < date_flt_end))
    tens_where = tens_where[0]
                        
    tens_ti = np.array(tens_time_arr[tens_where])
    tens = np.array(tens_data_arr[tens_where,:])
    
    tensor = []
    for p in range(len(tens_where)):
        
        Txx = tens[p,0]
        Tyy = tens[p,1]
        Tzz = tens[p,2]
        Txy = tens[p,3]
        Txz = tens[p,4]
        Tyz = tens[p,5]
        
        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
        
        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
        tensor.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))

    tensor = np.array(tensor)
    
    tens_where_hr = np.where((tens_time_arr > date_flt_str-3600.) & (tens_time_arr < date_flt_str))
    tens_where_hr = tens_where_hr[0]
    
    tens_hr = np.array(tens_data_arr[tens_where_hr,:])
    
    tensor_hr = []
    for p in range(len(tens_where_hr)):
        
        Txx = tens_hr[p,0]
        Tyy = tens_hr[p,1]
        Tzz = tens_hr[p,2]
        Txy = tens_hr[p,3]
        Txz = tens_hr[p,4]
        Tyz = tens_hr[p,5]
        
        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
        
        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
        tensor_hr.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))

    tensor_hr = np.array(tensor_hr)
    
    """ proton density data """
    
    dens_where = np.where((dens_time_arr > date_flt_str) & (dens_time_arr < date_flt_end))
    dens_where = dens_where[0]
    
    dens_ti = np.array(dens_time_arr[dens_where])
    dens = np.array(dens_data_arr[dens_where])
    
    dens_where_hr = np.where((dens_time_arr > date_flt_str-3600.) & (dens_time_arr < date_flt_str))
    dens_where_hr = dens_where_hr[0]
    
    dens_ti_hr = np.array(dens_time_arr[dens_where_hr])
    dens_hr = np.array(dens_data_arr[dens_where_hr])
    
    """ mag data """

    mag_where = np.where((mag_time_arr > date_flt_str) & (mag_time_arr < date_flt_end))
    mag_where = mag_where[0]
    
    mag_ti = np.array(mag_time_arr[mag_where])
    mag_scx = np.array(mag_data_arr[mag_where,0])
    mag_scy = np.array(mag_data_arr[mag_where,1])
    mag_scz = np.array(mag_data_arr[mag_where,2])
    
    mag_where_hr = np.where((mag_time_arr > date_flt_str-3600.) & (mag_time_arr < date_flt_str))
    mag_where_hr = mag_where_hr[0]
    
    mag_scx_hr = np.array(mag_data_arr[mag_where_hr,0])
    mag_scy_hr = np.array(mag_data_arr[mag_where_hr,1])
    mag_scz_hr = np.array(mag_data_arr[mag_where_hr,2])
    
    """ core drift data """
                    
    drift_where = np.where((drift_time_arr >date_flt_str) & (drift_time_arr < date_flt_end))
    drift_where = drift_where[0]

    drift_ti = np.array(drift_time_arr[drift_where])
    drift_data = np.array(drift_data_arr[drift_where])
    
    drift_where_hr = np.where((drift_time_arr >date_flt_str-3600.) & (drift_time_arr < date_flt_str))
    drift_where_hr = drift_where_hr[0]
    
    drift_data_hr = np.array(drift_data_arr[drift_where_hr])
    
    """ core temp data """
    
    tpar_where = np.where((tpar_time_arr > date_flt_str) & (tpar_time_arr < date_flt_end))
    tpar_where = tpar_where[0]
    
    tpar_ti = np.array(tpar_time_arr[tpar_where])
    tpar_data = np.array(tpar_data_arr[tpar_where])
    
    tper_where = np.where((tper_time_arr > date_flt_str) & (tper_time_arr < date_flt_end))
    tper_where = tper_where[0]
    
    tper_ti = np.array(tper_time_arr[tper_where])
    tper_data = np.array(tper_data_arr[tper_where])
    
    temp_data = (2*tper_data+tpar_data)/3
    
    tpar_where_hr = np.where((tpar_time_arr > date_flt_str-3600.) & (tpar_time_arr < date_flt_str))
    tpar_where_hr = tpar_where_hr[0]
    
    tpar_data_hr = np.array(tpar_data_arr[tpar_where_hr])
    
    tper_where_hr = np.where((tper_time_arr > date_flt_str-3600.) & (tper_time_arr < date_flt_str))
    tper_where_hr = tper_where_hr[0]
    
    tper_data_hr = np.array(tper_data_arr[tper_where_hr])
    
    temp_data_hr = (2*tper_data_hr + tpar_data_hr)/3
    
    """ magnetic field normalization """
    
    n_tens_bins = len(tens_ti)
    mag_bin_size = len(mag_ti)/n_tens_bins
    ind_arr = np.arange(len(mag_ti))
    norm_mag_time = np.zeros((n_tens_bins,))
    norm_mag_scx_val = np.zeros((n_tens_bins,))
    norm_mag_scy_val = np.zeros((n_tens_bins,))
    norm_mag_scz_val = np.zeros((n_tens_bins,))
    
    tens_fa = []    
    
    for m in range(n_tens_bins):
        
        
        win_str_mag = m*mag_bin_size
        win_end_mag = (m+1)*mag_bin_size
        wind_where = np.where((ind_arr>=win_str_mag)&(ind_arr<win_end_mag))
        wind_where = wind_where[0]
        mag_time_val_tmp = np.median(mag_ti[wind_where])
        mag_scx_val_tmp = np.median(mag_scx[wind_where])
        mag_scy_val_tmp = np.median(mag_scy[wind_where])
        mag_scz_val_tmp = np.median(mag_scz[wind_where])
        norm_mag_time[m] = mag_time_val_tmp
        norm_mag_scx_val[m] = mag_scx_val_tmp
        norm_mag_scy_val[m] = mag_scy_val_tmp
        norm_mag_scz_val[m] = mag_scz_val_tmp
        
        Bx = mag_scx_val_tmp
        By = mag_scy_val_tmp
        Bz = mag_scz_val_tmp
        B = np.sqrt(Bx**2+By**2+Bz**2)
        Bx_By = np.sqrt(Bx**2+By**2)
        
        rot_mat = np.array([[Bx/B,    By/B,                         Bz/B],\
                           [0,        Bx_By*Bz/(By**2+Bz**2),      -Bx_By*By/(By**2+Bz**2)],\
                           [-Bx_By/B, Bx*By*Bx_By/(B*(By**2+Bz**2)),Bx*Bz*Bx_By/(B*(By**2+Bz**2))]])#rotation matrix from SC to FA
            
        rot_inv = inv(rot_mat)
        tmp_tens = np.dot(rot_mat,tensor[m]) #first calculation of T' = R T R^-1
        tens_fa.append(np.dot(tmp_tens,rot_inv)) #temperature tensor rotated to FA coords.

    tens_fa = np.array(tens_fa)
    
    T_perp_I = np.array(tens_fa[:,1,1]+tens_fa[:,2,2])/2
    T_par_I = np.array(tens_fa[:,0,0])
    T_anis = T_perp_I/T_par_I
    
    """ first hour mag normalization """
    
    n_tens_bins_hr = len(temp_data_hr)
    mag_bin_hr_size = len(mag_scx_hr)/n_tens_bins_hr
    ind_hr_arr = np.arange(len(mag_scx_hr))
    norm_mag_time = np.zeros((n_tens_bins_hr,))
    norm_mag_scx_val = np.zeros((n_tens_bins_hr,))
    norm_mag_scy_val = np.zeros((n_tens_bins_hr,))
    norm_mag_scz_val = np.zeros((n_tens_bins_hr,))
    
    tens_fa_hr = []    
    
    for m in range(n_tens_bins_hr):
        
        
        win_str_mag = m*mag_bin_hr_size
        win_end_mag = (m+1)*mag_bin_hr_size
        wind_where = np.where((ind_hr_arr>=win_str_mag)&(ind_hr_arr<win_end_mag))
        wind_where = wind_where[0]
        #mag_time_val_tmp = np.median(mag_time_arr[wind_where])
        mag_scx_val_tmp = np.median(mag_scx_hr[wind_where])
        mag_scy_val_tmp = np.median(mag_scy_hr[wind_where])
        mag_scz_val_tmp = np.median(mag_scz_hr[wind_where])
        #norm_mag_time[m] = mag_time_val_tmp
        norm_mag_scx_val[m] = mag_scx_val_tmp
        norm_mag_scy_val[m] = mag_scy_val_tmp
        norm_mag_scz_val[m] = mag_scz_val_tmp
        
        Bx = mag_scx_val_tmp
        By = mag_scy_val_tmp
        Bz = mag_scz_val_tmp
        B = np.sqrt(Bx**2+By**2+Bz**2)
        Bx_By = np.sqrt(Bx**2+By**2)
        
        rot_mat = np.array([[Bx/B,    By/B,                         Bz/B],\
                           [0,        Bx_By*Bz/(By**2+Bz**2),      -Bx_By*By/(By**2+Bz**2)],\
                           [-Bx_By/B, Bx*By*Bx_By/(B*(By**2+Bz**2)),Bx*Bz*Bx_By/(B*(By**2+Bz**2))]])#rotation matrix from SC to FA
            
        rot_inv = inv(rot_mat)
        tmp_tens = np.dot(rot_mat,tensor_hr[m]) #first calculation of T' = R T R^-1
        tens_fa_hr.append(np.dot(tmp_tens,rot_inv)) #temperature tensor rotated to FA coords.

    tens_fa_hr = np.array(tens_fa_hr)
    
    T_perp_I_hr = np.array(tens_fa_hr[:,1,1]+tens_fa_hr[:,2,2])/2
    T_par_I_hr = np.array(tens_fa_hr[:,0,0])
    T_anis_hr = T_perp_I_hr/T_par_I_hr
    
    """ distribution plots """
    
    range1 = -400
    range2 = 400
    res = 1
    
    
    
    v_drift_hr = -np.nanmedian(drift_data_hr) #km/s
    vth_par_e_hr = np.sqrt((2/me)*np.nanmedian(tpar_data_hr)*evtoJ)*1e-3 #km/s
    vth_per_e_hr = np.sqrt((2/me)*np.nanmedian(tper_data_hr)*evtoJ)*1e-3 #km/s

    v_par = np.arange(range1, range2, res)
    v_per = np.arange(range1, range2, res)
    v_par, v_per = np.meshgrid(v_par, v_per)

    
    #surf = ax.plot_surface(v_par, v_per, core_dist_d, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fe_hr = lambda y, x: np.e**(-(x - v_drift_hr)**2/(vth_par_e_hr**2) - y**2/vth_per_e_hr**2)
    elec_integr_hr = integrate.dblquad(fe_hr, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    vth_par_I_hr = np.sqrt((2/mp)*np.nanmedian(T_par_I_hr)*evtoJ)*1e-3 #km/s
    vth_per_I_hr = np.sqrt((2/mp)*np.nanmedian(T_perp_I_hr)*evtoJ)*1e-3 #km/s

    fi_hr = lambda y, x: np.e**(-(x)**2/(vth_par_I_hr**2) - y**2/vth_per_I_hr**2)
    ion_integr_hr = integrate.dblquad(fi_hr, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    
    Ai_hr = np.median(dens_hr)/(np.pi*vth_par_I_hr*vth_per_I_hr)
    Ae_hr = ion_integr_hr[0]/elec_integr_hr[0]*Ai_hr

    core_dist_hr = Ae_hr*np.e**(-(v_par - v_drift_hr)**2/(vth_par_e_hr**2) - v_per**2/vth_per_e_hr**2)

    v_par = np.arange(range1, range2, res)
    v_per = np.arange(range1, range2, res)
    v_par, v_per = np.meshgrid(v_par, v_per)
    ion_dist_hr = Ai_hr*np.e**(-(v_par)**2/(vth_par_I_hr**2) - v_per**2/vth_per_I_hr**2)
    
    
    fig, ax = plt.subplots(figsize=(15,15)) #subplot_kw={"projection": "3d"},
    
    v_drift = -np.nanmedian(drift_data) #km/s
    vth_par_e = np.sqrt((2/me)*np.nanmedian(tpar_data)*evtoJ)*1e-3 #km/s
    vth_per_e = np.sqrt((2/me)*np.nanmedian(tper_data)*evtoJ)*1e-3 #km/s

    v_par = np.arange(range1, range2, res)
    v_per = np.arange(range1, range2, res)
    v_par, v_per = np.meshgrid(v_par, v_per)

    
    fe = lambda y, x: np.e**(-(x - v_drift)**2/(vth_par_e**2) - y**2/vth_per_e**2)
    elec_integr = integrate.dblquad(fe, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)

    vth_par_I = np.sqrt((2/mp)*np.nanmedian(T_par_I)*evtoJ)*1e-3 #km/s
    vth_per_I = np.sqrt((2/mp)*np.nanmedian(T_perp_I)*evtoJ)*1e-3 #km/s

    fi = lambda y, x: np.e**(-(x)**2/(vth_par_I**2) - y**2/vth_per_I**2)
    ion_integr = integrate.dblquad(fi, -np.inf, np.inf, lambda x: -np.inf, lambda x: np.inf)
    
    Ai = np.median(dens)/(np.pi*vth_par_I*vth_per_I)
    Ae = ion_integr[0]/elec_integr[0]*Ai
    
    core_dist = Ae*np.e**(-(v_par - v_drift)**2/(vth_par_e**2) - v_per**2/vth_per_e**2)
    v_par = np.arange(range1, range2, res)
    v_per = np.arange(range1, range2, res)
    v_par, v_per = np.meshgrid(v_par, v_per)
    ion_dist = Ai*np.e**(-(v_par)**2/(vth_par_I**2) - v_per**2/vth_per_I**2)
    #print(ion_dist.shape)
    
    surf1 = ax.contour(v_par,v_per,((core_dist_hr+ion_dist_hr) - (core_dist+ion_dist)),75)
    fig.colorbar(surf1, ax=ax)
    #surf1 = ax.plot_surface(v_per, v_par, (core_dist_d) - (core_dist), cmap=cm.coolwarm, linewidth=0, antialiased=False)
    #ax.clabel(surf1, inline=True, fontsize=8)
    ax.set_title('Outside Distribution minus Inside Distribution '+trange[0])
    ax.set_xlabel('Vperp')
    ax.set_ylabel('Vpara')
    #ax.view_init(-140, 60)
    
    #ax.set_zlim(0, 1.2)
    plt.show()
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"},figsize=(15,15))
    #vmax = 0.001
    surf1 = ax.plot_surface(v_par,v_per, (ion_dist_hr+core_dist_hr)-(ion_dist+core_dist), cmap=cm.coolwarm, linewidth=0, antialiased=False)#(core_dist_hr+ion_dist_hr) - 
    #ax.view_init(azim=10, elev=35)
    #ax.set_zlim(0, vmax)
    ax.set_title('Outside Distribution minus Inside Distribution ')
    ax.set_xlabel('Vperp')
    ax.set_ylabel('Vpara')
    plt.show()
    
    v_par = np.arange(range1, range2, res)
    v_per = np.arange(range1, range2, res)
    urmom = (ion_dist_hr+core_dist_hr)-(ion_dist+core_dist)
    
    return v_par,v_per,urmom
    
    