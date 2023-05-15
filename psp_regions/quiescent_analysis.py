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

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

from .davidtensor import david_rot_mat
from .davidtensor import david_anis
from .davidtensor import steven_anis

from numpy.linalg import inv

import matplotlib.patches as mpatch

import re
import pandas as pd

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

Rs_to_AU = 0.00465047 # 1 Rs = 0.00465047 AU

def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z
    
# def t_r_plot(enc='all',enc_radius=67,data='p',anis=False,drift=False,tper=False,tpar=False):
def t_r_plot(enc='all',enc_radius=67,atype='v'): 
    
    #SOON TO COME: 'estrahl'
    
    ptypes = ['ptemp','panis','ptpar','ptper','pdens']
    etypes = ['etemp','eanis','edrift']
    qtntypes = ['qtndens','qtntemp']
    vtypes = ['v']
    
    
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
    Rtfull = np.array([])
    Rt_AU = np.array([])
    timefull = np.array([])
    
    for i in enc:
        
        # print(i)
        
        enc_ind = (i-1)
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        if i == 7:
            enc_str ='2021-01-13/04:48:00'
            enc_end ='2021-01-18/00:00:00'
            enc_ends[0] = pys.time_float('2021-01-13/04:48:00')
            enc_ends[1] = pys.time_float('2021-01-18/00:00:00')
            
        
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
            
            psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00')
            # vel_data = pyt.get_data('VEL_RTN_SUN')
            
            if atype in ['panis','ptpar','ptper']:
                tens_data = pyt.get_data('T_TENSOR_INST')
                
                tens_time_arr = tens_data[0]
                tens_data_arr = tens_data[1]
                
                Txx = tens_data_arr[:,0]
                Tyy = tens_data_arr[:,1]
                Tzz = tens_data_arr[:,2]
                Txy = tens_data_arr[:,3]
                Txz = tens_data_arr[:,4]
                Tyz = tens_data_arr[:,5]
                
                tens = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                
                mag_data = pyt.get_data('MAGF_INST')
                
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
                
                temp_data = pyt.get_data('TEMP')
                
                time_arr = temp_data[0]
                data_arr = temp_data[1]
                
            else:
                dens_data = pyt.get_data('DENS')
                
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
            
            if i in [1]: #choose encounters for which to use SPC instead of SPAN. In this case, the first orbit family: Enc 1-3

                psp.spc(trange=[t0,tf], level='L3')
                vel_data = pyt.get_data('vp_fit_RTN')
            
            else:
                psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00')
                vel_data = pyt.get_data('VEL_RTN_SUN')
            
            
            vel_time_arr = vel_data[0]
            vel_data_arr = vel_data[1]
            
            vr = vel_data_arr[:,0]
            vt = vel_data_arr[:,1]
            vn = vel_data_arr[:,2]
            
            V = np.sqrt(vr**2+vt**2+vn**2)
            
            time_arr = vel_time_arr
            data_arr = V
            
        
        # vr = vel_data_arr[:,0]
        # vt = vel_data_arr[:,1]
        # vn = vel_data_arr[:,2]
        
        # V = np.sqrt(vr**2+vt**2+vn**2)
        
        # T = data_arr
        
        shape_arr.append(time_arr.shape)
        
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
        drange = (150,700)
        title_lab = '|Vsw|'
        
        
    if len(enc)>1:
        title = title_lab+' vs R, Enc '+str(enc[0])+' thru '+str(enc[-1])
    else:
        title = title_lab+' vs R, Enc '+str(enc[0])
    
    
    fig = plt.figure(figsize=(12,6))
    
    axs = fig.add_subplot(111)
    # axs.hist2d(Rfull,datafull, bins=40,range=[[10, 50], drange])
    # hist2d,xedge,yedge = np.histogram2d(Rfull,datafull, bins=40,range=[[10, 50], drange])
    hist2d,xedge,yedge = np.histogram2d(R_to_hist,data_to_hist, bins=40,range=[[10, 50], drange])
    
    hist2d[hist2d<60]=np.nan
    
    histo = np.transpose(hist2d)
    
    # breakpoint()
    
    xcenters = (xedge[:-1] + xedge[1:]) / 2
    ycenters = (yedge[:-1] + yedge[1:]) / 2
    
    # axs.scatter(Rfull,datafull,s=1,color='tab:blue',marker=",",label='Full Mission Data',zorder=1)
    axs.scatter(Rfull,datafull,s=1,color='tab:blue',marker=",",label='Encounter 7 Data',zorder=1)
    # axs.pcolormesh(hist2d, interpolation='nearest', origin='lower',extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # r_color = axs.contour(xcenters,ycenters,histo, cmap='winter', levels=15,vmax=np.nanmax(histo),zorder=2)
    
    # axs.hist2d(Rfull[where_ap],datafull[where_ap], bins=30,range=[[10, 50], [100, 600]]) 
    
    # axs.scatter(Rfull,datafull,s=1,color='tab:blue',marker=",",label='Full Mission Data')
    axs.scatter(Rfull[where_ap],datafull[where_ap],s=1,color='tab:orange',marker=",",label='Quiescent Regions',zorder=3)
    
    
    axs.set_title(title,fontsize=20)
    axs.set_ylabel(part_title,fontsize=18)
    axs.set_xlabel('Radial Position of PSP (Rs)',fontsize=18)
    axs.tick_params(axis='both', which='major', labelsize=18)
    # if 'dens' not in atype:
    axs.set_ylim(drange)
    # axs.set_xlim(15,55)
    axs.set_xlim(17,48)
    lgnd = axs.legend(fontsize=14)
    
    
    lgnd.legendHandles[0]._sizes = [10]
    lgnd.legendHandles[1]._sizes = [10]
    
    plt.show()
    
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