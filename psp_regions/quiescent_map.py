#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:28:14 2022

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
import _pickle as cpkl
from streamtracer import StreamTracer, VectorGrid

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import sunpy.map
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.coordinates.sun import carrington_rotation_number as crn
import pfsspy
import pfsspy.utils
from pfsspy import coords, tracing
from pfsspy.sample_data import get_gong_map

import matplotlib.patches as mpatch

from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

import warnings

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

sweap_id = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield cpkl.load(f)
            except EOFError:
                break


def group_elements(A, B):
    """
    Group elements in Array A to elements in Array B
    by which element in Array B is closest to the element in Array A.
    Returns an array with the indices of the closest elements in Array B
    for each element in Array A.
    """
    indices = np.zeros_like(A, dtype=np.int32)
    for i, a in enumerate(A):
        indices[i] = np.argmin(np.abs(B - a))
    return indices

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
    
    

def q_test(enc=1):


    if enc is not None:
        print(5)    
    
    return enc_flt[1]
    
def quiescent_map(t0='2020-01-29',tf=None,enc=None,rss=2.5,r_tmp=None,plot=False,save_coords=False,rlim=55):
    
    Rs_km = 6.957e5 #solar radius in km  
    Rs = Rs_km*10**3
    
    if enc is not None:
        # enc_num = enc-1
        # encounter_flts = enc_flt[enc_num]
        # encounter_strs = pys.time_string(encounter_flts)
        
        # t0 = encounter_strs[0]
        # tf = encounter_strs[1]

        # enc_num = enc
        
        hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/' #historical poaition
        pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        hpos = pyt.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc-1)
        
        enc_sel = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_sel[0])
        enc_end = pys.time_string(enc_sel[1])
        
        hpos_where = np.where((hpos_time_arr>enc_sel[0])&(hpos_time_arr<enc_sel[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = ((np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs_km)/Rs_km)
        
        R_where = np.where(R<rlim)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
    
    
    # breakpoint()
    if r_tmp==None:
        r_tmp=rss
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)
        
        
    def set_axes_lims(ax):
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)

        

    #-------------------------------IMPORT DATA-------------------------------#
    
    # pos_in = psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True) #going to be used to plot parker position
    pos_in = psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',username=fields_id,password=fields_pass,last_version=True) #going to be used to plot parker position
    pos_data = pyt.get_data('position')
    
    pos_time_arr = pos_data[0]
    pos_data_arr = pos_data[1]
    
    x = pos_data_arr[:,0]
    y = pos_data_arr[:,1]
    z = pos_data_arr[:,2]
    
    spc_in = psp.spc(trange=[t0,tf],level='L3')
    spc_data = pyt.get_data('vp_fit_RTN')
    
    spc_time_arr = spc_data[0]
    spc_data_arr = spc_data[1]
    
    res_check = np.diff(spc_time_arr)
    
    # breakpoint()
    
    vr_spc = spc_data_arr[:,0]
    
    vel_in = psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
    vel_data = pyt.get_data('VEL_RTN_SUN')
    
    vel_time_arr = vel_data[0]
    vel_data_arr = vel_data[1]
    
    vr = vel_data_arr[:,0]
    vt = vel_data_arr[:,1]
    vn = vel_data_arr[:,2]


    r0 = np.sqrt(x**2+y**2+z**2) #parker spiral source height in kilometers
    r0_Rs = np.sqrt(x**2+y**2+z**2)/Rs_km #height from center of sun
    
    # Vsw = np.nanmean(np.sqrt(vx**2+vy**2+vz**2))
    
    spc_len = len(vr_spc)
    pos_len = len(pos_time_arr)
    
    spc_bin_size = spc_len // pos_len
    
    total_mean=np.nanmean(vr_spc)
    
    v_mean_arr = total_mean*np.ones(pos_len)
    
    for ii in range(pos_len):
        
        start_index = ii * spc_bin_size
        end_index = start_index + spc_bin_size
        
        nan_check = np.isnan(np.nanmean(vr_spc[start_index:end_index]))
        
        if not nan_check:
            v_mean_arr[ii] = np.nanmean(vr_spc[start_index:end_index])
    
    
    v_r_xp = np.arange(len(vr))
    v_r_x = np.arange(pos_len)/pos_len*(len(vr)-1)
    vr_i = np.interp(v_r_x,v_r_xp,vr)
    
    vr_span = vr_i
    
    # Vsw = 360 #km/s
    # Vsw_Rs = Vsw/Rs_km
    
    Vsw = 0.5*(v_mean_arr+vr_span)
    Vsw_Rs = Vsw/Rs_km
    
    # breakpoint()
    
    del_time = np.abs(rss-r0_Rs)/Vsw_Rs #should be in seconds.
    
    corrected_time = pos_time_arr-del_time
    
    carr_lon_psp = 2*np.arctan(y/(np.sqrt(x**2+y**2)+x))*(180/np.pi)
    
    carr_lat_psp = np.arcsin(z/r0)*(180/np.pi)
    
    w = 360/(25.38*86400) # angular frequency of the sun in degrees/sec
    # w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec
    
    sintheta = np.sin((90-carr_lat_psp)*np.pi/180) #sin of the azimuthal angle, which is 90 degrees minus the latitude
    
    src_lon = carr_lon_psp - (w*sintheta/Vsw_Rs)*(rss-r0_Rs) #carrington longitude of the PSP traced by a Parker Spiral on the source surface

    t0_del = pys.time_string(corrected_time[0])
    tf_del = pys.time_string(corrected_time[-1])
   
    #--------------------------PFSS MODEL START-----------------------------#
    

    gong_fname = pys.gong.synomap(trange=[t0_del,tf_del])
    
    # tf0 = pys.time_string(pys.time_float(t0)+2*86400)
    # gong_fname = pys.gong.synomap(trange=[t0,tf0])
    
    first_file = gong_fname[0]
    
    ###############################################################################
    # The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
    # rho = ln(r), and r is the standard spherical radial coordinate. We need to
    # define the number of rho grid points, and the source surface radius.
    nrho = 35
    rss = rss

    ###NEED TO CONTINUE WORKING HERE. IMPORTING EACH GONG MAP BUT STILL NEED TO CALCULATE PFSS SOLUTIONS.###
    
    gong_dates = []
    pfss_ins = []
    pfss_outs = []
    
    for iii in range(len(gong_fname)):
        date_tmp = gong_fname[iii]
        yyyy = '20'+date_tmp[57:59]
        mm = date_tmp[59:61]
        dd = date_tmp[61:63]
        HH = date_tmp[64:66]
        MM = date_tmp[66:68]
        gong_dates.append(yyyy+'-'+mm+'-'+dd+'/'+HH+':'+MM+':00')
        
        
        # gong_map_tmp = sunpy.map.Map(gong_fname[iii])
        # in_tmp=pfsspy.Input(gong_map_tmp, nrho, rss)
        # pfss_ins.append(in_tmp)
        # out_tmp = pfsspy.pfss(in_tmp)  #this makes the loop take a long time.
        # pfss_outs.append(out_tmp)      #basically calculating a PFSS solution for each GONG map. This takes a while.
    
    
    gong_floats = pys.time_float(gong_dates)
    
    indices = group_elements(corrected_time,gong_floats)
    
    unique_indices, ind_count = np.unique(indices,return_counts=True)
    
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/'
    fieldline_savepath = "/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/"
    
    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
        fieldline_savename = 'Enc_'+str(enc)+'_field_lines.pkl'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        fieldline_savename = t0+'_'+tf+'_field_lines.pkl'
    
    progress = np.linspace(0,pos_len,21)
    i=0
    ftprnt_lon = []
    ftprnt_lat = []
    i_non_breaks = []
    # field_lines = []
    with open(fieldline_savepath+fieldline_savename,'wb') as file:
        
        for ind in unique_indices:
            
            long_where = np.where(indices==ind)
            
            gong_map = sunpy.map.Map(gong_fname[ind])
            # gong_map.meta['rsun'] = sunpy.sun.constants.radius.value/gong_map.meta['cdelt1']
            # breakpoint()
            pfss_in = pfsspy.Input(gong_map, nrho, rss)
            
            pfss_out = pfsspy.pfss(pfss_in)
            
    
            tracer = tracing.FortranTracer()
            r = r_tmp * const.R_sun
            
            r2 = (r_tmp-0.2)*const.R_sun
            
            lat_tmp = carr_lat_psp[long_where]
            lon_tmp = src_lon[long_where]
            
            lat_tmp, lon_tmp = lat_tmp.ravel() * u.deg, lon_tmp.ravel() * u.deg
            
            seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)
    
            field_lines_tmp = tracer.trace(seeds, pfss_out)
            
            # pkl.dump(save_lines, file)
            cpkl.dump(field_lines_tmp, file)

            for field_line in field_lines_tmp:
                # cpkl.dump(field_line, file)
                if i in progress:
                    print(int(100*i/pos_len),'%')
                
                coord_check = field_line.coords
                if coord_check.shape != (0,):
                    coords = field_line.solar_footpoint

                    ftprnt_lon.append(float(coords.lon/u.deg))
                    ftprnt_lat.append(float(coords.lat/u.deg))
                    
                    i_non_breaks.append(i)

                i+=1

    
    # field_lines = pfsspy.fieldline.FieldLines(field_lines)
    #
    # breakpoint()
    #
    # gong_map = sunpy.map.Map(gong_fname[int(len(gong_fname)/2)])

    # ###############################################################################
    # # From the boundary condition, number of radial grid points, and source
    # # surface, we now construct an Input object that stores this information
    # pfss_in = pfsspy.Input(gong_map, nrho, rss)


    # # def set_axes_lims(ax):
    # #     ax.set_xlim(0, 360)
    # #     ax.set_ylim(0, 180)

    # ###############################################################################
    # # Now calculate the PFSS solution
    
    # pfss_out = pfsspy.pfss(pfss_in)
    

    # tracer = tracing.FortranTracer()
    # r = r_tmp * const.R_sun
    
    # r2 = (r_tmp-0.2)*const.R_sun
    
    # lat = carr_lat_psp
    # lon = src_lon
    
    # lat, lon = lat.ravel() * u.deg, lon.ravel() * u.deg
    
    # seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)

    # field_lines = tracer.trace(seeds, pfss_out)
    
    # breakpoint()
    
    # lat2 = np.linspace(-np.pi / 2, np.pi / 2,8, endpoint=False)
    # # lon2 = np.linspace(np.pi/2, np.pi*(5/6), 6, endpoint=False)
    # lon2 = np.linspace(0, 2*np.pi, 8, endpoint=False)
    # lat2, lon2 = np.meshgrid(lat2, lon2, indexing='ij')
    

    # lat2, lon2 = lat2.ravel() * u.rad, lon2.ravel() * u.rad


    # seeds2 = SkyCoord(lon2, lat2, r2, frame=pfss_out.coordinate_frame)
    
    # field_lines2 = tracer.trace(seeds2, pfss_out)


    # breakpoint()

    lat = carr_lat_psp
    lon = src_lon

    #------------------SAVING THE FOOTPOINT COORDINATES-------------------#
        
    # progress = np.linspace(0,pos_len,21)
    # # i_break = 45468

    # ftprnt_lon = []
    # ftprnt_lat = []
    # i_non_breaks = []
    
    # i=0
    # for field_line in field_lines:
    #     if i in progress:
    #         print(int(100*i/pos_len),'%')
        
    #     coord_check = field_line.coords
    #     if coord_check.shape != (0,):
    #         coords = field_line.solar_footpoint

    #         ftprnt_lon.append(float(coords.lon/u.deg))
    #         ftprnt_lat.append(float(coords.lat/u.deg))
            
    #         i_non_breaks.append(i)

    #     i+=1
    

    sol_long = np.array(ftprnt_lon)
    sol_lat = np.array(ftprnt_lat)
    dates = np.array(pys.time_string(pos_time_arr[i_non_breaks]))
    date_flts = np.array(pos_time_arr[i_non_breaks])
    PSP_lon = np.array(carr_lon_psp[i_non_breaks])
    PSP_lat = np.array(carr_lat_psp[i_non_breaks])
    PSP_Rs = np.array(r0_Rs[i_non_breaks])
    Vsw_Rs_new = np.array(Vsw_Rs[i_non_breaks])
        
    
    
    # if enc != None:    
    #     tplot_savename = 'Enc_'+str(enc)+'_quiescent_coords.cdf'
    # else:
    #     tplot_savename = t0+'_'+tf+'_quiescent_coords.cdf'

    # tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/'

    tplot_time = date_flts
    
    tplot_sol_lon = sol_long
    tplot_sol_lat = sol_lat
    tplot_carr_lon = PSP_lon
    tplot_carr_lat = PSP_lat
    tplot_r0_Rs = PSP_Rs
    tplot_rss = np.array([rss])

    tplot_Vsw_Rs = Vsw_Rs_new
    

    # # breakpoint()
    
    pyt.store_data("solar_lon", data={'x':tplot_time, 'y':tplot_sol_lon})
    pyt.store_data("solar_lat", data={'x':tplot_time, 'y':tplot_sol_lat})
    pyt.store_data("PSP_lon", data={'x':tplot_time, 'y':tplot_carr_lon})
    pyt.store_data("PSP_lat", data={'x':tplot_time, 'y':tplot_carr_lat})
    pyt.store_data("PSP_Rs", data={'x':tplot_time, 'y':tplot_r0_Rs})
    pyt.store_data("rss",data={'x':tplot_rss, 'y':tplot_rss})
    pyt.store_data("Vsw_Rs",data={'x':tplot_time, 'y':tplot_Vsw_Rs})

    cdf_var_list = ["solar_lon","solar_lat","PSP_lon","PSP_lat","PSP_Rs","rss","Vsw_Rs"]
    
    pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
    
    
    #--------------------SAVING THE WHOLE PFSS OUTPUT---------------------#
    
    # save_lines = field_lines
    # fieldline_savepath = "/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/"
    # if enc != None:    
    #     fieldline_savename = 'Enc_'+str(enc)+'_field_lines.pkl'
    # else:
    #     fieldline_savename = t0+'_'+tf+'_field_lines.pkl'
    
    # with open(fieldline_savepath+fieldline_savename,'wb') as file:
    #     # pkl.dump(save_lines, file)
    #     cpkl.dump(save_lines, file)

def quiescent_plots(t0='2020-01-29',tf=None,enc=None,save_coords=False,plot=True):
    
    Rs_km = 6.957e5 #solar radius in km  
    Rs = Rs_km*10**3
    w = 360/(25.38*86400) # angular frequency of the sun in degrees/sec
    # w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    
    fieldline_savepath = "/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/"
    if enc != None:    
        fieldline_savename = 'Enc_'+str(enc)+'_field_lines.pkl'
    else:
        fieldline_savename = t0+'_'+tf+'_field_lines.pkl'
    
    


    field_lines = loadall(fieldline_savepath+fieldline_savename)
    # with open(fieldline_savepath+fieldline_savename, 'rb') as file:
    #     # field_lines = pkl.load(file)
    #     field_lines = cpkl.load(file)

   
    
    # field_lines = list(field_lines)

    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/'
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    # breakpoint()
    
    rss_data = pyt.get_data('rss')
    rss = rss_data[0]
    rss = rss[0]
    
    carr_lon_data = pyt.get_data('PSP_lon')
    carr_lon_time = carr_lon_data[0]
    carr_lon = carr_lon_data[1]
    
    carr_lat_data = pyt.get_data('PSP_lat')
    carr_lat = carr_lat_data[1]
    
    sintheta = np.sin((90-carr_lat)*np.pi/180) #sin of the azimuthal angle, which is 90 degrees minus the latitude
    
    r0_Rs_data = pyt.get_data('PSP_Rs')
    r0_Rs = r0_Rs_data[1]
    
    Vsw_Rs_data = pyt.get_data('Vsw_Rs')
    Vsw_Rs = Vsw_Rs_data[1]
    
    
    t0p = pys.time_string(carr_lon_time[0])
    tfp = pys.time_string(carr_lon_time[-1])
    
    pos_in = psp.fields(trange=[t0p,tfp], datatype='ephem_spp_hg', level='l1',last_version=True) #going to be used to plot parker position
    pos_data = pyt.get_data('position')
    
    pos_time_arr = pos_data[0]
    pos_data_arr = pos_data[1]
    
    x = pos_data_arr[:,0]
    y = pos_data_arr[:,1]
    z = pos_data_arr[:,2]

        
    #-----------------------------PLOTTING ROUTINES---------------------------#
        
    if plot:
    
        #----------------# PFSS FIELD MODEL FIGURE #----------------#
        
        
        fig = plt.figure(figsize=(17.5,7))
        ax1 = fig.add_subplot(121, projection='3d')
        
        n_ind = list(range(len(carr_lon)))
        n_lines = 25
        
        idx = np.round(np.linspace(0, max(n_ind) - 1, n_lines)).astype(int) # number of lines
        
        j, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x1 = np.cos(j)*np.sin(v)
        y1 = np.sin(j)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="w", edgecolor="gray")
    
        ax2 = fig.add_subplot(122)
    
        sub_ind = 0
    
        for i in field_lines:
            # field_lines_2_plot = field_lines[idx]
            
            # breakpoint()
            field_lines_2_plot = i
            # field_lines_2_plot = np.append(field_lines_2_plot,field_lines2)
            
            for field_line in field_lines_2_plot:
                    
                if sub_ind in idx:
                    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
                    coords = field_line.coords
                    coords.representation_type = 'cartesian'
                    ax1.plot(coords.x / const.R_sun,
                            coords.y / const.R_sun,
                            coords.z / const.R_sun,
                            color=color, linewidth=1)
        
                
                    # breakpoint()

            
                    #-----------------# PARKER SPIRAL FIGURE #------------------#
                    
                    # fig = plt.figure(figsize=(10,10))
                    # ax2 = fig.add_subplot(122)
        

            
                    # ib = idx[j]
                    r = np.linspace(rss,r0_Rs[sub_ind],100)
                    spiral_plot_lon = carr_lon[sub_ind] - (w*sintheta[sub_ind]/Vsw_Rs[sub_ind])*(r-r0_Rs[sub_ind])
                    # spiral_plot_lon = carr_lon_psp[ib] - (w*sintheta[ib]/Vsw_Rs)*(r-r0_Rs[ib])
                    
                    spiral_x = r*np.cos(spiral_plot_lon*np.pi/180)
                    spiral_y = r*np.sin(spiral_plot_lon*np.pi/180)
            
                    # if field_lines_2_plot[sub_ind].polarity == 1:
                    if field_line.polarity == 1:
                        color = 'tab:red'
                    else:
                        color = 'tab:blue'
            
                    ax2.plot(spiral_x,spiral_y,color=color)
                
                sub_ind+=1
                
            
        # ax1.view_init(10, 40) #VIEW THAT SAM USED FOR ENC 1
        # ax1.view_init(10, 60)
        ax1.view_init(40, 280)
        # ax1.set_title('PFSS solution with Source Surface at '+str(rss)+' Rs',fontsize=20)
        ax1.set_title(r"PFSS solution with Source Surface at "+str(rss)+" Rs", fontsize=20)
        
        ax1.text2D(0.5,-0.08,"using GONG Synoptic Map",fontsize=20,transform=ax1.transAxes,horizontalalignment='center')
        # text2D(0.05, 0.95, "2D Text", transform=ax.transAxes
        
    
        set_axes_equal(ax1)
        
        # Make panes transparent
        ax1.xaxis.pane.fill = False # Left pane
        ax1.yaxis.pane.fill = False # Right pane
        
        # Remove grid lines
        ax1.grid(False)
        
        # Remove tick labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        
        # # Transparent spines
        # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent spines
        ax1.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # # Transparent panes
        # ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent panes
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # No ticks
        ax1.set_xticks([]) 
        ax1.set_yticks([]) 
        ax1.set_zticks([])
        
        #scale up axes
        
        # x_scale=4
        # y_scale=4
        # z_scale=4
        
        # scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        # scale=scale*(1.0/scale.max())
        # scale[3,3]=1.0
        
        # def short_proj():
        #   return np.dot(Axes3D.get_proj(ax1), scale)
        
        # ax1.get_proj=short_proj
        
        ax1.set_xlim3d(-1.75, 1.75)
        ax1.set_ylim3d(-1.75, 1.75)
        ax1.set_zlim3d(-1.4, 1.4)
            
        ax2.set_title('Parker Spiral Propagation to PSP Position',fontsize=20)
        ax2.plot(x/Rs_km,y/Rs_km,color='black')
        ax2.set_ylim(-150,100)
        ax2.set_xlim(-150,150)
        ax2.set_xlabel('Carrington X in Rs',fontsize=16)
        ax2.set_ylabel('Carrington Y in Rs',fontsize=16)
        
        # We change the fontsize of minor ticks label 
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.tick_params(axis='both', which='minor', labelsize=8)
        # ax.set_aspect("equal")
        plt.show()

def hmi_ex(t0='2018-11-05',rss=2.5):
    ###############################################################################
    # Set up the search.
    #
    # Note that for SunPy versions earlier than 2.0, a time attribute is needed to
    # do the search, even if (in this case) it isn't used, as the synoptic maps are
    # labelled by Carrington rotation number instead of time
    time = a.Time(t0, t0)
    series = a.jsoc.Series('hmi.synoptic_mr_polfil_720s')
    
    crot_tmp = int(np.floor(crn(t=t0)))
    print(crot_tmp)
    crot = a.jsoc.PrimeKey('CAR_ROT', crot_tmp)

    ###############################################################################
    # Do the search.
    #
    # If you use this code, please replace this email address
    # with your own one, registered here:
    # http://jsoc.stanford.edu/ajax/register_email.html
    result = Fido.search(time, series, crot, a.jsoc.Notify("besh2109@colorado.edu"))
    files = Fido.fetch(result)

    ###############################################################################
    # Read in a file. This will read in the first file downloaded to a sunpy Map
    # object
    hmi_map = sunpy.map.Map(files[0])
    print('Data shape: ', hmi_map.data.shape)

    ###############################################################################
    # Since this map is far to big to calculate a PFSS solution quickly, lets
    # resample it down to a smaller size.
    hmi_map = hmi_map.resample([360, 180] * u.pix)
    print('New shape: ', hmi_map.data.shape)

    ###############################################################################
    # Now calculate the PFSS solution
    nrho = 35
    rss = rss
    pfss_in = pfsspy.Input(hmi_map, nrho, rss)
    pfss_out = pfsspy.pfss(pfss_in)

    ###############################################################################
    # Using the Output object we can plot the source surface field, and the
    # polarity inversion line.
    ss_br = pfss_out.source_surface_br
    # Create the figure and axes
    
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(projection=ss_br)

    # Plot the source surface map
    ss_br.plot()
    # Plot the polarity inversion line
    ax.plot_coord(pfss_out.source_surface_pils[0])
    # Plot formatting
    plt.colorbar()
    ax.set_title('Source surface magnetic field')

    plt.show()

def gong_ex(t0='2020-01-29',rss=2.5):
    
    ###############################################################################
    # Load a GONG magnetic field map
    # gong_fname = get_gong_map()
    
    tf = pys.time_string(pys.time_float(t0)+86400)
    
    gong_fname = pys.gong.synomap(trange=[t0,tf])
    
    gong_map = sunpy.map.Map(gong_fname[0])

    ###############################################################################
    # The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
    # rho = ln(r), and r is the standard spherical radial coordinate. We need to
    # define the number of rho grid points, and the source surface radius.
    nrho = 35
    rss = rss

    ###############################################################################
    # From the boundary condition, number of radial grid points, and source
    # surface, we now construct an Input object that stores this information
    pfss_in = pfsspy.Input(gong_map, nrho, rss)


    def set_axes_lims(ax):
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)


    ###############################################################################
    # Using the Input object, plot the input field
    # m = pfss_in.map
    # fig = plt.figure()
    # ax = plt.subplot(projection=m)
    # m.plot()
    # plt.colorbar()
    # ax.set_title('Input field at '+str(rss)+' Rs')
    # set_axes_lims(ax)

    ###############################################################################
    # Now calculate the PFSS solution
    pfss_out = pfsspy.pfss(pfss_in)

    ###############################################################################
    # Using the Output object we can plot the source surface field, and the
    # polarity inversion line.
    ss_br = pfss_out.source_surface_br
    # # Create the figure and axes
    
    # breakpoint()
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(projection=ss_br)

    # Plot the source surface map
    ss_br.plot()
    # Plot the polarity inversion line
    ax.plot_coord(pfss_out.source_surface_pils[0])
    # Plot formatting
    plt.colorbar()
    ax.set_title('Source surface magnetic field at '+str(rss)+' Rs')
    set_axes_lims(ax)

    ###############################################################################
    # It is also easy to plot the magnetic field at an arbitrary height within
    # the PFSS solution.

    # Get the radial magnetic field at a given height
    ridx = 15
    br = pfss_out.bc[0][:, :, ridx]
    # Create a sunpy Map object using output WCS
    br = sunpy.map.Map(br.T, pfss_out.source_surface_br.wcs)
    # Get the radial coordinate
    r = np.exp(pfss_out.grid.rc[ridx])

    # Create the figure and axes
    fig = plt.figure(figsize=(15,5))
    ax = plt.subplot(projection=br)

    # Plot the source surface map
    br.plot(cmap='RdBu')
    # Plot formatting
    plt.colorbar()
    ax.set_title('$B_{r}$ ' + f'at r={r:.2f}' + '$r_{\\odot}$')
    set_axes_lims(ax)


    ###############################################################################
    # Finally, using the 3D magnetic field solution we can trace some field lines.
    # In this case 64 points equally gridded in theta and phi are chosen and
    # traced from the source surface outwards.
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    # ax.set_aspect("equal")

    tracer = tracing.FortranTracer()
    # tracer = tracing.PythonTracer()
    r = 1.2 * const.R_sun
    lat = np.linspace(-np.pi / 2, np.pi / 2,8, endpoint=False)
    # lat = np.linspace(0, 0, 8, endpoint=False)
    lon = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    lat, lon = np.meshgrid(lat, lon, indexing='ij')
    

    lat, lon = lat.ravel() * u.rad, lon.ravel() * u.rad

    seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)
    # print('ey')
    field_lines = tracer.trace(seeds, pfss_out)
    # print('yey')
    for field_line in field_lines:
        color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
        coords = field_line.coords
        coords.representation_type = 'cartesian'
        ax.plot(coords.x / const.R_sun,
                coords.y / const.R_sun,
                coords.z / const.R_sun,
                color=color, linewidth=1)


    ax.set_title('PFSS solution at '+str(rss)+' Rs')
    plt.show()
    
    # fig, ax = plt.subplots(figsize=(10,10))
    # ax.set_aspect('equal')
    
    # # Take 32 start points spaced equally in theta
    # # r_tmp = 1.5
    # r = r_tmp * const.R_sun
    # lon = np.pi / 2 * u.rad
    # lat = np.linspace(-np.pi / 2, np.pi / 2, 33) * u.rad
    # seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)
    
    # tracer = pfsspy.tracing.FortranTracer()
    # field_lines = tracer.trace(seeds, pfss_out)
    
    # for field_line in field_lines:
    #     coords = field_line.coords
    #     coords.representation_type = 'cartesian'
    #     color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    #     ax.plot(coords.y / const.R_sun,
    #             coords.z / const.R_sun, color=color)

    # # Add inner and outer boundary circles
    # ax.add_patch(mpatch.Circle((0, 0), 1, color='k', fill=False))
    # ax.add_patch(mpatch.Circle((0, 0), r_tmp, color='k', fill=False,linestyle='dotted'))
    # ax.add_patch(mpatch.Circle((0, 0), pfss_in.grid.rss, color='k', linestyle='--',
    #                             fill=False))
    # ax.set_title('PFSS solution with seed height at '+str(r_tmp)+' Rs')
    # plt.show()
    
    
    # breakpoint()
    # sphinx_gallery_thumbnail_number = 4
