#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:47:09 2025

@author: besh2109
"""

from .config import CONFIG
from .config import enc_flt

import numpy as np

import os
import pyspedas as pys
import pytplot as pyt

import astropy.units as u
import pandas as pd
import csv

import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.time import parse_time

from datetime import datetime, timedelta

import parkersolarwind as psw
from multiprocessing import Pool

from scipy.optimize import minimize
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
import sys

import _pickle as cpkl

from sunkit_magex import pfss
import astropy.units as u
import astropy.constants as const

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.markers import MarkerStyle
import matplotlib.dates as mdates

from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle

def find_every_i_days(dates, reference_index,day_num=3):
    reference_date = dates[reference_index]
    # three_days = timedelta(days=day_num)

    # Find dates going forward
    forward_dates = [(i, date) for i, date in enumerate(dates[reference_index:], start=reference_index)
                     if (date - reference_date).days % day_num == 0]

    # Find dates going backward
    backward_dates = [(i, date) for i, date in enumerate(dates[:reference_index])
                      if (reference_date - date).days % day_num == 0]
    
    result = backward_dates + forward_dates
    # Combine results, maintaining original order
    dates_result = [dt[1] for dt in result]
    index_result = [i[0] for i in result]
    
    return dates_result,index_result

def checkKey(dic, key):
    if key in dic.keys():
        return True
    else:
        return False

def _observer_coord_meta(observer_coord):
    """
    Convert an observer coordinate into FITS metadata.
    """
    new_obs_frame = sunpy.coordinates.HeliographicStonyhurst(obstime=observer_coord.obstime)
    observer_coord = observer_coord.transform_to(new_obs_frame)

    new_meta = {}
    new_meta['HGLT_OBS'] = observer_coord.lat.to_value(u.deg)
    new_meta['HGLN_OBS'] = observer_coord.lon.to_value(u.deg)
    new_meta['DSUN_OBS'] = observer_coord.radius.to_value(u.km)
    # new_meta['HGLT_OBS'] = f"{observer_coord.lat.to_value(u.deg):.6f}"
    # new_meta['HGLN_OBS'] = f"{observer_coord.lon.to_value(u.deg):.6f}"
    # new_meta['DSUN_OBS'] = f"{observer_coord.radius.to_value(u.m):.6f}"

    return new_meta

def earth_obs_coord_meta(obstime):
    """
    Return metadata for an Earth obeserver coordinate.
    """
    return _observer_coord_meta(sunpy.coordinates.get_earth(obstime))

def fix_hmi_meta(header):

    if header['cunit1'] == 'Degree':
        header['cunit1'] = 'deg'

    if header['cunit2'] == 'Sine Latitude' or header['cunit2'] == 'sin(deg)':
        header['cunit2'] = 'deg'   
        header['cdelt2'] = 180 / np.pi * header['cdelt2']
        header['cdelt1'] = np.abs(header['cdelt1'])
    
    if header['bunit'] == 'Mx/cm^2':
        header['bunit'] = 'G'
    
    if checkKey(header,'DATE-OBS') is False:
        date_str = header['T_OBS']
        # Parse the original date string
        dt = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")
        # Format the datetime object to the desired format
        formatted_date = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        header['DATE-OBS'] = formatted_date
    
    if checkKey(header,'HGLT_OBS') is False:
        # hmi_map.meta.update(pfss.map._earth_obs_coord_meta(hmi_map.meta['date-obs']))
        meta_update = earth_obs_coord_meta(header['DATE-OBS'])
        header['HGLT_OBS'] = meta_update['HGLT_OBS']
        header['HGLN_OBS'] = meta_update['HGLN_OBS']
        header['DSUN_OBS'] = meta_update['DSUN_OBS']
        
    header['DSUN_OBS'] = header['DSUN_OBS']*10**3 #convert this to meters, as written its in km.
    
    header['CRVAL1'] = 120 + sunpy.coordinates.sun.L0(time=header['DATE-OBS']).value
    header['CUNIT1'] = 'deg'
    
    return header

def cost_function(T0, r_obs, v_obs,Tp_obs,gamma,model):
    """Computes the squared difference between observed and model velocities for a single r."""
    # parker_model = ParkerSolution(T0[0])  # Ensure T0 is treated as a scalar
    # print('gg')
    r_in = np.linspace(1,200,3000)*u.R_sun
    T0_in = T0[0]*u.MK
    # print(T0_in)
    # sys.stdout.flush()
    if model=='iso':
        sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(r_in,T0_in)
    if model=='poly':
        sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_polytropic(r_in,T0_in,gamma)
    if model=='isolayer':
        # breakpoint()
        K = Tp_obs/(r_obs**(-(2*(gamma-1))))
        r_iso = (T0_in/K)**(-1/(2*(gamma-1)))
        if r_iso.value > 150: #do not allow R_iso to go above 150 Rs. The truest answer should be ~10-30 Rs.
            r_iso = 150*u.R_sun

        result = psw.solve_isothermal_layer(r_in, r_iso, T0_in, gamma)
        sol_pos_iso = result[0]
        sol_pos_poly = result[4]
        sol_pos = np.concatenate((sol_pos_iso[:-1].value, sol_pos_poly.value)) * u.R_sun
        sol_vel_iso = result[2]
        sol_vel_poly = result[6]
        sol_vel = np.concatenate((sol_vel_iso[:-1].value, sol_vel_poly.value)) * u.km/u.s


        # try:
        #     result = psw.solve_isothermal_layer(r_in, r_iso, T0_in, gamma)
        #     sol_pos_iso = result[0]
        #     sol_pos_poly = result[4]
        #     sol_pos = np.concatenate((sol_pos_iso[:-1].value, sol_pos_poly.value)) * u.R_sun
        #     sol_vel_iso = result[2]
        #     sol_vel_poly = result[6]
        #     sol_vel = np.concatenate((sol_vel_iso[:-1].value, sol_vel_poly.value)) * u.km/u.s
        # except Exception as e:
        #     print(f"Error in solve_isothermal_layer with r_iso={r_iso:.1f}: {str(e)}", 
        #           file=sys.stdout, flush=True)
        #     print(r_obs,v_obs,Tp_obs,T0_in,file=sys.stdout, flush=True)
        #     print('Checking K and R-iso:',K,r_iso,file=sys.stdout, flush=True)
        #     raise 
        #     # Return a large penalty value to guide minimization away from this point
        

    # Find the index of the closest element
    closest_index = np.abs(sol_pos.value - r_obs.value).argmin()
    v_model = sol_vel[closest_index]
    
    cost = (v_model - v_obs) ** 2 # Squared error
    
    # if np.isnan(cost.value) or np.isinf(cost.value):
    #     print(f"Cost function returning NaN or Inf: v_model={v_model}, v_obs={v_obs}, cost={cost}", 
    #       file=sys.stdout, flush=True)
    
    return cost  

def fit_single_T0(args):
    r, v, tp, gamma, model = args
    
    K = tp/(r**(-(2*(gamma-1))))
    
    if np.any(np.isnan([r.value, v.value, tp.value])):
        return np.nan, np.nan*u.R_sun  # Skip minimize by returning NaN
    
    def cost_wrapper(T0, *args):
        # print(f"Evaluating T0={T0}", file=sys.stdout, flush=True)
        return cost_function(T0, *args)
    
    result = minimize(cost_wrapper, x0=[1.0], args=(r, v, tp, gamma, model), method="L-BFGS-B", bounds=[(0.3, 6)], options={'maxiter': 80, 'gtol': 1e-5})
    r_iso = (result.x[0]*u.MK/K)**(-1/(2*(gamma-1)))
    # print(r_iso, 'R_iso')

    return result.x[0], r_iso

def fit_T0_parallel(r_obs, v_obs, Tp_obs, gamma, model):
    n = len(r_obs)
    model = [model]*n
    gamma = [gamma]*n
    
    with Pool() as pool:
        # Use imap for efficiency with a generator
        T0_fitted = []
        r2_fitted = []
        # Iterate over imap with the index to print progress
        for i, T0 in enumerate(pool.imap(fit_single_T0, zip(r_obs, v_obs, Tp_obs, gamma, model))):
            T0_fitted.append(T0[0])
            
            r_in = r_obs
            r_iso = T0[1]
            if not np.isnan(r_iso.value):
                if r_iso>np.nanmax(r_in):
                    
                    r_iso = np.nanmax(r_in)
                
                if model[i]=='iso':
                    sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(r_in,T0[0]*u.MK)
                if model[i]=='poly':
                    sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_polytropic(r_in,T0[0]*u.MK,gamma[i])
                if model[i]=='isolayer':
                    result = psw.solve_isothermal_layer(r_in,r_iso,T0[0]*u.MK,gamma[i])
                    sol_vel_iso = result[2]
                    sol_vel_poly = result[6]
                    sol_vel = np.concatenate((sol_vel_iso[:-1].value,sol_vel_poly.value))*u.km/u.s
                
                # Create mask to keep only non-NaN pairs
                mask = ~np.isnan(v_obs.value) & ~np.isnan(sol_vel.value)
                
                # Filter both arrays
                v_obs_clean = v_obs[mask].value
                v_pred_clean = sol_vel[mask].value
                
                # Calculate R^2
                r2 = r2_score(v_obs_clean, v_pred_clean)
                r2_fitted.append(r2)
            else:
                r2_fitted.append(np.nan)
            
            # Print the percentage progress every 1% complete
            if (i + 1) % (n // 100) == 0:  # Print every 1% of the tasks
                percent_complete = (i + 1) / n * 100
                print(f"Progress: {percent_complete:.1f}% complete")
                sys.stdout.flush()
    return np.array(T0_fitted), np.array(r2_fitted)

def fit_T0_sequential(r_obs, v_obs, Tp_obs, gamma, model):
    n = len(r_obs)
    model = [model] * n  # Replicate model for each observation
    gamma = [gamma] * n  # Replicate gamma for each observation
    
    # Fit T0 sequentially
    T0_fitted = []
    r2_fitted = []
    for i, args in enumerate(zip(r_obs, v_obs, Tp_obs, gamma, model)):
        T0 = fit_single_T0(args)
        T0_fitted.append(T0[0])
                    
        r_in = r_obs
        r_iso = T0[1]
        
        if not np.isnan(r_iso.value):
            if r_iso>np.nanmax(r_in):
                
                r_iso = np.nanmax(r_in)
            
            if model[i]=='iso':
                sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(r_in,T0[0]*u.MK)
            if model[i]=='poly':
                sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_polytropic(r_in,T0[0]*u.MK,gamma[i])
            if model[i]=='isolayer':
                result = psw.solve_isothermal_layer(r_in,r_iso,T0[0]*u.MK,gamma[i])
                sol_vel_iso = result[2]
                sol_vel_poly = result[6]
                sol_vel = np.concatenate((sol_vel_iso[:-1].value,sol_vel_poly.value))*u.km/u.s
            
            # Create mask to keep only non-NaN pairs
            mask = ~np.isnan(v_obs.value) & ~np.isnan(sol_vel.value)
            
            # Filter both arrays
            v_obs_clean = v_obs[mask].value
            v_pred_clean = sol_vel[mask].value
            
            # Calculate R^2
            r2 = r2_score(v_obs_clean, v_pred_clean)
            r2_fitted.append(r2)
        else:
            r2_fitted.append(np.nan)

        # Print progress every 1% complete
        if (i + 1) % (n // 100) == 0:  # Avoid division by zero for small n
            percent_complete = (i + 1) / n * 100
            print(f"Progress: {percent_complete:.1f}% complete")
            sys.stdout.flush()
    
    return np.array(T0_fitted), np.array(r2_fitted)

def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

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

def sliding_median(array, window_size):
    
    # Calculate padding amounts
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1  # Ensures total padding equals window_size - 1

    # Pad the array asymmetrically if window_size is even
    padded_array = np.pad(array, (pad_left, pad_right), mode='edge')

    # Create a 2D array of sliding windows
    rolling_window = np.lib.stride_tricks.sliding_window_view(padded_array, window_shape=window_size)

    # Compute the median along the window axis
    medians = np.nanmedian(rolling_window, axis=1)
    
    # Determine if the number of NaNs exceeds half the window size
    num_nans = np.sum(np.isnan(rolling_window), axis=1)
    # too_many_nans = num_nans > 3*window_size / 4
    too_many_nans = num_nans > 1*window_size / 4
    
    # Set medians to np.nan where there are too many NaNs
    medians[too_many_nans] = np.nan
    
    return medians

def sliding_average(data, window_size, overlap_percentage):
    """
    Compute sliding average with adjustable window overlap.
    
    Parameters:
    - data: List or numpy array of numerical data
    - window_size: Size of the moving window (positive integer)
    - overlap_percentage: Overlap as a fraction (0 to 1, e.g., 0.5 for 50%)
    
    Returns:
    - averages: List of averages for each window
    - indices: List of starting indices for each window
    """
    if window_size <= 0 or window_size > len(data):
        raise ValueError("Window size must be positive and not exceed data length")
    if not 0 <= overlap_percentage < 1:
        raise ValueError("Overlap percentage must be between 0 and 1")
    
    # Calculate step size based on overlap percentage
    step_size = max(1, int(window_size * (1 - overlap_percentage)))  # Ensure step_size >= 1
    averages = []
    indices = []
    
    # Slide the window with calculated step size
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data[start:start + window_size]
        averages.append(np.nanmean(window))
        indices.append(start)
    
    return averages, indices


def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

def SPC_SPI_Construct(enc,spi_time,spi_data,spc_time,spc_data):
    
    span_check_savename = 'SPAN_SPC_QTN_flags_enc_'+str(enc)+'.cdf'
    span_check_savepath = '/Users/besh2109/Documents/SPAN Checks/'
    
    pyt.tplot_restore(span_check_savepath+span_check_savename)

    SPAN_QTN_flag = pyt.get_data('SPAN_qual_flag')
    SPAN_flag_time = SPAN_QTN_flag[0]
    SPAN_flag = SPAN_QTN_flag[1]
    
    # SPC_QTN_flag = pyt.get_data('SPC_qual_flag')
    # SPC_flag_time = SPC_QTN_flag[0]
    # SPC_flag = SPC_QTN_flag[1]
    
    flag_zero_group = group_zeros(SPAN_flag)
    
    # bad_span_times = []
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
            
            spc_where = np.where((spc_time>t0i)&(spc_time<tfi))
            spc_where = spc_where[0]
            
            vel_time_construct = np.append(vel_time_construct,spc_time[spc_where])
            vel_mag_data_construct = np.append(vel_mag_data_construct,spc_data[spc_where])
            
            good_bad = 1 #alternate between good and bad times. Should start with bad.
            # breakpoint()
        elif good_bad ==1:
            
            spi_where = np.where((spi_time>t0i)&(spi_time<tfi))
            spi_where = spi_where[0]
            
            vel_time_construct = np.append(vel_time_construct,spi_time[spi_where])
            vel_mag_data_construct = np.append(vel_mag_data_construct,spi_data[spi_where])
            
            good_bad = 0
            
    # breakpoint()
    
    #--------------check for Nans, and fill in with SPI if possible--------------#
    
    a = np.where(np.isnan(vel_mag_data_construct))
    
    one_arr = np.ones(vel_mag_data_construct.shape)
    one_arr[a] = 0
    
    groups = group_zeros(one_arr)
    index_list = []
    
    for group in groups:
        
        index_list.append(list(range(group[0],group[1])))
        # print(pys.time_string(t0_group),pys.time_string(tf_group))
        # check = np.delete(check,np.s_[group[0]:group[1]])
    
    indices_to_delete = [index for chunk in index_list for index in chunk]
    
    vel_time_temp = np.delete(vel_time_construct, indices_to_delete)
    vel_temp = np.delete(vel_mag_data_construct, indices_to_delete)
    
    for group in groups:
        t0 = vel_time_construct[group[0]]
        tf = vel_time_construct[group[1]]
        
        spi_where = np.where((spi_time>t0)&(spi_time<tf))
        spi_where = spi_where[0]
        
        vel_time_temp = np.append(vel_time_temp,spi_time[spi_where])
        vel_temp = np.append(vel_temp,spi_data[spi_where])
        
    
    # Get the sorted indices based on the time array
    sorted_indices = np.argsort(vel_time_temp)
    
    # Sort both arrays using the sorted indices
    sorted_time = vel_time_temp[sorted_indices]
    sorted_data = vel_temp[sorted_indices]
    
    #--------------output the result---------------------#
    
    time_arr = sorted_time
    data_arr = sorted_data
    
    return time_arr, data_arr

def encounter_check(day):
    
    i_day = pys.time_float(day)
    enci = 0
    for enc in enc_flt:          
        if (i_day>enc[0]) and (i_day<enc[1]): 
            encounter = enci+1 #determine which encounter input day is part of
        enci+=1
    
    return encounter

def encounter_dates(enc,rlim):
    
    Rs_km = 6.957e5*u.km #solar radius in km  
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/' #historical position
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v42.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    enc_ind = (enc-1)
    
    enc_sel = enc_flt[enc_ind]
    
    
    hpos_where = np.where((hpos_time_arr>enc_sel[0])&(hpos_time_arr<enc_sel[1]))
    hpos_where = hpos_where[0]
    
    hpos_time = hpos_time_arr[hpos_where]
    
    hposx_data = hpos_data_arr[hpos_where,0]*u.km
    hposy_data = hpos_data_arr[hpos_where,1]*u.km
    hposz_data = hpos_data_arr[hpos_where,2]*u.km
    
    R = ((np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs_km)/Rs_km)
    
    R_where = np.where(R<rlim)
    R_where = R_where[0]
    
    time_select = hpos_time[R_where]
    
    t0 = pys.time_string(time_select[0])
    tf = pys.time_string(time_select[-1])
    
    return t0, tf

def jsoc_check(enc_start=1,enc_end=22,rlim=70,enc_list=None):
    
        if enc_list == None:
            for i in range(enc_start,enc_end+1):
        
                t0,tf = encounter_dates(i,rlim)
                t_start_hmi = parse_time(t0)
                t_end_hmi = parse_time(tf)
                
                results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
                
                print("Encounter ",str(i))
                print(results_hmi)
                
                try:
                    filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
                    print(f"Download successful for Encounter {i}")
                except:
                    # Skip this iteration if an exception occurs
                    print(f"Skipping {i} due to error.")
        else:
             for i in enc_list:
         
                 t0,tf = encounter_dates(i,60)
                 t_start_hmi = parse_time(t0)
                 t_end_hmi = parse_time(tf)
                 
                 results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
                 
                 print("Encounter ",str(i))
                 print(results_hmi)
                 # breakpoint()
                 
                 try:
                    filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
                    print(f"Download successful for Encounter {i}")
                 except:
                    # Skip this iteration if an exception occurs
                    print(f"Skipping {i} due to error.")
                    res_atrs = results_hmi['JSOC']
                    result_times = res_atrs['T_REC']
                    
                    filenames = []
                    path_check = []
                    for i in result_times:
                        tmp_str = str(i)
                        full_path = 'hmi.mrdailysynframe_small_720s.'+tmp_str[:4]+tmp_str[5:7]+tmp_str[8:13]+tmp_str[14:16]+tmp_str[17:23]+'.data.fits'
                        filenames.append(full_path)
                    
                    print(filenames)
                    
                    with open('/Users/besh2109/Desktop/output.txt', 'a') as file:
                        file.writelines(string + '\n' for string in filenames)
                        
def PFSS_Br_estimation(pfss_out,seeds,r0_Rs,rss,A_scale=6.90):
        
    ss_br = pfss_out.source_surface_br
    # Convert Carrington coordinates to pixel coordinates
    pixel_coords = ss_br.world_to_pixel(seeds)

    # Retrieve data values at the pixel locations
    # Note: Pixel coordinates are in floating point; round or floor for integer indices
    
    x_pixels = np.round(pixel_coords.x)
    y_pixels = np.round(pixel_coords.y)
    
    
    nan_where = np.where(np.isnan(x_pixels))
    nan_where = nan_where[0]
    
    
    x_pixels[nan_where] = 0 #set nans equal to zero to make the next step possible.
    y_pixels[nan_where] = 0 #we'll throw these data points away afterward
    
    x_pixels = np.array(x_pixels,dtype=int) #convert np array to integers, i.e. which can be used as indices
    y_pixels = np.array(y_pixels,dtype=int)
    
    pfss_ss_br = ss_br.data[y_pixels,x_pixels] # retrieve values using pixel indices
    
    A_scale = A_scale #scaling factor for PFSS model B to match PSP data.
    br_pfss = A_scale*np.array(pfss_ss_br)*(rss/r0_Rs)**2 * u.G # in Gauss, propagate out to PSP distances.
    br_pfss[nan_where] = np.nan # put the nans back in
    # breakpoint()
    br_pfss = br_pfss.to(u.nT) #convert to nT

    return br_pfss

def read_in_footpoints(t0='2020-01-29',tf=None,enc=None,save=True):
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
        qregion_savename = 'enc_'+str(enc)+'_regions_raw.csv'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        qregion_savename = t0+'_'+tf+'_regions_raw.csv'
        
        t0float = pys.time_float(t0)
        enc = 1
        for i in enc_flt:
            # if t0float>i[0] and t0float<i[1]:
            #     encounter = enc
            enc+=1
        if tf==None:
            tffloat = pys.time_float(t0)+86400
        else:
            tffloat=pys.time_float(tf)
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
    qregion_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    
    #------------------------------ Read in Tplot Variables ----------------------------------#
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    solar_lon_data = pyt.get_data('solar_lon')
    solar_lon_time = solar_lon_data[0]
    sol_lon = solar_lon_data[1]
    
    sol_lat_data = pyt.get_data('solar_lat')
    sol_lat = sol_lat_data[1]
    
    
    return solar_lon_time, sol_lat, sol_lon

def read_in_parker_fits(enc,model='iso'):

    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/fit_T0/'
    tplot_savename = 'Enc_'+str(enc)+'_fit_T0_'+model+'.cdf'
    
    #------------------------------ Read in Tplot Variables ----------------------------------#
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    cdf_var = [model+"_T0_coronal_temp_fit",model+"_T0_coronal_temp_fit_max_err",model+"_T0_coronal_temp_fit_min_err",
               model+"_r2_scores",model+"_r2_scores_max_err",model+"_r2_scores_min_err",
               "PSP_Rs","Vsw_km","Tp_MK"]
    
    if model == 'isolayer':
        
        isolayer_list = ["isothermal_layer_height","isothermal_layer_height_max_err","isothermal_layer_height_min_err"]
        for var in isolayer_list:
            cdf_var.append(var)
        
    
    # Dictionary to hold the tuples
    tplot_dict = {}

    # Get the list of all Tplot variable names
    all_variable_names = pyt.tplot_names()
    # Check if anything was loaded
    if not all_variable_names:
        print("No variables were loaded from the file.")
        return tplot_dict  # Return empty dict if nothing loaded
    else:
        # print(f"Loaded {len(cdf_var)} variables: {cdf_var}")
        pass
        
    # Loop through all variable names and create tuples
    for var_name in cdf_var:
        # Get the data for this variable
        data = pyt.get_data(var_name)
        
        if data is not None:
            # Create a tuple of (time_array, data_array)
            tplot_dict[var_name] = (data[0], data[1])
            
            # Optional: Print to confirm
            # print(f"Added {var_name}:")
            # print(f"  Time array shape: {data[0].shape}")
            # print(f"  Data array shape: {data[1].shape}")
        else:
            print(f"Variable: {var_name} has no data or is not a standard Tplot variable.")
    
    return tplot_dict

def find_quiescent_points(enc,in_time):
    
    qregion_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    qregion_savename = 'enc_'+str(enc)+'_regions_raw.csv'
    
    qregion_df = pd.read_csv(qregion_savepath+qregion_savename)
    qregion_array = qregion_df.to_numpy()
    
    regions_arr = qregion_array
    #read in PSP fit and other data
    
    #---------------- find footpoints associated with quiescent regions --------------------#
    
    q_time = np.array([])
    
    q_index_arr = np.array([],dtype=int)
    
    for k in regions_arr:

        pre_flt = k[:2]
        reg_flt = pys.time_float(pre_flt)
        point_where = np.where((in_time>reg_flt[0])&(in_time<reg_flt[1]))
        point_where = point_where[0]
        
        q_index_arr = np.append(q_index_arr,point_where)
        
        region_time_tmp = in_time[point_where]

        q_time = np.append(q_time,region_time_tmp)
    
    indices = np.linspace(0,len(in_time)-1,num=len(in_time),dtype=int) #indices for whole set of encounter data points.
    non_q_index_arr = np.setdiff1d(indices, q_index_arr) #indices for non-quiescent wind.

    non_q_time = in_time[non_q_index_arr]
    
    return q_time, non_q_time, q_index_arr

def sort_by_footpoint_coords(enc,lon_range=None,lat_range=None,obstime=None):
    
    #------------------------------ Set path to tplot and quiescent region files ----------------------------------#

    tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords_hmi_rss_3_1.cdf'
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
    #------------------------------ Read in Tplot Variables ----------------------------------#
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    
    solar_lon_data = pyt.get_data('solar_lon')
    solar_lon_time = solar_lon_data[0]
    sol_lon = solar_lon_data[1]

    
    sol_lat_data = pyt.get_data('solar_lat')
    # sol_lat_time = sol_lat_data[0]
    sol_lat = sol_lat_data[1]
    
    r0_Rs_data = pyt.get_data('PSP_Rs')
    r0_Rs_time = r0_Rs_data[0]
    r0_Rs = r0_Rs_data[1]
    
    #--------------- sort by lon-lat range------------------#
    if not lon_range:
        lanes_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/fits/'
        lanes_savename = 'lanes_gaussian_fwhm7_run_avg.fits'
        
        lanes_map = sunpy.map.Map(lanes_savepath+lanes_savename)
        
        # Get the map's spatial extent in world coordinates
        lon_min, lon_max = lanes_map.meta['crval1'] - lanes_map.meta['cdelt1'] * lanes_map.data.shape[1] / 2, \
                            lanes_map.meta['crval1'] + lanes_map.meta['cdelt1'] * lanes_map.data.shape[1] / 2
        lat_min, lat_max = lanes_map.meta['crval2'] - lanes_map.meta['cdelt2'] * lanes_map.data.shape[0] / 2, \
                            lanes_map.meta['crval2'] + lanes_map.meta['cdelt2'] * lanes_map.data.shape[0] / 2
                            
        obstime = lanes_map.date
                            
    else:
        lon_min = lon_range[0]
        lon_max = lon_range[1]
        lat_min = lat_range[0]
        lat_max = lat_range[1]
        obstime = obstime
    # Define the bounds (already in decimal degrees)
    lon_min = lon_min * u.deg
    lon_max = lon_max * u.deg
    lat_min = lat_min * u.deg
    lat_max = lat_max * u.deg
    
    # # Define the corner coordinates of the map bounds
    # bounds_coords = SkyCoord([lon_min, lon_max], [lat_min, lat_max], 
    #                          frame="heliographic_carrington", 
    #                          obstime=obstime, 
    #                          observer="earth")
    
    # Convert to pixel coordinates
    # bounds_pixel = lanes_map.world_to_pixel(bounds_coords)
    
    sol_coords = SkyCoord(sol_lon*u.deg,sol_lat*u.deg,
                          frame='heliographic_carrington',
                          obstime=obstime,
                          observer="earth")
    
    # # Extract pixel limits
    # x_pixel_min, x_pixel_max = bounds_pixel.x.value[0], bounds_pixel.x.value[1]
    # y_pixel_min, y_pixel_max = bounds_pixel.y.value[0], bounds_pixel.y.value[1]
    
    # Wrap longitudes around 0–360 degrees if necessary
    data_coords_lon = sol_coords.lon.wrap_at(360 * u.deg)
    
    # Create masks for the range
    lon_mask = (data_coords_lon >= lon_min) & (data_coords_lon <= lon_max)
    lat_mask = (sol_coords.lat >= lat_min) & (sol_coords.lat <= lat_max)
    
    # Combine the masks
    mask = lon_mask & lat_mask
    
    # Filter all SkyCoord objects
    sol_filtered_time = solar_lon_time[mask]
    sol_filtered_coords = sol_coords[mask]

    # # Convert filtered coordinates to pixel space
    # sol_pixel_coords = lanes_map.world_to_pixel(sol_filtered_coords)
    
    return sol_filtered_time, sol_filtered_coords, mask

def find_distance_to_nearest_lane(modified_map, sol_filtered_coords, q_filtered_coords):
    
        
    # Step 1: Find pixels in modified_map where value == 1
    # Get indices of pixels with value 1
    y_idx, x_idx = np.where(modified_map.data == 1)
    
    # Step 2: Convert these pixel indices to world coordinates
    # Create pixel coordinates as arrays
    pixel_coords = np.array([x_idx, y_idx]).T  # Shape: (N, 2) for N pixels
    # Convert to SkyCoord using pixel_to_world
    ones_coords = modified_map.pixel_to_world(x_idx * u.pix, y_idx * u.pix)
    # ones_coords is a SkyCoord array with Carrington coordinates
    
    # Step 3: Compute distances for sol_filtered_coords
    sol_distances = []
    for coord in sol_filtered_coords:
        # Calculate angular separation to all "1" pixels
        separations = coord.separation(ones_coords)  # Returns Quantity array in degrees
        # Find the minimum separation
        min_distance = np.min(separations)
        sol_distances.append(min_distance)
    
    # Step 4: Compute distances for q_filtered_coords
    q_distances = []
    for coord in q_filtered_coords:
        # Calculate angular separation to all "1" pixels
        separations = coord.separation(ones_coords)
        min_distance = np.min(separations)
        q_distances.append(min_distance)
    
    # Convert distances to numpy arrays for convenience
    sol_distances = np.array([d.value for d in sol_distances]) * u.deg
    q_distances = np.array([d.value for d in q_distances]) * u.deg
    
    # Optional: Convert angular distances to physical distances (Mm)
    # Assuming solar radius ~696,000 km, 1 deg ≈ 12.1 Mm at solar surface
    solar_radius = 696000 * u.km
    deg_to_mm = (solar_radius * (2 * np.pi / (360*u.deg))).to(u.Mm / u.deg)  # ~12.1 Mm/deg
    sol_distances_mm = sol_distances * deg_to_mm
    q_distances_mm = q_distances * deg_to_mm
    
    return sol_distances_mm, q_distances_mm