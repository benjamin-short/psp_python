#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 13:27:49 2021

@author: benshort
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
import cartopy.crs as ccrs
import pyspedas.stereo as ste
from matplotlib import gridspec

from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as mticker

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

sweap_id = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']


class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.0e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)

def auto_plot(fd="2018-10-03",fast=True):
    first_day = fd
    current_day = date.today().strftime("%Y-%m-%d")
    
    i_day = pys.time_float(first_day)
    
    loop_day = pys.time_float(current_day) - 86400.
    
    while i_day < loop_day:
            
        t0 = pys.time_string(i_day)
        tf = pys.time_string(i_day + 86400.)
        
        mag_list = ['mag_RTN','mag_rtn']
        

        
        if os.environ.get('PSP_FIELDS_ID'):
            mag_data_type = mag_list[0]
        else:
            mag_data_type = mag_list[1]
            
        if fast:
            mag_data_type=mag_data_type+'_4_Sa_per_Cyc'
        
        mag_in = psp.fields(trange=[t0,tf], datatype=mag_data_type, level='l2',last_version=True,username=fields_id,password=fields_pass)    
        acspec_in = psp.fields(trange=[t0,tf], datatype='dfb_ac_spec', level='l2',username=fields_id,password=fields_pass)
        dcspec_in = psp.fields(trange=[t0,tf], datatype='dfb_dc_spec', level='l2',username=fields_id,password=fields_pass)
        #burst = psp.fields(trange=[t0,tf], datatype='dfb_dbm_dvac', level='l2')
        spc_in = psp.spc(trange=[t0, tf], datatype='l3i', level='l3',username=sweap_id,password=sweap_pass)
        
        
        
        """ Historical Position Data """
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pyt.tplot_names())
        hpos = pyt.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        hposx_data = hpos_data_arr[:,0]
        hposy_data = hpos_data_arr[:,1]
        
        enc_time = []
        encx_data = []
        ency_data = []
        enci = 0
        for enc in enc_flt:
            hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1])) #for each encounter, find the data for that orbit.
            hpos_where = hpos_where[0]
        
            enc_time.append(hpos_time_arr[hpos_where])
            encx_data.append(hposx_data[hpos_where]) 
            ency_data.append(hposy_data[hpos_where])
            
            if (i_day>enc[0]) and (i_day<enc[1]): #check which encounter today is in
                encounter = enci+1
                per_date = per_flt[enci]
            enci+=1
        
        #print('Encounter',encounter+1)
        encx = encx_data[encounter-1]
        ency = ency_data[encounter-1]
        
        """ STEREO and SOHO Image Data, 171 Angstrom """
        download171 = ste.secchi(trange=[t0,tf],chn='171')
        download = np.array(download171)
    
        
        """ PSP wave data """
        mag_data_get_type = 'psp_fld_l2_mag_RTN'
        if fast:
            mag_data_get_type = mag_data_get_type+'_4_Sa_per_Cyc'
        
        mag_data = pyt.get_data(mag_data_get_type)
        acspec_data = pyt.get_data('psp_fld_l2_dfb_ac_spec_dV12hg')
        dcspec_data = pyt.get_data('psp_fld_l2_dfb_dc_spec_dV12hg')
        
        """ PSP particle data """
        dens_data = pyt.get_data('np_fit')
        vel_data = pyt.get_data('vp_fit_RTN')
        
        """ PSP Positional data """
        carr_lat_data = pyt.get_data('carr_latitude')
        carr_lon_data = pyt.get_data('carr_longitude')
        
        pos2_in = psp.fields(trange=[t0,tf], datatype='ephem_eclipj2000', level='l1',username=fields_id,password=fields_pass) #going to be used to plot parker position
        pos2_data = pyt.get_data('position')
        
        pos_data = pyt.get_data('sc_pos_HCI')
    
        ac_chan = 'V12'
        dc_chan = 'V12'
        
        if acspec_data == None:
            acspec_data = pyt.get_data('psp_fld_l2_dfb_ac_spec_dV34hg')
            ac_chan = 'V34'
        if dcspec_data == None:
            dcspec_data = pyt.get_data('psp_fld_l2_dfb_dc_spec_dV34hg')
            dc_chan = 'V34'
            
        wave_data_check = [mag_data,acspec_data,dcspec_data]
        part_data_check = [dens_data,vel_data]
        pos_data_check = [carr_lat_data,carr_lon_data,pos_data]
        im_data_check = [len(download171)]
        
        if None in wave_data_check:
            i_min = 10e4
            
        else:
            mag_time_arr = mag_data[0]
            mag_data_arr = mag_data[1]
            
            acspec_time_arr = acspec_data[0]
            ac_tmp = acspec_data[1]
            ac_tmp[ac_tmp==0] = np.nan
            acspec_data_arr = ac_tmp
            
            for f in range(len(ac_tmp[0,:])):
                noise = median(ac_tmp[:,f])
                acspec_data_arr[:,f] = 10*np.log10(ac_tmp[:,f]/noise)
            
            acspec_data_arr = np.transpose(acspec_data_arr)
            acspec_freq_arr = np.transpose(acspec_data[2])
            acspec_freq_lst = acspec_freq_arr[:,0]
            
            dcspec_time_arr = dcspec_data[0]
            dc_tmp = dcspec_data[1]
            dc_tmp[dc_tmp==0] = np.nan
            dcspec_data_arr = dc_tmp
            
            for f in range(len(dc_tmp[0,:])):
                noise = median(dc_tmp[:,f])
                dcspec_data_arr[:,f] = 10*np.log10(dc_tmp[:,f]/noise)
            
            dcspec_data_arr = np.transpose(dcspec_data_arr)
            dcspec_freq_arr = np.transpose(dcspec_data[2])
            dcspec_freq_lst = dcspec_freq_arr[:,0]
            
            if None in part_data_check:
                dens_time_arr = np.array([])
                dens_data_arr = np.array([])
                
                vel_time_arr = np.array([])
                vel_data_arr = np.array([])
            else:
                dens_time_arr = dens_data[0]
                dens_data_arr = dens_data[1]
                
                vel_time_arr = vel_data[0]
                vel_data_arr = vel_data[1]
            
            
            if None in pos_data_check:
                lat_time_arr = np.array([])
                lat_data_arr = np.array([])
                
                lon_time_arr = np.array([])
                lon_data_arr = np.array([])
                
                pos_time_arr = np.array([])
                pos_data_arr = np.array([])
                
            else:
                lat_time_arr = carr_lat_data[0]
                lat_data_arr = carr_lat_data[1]
                
                lon_time_arr = carr_lon_data[0]
                lon_data_arr = carr_lon_data[1]
                
                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]            
            
            
            pos2_time_arr = pos2_data[0]
            pos2_data_arr = pos2_data[1]
            
            if len(download171) > 0:
                im_date_arr = []
                for x in download171: #/Users/benshort/spedas_data/stereo/secchi/2020/06/07/171/20200607_200900_171.jpg benshort, besh2109
                    
                    im_date = x[len(x)-23:len(x)-19]+'-'+x[len(x)-19:len(x)-17]+'-'+x[len(x)-17:len(x)-15]+'/'+x[len(x)-14:len(x)-12]+':'+x[len(x)-12:len(x)-10]+':'+x[len(x)-10:len(x)-8]
                    im_date_arr.append(im_date)
                im_time_arr = np.array(pys.time_float(im_date_arr))
    
                
            i_min = 0
        
        #unset newly redundant data variables, for RAM conservation purposes        
        del mag_in
        del acspec_in
        del dcspec_in
        del mag_data
        del acspec_data
        del dcspec_data
            
        while i_min <= 85200.0: #85200 is 86400 seconds minus 20 minutes.
               
            window_skip = False
            
            ti_doub = pys.time_double(t0) + i_min
            tf_doub = ti_doub + 1200. #20 minute window. 1200 seconds in 20 minutes
            
            #print('ti_doub: ',ti_doub)
            #print('tf_doub: ',tf_doub)
            
            mag_where = np.where((mag_time_arr > ti_doub) & (mag_time_arr < tf_doub))
            mag_where = mag_where[0]
            #print("Length of mag_where:", len(mag_where))
            
            ac_where = np.where((acspec_time_arr > ti_doub) & (acspec_time_arr < tf_doub))    
            ac_where = ac_where[0]
            #print("Length of ac_where:", len(ac_where))
            
            dc_where = np.where((dcspec_time_arr > ti_doub) & (dcspec_time_arr < tf_doub))
            dc_where = dc_where[0]
            #print("Length of dc_where:", len(dc_where))
            
            dens_where = np.where((dens_time_arr > ti_doub) & (dens_time_arr < tf_doub))
            dens_where = dens_where[0]
            
            vel_where = np.where((vel_time_arr > ti_doub) & (vel_time_arr < tf_doub))
            vel_where = vel_where[0]
            
            lat_where = np.where((lat_time_arr > ti_doub)&(lat_time_arr < tf_doub))
            lat_where = lat_where[0]
            
            lon_where = np.where((lon_time_arr > ti_doub)&(lon_time_arr < tf_doub))
            lon_where = lon_where[0]
            
            pos_where = np.where((pos_time_arr > ti_doub)&(pos_time_arr < tf_doub))
            pos_where = pos_where[0]
            
            pos2_where = np.where((pos2_time_arr > ti_doub)&(pos2_time_arr < tf_doub))
            pos2_where = pos2_where[0]
            
            if len(download171) > 0:
                
                im_time_dif = np.array(abs(im_time_arr-ti_doub))
                im_where = np.where(im_time_dif == min(im_time_dif))
                im_where = im_where[0]
                impath = download[im_where[0]]
                #impath = impath[0]
                #print(impath)
                
            wave_check = [len(mag_where),len(ac_where),len(dc_where)]
            
            if min(wave_check) < 10:
                window_skip = True
                print("Data missing, skipping window: "+pys.time_string(ti_doub,fmt='%Y-%m-%d/%H:%M'))
            else:
                window_skip = False
                
            part_check = [len(dens_where),len(vel_where)]
            
            if min(part_check) < 5:
                no_part_data = True
            else:
                no_part_data = False
            
            pos_part_check = [len(lat_where),len(lon_where),len(pos_where),len(dens_where),len(vel_where)]
            
            if min(pos_part_check) < 5:
                no_pos_part_data = True
            else:
                no_pos_part_data = False
            
            #formatter = ticker.ScalarFormatter(useOffset=False,useMathText=True)
            #formatter.set_scientific(True) 
            #formatter.set_powerlimits((-1,1)) 
            
            # f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
            # g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
            
            
            # create stacked plot of data
            if window_skip == False:
                plt.rcParams['font.size']='20'
                B_time_tmp = mag_time_arr[mag_where]
                B_data_tmp = mag_data_arr[mag_where,:]
                
                Bmag = np.sqrt(B_data_tmp[:,0]**2+B_data_tmp[:,1]**2+B_data_tmp[:,2]**2) #nT; magnitude of B
                Bmag = Bmag*10e-10 #T
                q = 1.60218*10e-20 #Coulombs
                me =  9.10938*10e-32 #kg
                mp =  1.67262*10e-28 #kg
                Rs = 6.957e5 #solar radius in km    
                fce = q*Bmag/(2*math.pi*me) #Hz
                fcp = q*Bmag/(2*math.pi*mp) #Hz
            
                ac_time_tmp = acspec_time_arr[ac_where]
                ac_data_tmp = acspec_data_arr[:,ac_where]
                ac_freq_tmp = acspec_freq_arr[:,ac_where[0]]
            
                dc_time_tmp = dcspec_time_arr[dc_where]
                dc_data_tmp = dcspec_data_arr[:,dc_where]
                dc_freq_tmp = dcspec_freq_arr[:,dc_where[0]]
                
                pos2_time_tmp = pos2_time_arr[pos2_where]
                pos2_data_tmp = pos2_data_arr[pos2_where,:]
                
                fig = plt.figure(figsize=(25,20))
                gs = gridspec.GridSpec(4,2, width_ratios=[1, 1],height_ratios=[1,1,1,1.5]) 
                
                #fig, axs = plt.subplots(4, 1, figsize=(20,20))
                axs1 = fig.add_subplot(gs[0, :])
                axs1.set(title=pys.time_string(B_time_tmp[0],fmt='%Y-%m-%d/%H:%M:%S') + ' Encounter '+str(encounter))
                
                #axs[0].plot(mag_data_tmp,linewidth=0.2)
                
                if fast:
                    linewidth = 0.4
                else:
                    linewidth = 0.2
                
                axs1.plot(B_data_tmp[:,0],linewidth=linewidth,label='Br')
                axs1.plot(B_data_tmp[:,1],linewidth=linewidth,label='Bt')
                axs1.plot(B_data_tmp[:,2],linewidth=linewidth,label='Bn')
    
                leg = axs1.legend(loc='upper right')
                leg.legend_handles[0].set_linewidth(1.0)
                leg.legend_handles[1].set_linewidth(1.0)
                leg.legend_handles[2].set_linewidth(1.0)
                
                axs1.set_xlim(0, len(B_time_tmp))
                axs1.set_ylabel('mag_rtn (nT)')
                axs1.grid(True)
                axs1.yaxis.set_minor_locator(AutoMinorLocator())
                axs1.set_xticks([0,round(len(B_time_tmp)/3),\
                                   round(2*len(B_time_tmp)/3),(len(B_time_tmp)-1)])
                x_label_list = [pys.time_string(B_time_tmp[0],fmt='%H:%M:%S'),\
                                pys.time_string(B_time_tmp[round(len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[round(2*len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[len(B_time_tmp)-1],fmt='%H:%M:%S')]
                    
                axs1.set_xticklabels(x_label_list)
                
                axs2 = fig.add_subplot(gs[1, :])
                
                acspec = axs2.pcolormesh(ac_time_tmp,ac_freq_tmp,ac_data_tmp,cmap='nipy_spectral',shading='nearest',vmin=-5,vmax=35) #ac_time_tmp,ac_freq_tmp,
                
                box = axs2.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                
                fig.colorbar(acspec,cax=axColor, label='dB')
                
                axs2.plot(B_time_tmp,fce,label='fce', color='cyan',linewidth=0.5)
                #axs[1].plot(B_time_tmp,fcp,label='fcp', color='red',linewidth=0.5)
                #axs[1].legend(loc='upper right')
                axs2.set_xlim(B_time_tmp[0], B_time_tmp[len(B_time_tmp)-1])
                axs2.set_ylabel('ac_spec_'+ac_chan+' (Hz)')
                axs2.set_yscale("log")
                axs2.set_ylim(ac_freq_tmp[2],ac_freq_tmp[51])
                axs2.set_yticks([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
                axs2.set_yticklabels([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
                # axs2.yaxis.set_major_formatter(ticker.FuncFormatter(g))
                
                # axs2.yaxis.set_major_formatter(FormatStrFormatter('%1.0e'))
                
                # axs2.ticklabel_format(axis='y',style='sci')
                
                axs2.yaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
                
                
                axs2.set_xticks([B_time_tmp[0],B_time_tmp[round(len(B_time_tmp)/3)],\
                                   B_time_tmp[round(2*len(B_time_tmp)/3)],B_time_tmp[(len(B_time_tmp)-1)]])
                
                x_label_list = [pys.time_string(B_time_tmp[0],fmt='%H:%M:%S'),\
                                pys.time_string(B_time_tmp[round(len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[round(2*len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[len(B_time_tmp)-1],fmt='%H:%M:%S')]
                    
                axs2.set_xticklabels(x_label_list)
                #axs[1].clim([-4,40])
                
                axs3 = fig.add_subplot(gs[2, :])
                
                dcspec = axs3.pcolormesh(dc_time_tmp,dc_freq_tmp,dc_data_tmp,cmap='nipy_spectral',shading='nearest',vmax=25) #dc_time_tmp,dc_freq_tmp,
                
                box = axs3.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                
                fig.colorbar(dcspec,cax=axColor, label='dB')
                #fig.colorbar(dcspec,ax=axs[2],fraction=0.0001, label='dB')
                #fig.colorbar(dcspec,cax=axs[2],fraction=0.01, label='dB')
                
                axs3.plot(B_time_tmp,fce,label='fce', color='cyan',linewidth=0.5)
                #axs[2].plot(B_time_tmp,fcp,label='fcp', color='red',linewidth=0.5)
                #axs[2].legend(loc='upper right')
                axs3.set_xlim(B_time_tmp[0], B_time_tmp[len(B_time_tmp)-1])
                axs3.set_ylabel('dc_spec_'+dc_chan+' (Hz)')
                axs3.set_xlabel('time')
                axs3.set_yscale("log")
                axs3.set_ylim(dc_freq_tmp[2],dc_freq_tmp[51])
                axs3.set_yticks([20,100,200,500,1000,2000])
                axs3.set_yticklabels([20,100,200,500,1000,2000])
                #axs[2].set_xlim(0, len(dc_time_tmp))
                axs3.set_xticks([B_time_tmp[0],B_time_tmp[round(len(B_time_tmp)/3)],\
                                   B_time_tmp[round(2*len(B_time_tmp)/3)],B_time_tmp[(len(B_time_tmp)-1)]])
                    
                
                
                x_label_list = [pys.time_string(B_time_tmp[0],fmt='%H:%M:%S'),\
                                pys.time_string(B_time_tmp[round(len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[round(2*len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                                pys.time_string(B_time_tmp[len(B_time_tmp)-1],fmt='%H:%M:%S')]
                    
                axs3.set_xticklabels(x_label_list)
                #axs[2].clim([-4,30])
                
                #axs4 = plt.subplot(4,3,11)
                #gs = gridspec.GridSpec(4,2, width_ratios=[1, 1],height_ratios=[1,1,1,1.5]) 
                axs5 = fig.add_subplot(gs[3,1])
                
                axs5.plot(encx/Rs, ency/Rs, color='whitesmoke')
                axs5.plot(0,0, color='goldenrod', marker='o')
                axs5.set_facecolor('xkcd:navy')
                po = axs5.plot(pos2_data_tmp[:,0]/Rs,pos2_data_tmp[:,1]/Rs, marker='^', color='xkcd:blood orange')        
                # breakpoint()
                psp_r = str(np.sqrt((pos2_data_tmp[0,0]/Rs)**2+(pos2_data_tmp[0,1]/Rs)**2)) #convert radial distance into a string
                #psp_r = str(psp_r[0])
                axs5.set_adjustable('box')
                axs5.text(5,-75,'PSP Radial Distance: '+psp_r[0:6]+' Rs', color='whitesmoke', fontsize=16.)
                axs5.set_aspect(1)
                axs5.set(title='PSP Position in Orbit')
                #axs4.legend(po, ['Date: {0}'.format(date)], loc='lower right', fontsize=16)
                
                #axs4.plot([3,2,1])
                #axs5.plot([3,2,6,7,8])
                
                if no_part_data == False:
                    
                    dens_time_tmp = dens_time_arr[dens_where]
                    dens_data_tmp = dens_data_arr[dens_where]
                    
                    vel_time_tmp = vel_time_arr[vel_where]
                    vel_data_tmp = vel_data_arr[vel_where,:]
                    
                    dens = dens_data_tmp*10**6 #convert to 1/m^3
                    eps0 = 8.8541878128e-12 #F/m
                    mp = 1.67262e-27
                    
                    
                    wpp = np.sqrt((dens*q**2)/(eps0*mp))
                    fpp = wpp/(2*math.pi)
                    axs2.plot(dens_time_tmp,fpp,label='fpp', color='pink',linewidth=1.5)
                    axs3.plot(dens_time_tmp,fpp,label='fpp', color='pink',linewidth=1.5)
                    
                    
                    #axs4.plot(vel_time_tmp,vel_data_tmp[:,0],linewidth=1.2,label='Vr')
                    #axs4.plot(vel_time_tmp,vel_data_tmp[:,1],linewidth=1.2,label='Vt')
                    #axs4.plot(vel_time_tmp,vel_data_tmp[:,2],linewidth=1.2,label='Vn')
                    
                if no_pos_part_data == False:    
                    
                    
                    lat_time_tmp = lat_time_arr[lat_where]
                    lat_data_tmp = lat_data_arr[lat_where]
                    
                    lon_time_tmp = lon_time_arr[lon_where]
                    lon_data_tmp = lon_data_arr[lon_where]
                    
                    pos_time_tmp = pos_time_arr[pos_where]
                    pos_data_tmp = pos_data_arr[pos_where,:]
                    
                    
                    if len(download171) > 0:
                        
                        sinlat = np.sin(np.radians(90. - lat_data_tmp)) #sin of latitude in terms of polar angle theta
                    
                         
                        R0 = Rs #again, solar radius  
                        w = 360/(27*86400) # angular frequency of the sun in degrees/sec
                    
                        lon = lon_data_tmp
                        lat = lat_data_tmp
                    
                        r = np.sqrt(pos_data_tmp[:,0]**2+pos_data_tmp[:,1]**2+pos_data_tmp[:,2]**2)/Rs #parker solar probe position in terms of solar radius     
             
                        vel_mag = np.mean(np.sqrt(vel_data_tmp[:,0]**2+vel_data_tmp[:,1]**2+vel_data_tmp[:,2]**2)/Rs) #solar wind velocity in terms of Rs/s             
                        phi0 = lon + (r-R0)*w*sinlat/vel_mag  #back calculate the source carrington degrees longitude from a parker spiral
                    
                        srce_lat = lat_data_tmp
                        srce_lon = phi0
                    
                        t0tmp = pys.time_string(ti_doub)
                        tftmp = pys.time_string(tf_doub)
           
                    else:
                        impath = '/Users/besh2109/spedas_data/stereo/secchi/nodata/nodata.jpg'
                        srce_lat = np.array([0])
                        srce_lon = np.array([180])
                        lon = lon_data_tmp
                        lat = lat_data_tmp
                    
    
                    
                    img1 = plt.imread(impath)
                    
                    axs4 = fig.add_subplot(gs[3,0], projection=ccrs.Orthographic(srce_lon[0], srce_lat[0]))
                    
                    
                    axs4.gridlines(color='black', linestyle='dotted')
                    axs4.imshow(img1, origin="upper", extent=(0, 360, -90, 90),transform=ccrs.PlateCarree())  # Important
                    axs4.title.set_text('171 A, PSP Longitude: '+str(lon[0]))
                    axs4.plot(srce_lon,srce_lat)
                    axs4.set_adjustable('box')
                    axs4.set_aspect(1)
                    #axs3.set(title='PSP Parker Spiral Footprint')
                    #axs3.set_extent([srce_lon[0]-10, srce_lon[0]+10, srce_lat[0]-5, srce_lat[0]+5])
                
    
                else:
                    # lat_time_tmp = lat_time_arr[lat_where]
                    # lat_data_tmp = lat_data_arr[lat_where]
                    
                    # lon_time_tmp = lon_time_arr[lon_where]
                    # lon_data_tmp = lon_data_arr[lon_where]
                    
                    # pos_time_tmp = pos_time_arr[pos_where]
                    # pos_data_tmp = pos_data_arr[pos_where,:]
                    
                    impath = '/Users/besh2109/spedas_data/stereo/secchi/nodata/nodata.jpg'
                    srce_lat = np.array([0])
                    # srce_lat = lat_data_tmp
                    
                    srce_lon = np.array([180])
                    # srce_lon = lon_data_tmp
                    
                    img1 = plt.imread(impath)
                    axs4 = fig.add_subplot(gs[3,0], projection=ccrs.Orthographic(srce_lon[0], srce_lat[0]))
                    # axs4 = fig.add_subplot(gs[3,0], projection=ccrs.Orthographic(srce_lon, srce_lat))
                    axs4.gridlines(color='black', linestyle='dotted')
                    axs4.imshow(img1, origin="upper", extent=(0, 360, -90, 90),transform=ccrs.PlateCarree())
                    axs4.title.set_text('No particle data to calculate Parker Spiral.')
                    axs4.set_aspect(1)
    
    
                axs2.legend(loc='upper right')
                axs3.legend(loc='upper right')            
                
                #plt.show()
                savepath = "/Users/besh2109/Desktop/psp_pictures/"+pys.time_string(ti_doub,fmt='%Y/%m/%d/')
                savename = "psp_"+pys.time_string(ti_doub,fmt='%Y%m%d%H%M')+"_plot.png"
                
                isdir = os.path.isdir(savepath)
                
                if isdir == False:
                    os.makedirs(savepath)
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig)
                print('Saved:'+' '+pys.time_string(ti_doub,fmt='%Y-%m-%d/%H:%M'))
                
                del fig
                del axs1
                del axs2
                del axs3
                del axs4
                del axs5
                del acspec
                del dcspec
                del axColor
                del box
                del x_label_list
                
            i_min += 1200.
        #gc.collect()
        i_day += 86400.

def spec_plot(t0="2018-10-03",tf="2018-10-04"):
    
    t0_flt = pys.time_float(t0)
    tf_flt = pys.time_float(tf)
    
    mag_in = psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True)    
    acspec_in = psp.fields(trange=[t0,tf], datatype='dfb_ac_spec', level='l2')
    
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
    acspec_data = pyt.get_data('psp_fld_l2_dfb_ac_spec_dV12hg')
    # pyt.tplot_names()
    mag_time_arr = mag_data[0]
    mag_data_arr = mag_data[1]
    
    acspec_time_arr = acspec_data[0]
    ac_tmp = acspec_data[1]
    ac_tmp[ac_tmp==0] = np.nan
    acspec_data_arr = ac_tmp
    
    for f in range(len(ac_tmp[0,:])):
        noise = median(ac_tmp[:,f])
        acspec_data_arr[:,f] = 10*np.log10(ac_tmp[:,f]/noise)
    
    acspec_data_arr = np.transpose(acspec_data_arr)
    acspec_freq_arr = np.transpose(acspec_data[2])
    acspec_freq_lst = acspec_freq_arr[:,0]
    
    mag_where = np.where((mag_time_arr > t0_flt) & (mag_time_arr < tf_flt))
    mag_where = mag_where[0]
    
    ac_where = np.where((acspec_time_arr > t0_flt) & (acspec_time_arr < tf_flt))    
    ac_where = ac_where[0]
    
    plt.rcParams['font.size']='20'
    B_time_tmp = mag_time_arr[mag_where]
    B_data_tmp = mag_data_arr[mag_where,:]
    
    Bmag = np.sqrt(B_data_tmp[:,0]**2+B_data_tmp[:,1]**2+B_data_tmp[:,2]**2) #nT; magnitude of B
    # Bmag = Bmag*10e-10 #T
    q = 1.60218*10e-20 #Coulombs
    me =  9.10938*10e-32 #kg
    mp =  1.67262*10e-28 #kg
    Rs = 6.957e5 #solar radius in km    
    fce = q*Bmag/(2*math.pi*me) #Hz
    fcp = q*Bmag/(2*math.pi*mp) #Hz

    ac_time_tmp = acspec_time_arr[ac_where]
    ac_data_tmp = acspec_data_arr[:,ac_where]
    ac_freq_tmp = acspec_freq_arr[:,ac_where[0]]
    
    fig = plt.figure(figsize=(25,13))
    axs1 = fig.add_subplot(211)
    
    acspec = axs1.pcolormesh(ac_time_tmp,ac_freq_tmp,ac_data_tmp,cmap='nipy_spectral',shading='nearest',vmin=-5,vmax=35) #ac_time_tmp,ac_freq_tmp,
    
    box = axs1.get_position()
    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
    
    cbar = fig.colorbar(acspec,cax=axColor, label='dB')
    
    cbar.ax.tick_params(labelsize=28)
    cbar.set_label(label='Decibels (dB)', size=28)
    
    axs1.set_title("2020-01-29/10:37:00 Quiescent Region",fontsize=34)
    # fig.suptitle("2020-01-29/10:35:00 Quiescent Region")
    
    axs1.set_ylabel('V12 AC Spectrum (Hz)',fontsize=26)
    
    axs1.set_yscale('log')
    axs1.set_ylim([0.5*10e2,7.5*10e3])
    axs1.set_xticks([])
    axs1.set_xticklabels([])
    axs1.tick_params(which='major',width=3,length=9)
    axs1.tick_params(which='minor',width=2,length=6)
    axs1.tick_params(axis='y',labelsize=32)
    
    axs2 = fig.add_subplot(212)
    
    # axs2.plot(B_data_tmp[:,0]/Bmag,linewidth=2,label='Br')
    # axs2.plot(B_data_tmp[:,1]/Bmag,linewidth=2,label='Bt')
    # axs2.plot(B_data_tmp[:,2]/Bmag,linewidth=2,label='Bn')
    
    axs2.plot(B_data_tmp[:,0],linewidth=2,label='Br')
    axs2.plot(B_data_tmp[:,1],linewidth=2,label='Bt')
    axs2.plot(B_data_tmp[:,2],linewidth=2,label='Bn')
    axs2.plot(Bmag,linewidth=2,label='|B|',color='black')
    
    axs2.tick_params(which='major',width=3,length=9)
    axs2.tick_params(which='minor',width=2,length=6)
    axs2.tick_params(axis='both',labelsize=26)
    leg = axs2.legend(loc='upper left',fontsize=26)
    # leg.get_title().set_fontsize('26')
    leg.legendHandles[0].set_linewidth(3.0)
    leg.legendHandles[1].set_linewidth(3.0)
    leg.legendHandles[2].set_linewidth(3.0)
    
    axs2.set_xlim(0, len(B_time_tmp))
    axs2.set_xlabel('UTC',fontsize=30)
    axs2.set_ylabel('Magnetic Field Vector B (nT)',fontsize=26)
    axs2.grid(True)
    axs2.yaxis.set_minor_locator(AutoMinorLocator())
    axs2.set_xticks([0,round(len(B_time_tmp)/3),\
                       round(2*len(B_time_tmp)/3),(len(B_time_tmp)-1)])
    x_label_list = [pys.time_string(B_time_tmp[0],fmt='%H:%M:%S'),\
                    pys.time_string(B_time_tmp[round(len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                    pys.time_string(B_time_tmp[round(2*len(B_time_tmp)/3)],fmt='%H:%M:%S'), \
                    pys.time_string(B_time_tmp[len(B_time_tmp)-1],fmt='%H:%M:%S')]
        
    axs2.set_xticklabels(x_label_list)
    # print(B_data_tmp[:,0])
    plt.subplots_adjust(wspace=3, hspace=0.08)
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
