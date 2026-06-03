import os
import pyspedas.projects.psp as psp
import pyspedas as pys
import matplotlib.pyplot as plt
import numpy as np
from .config import CONFIG
from .config import enc_flt
import pandas as pd
import matplotlib.colors
from pathlib import Path

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

# sweap_id = os.environ['PSP_SWEAP_ID']
# sweap_pass = os.environ['PSP_SWEAP_PW']

sweap_id = os.environ['PSP_SWEAP_ID_BERK']
sweap_pass = os.environ['PSP_SWEAP_PW_BERK']

jsoc_email = os.environ['JSOC_EMAIL']

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3 #solar radius in m
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

def split_data_evenly_in_time(time_array, data_array, num_segments):
    # Calculate the total time span
    total_time_span = time_array[-1] - time_array[0]

    # Calculate the time interval for each segment
    segment_interval = total_time_span / num_segments

    # Compute the indices corresponding to the boundaries of each segment
    segment_indices = [np.searchsorted(time_array, time_array[0] + i * segment_interval) for i in range(1, num_segments)]

    # Split the data into segments based on the computed indices
    time_segments = np.split(time_array, segment_indices)
    data_segments = np.split(data_array, segment_indices)

    return time_segments, data_segments

def assign_closest_index(arr_one, arr_two):
    # Sort arr_one for binary search
    sorted_arr_two = np.sort(arr_two)

    # Initialize an empty array to store assigned values
    assigned_index = np.empty_like(arr_one,dtype=int)

    # Iterate over each value in arr_one
    for i, val in enumerate(arr_one):
        # Find the index of the closest value in arr_two using binary search
        closest_index = np.abs(sorted_arr_two - val).argmin()
        
        if i%55000==0:
            print(round(i*100/len(arr_one)),'%')
        # Assign the closest index from arr_one to arr_two
        assigned_index[i] = closest_index

    return assigned_index

def SPAN_FOV(enc=1,enc_radius=45,plot=False, store=False):

    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc_arr = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc_arr = list(range(2,13))
        enc_arr.append(14)
        enc_arr.append(15)
        enc_arr.append(16)
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc_arr = list(range(2,15)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc_arr = [enc]
        
    else:
        enc_arr = enc
        
    # breakpoint()
        
    for encs in enc_arr:
        
        enc_num = encs
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pys.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pys.tplot_names())
        hpos = pys.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc_num-1)
        
        enc = enc_flt[enc_ind]
        
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
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)

        #specify time range in the form ['yyyy-mm-dd/hh:mm:ss','yyyy-mm-dd/hh:mm:ss']
        trange=[t0,tf]
        
        #specify data type to plot
        datatype='spi_sf00' #protons
        spi_vars = psp.spi(trange=trange, datatype=datatype, level='L3', time_clip=True,username=sweap_id,password=sweap_pass,last_version=True)
        
        prefix='psp_spi_'
        # pys.tplot([prefix+'SUN_DIST',prefix+'EFLUX_VS_ENERGY',prefix+'EFLUX_VS_THETA',prefix+'EFLUX_VS_PHI'])
    
    
        #define variables
        eflux_phi_data=pys.get_data(prefix+'EFLUX_VS_PHI')
        times_unix=eflux_phi_data.times
        eflux = eflux_phi_data.y
        phi = eflux_phi_data.v
        
        rad_data = pys.get_data('psp_spi_SUN_DIST')
        rad = rad_data.y
        rad = np.array(rad-Rs)/Rs
        
        # breakpoint()
        
        #determine phi angle with max eflux
        max_phi_ind = np.argmax(eflux, axis=1)
        max_phi = phi[0, max_phi_ind]
        # print(max_phi)
        
        t_string = pys.time_string(times_unix)
        
        times = pd.to_datetime(t_string)
        times = times.to_numpy()
        
        # times = [dateutil.parser.parse(t) for t in pys.time_string(times_unix)]     
        
        # breakpoint()
        
        # if min(rad) < 35:
        #     spline_r_35 = UnivariateSpline(times_unix, rad-35., s=0)
        #     r1x_35, r2x_35 = spline_r_35.roots()
        
        # root_times = [dateutil.parser.parse(t) for t in pys.time_string([r1x_35,r2x_35])]     
        
        #define fov array
        tlen = times_unix.shape[0]
        phi_fov=np.ones(tlen)
        phi_av_fov = np.ones(tlen)
        
        #set threshold 163.125 degrees
        phi_thresh = phi[0,1]
        
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
        point_ratios = []

        window_size = 515 #roughly 30 minute windows

        # Loop through the array to consider
        # every window of size 3
        while i < len(max_phi):

            # Store elements from i to i+window_size
            # in list to get the current window
            
            if i < int(window_size/2):
                window = max_phi[i : i + int(window_size/2)]
                win_len = len(window)
            elif (i+int(window_size/2)) > len(max_phi):
                window = max_phi[i - int(window_size/2) : i]
                win_len = len(window)
            else:
               window = max_phi[i - int(window_size/2) : i + int(window_size/2)]
               win_len = window_size

            # Calculate the average of current window
            window_average = round(sum(window) / win_len, 2)

            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
            
            # Check ratio of good points to bad points in window
            
            good = np.where(window<phi_thresh)
            good = good[0]
            
            bad = np.where(window>=phi_thresh)
            bad = bad[0]
            
            if len(bad) != 0:
                
                if len(good)/len(bad)>300.:
                    good_bad_ratio = 300.
                
                else:    
                    good_bad_ratio = len(good)/len(bad)
            
            else:
                good_bad_ratio = 300.
                
            point_ratios.append(good_bad_ratio)

            # Shift window to right by one position
            i += 1
        
        moving_avs = np.array(moving_averages)
        
        point_ratios = np.array(point_ratios)
        
        exact_where = np.where(max_phi<phi_thresh)
        exact_where = exact_where[0]
        
        phi_fov[exact_where] = 0
        
        av_where = np.where(moving_avs<155)
        av_where = av_where[0]
        
        phi_av_fov[av_where] = 0

        if plot:
                
            fig, ax = plt.subplots(figsize=(12, 5))
        
            #print(times.shape)
            # start_tind = 66500
            # stop_tind = 67000
            if np.isnan(phi[0,0]):
                phi[0,0] = 174.375
            # breakpoint()
            p = ax.pcolormesh(times, phi[0,:], eflux.T, norm=matplotlib.colors.LogNorm())
            plt.colorbar(p, ax=ax, label=f'$(cm^2 \\ s \\ sr \\ eV)^{-1}$')
            # ax.plot(times, max_phi, 'k')
            ax.plot(times, moving_avs, 'k')
            
            # ax.plot(times,phi_fov*160,'r+')
            
            ax.plot(times,phi_av_fov*120,'b+')
            
            ax.plot(times,(point_ratios*0.2)+100,'y')
            
            # ax.axvspan(root_times[0], root_times[1], facecolor='g', alpha=0.5)
            
            ax.set_title("Encounter "+str(enc_num)+" SPAN-Ion Phi-direction.")
                    
            ax.set(ylim=(100, 185), xlabel='Time', ylabel=f"$\\phi$ [deg]")
            
            plt.show()
        
        #------------------store quality flags-------------------#
        
        if store:
        
            tplot_savename = 'SPAN_ion_fov_flags'+'_enc_'+str(enc_num)+'.cdf'
            tplot_savepath = str(Path('~/Documents/SPAN Checks/FOV flags/').expanduser())
    
            tplot_time = times_unix
            
            tplot_phi_fov = phi_fov
            
            tplot_phi_fov_av = phi_av_fov
            
            tplot_phi_ratio = point_ratios
            
            tplot_r_Rs = rad
            
            # breakpoint()
            
            pys.store_data("phi_fov", data={'x':tplot_time, 'y':tplot_phi_fov})
            pys.store_data("phi_fov_average", data={'x':tplot_time, 'y':tplot_phi_fov_av})
            pys.store_data("phi_fov_ratio", data={'x':tplot_time, 'y':tplot_phi_ratio})
            pys.store_data("psp_radial_dist_Rs",data={'x':tplot_time,'y':tplot_r_Rs})
    
            cdf_var_list = ["phi_fov","phi_fov_average","phi_fov_ratio","psp_radial_dist_Rs"]
            
            pys.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
            
            
            pys.del_data()
            
def SPAN_SPC_QTN(enc=6,enc_radius=60,plot=False, store=False):
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc_arr = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc_arr = list(range(2,13))
        enc_arr.append(14)
        enc_arr.append(15)
        enc_arr.append(16)
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc_arr = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc_arr = [enc]
        
    else:
        enc_arr = enc
        
    # breakpoint()
        
    for encs in enc_arr:
        
        enc_num = encs
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pys.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pys.tplot_names())
        hpos = pys.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc_num-1)
        
        enc = enc_flt[enc_ind]
        
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
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)

        #specify time range in the form ['yyyy-mm-dd/hh:mm:ss','yyyy-mm-dd/hh:mm:ss']
        trange=[t0,tf]
        
        psp.spi(trange=trange,level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        spi_data = pys.get_data('psp_spi_DENS')
        spi_time = spi_data[0]
        spi_dens = spi_data[1]
        
        psp.spc(trange=trange, level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        spc_data = pys.get_data('psp_spc_np_fit')
        
        if spc_data==None:
            spc_data = pys.get_data('spp_spc_np_fit')
        
        spc_time = spc_data[0]
        spc_dens = spc_data[1]
        
        psp.fields(trange=trange, datatype='sqtn_rfs_V1V2', level='L3',last_version=True,username=fields_id,password=fields_pass)
        qtn_data = pys.get_data('electron_density')
        
        if qtn_data is None:
            qtn_data_exists = False
        else:
            qtn_data_exists = True
        
            qtn_time = qtn_data[0]
            qtn_dens = qtn_data[1]
            
            qtn_errs = pys.get_data('electron_density_delta')
            qtn_err_time = qtn_errs[0]
            qtn_errs = qtn_errs[1]

        # breakpoint()
        #------------------ downsample and clean SPC data ------------------#
        
        # spc_tmp = []
        # for i in range(len(spi_time)):
        #     spi_window_start = spi_time[i]-30
        #     spi_window_end = spi_time[i]+30
        #     spc_where = np.where((spc_time>spi_window_start) & (spc_time<spi_window_end))
        #     spc_where = spc_where[0]
        #     nan_total = sum(np.isnan(spc_dens[spc_where]))
        #     if nan_total > len(spc_where)/2:
        #         spc_tmp.append(np.nan)
        #     else:
        #         spc_tmp.append(np.nanmedian(spc_dens[spc_where]))
        #     # if i%2500 == 0:
        #     #     print(round(i/len(spi_time)*100,2),"%")
                
        # # plt.plot(spc_tmp)
        
        # spc_time_down = spi_time         
        # spc_dens_down = np.array(spc_tmp) #downsampled SPC data
        

        
        # Define the window size in time (delta_t), 1 minute or 60 seconds
        delta_t = 180  # Adjust this as needed
        
        # breakpoint()
        # print('ey1')
        
        spc_time_down = np.array([])
        spc_dens_down = np.array([])
        
        n_arrays = 100
        # break data into 100 chunks so that can be looped through. 
        # This saves time and RAM.
        
        spc_time_segs, spc_dens_segs = split_data_evenly_in_time(spc_time, spc_dens, n_arrays)
        spi_time_segs, spi_dens_segs = split_data_evenly_in_time(spi_time, spi_dens, n_arrays)
        
        
        for i in range(n_arrays):
            
            # split_arrays = np.array_split(array1, n_arrays)
            spc_split_time_arr = spc_time_segs[i]
            spc_split_data_arr = spc_dens_segs[i]
            spi_split_time_arr = spi_time_segs[i]
                          
            time_diff = spc_split_time_arr[:, None] - spi_split_time_arr

            # Use boolean indexing to create a 2D mask
            window_mask = (time_diff >= -delta_t/2) & (time_diff <= delta_t/2)
            
            window_values = np.where(window_mask.T, spc_split_data_arr, np.nan)
            spc_medians = np.nanmedian(window_values, axis=1)
            
            # Compute the median along the second axis (axis=1) ignoring NaN values
            spc_time_down = np.append(spc_time_down,spi_split_time_arr)
            spc_dens_down = np.append(spc_dens_down,spc_medians)
            # print(i)
        
        # print('ey2')

        # breakpoint()
        #------------------construct SPAN quality flag---------------------#
        
        span_check_savename = 'SPAN_ion_fov_flags_enc_'+str(enc_num)+'.cdf'
        span_check_savepath = str(Path('~/Documents/SPAN Checks/FOV flags/').expanduser())
        
        pys.tplot_restore(span_check_savepath+span_check_savename)
        
        fov_flag_average = pys.get_data('phi_fov_average')
        fov_av_time = fov_flag_average[0]
        fov_flag_av = fov_flag_average[1]
        
        fov_where = np.where(np.isin(fov_av_time,spi_time))
        fov_where = fov_where[0]
        
        fov_flag_inv = np.array(fov_flag_av[fov_where],dtype=int)
        fov_flag = (1-fov_flag_inv)*2
        
        #--------------------compare both SPAN and SPC to QTN density-------#
        
        if qtn_data_exists:
        
            qtn_indices = assign_closest_index(spi_time,qtn_time)
    
            qtn_values_check = qtn_dens[qtn_indices]
            
            qtn_spc_diff = abs(qtn_values_check - spc_dens_down)
            qtn_spi_diff = abs(qtn_values_check - spi_dens)
            
            spc_per_diff = qtn_spc_diff/qtn_values_check*100 #percent difference of both SPC and SPI
            spi_per_diff = qtn_spi_diff/qtn_values_check*100
            
            # breakpoint()
            
            dens_cor_where = np.where(np.isin(spi_time,fov_av_time))
            dens_cor_where = dens_cor_where[0]
            
            spi_per_diff_cor = spi_per_diff[dens_cor_where]
            
            
            qtn_err_pers = qtn_errs[qtn_indices,:]/qtn_values_check[:,np.newaxis]*100
            
            qtn_err_pers_corr = qtn_err_pers[dens_cor_where,:]
            
            one_sig_max = np.max(qtn_err_pers,axis=1)
            two_sig_max = 2*np.max(qtn_err_pers,axis=1)
            three_sig_max = 3*np.max(qtn_err_pers,axis=1)
            
            one_sig_max_corr = np.max(qtn_err_pers_corr,axis=1)
            two_sig_max_corr = 2*np.max(qtn_err_pers_corr,axis=1)
            three_sig_max_corr = 3*np.max(qtn_err_pers_corr,axis=1)
            
            # spi_one_sig_where = np.where(spi_per_diff<one_sig_max)
            # spi_one_sig_where = spi_one_sig_where[0]
            # spi_two_sig_where = np.where(spi_per_diff<two_sig_max)
            # spi_two_sig_where = spi_two_sig_where[0]
            # spi_three_sig_where = np.where(spi_per_diff<three_sig_max)
            # spi_three_sig_where = spi_three_sig_where[0]
            
            spi_one_sig_where = np.where(spi_per_diff_cor<one_sig_max_corr)
            spi_one_sig_where = spi_one_sig_where[0]
            spi_two_sig_where = np.where(spi_per_diff_cor<two_sig_max_corr)
            spi_two_sig_where = spi_two_sig_where[0]
            spi_three_sig_where = np.where(spi_per_diff_cor<three_sig_max_corr)
            spi_three_sig_where = spi_three_sig_where[0]
            
            spc_one_sig_where = np.where(spc_per_diff<one_sig_max)
            spc_one_sig_where = spc_one_sig_where[0]
            spc_two_sig_where = np.where(spc_per_diff<two_sig_max)
            spc_two_sig_where = spc_two_sig_where[0]
            spc_three_sig_where = np.where(spc_per_diff<three_sig_max)
            spc_three_sig_where = spc_three_sig_where[0]
        
        dens_cor_where = np.where(np.isin(spi_time,fov_av_time))
        dens_cor_where = dens_cor_where[0]
        
        spi_dens_time_cor = spi_time[dens_cor_where]
        spi_dens_corrected = spi_dens[dens_cor_where]
        
        # breakpoint()
        
        if qtn_data_exists:
        
            # span_quality_flag = np.empty_like(spi_time,dtype=float)
            span_quality_flag = np.array(fov_flag,dtype=float)
            nanwhere = np.where(np.isnan(spi_dens_corrected))
            nanwhere = nanwhere[0]
            # breakpoint()
            span_quality_flag[nanwhere] = np.nan
            span_quality_flag[spi_three_sig_where] += 1
            span_quality_flag[spi_two_sig_where] += 1
        
        else:
            # span_quality_flag = np.empty_like(spi_time)
            span_quality_flag = np.array(fov_flag,dtype=float)
            nanwhere = np.where(np.isnan(spi_dens_corrected))
            nanwhere = nanwhere[0]
            span_quality_flag[nanwhere] = np.nan
        
        #-------------------construct SPC quality flag---------------------#
        
        if qtn_data_exists:
        
            spc_quality_flag = np.zeros(spi_time.shape,dtype=float)
            nanwhere = np.where(np.isnan(spc_dens_down))
            nanwhere = nanwhere[0]
            spc_quality_flag[nanwhere] = np.nan
            # spc_quality_flag = fov_flag
            spc_quality_flag[spc_three_sig_where] += 1
            spc_quality_flag[spc_two_sig_where] += 1
        
        else:
            
            spc_quality_flag = np.zeros(spi_time.shape,dtype=float)
            nanwhere = np.where(np.isnan(spc_dens_down))
            nanwhere = nanwhere[0]
            spc_quality_flag[nanwhere] = np.nan
            
            
            #------------------store quality flags-------------------#
            
        if store:
            # breakpoint()
            tplot_savename = 'SPAN_SPC_QTN_flags'+'_enc_'+str(enc_num)+'.cdf'
            tplot_savepath = str(Path('~/Documents/SPAN Checks/').expanduser())
    
            tplot_time_spi = spi_dens_time_cor
            tplot_time_spc = spi_time
            
            tplot_span_quality_flag = span_quality_flag
            tplot_spc_quality_flag = spc_quality_flag
            
            pys.store_data("SPAN_qual_flag", data={'x':tplot_time_spi, 'y':tplot_span_quality_flag})
            pys.store_data("SPC_qual_flag", data={'x':tplot_time_spc, 'y':tplot_spc_quality_flag})
    
            cdf_var_list = ["SPAN_qual_flag","SPC_qual_flag"]
            
            pys.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
        
        #------------------check agreement with figure---------------------#
        
        if plot:
            
            # breakpoint()
            
            fig = plt.figure(figsize=(15,10))
            axs = fig.add_subplot(111)
            axs.set_title('QTN Density Comparison Encounter '+str(enc_num))
            axs.plot(spc_time_down,spc_dens_down,linewidth=0.5,label='SPC Proton Density',color='tab:orange')
            # axs.plot(spc_time,spc_dens,linewidth=0.1,label='SPC Proton Density',color='tab:orange')
            # axs.plot(spi_time,spi_dens,linewidth=0.1,label='SPAN-I Proton Density',color='tab:blue')
            # axs.plot(qtn_time,qtn_dens,linewidth=0.1,label='QTN Electron Density', color='black')
            
            axs.set_ylim(-3,5000)
            
            leg = plt.legend(loc='upper right',fontsize=18)
            
            for legobj in leg.legend_handles:
                legobj.set_linewidth(2.0)
            
            
            # leg.legend_handles[0].set_sizes = [30]
            # leg.legend_handles[1].set_sizes = [30]
            
            plt.show()
        
        pys.del_data() #clear all tplot variables
        
        # breakpoint()
