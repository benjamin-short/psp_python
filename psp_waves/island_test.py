#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 16:47:20 2021

@author: benshort
"""

import pyspedas as pys 
import pyspedas.psp as psp
import pytplot as pyt
from py_islands import islands
import numpy as np
from statistics import median
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.io import savemat


t0 = '2020-01-29/10:30:00'
tf = '2020-01-29/10:45:00'

tftmp = pys.time_string(pys.time_double(tf)+86400.)

psp.fields(trange=[t0[0:10],tftmp[0:10]],datatype='dfb_ac_spec',level='l2')

acspec_data = pyt.get_data('psp_fld_l2_dfb_ac_spec_dV12hg')

t0doub = pys.time_double(t0)
tfdoub = pys.time_double(tf)

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

ac_where = np.where((acspec_time_arr > t0doub) & (acspec_time_arr < tfdoub))
ac_where = ac_where[0]

acspec_time = acspec_time_arr[ac_where]
acspec_arr = acspec_data_arr[:,ac_where]
acspec_freq = acspec_freq_lst

thresh = 8

array = np.where(acspec_arr <= thresh,np.nan,acspec_arr)
array = np.where(array > thresh,1,array)  

arr_lib = {"a": array}
#savemat("island_test.mat",arr_lib)

isl = islands(acspec_arr, 8)

isl_arr = isl[0]
isl_arr = isl_arr.astype(float)
isl_arr[isl_arr == 0] = np.nan

breakpoint()

fig = plt.figure(figsize=(25,20))
axs1 = fig.add_subplot(311)
axs2 = fig.add_subplot(312)
axs3 = fig.add_subplot(313)

f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
# g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
acspec = axs1.pcolormesh(acspec_time,acspec_freq,acspec_arr,cmap='nipy_spectral',shading='nearest')

axs1.set_yscale("log")
axs1.set_ylim(acspec_freq_lst[2],acspec_freq_lst[51])
axs1.set_yticks([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
axs1.set_yticklabels([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
# axs1.yaxis.set_major_formatter(ticker.FuncFormatter(g))

                
box = axs1.get_position()
axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])

fig.colorbar(acspec,cax=axColor, label='dB')

checkspec = axs2.pcolormesh(acspec_time,acspec_freq,array,cmap='nipy_spectral',shading='nearest')

axs2.set_yscale("log")
axs2.set_ylim(acspec_freq_lst[2],acspec_freq_lst[51])
axs2.set_yticks([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
axs2.set_yticklabels([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
# axs2.yaxis.set_major_formatter(ticker.FuncFormatter(g))

                
box = axs2.get_position()
axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])

fig.colorbar(checkspec,cax=axColor, label='Where Check')

islspec = axs3.pcolormesh(acspec_time,acspec_freq,isl_arr,cmap='nipy_spectral',shading='nearest')

axs3.set_yscale("log")
axs3.set_ylim(acspec_freq_lst[2],acspec_freq_lst[51])
axs3.set_yticks([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
axs3.set_yticklabels([5*10e1,10e2,2*10e2,5*10e2,10e3,2*10e3,5*10e3])
# axs3.yaxis.set_major_formatter(ticker.FuncFormatter(g))

                
box = axs3.get_position()
axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])

fig.colorbar(islspec,cax=axColor, label='Island Number')

plt.show()
