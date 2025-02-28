#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:13:13 2022

@author: besh2109
"""

import os
from pyspedas.utilities.dailynames import dailynames
from pyspedas.utilities.download import download
from pyspedas.analysis.time_clip import time_clip as tclip
from pytplot import cdf_to_tplot

from .config import CONFIG

def load(trange=['2018-11-5', '2018-11-6'], 
         instrument='gong',
         adapt_source='gong',
         datatype='magnetogram',
         suffix='', 
         get_support_data=False, 
         varformat=None,
         varnames=[],
         downloadonly=True,
         notplot=False,
         no_update=False,
         time_clip=False,
         last_version=False,
         res=24*3600):
    
    """
    This function loads GONG magnetograms; this function is not meant 
    to be called directly; instead, see the wrappers:
        pyspedas.gong.mag
        pyspedas.gong.plastic

    """
    
    if instrument == 'gong':
        
        pathformat = '%Y%m/mrzqs%y%m%d/mrzqs%y%m%dt%H%Mc*.fits.gz'
        file = '/gong'
        server = CONFIG['gong_data_dir']
        
    if instrument == 'adapt':
        
        A = {"gong": 3, "hmi" : 4}.get(adapt_source)
        pathformat = '%Y/adapt40{A}*%Y%m%d%H%M*.fts.gz'
        file = '/adapt'
        server= CONFIG['adapt_data_dir']
        res = 2*3600
    out_files = []
    
    remote_names = dailynames(file_format=pathformat, trange=trange,res=res)

    files = download(remote_file=remote_names, remote_path=server, local_path=CONFIG['local_data_dir']+file, no_download=no_update)
    if files is not None:
        for file in files:
            out_files.append(file)

    if downloadonly:
        return out_files

    # tvars = cdf_to_tplot(out_files, suffix=suffix, get_support_data=get_support_data, varformat=varformat, varnames=varnames, notplot=notplot)

    # if notplot:
        # return tvars

    # if time_clip:
        # for new_var in tvars:
            # tclip(new_var, trange[0], trange[1], suffix='')

    # return tvars