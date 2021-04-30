import os
from pyspedas.utilities.dailynames import dailynames
from pyspedas.utilities.download import download
from pyspedas.analysis.time_clip import time_clip as tclip
from pytplot import cdf_to_tplot

from .config import CONFIG

def load(trange=['2013-11-5', '2013-11-6'], 
         probe='a',
         instrument='mag',
         level='l2',
         datatype='8hz',
         coord='RTN',
         suffix='', 
         get_support_data=False, 
         varformat=None,
         varnames=[],
         downloadonly=False,
         notplot=False,
         no_update=False,
         time_clip=False,
         res = 24*3600):
    """
    This function loads data from the STEREO mission; this function is not meant 
    to be called directly; instead, see the wrappers:
        pyspedas.stereo.mag
        pyspedas.stereo.plastic

    """

    out_files = []

    if not isinstance(probe, list):
        probe = [probe]

    if datatype == '32hz':
        burst = 'B'
    else:
        burst = ''

    for prb in probe:
        if prb == 'a':
            direction = 'ahead'
        elif prb == 'b':
            direction = 'behind'
        
        if instrument == 'mag':
            pathformat = 'impact/level1/'+direction+'/mag/'+coord+'/%Y/%m/ST'+prb.upper()+'_L1_MAG'+burst+'_'+coord+'_%Y%m%d_V??.cdf'
            file = '/mag'
        elif instrument == 'plastic':
            CONFIG['remote_data_dir'] = 'http://stereo-ssc.nascom.nasa.gov/data/ins_data/'
            if level == 'l2':
                pathformat = 'plastic/level2/Protons/Derived_from_1D_Maxwellian/'+direction+'/'+datatype+'/%Y/ST'+prb.upper()+'_L2_PLA_1DMax_'+datatype+'_%Y%m%d_V??.cdf'
            file = '/plastic'
        elif instrument[0:6] == 'secchi':
            CONFIG['remote_data_dir'] = 'https://stereo-ssc.nascom.nasa.gov/pub/browse_sphere/'
            if instrument[6:9] == '171':
                chnl = '171'
            elif instrument[6:9] == '195':
                chnl = '195'
            elif instrument[6:9] == '284':
                chnl = '284'
            elif instrument[6:9] == '304':
                chnl = '304'
            else:
                chnl = '171'
            pathformat = '%Y/%m/%d/'+chnl+'/%Y%m%d_*.jpg'
            file = '/secchi'
            downloadonly=True #downloading a jpg, no Tplot Variable
        # find the full remote path names using the trange
        remote_names = dailynames(file_format=pathformat, trange=trange,res=res)

        files = download(remote_file=remote_names, remote_path=CONFIG['remote_data_dir'], local_path=CONFIG['local_data_dir']+file, no_download=no_update)
        if files is not None:
            for file in files:
                out_files.append(file)

    out_files = sorted(out_files)

    if downloadonly:
        return out_files

    tvars = cdf_to_tplot(out_files, suffix=suffix, get_support_data=get_support_data, varformat=varformat, varnames=varnames, notplot=notplot)
    
    if notplot:
        return tvars

    if time_clip:
        for new_var in tvars:
            tclip(new_var, trange[0], trange[1], suffix='')

    return tvars
