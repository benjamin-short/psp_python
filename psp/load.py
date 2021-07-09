import os
from pyspedas.utilities.dailynames import dailynames
from pyspedas.utilities.download import download
from pyspedas.analysis.time_clip import time_clip as tclip
from pytplot import cdf_to_tplot

from .config import CONFIG

def load(trange=['2018-11-5', '2018-11-6'], 
         instrument='fields', 
         datatype='mag_rtn', 
         level='l2',
         suffix='', 
         get_support_data=False, 
         varformat=None,
         varnames=[],
         downloadonly=False,
         notplot=False,
         no_update=False,
         time_clip=False,
         last_version=False):
    """
    This function loads Parker Solar Probe data into tplot variables; this function is not 
    meant to be called directly; instead, see the wrappers: 
        psp.fields: FIELDS data
        psp.spc: Solar Probe Cup data
        psp.spe: SWEAP/SPAN-e data
        psp.spi: SWEAP/SPAN-i data
        psp.epihi: ISoIS/EPI-Hi data
        psp.epilo: ISoIS/EPI-Lo data
        psp.epi ISoIS/EPI (merged Hi-Lo) data
    
    """

    file_resolution = 24*3600.

    if instrument == 'fields':
        
        if os.environ.get('PSP_FIELDS_ID'):
            dateformat = '/%Y/%m/'

        else:
            dateformat = '/%Y/'
            
        if level == 'l1':
            nameform = 'spp'
        else:
            nameform = 'psp'
        """
        there might be a better way to do this, right now, if you have an
        environment variable PSP_FIELDS_ID, it assumes you're downloading
        from the secure site. - Ben Short
        """
        daytypelist = ['rfs_burst','rfs_hfr','rfs_lfr','tds_wf','dfb_ac_bpf','dfb_dc_bpf','dfb_ac_spec','dfb_dc_spec','dfb_ac_xspec','dfb_dc_xspec']
        othertypelist = ['mag_RTN_4_Sa_per_Cyc','mag_RTN_1min','mag_SC_1min','mag_SC_4_Sa_per_Cyc','f2_100bps','aeb1_hk','aeb2_hk']
        
        if datatype in daytypelist:
            dateres = '_%Y%m%d' #these datatypes do not have hour resolution.
            asterisk = '_*'
            
        elif (datatype in othertypelist) or (datatype[0:5] == 'ephem'):
            dateres = '_%Y%m%d'
            asterisk = ''
        
        else:
            dateres = '_%Y%m%d%H'
            asterisk =''

        pathformat = instrument + '/' + level + '/' + datatype + dateformat+nameform+'_fld_' + level + '_' + datatype + asterisk + dateres + '_v??.cdf'
        file_resolution = 6*3600.
        
    
    elif instrument == 'spc':
        if os.environ.get('PSP_SWEAP_ID'):
            dateformat = '/%Y/%m/'
            
            if level == 'L1':
                pathformat = 'sweap/spc/'+level+ dateformat + datatype + '/' + 'psp_swp_spc' + datatype + '_%Y%m%d_v??.cdf'
            else:
                pathformat = 'sweap/spc/'+level+ dateformat +'psp_swp_spc_l'+level[1]+'i_%Y%m%d_v??.cdf'
            
        
        else:
            pathformat = 'sweap/spc/' + level + '/' + datatype + '/%Y/psp_swp_spc_' + datatype + '_%Y%m%d_v??.cdf'
        
        
        
    elif instrument == 'spe':
        if os.environ.get('PSP_SWEAP_ID'):
            dateformat = '/%Y/%m/'
            
            if level == 'L1':
                pathformat = 'sweap/spe/'+level+'/' + datatype + dateformat + '/psp_swp_spc' + datatype + '_%Y%m%d_v??.cdf'
            elif level[0:2] == 'L2':
                pathformat = 'sweap/spe/'+level+'/'+ datatype + dateformat +'/psp_swp_'+datatype+'_'+level+'_*_%Y%m%d_v??.cdf'
            elif level == 'L3':
                pathformat = 'sweap/spe/'+level+'/'+ datatype + dateformat +'/psp_swp_'+datatype[0:7]+'_'+level+'_'+datatype[7:11]+'_%Y%m%d_v??.cdf'
        else:    
            pathformat = 'sweap/spe/' + level + '/' + datatype + '/%Y/psp_swp_sp?_*_%Y%m%d_v??.cdf'
            
    elif instrument == 'spi':
        
        if os.environ.get('PSP_SWEAP_ID'):
            dateformat = '/%Y/%m/'
            
            if level =='L1':
                pathformat = 'sweap/spi/'+level+'/'+datatype+dateformat+'psp_swp_'+datatype+'_'+level+'_%Y%m%d_v??.cdf'
            
            else:
                pathformat = 'sweap/spi/'+level+'/'+datatype+dateformat+'psp_swp_'+datatype+'_'+level+'_*_%Y%m%d_v??.cdf'
        
        else:
            pathformat = 'sweap/spi/' + level + '/' + datatype + '/%Y/psp_swp_spi_*_%Y%m%d_v??.cdf'
    
    
   
    elif instrument == 'epihi':
        pathformat = 'isois/epihi/' + level + '/' + datatype + '/%Y/psp_isois-epihi_' + level + '*_%Y%m%d_v??.cdf'
    elif instrument == 'epilo':
        pathformat = 'isois/epilo/' + level + '/' + datatype + '/%Y/psp_isois-epilo_' + level + '*_%Y%m%d_v??.cdf'
    elif instrument == 'epi':
        pathformat = 'isois/merged/' + level + '/' + datatype + '/%Y/psp_isois_' + level + '-' + datatype + '_%Y%m%d_v??.cdf'

    #credential management
    
    if instrument == 'fields':
        
        if os.environ.get('PSP_FIELDS_ID'):
            user = os.environ.get('PSP_FIELDS_ID')
            passw = os.environ.get('PSP_FIELDS_PW')
            server = CONFIG['secure_data_dir']

        else:
            user = None
            passw = None
            server = CONFIG['public_data_dir']
            
    
    elif instrument in ['spc','spe','spi']:
                
        if os.environ.get('PSP_SWEAP_ID'):
            user = os.environ.get('PSP_SWEAP_ID')
            passw = os.environ.get('PSP_SWEAP_PW')
            server = CONFIG['secure_data_dir']

        else:
            user = None
            passw = None
            server = CONFIG['public_data_dir']
    
    elif instrument in ['epihi','epilo','epi']:
        
        if os.environ.get('PSP_ISOIS_ID'):
            user = os.environ.get('PSP_ISOIS_ID')
            passw = os.environ.get('PSP_ISOIS_PW')
            server = CONFIG['secure_data_dir']

        else:
            user = None
            passw = None
            server = CONFIG['public_data_dir']



    # find the full remote path names using the trange
    remote_names = dailynames(file_format=pathformat, trange=trange, res=file_resolution)
    
    out_files = []

    files = download(remote_file=remote_names, remote_path=server, local_path=CONFIG['local_data_dir'], username=user, password=passw, no_download=no_update,last_version=last_version)
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
