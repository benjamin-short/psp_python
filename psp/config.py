import os

CONFIG = {'local_data_dir': 'psp_data/',
          'public_data_dir': 'https://spdf.gsfc.nasa.gov/pub/data/psp/',
          'secure_data_dir': 'http://research.ssl.berkeley.edu/data/psp/data/sci/'}

"""
if os.environ.get('PSP_FIELDS_ID'):
    CONFIG['remote_data_dir'] = 'http://research.ssl.berkeley.edu/data/psp/data/sci/'
else:
    print("Fields ID not set, using public data")
"""

# override local data directory with environment variables
if os.environ.get('SPEDAS_DATA_DIR'):
    CONFIG['local_data_dir'] = os.sep.join([os.environ['SPEDAS_DATA_DIR'], 'psp'])

if os.environ.get('PSP_DATA_DIR'):
    CONFIG['local_data_dir'] = os.environ['PSP_DATA_DIR']
