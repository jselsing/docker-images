import os
from grizli import utils

try:
    os.mkdir(utils.GRIZLI_PATH+'/CONF')
except:
    pass
    
utils.fetch_config_files(ACS=False, get_sky=False, get_stars=False, get_epsf=True)
utils.fetch_config_files(ACS=True, get_sky=False, get_stars=False, get_epsf=True)

os.system('rm {0}/CONF/*tar.gz'.format(utils.GRIZLI_PATH))
os.system('rm {0}/CONF/*sky*fits'.format(utils.GRIZLI_PATH))
os.system('rm {0}/CONF/*flat*fits'.format(utils.GRIZLI_PATH))

#utils.symlink_templates()
