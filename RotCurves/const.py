import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import os

# DIR paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLES_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_tables"

# constants
G_CONST = 4.30091e-6  # kpc (km/s)^2 Msun^-1
COSMOLOGY = FlatLambdaCDM(
    H0=70,
    Om0=0.3,
    Tcmb0=2.7255,
)
FWHM_TO_SIGMA = 2.35482


# lookup tables
def load_noor_lookuptables():
    dir = LOOKUP_TABLES_PATH+'/Noordermeer_lookup_tables'
    tables = {}
    for file in os.listdir(dir):
        n = float(file[file.find('_n')+2:file.find('_q')])
        q0 = float(file[file.find('_q')+2:file.find('.npy')])
        tables[n, q0] = np.load(
            os.path.join(
                dir,
                file
            )
        )
    return tables

def load_gaussian_tables():
    dir = LOOKUP_TABLES_PATH+'/GaussianRing_lookup_tables'
    dir_BT = LOOKUP_TABLES_PATH+'/GaussianRing_BTmin_lookup_tables'

    tables, BT_tables = {}, {}
    
    for file in os.listdir(dir):
        h = float(file[file.find('_h')+2:file.find('.npy')])
        tables[h] = np.load(
            os.path.join(
                dir,
                file
            )
        )

    bt_h_list = np.asarray(list(set([float(x[x.find('_invh') + 6:x.find('.csv')]) for x in os.listdir(dir_BT) if x.find('csv') > 0])))
    BT_tables['h_list'] = bt_h_list
    for h in bt_h_list:
        BT_tables[h] = np.load(dir_BT+f"/Gauss_BTmin_invh_{h:2.2f}.npy")

    return tables, BT_tables



