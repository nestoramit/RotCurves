import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
import os

# DIR paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOOKUP_TABLES_PATH = os.path.dirname(os.path.abspath(__file__)) + "/lookup_tables"

# constants and cosmology
G_CONST = 4.30091e-6  # kpc (km/s)^2 Msun^-1
COSMOLOGY = FlatLambdaCDM(
    H0=70,
    Om0=0.3,
    Tcmb0=2.7255,
)

# lookup tables
def load_noor_lookuptables():
    dir = LOOKUP_TABLES_PATH+'/Noordermeer_lookup_tables'

    tables = {}
    # q_list = np.asarray(
    #     list(set(
    #         [float(x[x.find('_q')+2:x.find('.npy')])
    #          for x in os.listdir(dir_path)
    #          if x.find('npy') > 0
    #          ]
    #     )))
    # n_list = np.asarray(
    #     list(set(
    #         [float(x[x.find('_n')+2:x.find('_q')])
    #          for x in os.listdir(dir_path)
    #          if x.find('npy') > 0
    #          ]
    #     )))
    for file in os.listdir(dir):
        if file.find(".npy") > 0:
            n = float(file[file.find('_n')+2:file.find('_q')])
            q0 = float(file[file.find('_q')+2:file.find('.npy')])
            tables[n, q0] = np.load(
                os.path.join(
                    dir,
                    file
                )
            )

    # for n in n_list:
    #     tables[n] = {}
    #     for q in q_list:
    #         try:
    #             tables[n][q] = np.load(os.path.join(dir_path, f"noor_n{n:2.2f}_q{q:2.2f}.npy"))
    #         except:
    #             pass

    return tables

def load_gaussian_tables():
    dir_path = LOOKUP_TABLES_PATH+'/GaussianRing_lookup_tables'
    dir_BT_path = LOOKUP_TABLES_PATH+'/GaussianRing_BTmin_lookup_tables'

    tables, BT_tables = {}, {}
    # TODO: switch from invh to h
    h_list = np.asarray(list(set([float(x[x.find('_invh') + 6:x.find('.csv')]) for x in os.listdir(dir_path) if x.find('csv') > 0])))
    tables['h_list'] = h_list
    for h in h_list:
        tables[h] = np.load(os.path.join(dir_path, f"Gauss_invh_{h:2.2f}.npy"))

    bt_h_list = np.asarray(list(set([float(x[x.find('_invh') + 6:x.find('.csv')]) for x in os.listdir(dir_BT_path) if x.find('csv') > 0])))
    BT_tables['h_list'] = bt_h_list
    for h in bt_h_list:
        BT_tables[h] = np.load(dir_BT_path+f"/Gauss_BTmin_invh_{h:2.2f}.npy")

    return tables, BT_tables



