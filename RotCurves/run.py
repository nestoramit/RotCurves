import os
import sys
import pandas as pd
import time
import numpy as np

from RotCurves.galaxy_model import GalaxyObject
from RotCurves.base_utils import make_pretty_plot

sys.path.insert(0, r"/mnt/sdceph/users/ycohen/Nestor/scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcmc_fitter import full_mcmc_run


def retrieve_run_info(
        num_walkers=200,
        num_burnins=100,
        burnin_factor=3,
        num_steps=1000,
        niter_per_loop=500,
        strecth_move_a=2.,
        mcmc_moves={"StretchMove": 1.},
        mp=True,
        Galaxies_to_run=None,
        show_plots=False,
        output=True,
        metadata_table_path=None,
        galaxies_outputs_dir=None
):

    if len(sys.argv) > 1:
        cluster = True
        num_burnins = 100 if num_burnins is None else num_burnins
        num_walkers = int(sys.argv[1]) if num_walkers is None else num_walkers
        num_steps = int(sys.argv[2]) if num_steps is None else num_steps
        strecth_move_a = float(sys.argv[3]) if strecth_move_a is None else strecth_move_a
        metadata_table_path = sys.argv[4] if metadata_table_path is None else metadata_table_path
        galaxies_outputs_dir = sys.argv[5] if galaxies_outputs_dir is None else galaxies_outputs_dir
        Galaxies_to_run = [x for x in sys.argv[6:]] if Galaxies_to_run is None else ["all"]
        mp = True
        niter_per_loop = niter_per_loop

    else:
        cluster = False
        if galaxies_outputs_dir is None:
            galaxies_outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_output_folder')
        Galaxies_to_run = Galaxies_to_run
        niter_per_loop = niter_per_loop

    mcmc_hyperparameters = {
        "nwalkers": num_walkers,
        "niter": [num_burnins, num_steps],
        "burnin_factor": burnin_factor,
        "multiprocessing": mp,
        "show plots": show_plots,
        "output files": output,
        "running in cluster": cluster,
        "niter_per_loop": niter_per_loop,
        "strecth_move_a": strecth_move_a,
        "moves": mcmc_moves,
        "tau_tol": 0.03,
        "aurocorrelation_steps_thersh": 50,
        'target_neff': 1000,
    }

    return metadata_table_path, galaxies_outputs_dir, Galaxies_to_run, mcmc_hyperparameters


def MCMC_run(galaxies_to_run=["all"], metadata_table_path=None, galaxies_outputs_dir=None, obsdata_dir=None, mp=True,
             num_walkers=300, num_burnins=100, num_steps=250):

    metadata_table_path, galaxies_outputs_dir, Galaxies_to_run, mcmc_hyperparameters =\
        retrieve_run_info(num_walkers=num_walkers, num_burnins=num_burnins, num_steps=num_steps, mp=mp,
                          Galaxies_to_run=galaxies_to_run, metadata_table_path=metadata_table_path, galaxies_outputs_dir=galaxies_outputs_dir)

    print("mcmc fitting:", Galaxies_to_run)

    galaxies_list = pd.read_excel(metadata_table_path, index_col="uniqID").index.values

    if Galaxies_to_run == "all" or Galaxies_to_run == ["all"]:
        N = len(galaxies_list)
    else:
        N = len(Galaxies_to_run)

    i = 1
    starttime = time.time()

    for galaxy_name in galaxies_list:
        run = False
        if (Galaxies_to_run == ["all"]) or (Galaxies_to_run == "all"):
            run = True
        else:
            for galaxy_to_run in Galaxies_to_run:
                if galaxy_to_run == galaxy_name:
                    run = True

        # if (galaxy_name in Galaxies_to_run) or (Galaxies_to_run == ["all"]) or (Galaxies_to_run == "all"):
        if run:
            print('Working on:', galaxy_name)
            Galaxy = GalaxyObject(galaxy_name, metadata_table_path=metadata_table_path,
                                  galaxy_outputs_folder=galaxies_outputs_dir, obsdata_dir=obsdata_dir,
                                  running_in_cluster=mcmc_hyperparameters["running in cluster"])
            print("AC is on!" if Galaxy.switches["adiabatic contraction"] else "AC is off...")

            results_table, walkers_results = full_mcmc_run(Galaxy, mcmc_hyperparameters)

            runtime = time.time() - starttime
            print("\nfinished: %s/%s. Time elapsed: %s:%s:%s \n" %
                  (i, N,  np.floor(runtime / 3600), np.floor(np.mod(runtime, 3600) / 60), round(np.mod(runtime, 60))))
            i += 1

        else:
            continue

if __name__   == '__main__':
    make_pretty_plot(dpi=300)

    # if cluster:
    #     metadata_table_path = r"C:\Users\amitn\OneDrive - Tel-Aviv University\RotCurvesMCMC\metadata_tables\metadata_table_rings.xlsx"
    #     obsdata_dir = "/mnt/sdceph/users/ycohen/Nestor/inputs/RC_raw_data"

    username = os.getlogin()
    MCMC_run(galaxies_to_run=['zC_406690-MassiveRing'],
             metadata_table_path=fr"C:\Users\{username}\OneDrive - Tel-Aviv University\RotCurvesMCMC\metadata_tables\metadata_table_rings.xlsx",
             obsdata_dir=fr"C:\Users\{username}\OneDrive - Tel-Aviv University\MPE\RC_raw_data",
             mp=False, num_walkers=100, num_burnins=1, num_steps=100)
