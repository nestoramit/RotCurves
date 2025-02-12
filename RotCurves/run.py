from utils import *
from classes import GalaxyObject

sys.path.insert(0, r"/mnt/sdceph/users/ycohen/Nestor/scripts")
# username = [x for x in ['amitn', 'Amit'] if x in os.listdir(r"C:\Users")][0]
# sys.path.insert(0, rf"C:\Users\{username}\OneDrive - Tel-Aviv University\Amit research\code\github-rotationcurves\rotationcurves\\")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import RotCurves.mcmc_functions as mc
import RotCurves.mass_models as models
# from rotationcurves.plotting import *
# import rotationcurves.mcmc.mcmc_functions as mc
# import rotationcurves.models.models as models

# ---------------------------------------------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------------------------------------------- #

def retrieve_run_info(num_walkers=300, num_burnins=100, num_steps=250, strecth_move_a=5., mp=True,
                      Galaxies_to_run=["all"], show_plots=False, output=True, metadata_table_path=None, galaxies_outputs_dir=None):
    if len(sys.argv) > 1:
        cluster = True
        num_burnins = 100
        num_walkers, num_steps = int(sys.argv[1]), int(sys.argv[2])
        strecth_move_a = float(sys.argv[3])
        metadata_table_path = sys.argv[4]
        if galaxies_outputs_dir is None:
            galaxies_outputs_dir = sys.argv[5]
        Galaxies_to_run = [x for x in sys.argv[6:]]
        mp = True
        niter_per_loop = 50

    else:
        cluster = False
        # if metadata_table_path is None:
        #     metadata_table_path = os.path.join(main_path, 'default_output_folder', 'metadata_tables', 'metadata_table_RC100.xlsx')
        if galaxies_outputs_dir is None:
            galaxies_outputs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'default_output_folder')
        Galaxies_to_run = Galaxies_to_run
        niter_per_loop = 50

    mcmc_hyperparameters = {
        "nwalkers": num_walkers,
        "niter": [num_burnins, num_steps],
        "multiprocessing": mp,
        "show plots": show_plots,
        "output files": output,
        "running in cluster": cluster,
        "niter_per_loop": niter_per_loop,
        "tau_tol": 0.05,
        "aurocorrelation_steps_thersh": 50,
        "strecth_move_a": strecth_move_a
    }

    return metadata_table_path, galaxies_outputs_dir, Galaxies_to_run, mcmc_hyperparameters


# def parameter_variance_run(selected_galaxies, parameter, parameter_range, show_plots, output, mp):
#     metadata_table_path, galaxies_outputs_dir, Galaxies_to_run, mcmc_hyperparameters = retrieve_run_info(Galaxies_to_run=selected_galaxies, show_plots=show_plots, output=output, mp=mp)
#
#     # determine galaxies list to run on
#     if Galaxies_to_run == ["all"]:
#         galaxies_list = pd.read_excel(metadata_table_path, index_col="uniqID").index
#     else:
#         galaxies_list = Galaxies_to_run
#
#     # set output file skeleton
#     columns = ["Re", "Re errplus", "Re errminus", "M_baryon", "M_baryon errplus", "M_baryon errminus", "M_vir",
#                "M_vir errplus", "M_vir errminus", "BT", "BT errplus", "BT errminus", "sigma", "sigma errplus",
#                "sigma errminus", "c", "c errplus", "c errminus", "alpha", "alpha errplus", "alpha errminus", "i",
#                "i errplus", "i errminus", "f", "f errplus", "f errminus"]
#     all_params = [x for x in models.GalaxyObject(galaxies_list[0], metadata_table_path=metadata_table_path, running_in_cluster=mcmc_hyperparameters["running in cluster"]).switches["parameters"]] + ["f"]
#
#     # loop over all of the galaxies, creating a separate output file for each one over the parameter space
#     for galaxy_name in galaxies_list:
#         idxs = ["-".join([galaxy_name, str(i)]) for i in range(len(parameter_range))]
#         galaxy_df = pd.DataFrame(np.zeros((len(idxs), len(columns))), index=idxs, columns=columns)
#         i = 0
#
#         # loop through the values of the parameter and insert them into the priors
#         for value in parameter_range:
#             galaxy = models.GalaxyObject(galaxy_name, metadata_table_path=metadata_table_path, galaxy_outputs_folder=galaxies_outputs_dir, running_in_cluster=mcmc_hyperparameters["running in cluster"])
#             galaxy.priors[parameter].initial = value
#             value *= galaxy.scales[parameter]
#
#             # verify that switch is turned off
#             if parameter in galaxy.switches["parameters"].keys():
#                 galaxy.switches["parameters"][parameter] = 0
#             else:
#                 pass
#
#             # run mcmc model
#             results_table, walkers_results = mc.full_mcmc_run(galaxy, mcmc_hyperparameters)
#
#             # insert results into df
#             for model_param in all_params:
#                 if model_param == "f":
#                     galaxy_df[model_param]["-".join([galaxy_name, str(i)])] = np.round(results_table["median"][model_param], 4)
#                     galaxy_df[model_param + " errplus"]["-".join([galaxy_name, str(i)])] = np.round(results_table["errplus"][model_param], 4)
#                     galaxy_df[model_param + " errminus"]["-".join([galaxy_name, str(i)])] = np.round(results_table["errminus"][model_param], 4)
#                 elif galaxy.switches["parameters"][model_param]:
#                     galaxy_df[model_param]["-".join([galaxy_name, str(i)])] = np.round(results_table["median"][model_param], 4)
#                     galaxy_df[model_param + " errplus"]["-".join([galaxy_name, str(i)])] = np.round(results_table["errplus"][model_param], 4)
#                     galaxy_df[model_param + " errminus"]["-".join([galaxy_name, str(i)])] = np.round(results_table["errminus"][model_param], 4)
#                 else:
#                     galaxy_df[model_param]["-".join([galaxy_name, str(i)])] = np.round(galaxy.priors[model_param].initial, 4)
#                     galaxy_df[model_param + " errplus"]["-".join([galaxy_name, str(i)])] = 0
#                     galaxy_df[model_param + " errminus"]["-".join([galaxy_name, str(i)])] = 0
#
#             i += 1
#
#         # export galaxy results to .csv file
#         galaxy_df.to_csv("/".join([galaxies_outputs_dir, galaxy_name + "_" + parameter + ".csv"]))


def MCMC_run(galaxies_to_run=["all"], metadata_table_path=None, galaxies_outputs_dir=None, obsdata_dir=None, mp=True,
             num_walkers=300, num_burnins=100, num_steps=250):

    # Specify what galaxies to run: list of NAMES or ["all"]
    # if running in cluster chosen galaxies are given using sys.argv statement in cmd line
    # if not, galaxies need to be stated specifically

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
                # if galaxy_to_run in galaxy_name:
                if galaxy_to_run == galaxy_name:
                    run = True

        # if (galaxy_name in Galaxies_to_run) or (Galaxies_to_run == ["all"]) or (Galaxies_to_run == "all"):
        if run:
            print('Working on:', galaxy_name)
            Galaxy = GalaxyObject(galaxy_name, metadata_table_path=metadata_table_path,
                                  galaxy_outputs_folder=galaxies_outputs_dir, obsdata_dir=obsdata_dir,
                                  running_in_cluster=mcmc_hyperparameters["running in cluster"])
            print("AC is on!" if Galaxy.switches["adiabatic contraction"] else "AC is off...")

            results_table, walkers_results = mc.full_mcmc_run(Galaxy, mcmc_hyperparameters)

            runtime = time.time() - starttime
            print("\nfinished: %s/%s. Time elapsed: %s:%s:%s \n" %
                  (i, N,  np.floor(runtime / 3600), np.floor(np.mod(runtime, 3600) / 60), round(np.mod(runtime, 60))))
            i += 1

        else:
            continue

if __name__ == '__main__':
    make_pretty_plot(dpi=300)

    # if cluster:
    #     metadata_table_path = r"C:\Users\amitn\OneDrive - Tel-Aviv University\RotCurvesMCMC\metadata_tables\metadata_table_rings.xlsx"
    #     obsdata_dir = "/mnt/sdceph/users/ycohen/Nestor/inputs/RC_raw_data"

    MCMC_run(galaxies_to_run=['zC_406690-MassiveRing'],
             metadata_table_path=r"C:\Users\amitn\OneDrive - Tel-Aviv University\RotCurvesMCMC\metadata_tables\metadata_table_rings.xlsx",
             obsdata_dir=r"C:\Users\amitn\OneDrive - Tel-Aviv University\MPE\RC_raw_data",
             mp=True, num_walkers=20, num_burnins=1, num_steps=5)
