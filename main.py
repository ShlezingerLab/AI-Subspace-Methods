"""
This script is used to run the simulation with the given parameters. The parameters can be set in the script or
by using the command line arguments. The script will run the simulation with the given parameters and save the
results to the results folder.

The script can be run with the following command line arguments:
    --snr: SNR value
    --N: Number of antennas
    --M: Number of sources
    --field_type: Field type
    --signal_nature: Signal nature
    --model_type: Model type
    --loss_type: Training loss type (rmspe, spectrum, unsupervised)
    --epochs: Number of training epochs
    --batch_size: Batch size
    --learning_rate: Learning rate
    --optimizer: Optimizer type
    --scheduler: Scheduler type
    --create: Create new dataset
    --wandb: Use wandb

"""
# Imports
import os
import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
from run_simulation import run_simulation
import argparse
import torch

#TODO: 
# 1. Right now, each experiment is saved alone (while optimizing the sotfmax window size).
# we need to create also a multi-case scenario where we can test different cross products of parameters.
# Something similiar to the old scenario_dict, but not necessirilly it. for the clearness of the code 
# we might want to create a new class for this, including the plotting capabilities (next point)

# 2. based on the multi-case results, create functions plotting relevanmt graphs from the paper to
# test our recreation. not such a big deal, please arrange it under one class.

#NOTE:
# points for the future:

# 1. The unsupervised loss suffers from problems at the endfire due to smaller number of samples in the window.
# We need to think about maybe mirroring at the edges or renormalizing the Jain's index somehow.

# 2. The losses minimize thr rmspe, leading to higher DoA accuracy, but the actual configuration paraameters
# diverge heavily. It'll be interesting to compare the learned steering matrix itself to the physical one although we will get
# probably the same conclusion.
# SUPER IMPORTANT IMPLICATION: If we don't really learn the rifht parameters, tracking with this garbage observations will not lead us far.

# Initialization
os.system("cls||clear")
plt.close("all")

scenario_dict = { # Fill in to do multiple experiments 
    # # Example 1: Single loss function sweep (original behavior)
    # "SNR_sweep_rmspe": {
    #     "parameter": "snr",
    #     "values": [-10, -5, 0, 5, 10, 15, 20, 25, 30],
    #     "fixed_params": {"T": 100},
    #     "plot_config": {
    #         "title": "RMSPE vs SNR (T=100, RMSPE Loss)",
    #         "x_label": "SNR (dB)", 
    #         "y_label": "RMSPE (degrees)",
    #         "save_name": "rmspe_vs_snr_rmspe_loss"
    #     }
    # },
    
    # # Example 2: Multi-loss function sweep
    # "SNR_sweep_multi_loss": {
    #     "parameter": "snr",
    #     "values": [-5, 0, 5, 10, 15, 20, 25, 30],
    #     "loss_functions": ["rmspe", "spectrum", "unsupervised"],  # This enables multi-loss mode
    #     "fixed_params": {"T": 100},
    #     "plot_config": {
    #         "title": "RMSPE vs SNR (T=100) - Loss Function Comparison",
    #         "x_label": "SNR (dB)", 
    #         "y_label": "RMSPE (degrees)",
    #         "save_name": "rmspe_vs_snr_multi_loss"
    #     }
    # },
    
    # # Example 3: T sweep with multi-loss
    # "T_sweep_multi_loss": {
    #     "parameter": "T", 
    #     "values": [20, 30, 50, 70, 100, 150, 200],
    #     "loss_functions": ["rmspe", "spectrum", "unsupervised"],
    #     "fixed_params": {"snr": 30},
    #     "plot_config": {
    #         "title": "RMSPE vs T (SNR=30dB) - Loss Function Comparison",
    #         "x_label": "T (snapshots)",
    #         "y_label": "RMSPE (degrees)", 
    #         "save_name": "rmspe_vs_T_multi_loss"
    #     }
    # }     
}

simulation_commands = {
    "create_data": True,
    "save_data": False,  # Save data after creation
    "plot_results": True,  # Plot data after creation
    "multi_loss_comparison": True,  # Enable multi-loss spectrum and learned parameters comparison, if multiple experiments then need to be False
    "spectrum_loss_functions": ["rmspe", "spectrum", "unsupervised"],  # Loss functions to compare
    "data_loading_path": "datasets/N:16_M:5_T:100_snr:10_location_pert_boundary:0.25_gain_perturbation_var:0.36_seed:42/03_06_2025_15_06/data.pkl"
    # This is the path to the data file, ONLY USED if CREATE_DATA is False!
    # By now, this gets set manually.
}

system_model_params = {
    "N": 16,  # number of antennas
    "M": 5,  # number of sources
    "T": 100,  # number of snapshots
    "snr": 10,  # if defined, values in scenario_dict will be ignored 
    "bias": 0, # steering vector bias error
    "sv_noise_var": 0.0, # steering vector addative gaussian error noise variance
    "doa_range": 60, # The range of the DOA values [-doa_range, doa_range]
    "doa_resolution": .5, # The resolution of the DOA values in degrees
    "wavelength": 1, # The carrier wavelength of the signal in meters, 1 can be fine for research,
    # 0.06 is for wifi 5 GHz for example.

    "location_perturbation": "wavelength/4",  # The boundaries of the location perturbation in meters, 
    # insert any valid float between 0 and wavelength/4 or "wavelength/n" to use with reference to the wavelength
    
    "gain_perturbation_var": 0.36, # The variance of the gain perturbation
    "seed": 42,  # Seed for reproducibility
    ############################### Fixed for now ##################################
    "field_type": "Far",  # Near, Far
    "signal_type": "Narrowband",  # Narrowband, broadband
    "signal_nature": "non-coherent"  # if defined, values in scenario_dict will be ignored
}

model_config = \
{
    "model_type": "diffMUSIC",  # or "MUSIC"
    
    # Case 1: Fixed integer window size (original behavior)
    # "softmax_window_size": 21,
    
    # Case 2: Relative window size (float between 0-1)
    "softmax_window_size": 0.1,  # % of angle grid length
    
    # Case 3: Window size optimization (array/list of values)
    # "softmax_window_size": np.arange(0.01, 0.4, 0.01),  # Relative sizes: [0.2, 0.25, 0.3, 0.35]
    # "softmax_window_size": [15, 21, 27, 33],  # Absolute sizes
    # "softmax_window_size": [0.25, 21, 0.35, 27],  # Mixed relative and absolute
}


training_params = {
    # "batch_size": 128,  # Note: This is legacy parameter, actual batch size is handled by snapshots
    "epochs": 100,
    "loss_type": "spectrum",  # rmspe, spectrum, unsupervised
    "optimizer": "Adam",  # Adam, SGD
    "scheduler": "ReduceLROnPlateau",  # StepLR, ReduceLROnPlateau
    "learning_rate": 0.001,
    "step_size": 50,
    "weight_decay": 0.0,
    "use_wandb": False
}



def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('-s', "--snr" ,type=int, help='SNR value', default=None)
    parser.add_argument('-n', "--number_of_sensors",type=int, help='Number of antennas', default=None)
    parser.add_argument('-m', "--number_of_sources", help='Number of sources, could be int or tuple for random case', default=None)
    parser.add_argument('-snap', "--number_of_snapshots", type=int, help='Number of snapshots', default=None)
    parser.add_argument('-eta', "--sv_error_var", type=float, help='Steering vector uniform error variance', default=None)
    parser.add_argument('-ft', '--field_type', type=str, help='Field type, far or near field.', default=None)
    parser.add_argument('-sn', '--signal_nature', type=str, help='Signal nature; non-coherent or coherent', default=None)
    parser.add_argument('-wav', '--wavelength', type=float, help='Wavelength of the signal in meters', default=None)
    parser.add_argument("--location_perturbation", type=float, help="Location perturbation variance", default=None)
    parser.add_argument("--gain_perturbation_var", type=float, help="Gain perturbation variance", default=None)
    parser.add_argument("--seed", type=int, help="Seed for reproducibility", default=None)

    parser.add_argument('-mt', '--model_type', type=str, help='Model type; diffMUSIC, MUSIC', default=None)
    parser.add_argument('-lt', '--loss_type', type=str, help='Loss type; rmspe, spectrum, unsupervised', default=None)

    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size (legacy parameter)', default=None)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('-op', '--optimizer', type=str, help='Optimizer; Adam, SGD', default=None)
    parser.add_argument('-sch', '--scheduler', type=str, help='Scheduler; StepLR, ReduceLROnPlateau', default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=None)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=None)
    parser.add_argument('-step', '--step_size', type=int, help='Step size', default=None)

    parser.add_argument('-w', '--wandb', action="store_true", help='Use wandb')
    parser.add_argument('-c', '--create', action="store_true", help='create a new dataset')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Update system model parameters from command line arguments
    if args.snr is not None:
        system_model_params["snr"] = args.snr
    if args.number_of_sensors is not None:
        system_model_params["N"] = args.number_of_sensors
    if args.number_of_sources is not None:
        # catch a case of random number of sources between two possible values
        str_m = args.number_of_sources
        if str_m.isnumeric():
            system_model_params["M"] = int(str_m)
        else:
            system_model_params["M"] = tuple(map(int, str_m.split(',')))
    if args.number_of_snapshots is not None:
        system_model_params["T"] = args.number_of_snapshots
    if args.sv_error_var is not None:
        system_model_params["eta"] = args.sv_error_var
    if args.field_type is not None:
        system_model_params["field_type"] = args.field_type
    if args.signal_nature is not None:
        system_model_params["signal_nature"] = args.signal_nature
    if args.wavelength is not None:
        system_model_params["wavelength"] = args.wavelength

    if args.location_perturbation is not None:
        system_model_params["location_perturbation"] = args.location_perturbation
    elif (isinstance(system_model_params["location_perturbation"], str)) and ("wavelength" in system_model_params["location_perturbation"]):
        int_idx = system_model_params["location_perturbation"].find("/") + 1
        system_model_params["location_perturbation"] = (system_model_params["wavelength"] /
                                                         float(system_model_params["location_perturbation"][int_idx:int_idx + 1]))

    if args.gain_perturbation_var is not None:
        system_model_params["gain_perturbation_var"] = args.gain_perturbation_var
    if args.seed is not None:
        system_model_params["seed"] = args.seed

    # Update model configuration
    if args.model_type is not None:
        model_config["model_type"] = args.model_type

    # Update training parameters
    if args.loss_type is not None:
        training_params["loss_type"] = args.loss_type
    if args.batch_size is not None:
        training_params["batch_size"] = args.batch_size
    if args.epochs is not None:
        training_params["epochs"] = args.epochs
    if args.optimizer is not None:
        training_params["optimizer"] = args.optimizer
    if args.scheduler is not None:
        training_params["scheduler"] = args.scheduler
    if args.learning_rate is not None:
        training_params["learning_rate"] = args.learning_rate
    if args.weight_decay is not None:
        training_params["weight_decay"] = args.weight_decay
    if args.step_size is not None:
        training_params["step_size"] = args.step_size

    if args.wandb:
        training_params["use_wandb"] = args.wandb
    if args.create:
        simulation_commands["create_data"] = args.create
    
    simulation_commands["load_data"] = not simulation_commands["create_data"]

    # Validate location perturbation
    if system_model_params["location_perturbation"] > system_model_params["wavelength"] / 4:
        raise ValueError("Location perturbation should be less than wavelength/4, "
                         "This may result in order switching between neighboring array sensors.")

    start = time.time()
    results = run_simulation(simulation_commands=simulation_commands,
                           system_model_params=system_model_params,
                           model_config=model_config,
                           training_params=training_params,
                           scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)
    
    # Print final results summary
    if simulation_commands.get("multi_loss_spectrum_comparison", False):
        print(f"\nMulti-Loss Spectrum Comparison Completed!")
        print(f"Loss functions compared: {simulation_commands['spectrum_loss_functions']}")
        print(f"Results saved and plots generated.")
    elif  isinstance(results, dict) and 'rmspe' in results:
        print(f"\nFinal Results:")
        print(f"RMSPE: {results['rmspe']:.6f} degrees")
        if 'steering_matrix_mse' in results:
            print(f"Steering Matrix MSE: {results['steering_matrix_mse']:.6f}")  # Add this line
        if 'estimated_angles' in results and 'true_angles' in results:
            print(f"True angles: {np.rad2deg(results['true_angles'])}")
            print(f"Estimated angles: {np.rad2deg(results['estimated_angles'])}")
        if 'final_train_loss' in results:
            print(f"Final training loss: {results['final_train_loss']:.6f}")  
        if "learned_antenna_positions" in results.keys() and "learned_antennas_gains" in results.keys():
            if model_config["model_type"].lower() == "music":
                print("These should be the tandard one:")
            print(f"Learned antennas positions: {np.round(results['learned_antenna_positions'], 4)}")
            print("Compared to the physical array positions:")
            print(f"Physical antennas positions: {results['physical_array']}")
            print(f"Learned antennas gains: {results['learned_antennas_gains']}")
            print(f"Learned antennas gains phase: {np.round(np.angle(results['learned_antennas_gains']), 4)}")
            print("Compared to the physical antennas gains:")
            print(f"Physical antennas gains: {results['physical_antennas_gains']}")
            print(f"Physical antennas gains phase: {np.round(np.angle(results['physical_antennas_gains']), 4)}")