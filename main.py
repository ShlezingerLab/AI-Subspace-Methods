"""
This script is used to run the simulation with the given parameters. The parameters can be set in the script or
by using the command line arguments. The script will run the simulation with the given parameters and save the
results to the results folder. The results will include the learning curves, RMSE results, and the accuracy results
of the evaluation. The results will be saved in the results folder in the project directory.

The script can be run with the following command line arguments:
    --snr: SNR value
    --N: Number of antennas
    --M: Number of sources
    --field_type: Field type
    --signal_nature: Signal nature
    --model_type: Model type
    --train: Train model
    --train_criteria: Training criteria
    --eval: Evaluate model
    --eval_criteria: Evaluation criteria
    --samples_size: Samples size
    --train_test_ratio: Train test ratio

"""
# Imports
import os
import warnings
import time
import matplotlib.pyplot as plt
from run_simulation import run_simulation
import argparse

# Initialization
os.system("cls||clear")
plt.close("all")

# if you want to run the simulation with different parameters, you can uncomment the following lines and set the values
scenario_dict = {
    # "SNR": [-10, -5, 0, 5, 10],
    # "T": [20, 30, 40, 50, 60],
    # "eta": [0.0, 0.01, 0.02, 0.03, 0.04],
    # "M": [2, 3, 4, 5, 6, 7],
}

simulation_commands = {
    "SAVE_TO_FILE": False, # if true, the logs will be saved to a file
    "CREATE_DATA": False, # if true, a new dataset will be created
    "SAVE_DATASET": True, # if true, the dataset will be saved to a file
    "LOAD_MODEL": False, # if true, the model will be loaded from a file
    "TRAIN_MODEL": True, # if true, the model will be trained
    "SAVE_MODEL": True, # if true, the model will be saved to a file
    "EVALUATE_MODE": True, # if true, a test will be performed
    "PLOT_RESULTS": True,  # if True, the learning curves will be plotted
    "PLOT_LOSS_RESULTS": True,  # if True, the RMSE results of evaluation will be plotted
    "PLOT_ACC_RESULTS": True,  # if True, the accuracy results of evaluation will be plotted
    "SAVE_PLOTS": True,  # if True, the plots will be saved to the results folder
}

system_model_params = {
    "N": 15,  # number of antennas
    "M": (2, 8),  # number of sources
    "T": 100,  # number of snapshots
    "snr": 0,  # if defined, values in scenario_dict will be ignored
    "field_type": "near",  # Near, Far
    "signal_type": "Narrowband",  # Narrowband, broadband
    "signal_nature": "coherent",  # if defined, values in scenario_dict will be ignored
    "eta": 0.0,  # steering vector uniform error variance with respect to the wavelength.
    "bias": 0, # steering vector bias error
    "sv_noise_var": 0.0, # steering vector addative gaussian error noise variance
    "doa_range": 60, # The range of the DOA values [-doa_range, doa_range]
    "doa_resolution": .5, # The resolution of the DOA values in degrees
    "max_range_ratio_to_limit": 0.5, # The ratio of the maximum range in respect to the Fraunhofer distance
    "range_resolution": 1, # The resolution of the range values in meters
    "wavelength": 1, # The carrier wavelength of the signal in meters
}
model_config = {
    "model_type": "SubspaceNet",  # SubspaceNet, DCD-MUSIC, DeepCNN, TransMUSIC, DR_MUSIC
    "model_params": {}
}
if model_config.get("model_type") == "SubspaceNet":
    model_config["model_params"]["diff_method"] = "music_2D"  # esprit, music_1D, music_2D, beamformer
    model_config["model_params"]["train_loss_type"] = "music_spectrum"  # music_spectrum, rmspe, beamformerloss
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["field_type"] = "Near"  # Far, Near
    model_config["model_params"]["regularization"] = "aic" # aic, mdl, threshold, None
    model_config["model_params"]["variant"] = "small"  # big, small
    model_config["model_params"]["norm_layer"] = True
    model_config["model_params"]["batch_norm"] = False
    model_config["model_params"]["skip_connection"] = True

elif model_config.get("model_type") == "DCD-MUSIC":
    model_config["model_params"]["tau"] = 8
    model_config["model_params"]["diff_method"] = ("esprit", "music_1D")  # ("esprit", "music_1D")
    model_config["model_params"]["train_loss_type"] = ("rmspe", "rmspe")  # ("rmspe", "rmspe"), ("rmspe",
    # "music_spectrum"), ("music_spectrum", "rmspe")
    model_config["model_params"]["regularization"] = None # aic, mdl, threshold, None
    model_config["model_params"]["variant"] = "small"  # big, small
    model_config["model_params"]["norm_layer"] = True

elif model_config.get("model_type") == "DeepCNN":
    model_config["model_params"]["grid_size"] = 361

training_params = {
    "samples_size": 40000,
    "train_test_ratio": .1,
    "training_objective": "angle, range",  # angle, range, source_estimation
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "Adam",  # Adam, SGD
    "scheduler": "ReduceLROnPlateau",  # StepLR, ReduceLROnPlateau
    "learning_rate": 0.001,
    "weight_decay": 1e-9,
    "step_size": 50,
    "gamma": 0.5,
    "true_doa_train": None,  # if set, this doa will be set to all samples in the train dataset
    "true_range_train": None,  # if set, this range will be set to all samples in the train dataset
    "true_doa_test": None,  # if set, this doa will be set to all samples in the test dataset
    "true_range_test": None,  # if set, this range will be set to all samples in the train dataset
    "use_wandb": False,
    "simulation_name": None,
}
evaluation_params = {
    "models": {
        # "TransMUSIC": {
        #                 "model_name": "TransMUSIC",
        #             },
        # "DCD-MUSIC": {
        #             "model_name": "DCD-MUSIC",
        #             "tau": 8,
        #             "diff_method": ("esprit", "music_1d"),
        #             "regularization": "threshold",
        #               },
        #  "DCD-MUSIC_V2": {
        #             "model_name": "DCD-MUSIC",
        #             "tau": 8,
        #             "diff_method": ("esprit", "music_1d"),
        #             "regularization": "aic",
        #             "variant": "big"
        #               },
        # "NFSubspaceNet": {
        #                 "model_name": "SubspaceNet",
        #                 "tau": 8,
        #                 "diff_method": "music_2D",
        #                 "train_loss_type": "music_spectrum",
        #                 "field_type": "near",
        #                 "regularization": "aic",
        #                 "skip_connection": True,
        #                 },
        # "NFSubspaceNet_V2": {
        #                 "model_name": "SubspaceNet",
        #                 "tau": 8,
        #                 "diff_method": "music_2D",
        #                 "train_loss_type": "music_spectrum",
        #                 "field_type": "near",
        #                 "regularization": "aic",
        #                 "variant": "big",
        #                 },
    },
    "augmented_methods": [
        # ("SubspaceNet", "beamformer", {"tau": 8, "diff_method": "music_2D", "train_loss_type": "music_spectrum", "field_type": "near"}),
        # ("SubspaceNet", "beamformer", {"tau": 8, "diff_method": "esprit", "train_loss_type": "rmspe", "field_type": "far"}),
        # ("SubspaceNet", "esprit", {"tau": 8, "diff_method": "esprit", "train_loss_type": "rmspe", "field_type": "far"}),
    ],
    "subspace_methods": [
        # "CCRB",
         "2D-MUSIC",
         "Beamformer",
        # "CS_Estimator",
        # "ESPRIT",
        # "1D-MUSIC",
        # "Root-MUSIC",
        # "TOPS",
    ]
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

    parser.add_argument('-mt', '--model_type', type=str, help='Model type; SubspaceNet, DCD-MUSIC, DeepCNN, TransMUSIC, DR_MUSIC', default=None)
    parser.add_argument('-reg', '--regularization', type=str, help='Regularization method for SubspaceNet of DCD', default=model_config["model_params"].get("regularization"))
    parser.add_argument('-tau', '--tau', type=int, help='Tau value for SubspaceNet or DCD-MUSIC', default=model_config["model_params"].get("tau"))
    parser.add_argument("-v", "--variant", type=str, help="Variant of the SubspaceNet model; big, small", default=model_config["model_params"].get("variant"))

    parser.add_argument('-ss', '--samples_size', type=int, help='Samples size', default=None)
    parser.add_argument('-ttr', '--train_test_ratio', type=float, help='Train test ratio', default=None)
    parser.add_argument('-to', '--training_objective', type=str, help='Training objective; angle, range or angle, range.', default=None)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=None)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs', default=None)
    parser.add_argument('-op', '--optimizer', type=str, help='Optimizer; Adam, SGD', default=None)
    parser.add_argument('-sch', '--scheduler', type=str, help='Scheduler; StepLR, ReduceLROnPlateau', default=None)
    parser.add_argument('-lr', '--learning_rate', type=float, help='Learning rate', default=None)
    parser.add_argument('-wd', '--weight_decay', type=float, help='Weight decay', default=None)
    parser.add_argument('-step', '--step_size', type=int, help='Step size', default=None)
    parser.add_argument('-g', '--gamma', type=float, help='Gamma', default=None)
    parser.add_argument('-w', '--wandb', action="store_true", help='Use wandb', default=training_params["use_wandb"])

    parser.add_argument('-t', '--train', action="store_true", help='Train model', default=simulation_commands["TRAIN_MODEL"])
    parser.add_argument('-no_t', "--no_train", action="store_false", help='Do not train model', dest='train')
    parser.add_argument('-e', '--eval', action="store_true", help='Evaluate model', default=simulation_commands["EVALUATE_MODE"])

    parser.add_argument('-c', '--create', action="store_true", help='create a new dataset', default=simulation_commands["CREATE_DATA"])
    parser.add_argument('-sv', '--save', action='store_true', help="save dataset", default=simulation_commands["SAVE_DATASET"])

    return parser.parse_args()


if __name__ == "__main__":
    # torch.set_printoptions(precision=12)

    args = parse_arguments()
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

    if args.model_type is not None:
        warnings.warn("Please make sure to configure the model parameters in the script.")
        model_config["model_type"] = args.model_type
    if model_config["model_type"] == "SubspaceNet":
        model_config["model_params"]["regularization"] = None if args.regularization == "None" else args.regularization
        model_config["model_params"]["tau"] = args.tau
        model_config["model_params"]["variant"] = args.variant

    if args.samples_size is not None:
        training_params["samples_size"] = args.samples_size
    if args.train_test_ratio is not None:
        training_params["train_test_ratio"] = args.train_test_ratio
    if args.training_objective is not None:
        if args.training_objective.startswith("angle,range"):
            training_params["training_objective"] = "angle, range"
        else:
            training_params["training_objective"] = args.training_objective
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
    if args.gamma is not None:
        training_params["gamma"] = args.gamma
    if args.wandb is not None:
        training_params["use_wandb"] = args.wandb

    simulation_commands["TRAIN_MODEL"] = args.train
    simulation_commands["EVALUATE_MODE"] = args.eval

    simulation_commands["CREATE_DATA"] = args.create
    simulation_commands["SAVE_DATASET"] = args.save

    start = time.time()
    loss = run_simulation(simulation_commands=simulation_commands,
                          system_model_params=system_model_params,
                          model_config=model_config,
                          training_params=training_params,
                          evaluation_params=evaluation_params,
                          scenario_dict=scenario_dict)
    print("Total time: ", time.time() - start)
