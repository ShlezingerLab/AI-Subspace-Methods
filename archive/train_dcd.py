"""
This is a runner script for training the DCD model.
It is a simple script that create the data, the model and train it over all 3 steps.

"""

import sys
from datetime import datetime
from pathlib import Path
from src.utils import set_unified_seed, initialize_data_paths
from src.training import Trainer, TrainingParamsNew
from src.data_handler import create_dataset, load_datasets
import argparse
from src.system_model import SystemModelParams
from src.signal_creation import Samples
from src.models import ModelGenerator

# default values for the argparse
number_sensors = 15
number_sources = "2"
number_snapshots = 100
snr = 10
field_type = "Near"
signal_type = "narrowband"
signal_nature = "coherent"
err_loc_sv = 0.0
wavelength = 1
tau = 8
sample_size = 4096
batch_size = 128
epochs = 50
optimizer = "Adam"
scheduler = "ReduceLROnPlateau"
learning_rate = 0.001
weight_decay = 1e-9
step_size = 10
gamma = 0.5
diff_method = ("esprit", "music_1d")
train_loss_type = ("rmspe", "rmspe")
regularization = None
variant = "small"
wandb_flag = False
skip_first_step = False
skip_second_step = False

def train_dcd_music(*args, **kwargs):
    SIMULATION_COMMANDS = kwargs["simulation_commands"]
    SYSTEM_MODEL_PARAMS = kwargs["system_model_params"]
    MODEL_PARAMS = kwargs["model_params"]
    TRAINING_PARAMS = kwargs["training_params"]
    save_to_file = SIMULATION_COMMANDS["SAVE_TO_FILE"]  # Saving results to file or present them over CMD
    create_data = SIMULATION_COMMANDS["CREATE_DATA"]  # Creating new dataset
    load_model = SIMULATION_COMMANDS["LOAD_MODEL"]  # Load specific model for training
    save_model = SIMULATION_COMMANDS["SAVE_MODEL"]  # Save model after training
    load_data = not create_data  # Loading data from exist dataset
    print("Running simulation...")
    print(f"Training model - DCD-MUSIC, {'all training steps' if not skip_first_step and not skip_second_step else 'partly training'}")

    now = datetime.now()
    plot_path = Path(__file__).parent / "plots"
    plot_path.mkdir(parents=True, exist_ok=True)
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")

    # Initialize seed
    set_unified_seed()

    # Initialize paths
    datasets_path, simulations_path, saving_path = initialize_data_paths(Path(__file__).parent / "data")

    # Saving simulation scores to external file
    suffix = ""
    suffix += f"_train_DCD-MUSIC_all_steps"
    suffix += (f"_{SYSTEM_MODEL_PARAMS['signal_nature']}_SNR_{SYSTEM_MODEL_PARAMS['snr']}_T_{SYSTEM_MODEL_PARAMS['T']}"
               f"_eta{SYSTEM_MODEL_PARAMS['eta']}.txt")

    if save_to_file:
        orig_stdout = sys.stdout
        file_path = (
                simulations_path / "results" / "scores" / Path(dt_string_for_save + suffix)
        )
        sys.stdout = open(file_path, "w")
    # Define system model parameters
    system_model_params = (
        SystemModelParams()
        .set_parameter("N", SYSTEM_MODEL_PARAMS["N"])
        .set_parameter("M", SYSTEM_MODEL_PARAMS["M"])
        .set_parameter("T", SYSTEM_MODEL_PARAMS["T"])
        .set_parameter("snr", SYSTEM_MODEL_PARAMS["snr"])
        .set_parameter("field_type", SYSTEM_MODEL_PARAMS["field_type"])
        .set_parameter("signal_type", SYSTEM_MODEL_PARAMS["signal_type"])
        .set_parameter("signal_nature", SYSTEM_MODEL_PARAMS["signal_nature"])
        .set_parameter("eta", SYSTEM_MODEL_PARAMS["eta"])
        .set_parameter("bias", SYSTEM_MODEL_PARAMS["bias"])
        .set_parameter("sv_noise_var", SYSTEM_MODEL_PARAMS["sv_noise_var"])
        .set_parameter("wavelength", SYSTEM_MODEL_PARAMS["wavelength"])
    )

    # Define samples size
    samples_size = TRAINING_PARAMS["samples_size"]  # Overall dateset size

    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")

    if load_data:
        try:
            train_dataset = load_datasets(
                    system_model_params=system_model_params,
                    samples_size=samples_size,
                    datasets_path=datasets_path,
                    is_training=True,
                )
        except Exception as e:
            print(e)
            print("#############################################")
            print("load_datasets: Error loading datasets")
            print("#############################################")
            create_data = True
            load_data = False
    if create_data and not load_data:
        # Define which datasets to generate
        print("Creating Data...")
        samples_model = Samples(system_model_params)
        # Generate training dataset
        train_dataset, _ = create_dataset(
            samples_model=samples_model,
            samples_size=samples_size,
            save_datasets=True,
            datasets_path=datasets_path,
            true_doa=TRAINING_PARAMS["true_doa_train"],
            true_range=TRAINING_PARAMS["true_range_train"],
            phase="train",
        )
    # Generate model configuration
    model_config = ModelGenerator()
    model_config.set_model_type("DCD-MUSIC")
    model_config.set_system_model(system_model_params)
    model_config.set_model_params({"tau": MODEL_PARAMS.get("tau"),
                                   "diff_method": diff_method,
                                   "regularization": MODEL_PARAMS.get("regularization"),
                                   "variant": MODEL_PARAMS.get("variant")})
    model_config.set_model()
    model_config.model.switch_train_mode()
    # model_config.model.update_train_mode("angle")

    trainingparams = TrainingParamsNew(learning_rate=TRAINING_PARAMS["learning_rate"],
                                       weight_decay=TRAINING_PARAMS["weight_decay"],
                                       epochs= 0 if skip_first_step else TRAINING_PARAMS["epochs"],
                                       optimizer=TRAINING_PARAMS["optimizer"],
                                       step_size=TRAINING_PARAMS["step_size"],
                                       gamma=TRAINING_PARAMS["gamma"],
                                       training_objective="angle",
                                       scheduler=TRAINING_PARAMS["scheduler"],
                                       batch_size=TRAINING_PARAMS["batch_size"]
                                       )
    train_dataloader, valid_dataloader = train_dataset.get_dataloaders(batch_size=TRAINING_PARAMS["batch_size"])
    trainer = Trainer(model=model_config.model, training_params=trainingparams, show_plots=True)
    model = trainer.train(train_dataloader, valid_dataloader,
                          use_wandb=TRAINING_PARAMS["use_wandb"],
                          save_final=save_model, load_model=load_model)

    print("END OF TRAINING - Step 1: angle branch training.")

    # Update model configuration
    model.switch_train_mode()
    # Assign the training parameters object
    trainingparams.update({"training_objective": "range"})
    trainingparams.update({"epochs": 0 if skip_second_step else TRAINING_PARAMS["epochs"]})
    trainer = Trainer(model=model, training_params=trainingparams, show_plots=True)
    model = trainer.train(train_dataloader, valid_dataloader,
                            use_wandb=TRAINING_PARAMS["use_wandb"],
                            save_final=save_model, load_model= load_model)

    print("END OF TRAINING - Step 2: distance branch training.")

    # Assign the training parameters object
    trainingparams.update({"training_objective": "angle, range",
                           "learning_rate": TRAINING_PARAMS["learning_rate"]})
    trainingparams.update({"epochs": TRAINING_PARAMS["epochs"]})
    model.init_model_train_params(init_eigenregularization_weight=1e-3, init_cell_size=0.2)
    model.switch_train_mode()
    trainer = Trainer(model=model, training_params=trainingparams, show_plots=True)
    model = trainer.train(train_dataloader, valid_dataloader,
                          use_wandb=TRAINING_PARAMS["use_wandb"],
                          save_final=save_model, load_model=False)
    print("END OF TRAINING - Step 3: adaption by position.")
    if save_to_file:
        sys.stdout = orig_stdout


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run simulation with optional parameters.")
    parser.add_argument('-n', '--number_sensors', type=int, help='Number of sensors', default=number_sensors)
    parser.add_argument('-m', '--number_sources', type=str, help='Number of sources', default=number_sources)
    parser.add_argument('-t', '--number_snapshots', type=int, help='Number of snapshots', default=number_snapshots)
    parser.add_argument('-s', '--snr', type=int, help='SNR value', default=snr)
    parser.add_argument('-ft', '--field_type', type=str, help='Field type', default=field_type)
    parser.add_argument('-st', '--signal_type', type=str, help='Signal type', default=signal_type)
    parser.add_argument('-sn', '--signal_nature', type=str, help='Signal nature', default=signal_nature)
    parser.add_argument('-eta', '--err_loc_sv', type=float, help="Error in sensors' locations", default=err_loc_sv)
    parser.add_argument('-wav', '--wavelength', type=float, help='Wavelength', default=wavelength)

    parser.add_argument('-tau', type=int, help="Number of autocorrelation features", default=tau)
    parser.add_argument("-reg", "--regularization", type=str, help="Regularization method", default=regularization)
    parser.add_argument("-v", "--variant", type=str, help="Model variant", default=variant)

    parser.add_argument('-size', '--sample_size', type=int, help='Samples size', default=sample_size)
    parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=batch_size)
    parser.add_argument('-ep', '--epochs', type=int, help='Number of epochs', default=epochs)
    parser.add_argument('-op', "--optimizer", type=str, help='Optimizer type', default=optimizer)
    parser.add_argument('-sch', "--scheduler", type=str, help='Scheduler type', default=scheduler)
    parser.add_argument('-lr', "--learning_rate", type=float, help='Learning rate', default=learning_rate)
    parser.add_argument('-wd', "--weight_decay", type=float, help='Weight decay for optimizer', default=weight_decay)
    parser.add_argument('-sp', "--step_size", type=int, help='Step size for schedular', default=step_size)
    parser.add_argument('-gm', "--gamma", type=float, help='Gamma value for schedular', default=gamma)
    parser.add_argument('-w', "--wandb", action="store_true", help='Use wandb', default=wandb_flag)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.number_sources.isnumeric():
        M = int(args.number_sources)
    else:
        M = tuple(map(int, args.number_sources.split(',')))
    system_model_params = {
        "N": args.number_sensors,  # number of antennas
        "M": M,  # number of sources
        "T": args.number_snapshots,  # number of snapshots
        "snr": args.snr,  # if defined, values in scenario_dict will be ignored
        "field_type": args.field_type,  # Near, Far
        "signal_type": args.signal_type,  # broadband, narrowband
        "signal_nature": args.signal_nature,  # if defined, values in scenario_dict will be ignored
        "eta": args.err_loc_sv,  # steering vector error
        "bias": 0,
        "sv_noise_var": 0.0,
        "wavelength": args.wavelength
    }
    model_params = {
        "tau": args.tau,
        "regularization": None if args.regularization == "None" else args.regularization,
        "variant": args.variant
    }
    training_params = {
        "samples_size": args.sample_size,
        "train_test_ratio": 0.0,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,  # Adam, SGD
        "scheduler": args.scheduler,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "step_size": args.step_size,
        "gamma": args.gamma,
        "true_doa_train": None,  # if set, this doa will be set to all samples in the train dataset
        "true_range_train": None,  # if set, this range will be set to all samples in the train dataset
        "true_doa_test": None,  # if set, this doa will be set to all samples in the test dataset
        "true_range_test": None,  # if set, this range will be set to all samples in the train dataset
        "use_wandb": args.wandb
    }
    simulation_commands = {
        "SAVE_TO_FILE": False,
        "CREATE_DATA": False,
        "LOAD_MODEL": False,
        "SAVE_MODEL": True,
    }
    train_dcd_music(simulation_commands=simulation_commands,
                    system_model_params=system_model_params,
                    model_params=model_params,
                    training_params=training_params)
