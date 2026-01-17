"""
This script is used to run an end to end simulation, including creating or loading data, training or loading
a NN model and do an evaluation for the algorithms.
The type of the run is based on the scenrio_dict. the avialble scrorios are:
- SNR: a list of SNR values to be tested
- T: a list of number of snapshots to be tested
- eta: a list of steering vector error values to be tested
- M: a list of number of sources to be tested


"""
# Imports
import sys
from src.data_handler import *
from src.training import *
from src.plotting import *
from src.evaluation import evaluate
from pathlib import Path
from src.models import ModelGenerator
from src.system_model import SystemModel, SystemModelParams
from src.utils import set_unified_seed, initialize_data_paths, print_loss_results_from_simulation
from src.signal_creation import Samples


def __run_simulation(**kwargs):
    # Initialize seed
    set_unified_seed()

    SIMULATION_COMMANDS = kwargs["simulation_commands"]
    SYSTEM_MODEL_PARAMS = kwargs["system_model_params"]
    MODEL_CONFIG = kwargs["model_config"]
    TRAINING_PARAMS = kwargs["training_params"]
    EVALUATION_PARAMS = kwargs["evaluation_params"]
    preloaded_test_dataset = kwargs.get("preloaded_test_dataset", None)  # Optional pre-loaded test dataset
    save_to_file = SIMULATION_COMMANDS["SAVE_TO_FILE"]  # Saving results to file or present them over CMD
    create_data = SIMULATION_COMMANDS["CREATE_DATA"]  # Creating new dataset
    load_model = SIMULATION_COMMANDS["LOAD_MODEL"]  # Load specific model for training
    train_model = SIMULATION_COMMANDS["TRAIN_MODEL"]  # Applying training operation
    save_model = SIMULATION_COMMANDS["SAVE_MODEL"]  # Saving tuned model
    evaluate_mode = SIMULATION_COMMANDS["EVALUATE_MODE"]  # Evaluating desired algorithms
    plot_mode = SIMULATION_COMMANDS["PLOT_RESULTS"]  # Plotting results
    save_plots = SIMULATION_COMMANDS["SAVE_PLOTS"]  # Saving plots
    load_data = not create_data  # Loading data from exist dataset
    if train_model:
        print("Training model - ", MODEL_CONFIG.get('model_type'))
        print("Training objective - ", TRAINING_PARAMS.get('training_objective'))

    now = datetime.now()
    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    # torch.set_printoptions(precision=12)

    # Initialize paths
    datasets_path, simulations_path = initialize_data_paths(Path(__file__).parent)

    # Saving simulation scores to external file
    suffix = ""
    if train_model:
        suffix += f"_train_{MODEL_CONFIG.get('model_type')}_{TRAINING_PARAMS.get('training_objective')}"
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
        .set_parameter("signal_nature", SYSTEM_MODEL_PARAMS["signal_nature"])
        .set_parameter("signal_type", SYSTEM_MODEL_PARAMS["signal_type"])
        .set_parameter("eta", SYSTEM_MODEL_PARAMS["eta"])
        .set_parameter("bias", SYSTEM_MODEL_PARAMS["bias"])
        .set_parameter("sv_noise_var", SYSTEM_MODEL_PARAMS["sv_noise_var"])
        .set_parameter("doa_range", SYSTEM_MODEL_PARAMS["doa_range"])
        .set_parameter("doa_resolution", SYSTEM_MODEL_PARAMS["doa_resolution"])
        .set_parameter("max_range_ratio_to_limit", SYSTEM_MODEL_PARAMS["max_range_ratio_to_limit"])
        .set_parameter("range_resolution", SYSTEM_MODEL_PARAMS["range_resolution"])
        .set_parameter("wavelength", SYSTEM_MODEL_PARAMS["wavelength"])
    )

    # Define samples size
    samples_size = TRAINING_PARAMS["samples_size"]  # Overall dateset size
    train_test_ratio = TRAINING_PARAMS["train_test_ratio"]  # training and testing datasets ratio
    # Print new simulation intro
    print("------------------------------------")
    print("---------- New Simulation ----------")
    print("------------------------------------")

    if load_data:
        if train_model:
            try:
                start = time.time()
                train_dataset = load_datasets(
                    system_model_params=system_model_params,
                    samples_size=samples_size,
                    datasets_path=datasets_path,
                    is_training=True,
                )
                print(f"Load the data took {time.time() - start} sec")
            except Exception as e:
                print(e)
                print("#############################################")
                print("load_datasets: Error loading train dataset, creating new dataset")
                print("#############################################")
                create_data = True
                load_data = False
        if evaluate_mode:
            if preloaded_test_dataset is not None:
                # Use pre-loaded test dataset (for varying samples_size experiments)
                generic_test_dataset = preloaded_test_dataset
                print("Using pre-loaded test dataset for consistent evaluation across dataset sizes")
            else:
                try:
                    generic_test_dataset = load_datasets(
                        system_model_params=system_model_params,
                        samples_size=samples_size * train_test_ratio,
                        datasets_path=datasets_path,
                        is_training=False,
                    )
                except Exception as e:
                    print(e)
                    print("#############################################")
                    print("load_datasets: Error loading test dataset, creating new dataset")
                    print("#############################################")
                    create_data = True
                    load_data = False
    if create_data and not load_data:
        # Define which datasets to generate
        print("Creating Data...")
        # init sample model
        samples_model = Samples(system_model_params)

        # If we need both train and test, generate the test dataset first with a dedicated seed
        # to avoid consuming the same RNG stream as the training data (prevents accidental overlap).
        if evaluate_mode:
            if preloaded_test_dataset is not None:
                # Use pre-loaded test dataset (for varying samples_size experiments)
                generic_test_dataset = preloaded_test_dataset
                print("Using pre-loaded test dataset for consistent evaluation across dataset sizes")
            else:
                # Generate test dataset first using a fixed seed to make it deterministic and separate
                # from the training RNG stream.
                start = time.time()
                set_unified_seed(1)
                generic_test_dataset, _ = create_dataset(
                    samples_model=samples_model,
                    samples_size=int(train_test_ratio * samples_size),
                    save_datasets=SIMULATION_COMMANDS["SAVE_DATASET"],
                    datasets_path=datasets_path,
                    true_doa=TRAINING_PARAMS["true_doa_test"],
                    true_range=TRAINING_PARAMS["true_range_test"],
                    phase="test",
                )
                # restore default deterministic seed for subsequent operations
                set_unified_seed()
                print(f"Create the test data took {time.time() - start} sec")

        # Now create the training dataset (after test creation) using the default seed.
        if train_model and TRAINING_PARAMS["epochs"] > 0:
            # Generate training dataset
            start = time.time()
            train_dataset, _ = create_dataset(
                samples_model=samples_model,
                samples_size=samples_size,
                save_datasets=SIMULATION_COMMANDS["SAVE_DATASET"],
                datasets_path=datasets_path,
                true_doa=TRAINING_PARAMS["true_doa_train"],
                true_range=TRAINING_PARAMS["true_range_train"],
                phase="train",
            )
            print(f"Create the train data took {time.time() - start} sec")

    if train_model:
        # Generate model configuration
        model_config = (
            ModelGenerator()
            .set_model_type(MODEL_CONFIG.get("model_type"))
            .set_system_model(system_model_params)
            .set_model_params(MODEL_CONFIG.get("model_params"))
            .set_samples_size(samples_size)  # Set samples_size for checkpoint naming
            .set_model()
        )

        trainingparams = TrainingParamsNew(learning_rate=TRAINING_PARAMS["learning_rate"],
                                           weight_decay=TRAINING_PARAMS["weight_decay"],
                                           epochs=TRAINING_PARAMS["epochs"],
                                           optimizer=TRAINING_PARAMS["optimizer"],
                                           step_size=TRAINING_PARAMS["step_size"],
                                           gamma=TRAINING_PARAMS["gamma"],
                                           training_objective=TRAINING_PARAMS["training_objective"],
                                           scheduler=TRAINING_PARAMS["scheduler"],
                                           batch_size=TRAINING_PARAMS["batch_size"],
                                           simulation_name=TRAINING_PARAMS["simulation_name"],
                                           )
        train_dataloader, valid_dataloader = train_dataset.get_dataloaders(batch_size=TRAINING_PARAMS["batch_size"])
        trainer = Trainer(model=model_config.model, training_params=trainingparams, show_plots=True)
        model = trainer.train(train_dataloader, valid_dataloader,
                              use_wandb=TRAINING_PARAMS["use_wandb"],
                              save_final=save_model, load_model=load_model)

    # Evaluation stage
    if evaluate_mode:
        if not train_model:
            model = None
        if isinstance(system_model_params.M, int): # if M is constant over the dataset, use a simple collate function
            generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                                                batch_size=100,
                                                                shuffle=False)
        else: # if M is not constant over the dataset, use a batch sampler to create batches of the same size
            batch_sampler_test = SameLengthBatchSampler(generic_test_dataset, batch_size=100)
            generic_test_dataset = torch.utils.data.DataLoader(generic_test_dataset,
                                                           collate_fn=collate_fn,
                                                           batch_sampler=batch_sampler_test,
                                                           shuffle=False)

        # Evaluate DNN models, augmented and subspace methods
        loss = evaluate(
            generic_test_dataset=generic_test_dataset,
            system_model_params=system_model_params,
            models=EVALUATION_PARAMS["models"],
            augmented_methods=EVALUATION_PARAMS["augmented_methods"],
            subspace_methods=EVALUATION_PARAMS["subspace_methods"],
            model_tmp=model,
            samples_size=samples_size  # Pass samples_size for checkpoint naming
        )
        print("-------------------------------------")
        print("--------- End of Evaluation ---------")
        print("-------------------------------------")
        if save_to_file:
            sys.stdout.close()
            sys.stdout = orig_stdout
        return loss
    return None


def run_simulation(**kwargs):
    """
    This function is used to run an end to end simulation, including creating or loading data, training or loading
    a NN model and do an evaluation for the algorithms.
    The type of the run is based on the scenrio_dict. the avialble scrorios are:
    - SNR: a list of SNR values to be tested
    - T: a list of number of snapshots to be tested
    - eta: a list of steering vector error values to be tested
    - M: a list of number of sources to be tested
    - samples_size: a list of dataset sizes to be tested
    """
    if kwargs["scenario_dict"] == {}:
        loss = __run_simulation(**kwargs)
        return loss
    loss_dict = {}
    default_snr = kwargs["system_model_params"]["snr"]
    default_T = kwargs["system_model_params"]["T"]
    default_eta = kwargs["system_model_params"]["eta"]
    default_m = kwargs["system_model_params"]["M"]
    default_samples_size = kwargs["training_params"]["samples_size"]
    default_true_range_test = kwargs["training_params"]["true_range_test"]
    for key, value in kwargs["scenario_dict"].items():
        if key == "SNR":
            loss_dict["SNR"] = {snr: None for snr in value}
            print(f"Testing SNR values: {value}")
            for snr in value:
                kwargs["system_model_params"]["snr"] = snr
                loss = __run_simulation(**kwargs)
                loss_dict["SNR"][snr] = loss
                kwargs["system_model_params"]["snr"] = default_snr
        if key == "T":
            loss_dict["T"] = {T: None for T in value}
            print(f"Testing T values: {value}")
            for T in value:
                kwargs["system_model_params"]["T"] = T
                loss = __run_simulation(**kwargs)
                loss_dict["T"][T] = loss
                kwargs["system_model_params"]["T"] = default_T
        if key == "eta":
            loss_dict["eta"] = {eta: None for eta in value}
            print(f"Testing eta values: {value}")
            for eta in value:
                kwargs["system_model_params"]["eta"] = eta
                loss = __run_simulation(**kwargs)
                loss_dict["eta"][eta] = loss
                kwargs["system_model_params"]["eta"] = default_eta
        if key == "M":
            loss_dict["M"] = {m: None for m in value}
            print(f"Testing M values: {value}")
            for m in value:
                kwargs["system_model_params"]["M"] = m
                loss = __run_simulation(**kwargs)
                loss_dict["M"][m] = loss
                kwargs["system_model_params"]["M"] = default_m
        if key == "samples_size":
            loss_dict["samples_size"] = {size: None for size in value}
            print(f"Testing samples_size values: {value}")
            
            # For dataset size variation, use the same test dataset for all experiments
            # Use the second dataset size to determine test dataset size (or a fixed reference)
            test_dataset_size = 1024
            print(f"Using fixed test dataset size: {test_dataset_size}")
            
            # Create/load test dataset once before the loop
            datasets_path, simulations_path = initialize_data_paths(Path(__file__).parent)
            system_model_params = (
                SystemModelParams()
                .set_parameter("N", kwargs["system_model_params"]["N"])
                .set_parameter("M", kwargs["system_model_params"]["M"])
                .set_parameter("T", kwargs["system_model_params"]["T"])
                .set_parameter("snr", kwargs["system_model_params"]["snr"])
                .set_parameter("field_type", kwargs["system_model_params"]["field_type"])
                .set_parameter("signal_nature", kwargs["system_model_params"]["signal_nature"])
                .set_parameter("signal_type", kwargs["system_model_params"]["signal_type"])
                .set_parameter("eta", kwargs["system_model_params"]["eta"])
                .set_parameter("bias", kwargs["system_model_params"]["bias"])
                .set_parameter("sv_noise_var", kwargs["system_model_params"]["sv_noise_var"])
                .set_parameter("doa_range", kwargs["system_model_params"]["doa_range"])
                .set_parameter("doa_resolution", kwargs["system_model_params"]["doa_resolution"])
                .set_parameter("max_range_ratio_to_limit", kwargs["system_model_params"]["max_range_ratio_to_limit"])
                .set_parameter("range_resolution", kwargs["system_model_params"]["range_resolution"])
                .set_parameter("wavelength", kwargs["system_model_params"]["wavelength"])
            )
            
            preloaded_test_dataset = None
            if kwargs["simulation_commands"]["EVALUATE_MODE"]:
                load_data = not kwargs["simulation_commands"]["CREATE_DATA"]
                try:
                    # if load_data:
                    #     preloaded_test_dataset = load_datasets(
                    #         system_model_params=system_model_params,
                    #         samples_size=test_dataset_size,
                    #         datasets_path=datasets_path,
                    #         is_training=False,
                    #     )
                    #     print(f"Loaded shared test dataset with size: {test_dataset_size}")
                    # else:
                    set_unified_seed(0)
                    samples_model = Samples(system_model_params)
                    preloaded_test_dataset, _ = create_dataset(
                        samples_model=samples_model,
                        samples_size=test_dataset_size,
                        save_datasets=kwargs["simulation_commands"]["SAVE_DATASET"],
                        datasets_path=datasets_path,
                        true_doa=kwargs["training_params"]["true_doa_test"],
                        true_range=kwargs["training_params"]["true_range_test"],
                        phase="test",
                    )
                    set_unified_seed()
                    print(f"Created shared test dataset with size: {test_dataset_size}")
                except Exception as e:
                    print(f"Warning: Could not pre-load test dataset: {e}")
                    print("Will create test dataset for each iteration instead")
                    preloaded_test_dataset = None
            
            for size in value:
                kwargs["training_params"]["samples_size"] = size
                kwargs["preloaded_test_dataset"] = preloaded_test_dataset
                loss = __run_simulation(**kwargs)
                loss_dict["samples_size"][size] = loss
                kwargs["training_params"]["samples_size"] = default_samples_size
            # Clean up
            if "preloaded_test_dataset" in kwargs:
                del kwargs["preloaded_test_dataset"]
        if key == "true_range_test":
            wavelength = kwargs["system_model_params"]["wavelength"]
            # Create display keys as multiples of wavelength
            display_keys = {tr: tr * wavelength for tr in value}
            loss_dict["true_range_test"] = {display_keys[tr]: None for tr in value}
            print(f"Testing true_range_test values: {[f'{tr * wavelength:.2f}λ' for tr in value]}")
            for true_range in value:
                kwargs["training_params"]["true_range_test"] = [true_range] * kwargs["system_model_params"]["M"]
                loss = __run_simulation(**kwargs)
                # Store with wavelength-scaled display key
                display_key = true_range * wavelength
                loss_dict["true_range_test"][display_key] = loss
                kwargs["training_params"]["true_range_test"] = default_true_range_test
        if key == "mask_init_cell_coeff":
            # Sweep over initial mask coefficient values. For each coefficient, instantiate the model
            # to compute the absolute mask cell size (used by MUSIC.__init_cells) and use that absolute
            # value as the key for storing/plotting results.
            loss_dict["mask_init_cell_coeff"] = {}
            print(f"Testing mask_init_cell_coeff values: {value}")
            # Create a shared ModelGenerator setup to avoid duplicating code
            for coeff in value:
                # update model params with the coefficient
                kwargs["evaluation_params"]["models"]["DCD-MUSIC"]["mask_init_cell_coeff"] = coeff
                # Run simulation and store results under the absolute key
                loss = __run_simulation(**kwargs)
                loss_dict["mask_init_cell_coeff"][coeff] = loss

            # Print results for mask_init_cell_coeff
            print("\nResults for mask_init_cell_coeff:")
            for coeff, result in loss_dict["mask_init_cell_coeff"].items():
                print(f"Absolute mask size: {coeff}, Loss: {result}")
            print("\n")
            # Remove the temporary model param to avoid side effects
            del kwargs["evaluation_params"]["models"]["DCD-MUSIC"]["mask_init_cell_coeff"]
            loss_dict["mask_init_cell_coeff"] = {}
    if None not in list(next(iter(loss_dict.values())).values()):
        print_loss_results_from_simulation(loss_dict)
        if kwargs["simulation_commands"]["PLOT_LOSS_RESULTS"]:
            plot_results(loss_dict, kwargs["system_model_params"]["field_type"],
                         plot_acc=kwargs["simulation_commands"]["PLOT_ACC_RESULTS"],
                         save_to_file=kwargs["simulation_commands"]["SAVE_PLOTS"],
                         system_model_params=kwargs["system_model_params"])

    return loss_dict


if __name__ == "__main__":
    now = datetime.now()
