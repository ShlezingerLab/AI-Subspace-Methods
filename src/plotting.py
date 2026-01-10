"""
Module for plotting simulation results.
"""
# Imports
from datetime import datetime
import os
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
from src.utils import plot_styles, parse_loss_results_for_plotting
import warnings

def plot_results(loss_dict: dict, field_type: str, plot_acc: bool = False, save_to_file: bool = False):
    """
    Plot the results of the simulation.
    The dict could be with several scenarios, each with different SNR values, or with different number of snapshots,
    or with diffetent noise to the steering matrix, or with different dataset sizes.

    Parameters
    ----------
    loss_dict

    Returns
    -------

    """
    now = datetime.now()
    base_plot_path = Path(__file__).parent.parent / "plots" / "simulations"
    snr_plot_path = base_plot_path / "SNR"
    snapshots_plot_path = base_plot_path / "Snapshots"
    steering_noise_plot_path = base_plot_path / "SteeringNoise"
    number_of_sources_plot_path = base_plot_path / "NumberOfSources"
    dataset_size_plot_path = base_plot_path / "DatasetSize"
    base_plot_path.mkdir(parents=True, exist_ok=True)
    snr_plot_path.mkdir(parents=True, exist_ok=True)
    snapshots_plot_path.mkdir(parents=True, exist_ok=True)
    steering_noise_plot_path.mkdir(parents=True, exist_ok=True)
    number_of_sources_plot_path.mkdir(parents=True, exist_ok=True)
    dataset_size_plot_path.mkdir(parents=True, exist_ok=True)
    plot_paths = {"SNR": snr_plot_path, "T": snapshots_plot_path, "eta": steering_noise_plot_path, "M": number_of_sources_plot_path, "samples_size": dataset_size_plot_path}


    dt_string_for_save = now.strftime("%d_%m_%Y_%H_%M")
    plt.rcParams.update({'font.size': 18})
    for scenario, dict_values in loss_dict.items():
        plot_path = os.path.join(plot_paths[scenario], dt_string_for_save)
        if field_type == "far":
            plot_test_results(scenario, dict_values, plot_path, tested_param="Angle", save_to_file=save_to_file, plot_acc=plot_acc)
        else:
            plot_test_results(scenario, dict_values, plot_path, save_to_file=save_to_file, plot_acc=plot_acc)
            # plot_test_results(scenario, dict_values, plot_path, tested_param="Angle", save_to_file=save_to_file, plot_acc=False)
            # plot_test_results(scenario, dict_values, plot_path, tested_param="Distance", save_to_file=save_to_file, plot_acc=False)
    return


def plot_test_results(test: str, res: dict, simulations_path: str, tested_param: str="Overall",
                      save_to_file=False, plot_acc: bool=False):
    """
    The input dict is a nested dict - the first level is for the snr values, the second level is for the methods,
    and the third level is for the loss values or accuracy.
    For example: res = {10: {"MUSIC": {"Overall": 0.1, "Accuracy": 0.9}, "RootMUSIC": {"Overall": 0.2, "Accuracy": 0.8}}
    Or, for near filed scenrio: res = {10: {"MUSIC": {"Overall": 0.1, "Angle": 0.2, "Distance": 0.3, "Accuracy": 0.9},
    "RootMUSIC": {"Overall": 0.2, "Angle": 0.3, "Distance": 0.4, "Accuracy": 0.8}}
    The possible test are: "SNR", "T", "eta", "M", "samples_size"
    """
    if tested_param not in ["Overall", "Angle", "Distance"]:
        raise ValueError(f"Unknown tested_param: {tested_param}")
    plot_rmse(test, res, simulations_path, tested_param, save_to_file, plot_acc=plot_acc)


def plot_rmse(test: str, res: dict, simulations_path: str, tested_param: str="Overall",
              save_to_file=False, plot_acc: bool=False):
    if tested_param == "Angle":
        units = "rad"
    else:
        units = "m"

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    test_values = res.keys()
    if test == "eta":
        test_values = np.array(list(res.keys())) * 2
    if test == "SNR":
        warnings.warn("SNR values in the plot are multiplied by 2, due to an error in the signal creation")
        test_values = np.array(list(res.keys())) * 2
    plt_res, plt_acc = parse_loss_results_for_plotting(res, tested_param)
    for method, loss_ in plt_res.items():
        # if loss_.get("Accuracy") is not None and method != "TransMUSIC" and test == "SNR":
        #     label = method + f": {np.mean(loss_['Accuracy']) * 100:.2f} %"
        # else:

        label = method.split("(")[0]
        if not np.isnan((loss_.get(tested_param))).any():
            try:
                ax.plot(test_values, loss_[tested_param], **plot_styles[label], label=label)
            except KeyError:
                print(f"{method} does not have plot style")
                ax.plot(test_values, loss_[tested_param], label=label)
    # decrease the size of the legend
    ax.legend(fontsize='x-small', loc="best")
    ax.grid()
    if test == "SNR":
        ax.set_xlabel("SNR [dB]")
    elif test == "T":
        ax.set_xlabel("T")
    elif test == "eta":
        ax.set_xlabel("$\eta[{\lambda}/{2}]$")
    elif test == "M":
        ax.set_xlabel("Number Of Sources")
    elif test == "samples_size":
        ax.set_xlabel("Dataset Size")
    ax.set_ylabel(f"RMSPE [{units}]")
    # ax.set_title("Overall RMSPE loss")
    if tested_param == "Angle":
        ax.set_yscale("log")
        ax.set_title("Angle RMSPE loss")
    else:
        ax.set_yscale("linear")
    if tested_param == "Distance":
        ax.set_title("Range RMSPE loss")
    ax.set_xticks(list(test_values))
    if test == "samples_size":
        ax.set_xscale("log", base=2)
        ax.set_xticks(list(test_values))
    fig.tight_layout()
    if save_to_file:
        if tested_param in ["Angle", "Distance"]:
            fig.savefig(simulations_path + f"_{tested_param}_loss.pdf", transparent=True, bbox_inches='tight')
        else:
            fig.savefig(simulations_path + "_loss.pdf", transparent=True, bbox_inches='tight')
    fig.show()
    if plt_acc and plot_acc:
        plot_acc_results(test, test_values, plt_res, simulations_path, save_to_file)


def plot_acc_results(test, test_values, plt_res, simulations_path, save_to_file=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    for method, loss_ in plt_res.items():
        if method in ["TransMUSIC", "DeepCNN"]:
            continue
        keys = list(loss_.keys())
        if loss_.get("Accuracy") is not None:
            ax.plot(test_values, loss_["Accuracy"], label=method, **plot_styles[method])
        elif any(key in ["AIC", "MDL", "Threshold"] for key in keys):
            for key in keys:
                if key in ["AIC", "MDL", "Threshold"]:
                    style = method.split("(")[0]
                    ax.plot(test_values, plt_res[method][key]["Accuracy"], label=key, **plot_styles[style])
        else:
            continue
    ax.legend()
    ax.grid()
    if test == "SNR":
        ax.set_xlabel("SNR [dB]")
    elif test == "T":
        ax.set_xlabel("T")
    elif test == "eta":
        ax.set_xlabel("eta")
    elif test == "M":
        ax.set_xlabel("Number Of Sources")
    elif test == "samples_size":
        ax.set_xlabel("Dataset Size")
    ax.set_ylabel("Accuracy [%]")
    # ax.set_title("Accuracy")
    ax.set_yscale("linear")
    if save_to_file:
        fig.savefig(simulations_path + "_acc.pdf", transparent=True, bbox_inches='tight')
    fig.show()
