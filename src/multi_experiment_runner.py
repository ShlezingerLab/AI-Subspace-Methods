"""
Multi-Experiment Runner - Handles parameter sweeps and multi-experiment orchestration
"""

import torch
import numpy as np
import time
from pathlib import Path
from typing import Dict, Any, List, Union
from tqdm import tqdm
import copy

from doa_runner import DoARunner
from src.signal_creation import SystemModelParams
import src.utils as utils


class MultiExperimentRunner:
    """
    Orchestrates parameter sweeps across multiple experiments
    """
    
    def __init__(self, sweep_config: Dict[str, Any], base_kwargs: Dict[str, Any]):
        """
        Initialize Multi-Experiment Runner
        
        Args:
            sweep_config: Configuration for the parameter sweep
            base_kwargs: Base configuration (system_model_params, model_config, etc.)
        """
        self.sweep_config = sweep_config
        self.base_kwargs = base_kwargs
        self.sweep_parameter = sweep_config["parameter"]
        self.sweep_values = sweep_config["values"]
        self.fixed_params = sweep_config.get("fixed_params", {})
        self.plot_config = sweep_config.get("plot_config", {})
        
        # Multi-loss configuration
        self.loss_functions = sweep_config.get("loss_functions", None)
        self.multi_loss_mode = self.loss_functions is not None
        
        # Initialize results manager
        self.results_manager = SweepResultsManager(sweep_config)
        
        # Setup paths
        self.base_path = Path(__file__).parent.parent
        self.sweep_results_path = None
        
    def run_sweep(self) -> Dict[str, Any]:
        """
        Execute the complete parameter sweep
        Supports both single loss and multi-loss modes
        
        Returns:
            Aggregated sweep results
        """
        if self.multi_loss_mode:
            return self._run_multi_loss_sweep()
        else:
            return self._run_single_loss_sweep()
    
    def _run_single_loss_sweep(self) -> Dict[str, Any]:
        """Execute parameter sweep with single loss function"""
        print(f"Starting parameter sweep: {self.sweep_parameter}")
        print(f"Values: {self.sweep_values}")
        print(f"Fixed parameters: {self.fixed_params}")
        
        # Setup results directory
        self._setup_results_directory()
        
        # Initialize progress tracking
        sweep_start_time = time.time()
        individual_times = []
        
        # Run experiments for each parameter value
        for i, param_value in enumerate(tqdm(self.sweep_values, desc=f"Sweeping {self.sweep_parameter}")):
            exp_start_time = time.time()
            
            print(f"\nExperiment {i+1}/{len(self.sweep_values)}: {self.sweep_parameter}={param_value}")
            
            # Run single experiment
            try:
                experiment_kwargs = self._prepare_experiment_kwargs(param_value)
                results = self._run_single_experiment(experiment_kwargs)
                
                # Store results
                self.results_manager.add_experiment_result(param_value, results)
                
                exp_duration = time.time() - exp_start_time
                individual_times.append(exp_duration)
                
                print(f"  RMSPE: {results.get('rmspe', 'N/A'):.6f} (Time: {exp_duration:.2f}s)")
                
            except Exception as e:
                print(f"  Error in experiment {param_value}: {e}")
                # Store error result
                self.results_manager.add_experiment_result(param_value, {
                    'error': str(e),
                    'rmspe': float('inf')
                })
                individual_times.append(0)
        
        # Finalize results
        total_time = time.time() - sweep_start_time
        sweep_results = self.results_manager.finalize_results(total_time, individual_times)
        
        # Save results and generate plots
        self._save_sweep_results(sweep_results)
        self._generate_plots(sweep_results)
        
        print(f"\nSweep completed in {total_time:.2f}s")
        print(f"Results saved to: {self.sweep_results_path}")
        
        return sweep_results
    
    def _run_multi_loss_sweep(self) -> Dict[str, Any]:
        """Execute parameter sweep with multiple loss functions"""
        print(f"Starting MULTI-LOSS parameter sweep: {self.sweep_parameter}")
        print(f"Parameter values: {self.sweep_values}")
        print(f"Loss functions: {self.loss_functions}")
        print(f"Fixed parameters: {self.fixed_params}")
        
        # Setup results directory
        self._setup_results_directory()
        
        # Initialize multi-loss results manager
        multi_loss_results = MultiLossResultsManager(self.sweep_config)
        
        total_experiments = len(self.sweep_values) * len(self.loss_functions)
        experiment_count = 0
        
        sweep_start_time = time.time()
        
        # Run experiments for each loss function
        for loss_idx, loss_function in enumerate(self.loss_functions):
            print(f"\n{'='*50}")
            print(f"Running with Loss Function: {loss_function} ({loss_idx+1}/{len(self.loss_functions)})")
            print(f"{'='*50}")
            
            loss_results = {}
            loss_times = []
            
            # Run parameter sweep for this loss function
            for param_idx, param_value in enumerate(self.sweep_values):
                experiment_count += 1
                exp_start_time = time.time()
                
                print(f"Experiment {experiment_count}/{total_experiments}: "
                      f"{self.sweep_parameter}={param_value}, loss={loss_function}")
                
                try:
                    # Prepare experiment with specific loss function
                    experiment_kwargs = self._prepare_experiment_kwargs(param_value, loss_function)
                    results = self._run_single_experiment(experiment_kwargs)
                    
                    # Store results
                    loss_results[param_value] = results
                    
                    exp_duration = time.time() - exp_start_time
                    loss_times.append(exp_duration)
                    
                    print(f"  RMSPE: {results.get('rmspe', 'N/A'):.6f} (Time: {exp_duration:.2f}s)")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    loss_results[param_value] = {'error': str(e), 'rmspe': float('inf')}
                    loss_times.append(0)
            
            # Add results for this loss function
            multi_loss_results.add_loss_function_results(loss_function, loss_results, loss_times)
        
        # Finalize multi-loss results
        total_time = time.time() - sweep_start_time
        final_results = multi_loss_results.finalize_results(total_time)
        
        # Save results and generate plots
        self._save_sweep_results(final_results)
        self._generate_multi_loss_plots(final_results)
        
        print(f"\nMulti-loss sweep completed in {total_time:.2f}s")
        print(f"Results saved to: {self.sweep_results_path}")
        
        return final_results
    
    def _prepare_experiment_kwargs(self, param_value, loss_function=None) -> Dict[str, Any]:
        """
        Prepare kwargs for a single experiment with the specified parameter value
        
        Args:
            param_value: Value for the sweep parameter
            loss_function: Optional loss function override
            
        Returns:
            Modified kwargs for the experiment
        """
        # Deep copy to avoid modifying original
        experiment_kwargs = copy.deepcopy(self.base_kwargs)
        
        # Apply fixed parameters
        for param_name, param_val in self.fixed_params.items():
            if param_name in experiment_kwargs["system_model_params"]:
                experiment_kwargs["system_model_params"][param_name] = param_val
        
        # Apply sweep parameter
        if self.sweep_parameter in experiment_kwargs["system_model_params"]:
            experiment_kwargs["system_model_params"][self.sweep_parameter] = param_value
        
        # Apply loss function if specified
        if loss_function is not None:
            experiment_kwargs["training_params"]["loss_type"] = loss_function
        
        # Force data creation for sweeps (don't load from file)
        experiment_kwargs["simulation_commands"]["create_data"] = True
        experiment_kwargs["simulation_commands"]["load_data"] = False
        experiment_kwargs["simulation_commands"]["plot_results"] = False  # Handle plotting at sweep level
        
        return experiment_kwargs
    
    def _setup_results_directory(self):
        """Setup directory structure for sweep results"""
        from datetime import datetime
        
        # Create timestamp
        dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M")
        
        # Create sweep directory name
        sweep_name = f"{self.sweep_parameter}_sweep"
        if self.fixed_params:
            fixed_str = "_".join([f"{k}{v}" for k, v in self.fixed_params.items()])
            sweep_name += f"_{fixed_str}"
        
        # Add multi-loss indicator
        if self.multi_loss_mode:
            loss_str = "_".join(self.loss_functions)
            sweep_name += f"_losses_{loss_str}"
        
        sweep_name += f"_{dt_string}"
        
        # Setup paths
        self.sweep_results_path = self.base_path / "results" / "parameter_sweeps" / sweep_name
        self.sweep_results_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.sweep_results_path / "plots").mkdir(exist_ok=True)
        (self.sweep_results_path / "individual_experiments").mkdir(exist_ok=True)
    
    def _run_single_experiment(self, experiment_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experiment with the given parameters
        
        Args:
            experiment_kwargs: Experiment configuration
            
        Returns:
            Experiment results
        """
        # Import here to avoid circular imports
        from run_simulation import run_single_simulation
        
        return run_single_simulation(**experiment_kwargs)
    
    def _save_sweep_results(self, sweep_results: Dict[str, Any]):
        """Save sweep results to file"""
        results_file = self.sweep_results_path / "sweep_results.pkl"
        utils.save_pickle(sweep_results, results_file)
        
        # Also save a summary text file
        summary_file = self.sweep_results_path / "summary.txt"
        self._save_summary_text(sweep_results, summary_file)
    
    def _save_summary_text(self, sweep_results: Dict[str, Any], summary_file: Path):
        """Save human-readable summary"""
        with open(summary_file, 'w') as f:
            f.write(f"Parameter Sweep Summary\n")
            f.write(f"======================\n\n")
            f.write(f"Sweep Parameter: {sweep_results['sweep_parameter']}\n")
            f.write(f"Sweep Values: {sweep_results['sweep_values']}\n")
            f.write(f"Fixed Parameters: {sweep_results['fixed_params']}\n")
            
            # Handle multi-loss results
            if 'loss_functions' in sweep_results:
                f.write(f"Loss Functions: {sweep_results['loss_functions']}\n\n")
                
                # Write results for each loss function
                for loss_func in sweep_results['loss_functions']:
                    f.write(f"Results for {loss_func}:\n")
                    f.write(f"{''.join(['-'] * (len(loss_func) + 12))}\n")
                    
                    loss_metrics = sweep_results['aggregated_metrics'][loss_func]
                    for param_val, rmspe in zip(sweep_results['sweep_values'], 
                                               loss_metrics['rmspe_values']):
                        f.write(f"{sweep_results['sweep_parameter']}={param_val}: RMSPE={rmspe:.6f}\n")
                    
                    f.write(f"Mean RMSPE: {loss_metrics['mean_rmspe']:.6f}\n")
                    f.write(f"Std RMSPE: {loss_metrics['std_rmspe']:.6f}\n\n")
            else:
                # Single loss function
                f.write(f"\nResults:\n")
                f.write(f"--------\n")
                for param_val, rmspe in zip(sweep_results['sweep_values'], 
                                           sweep_results['aggregated_metrics']['rmspe_values']):
                    f.write(f"{sweep_results['sweep_parameter']}={param_val}: RMSPE={rmspe:.6f}\n")
                
                f.write(f"\nSummary Statistics:\n")
                f.write(f"Mean RMSPE: {sweep_results['aggregated_metrics']['mean_rmspe']:.6f}\n")
                f.write(f"Std RMSPE: {sweep_results['aggregated_metrics']['std_rmspe']:.6f}\n")
            
            f.write(f"Total Time: {sweep_results['aggregated_metrics']['total_execution_time']:.2f}s\n")
    
    def _generate_plots(self, sweep_results: Dict[str, Any]):
        """Generate plots for single loss function sweep results"""
        plotter = SweepPlotter(sweep_results, self.sweep_results_path / "plots")
        plotter.plot_rmspe_vs_parameter()
    
    def _generate_multi_loss_plots(self, sweep_results: Dict[str, Any]):
        """Generate plots for multi-loss function sweep results"""
        plotter = MultiLossSweepPlotter(sweep_results, self.sweep_results_path / "plots")
        plotter.plot_multi_loss_comparison()


class MultiLossResultsManager:
    """
    Manages aggregation and organization of multi-loss sweep results
    """
    
    def __init__(self, sweep_config: Dict[str, Any]):
        self.sweep_config = sweep_config
        self.loss_functions = sweep_config["loss_functions"]
        self.results_by_loss = {}
        self.times_by_loss = {}
        
    def add_loss_function_results(self, loss_function: str, results: Dict[str, Any], times: List[float]):
        """
        Add results for a specific loss function
        
        Args:
            loss_function: Name of the loss function
            results: Results dictionary for this loss function
            times: List of execution times for each experiment
        """
        self.results_by_loss[loss_function] = results
        self.times_by_loss[loss_function] = times
    
    def finalize_results(self, total_time: float) -> Dict[str, Any]:
        """
        Finalize and aggregate all multi-loss results
        
        Args:
            total_time: Total execution time for all experiments
            
        Returns:
            Complete aggregated multi-loss results dictionary
        """
        # Aggregate metrics for each loss function
        aggregated_metrics = {}
        
        for loss_function in self.loss_functions:
            results = self.results_by_loss[loss_function]
            times = self.times_by_loss[loss_function]
            
            # Extract RMSPE values in order
            rmspe_values = []
            valid_results = []
            
            for param_val in self.sweep_config["values"]:
                if param_val in results:
                    result = results[param_val]
                    if 'error' not in result:
                        rmspe_values.append(result.get('rmspe', float('inf')))
                        valid_results.append(result)
                    else:
                        rmspe_values.append(float('inf'))
            
            # Calculate aggregated metrics for this loss function
            valid_rmspe = [r for r in rmspe_values if r != float('inf')]
            aggregated_metrics[loss_function] = {
                'rmspe_values': rmspe_values,
                'mean_rmspe': np.mean(valid_rmspe) if valid_rmspe else float('inf'),
                'std_rmspe': np.std(valid_rmspe) if valid_rmspe else 0,
                'min_rmspe': np.min(valid_rmspe) if valid_rmspe else float('inf'),
                'max_rmspe': np.max(valid_rmspe) if valid_rmspe else float('inf'),
                'execution_times': times,
                'average_time_per_experiment': np.mean(times) if times else 0,
                'num_successful_experiments': len(valid_results),
                'num_failed_experiments': len(results) - len(valid_results)
            }
        
        # Overall statistics
        aggregated_metrics['total_execution_time'] = total_time
        aggregated_metrics['total_experiments'] = len(self.sweep_config["values"]) * len(self.loss_functions)
        
        # Compile final results
        final_results = {
            'sweep_type': 'multi_loss_parameter_sweep',
            'sweep_parameter': self.sweep_config['parameter'],
            'sweep_values': self.sweep_config['values'],
            'loss_functions': self.loss_functions,
            'fixed_params': self.sweep_config.get('fixed_params', {}),
            'plot_config': self.sweep_config.get('plot_config', {}),
            'results_by_loss': self.results_by_loss,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': time.strftime("%d_%m_%Y_%H_%M")
        }
        
        return final_results


# Keep the original SweepResultsManager for single loss function sweeps
class SweepResultsManager:
    """
    Manages aggregation and organization of sweep results
    """
    
    def __init__(self, sweep_config: Dict[str, Any]):
        self.sweep_config = sweep_config
        self.results = {}
        
    def add_experiment_result(self, param_value: Union[int, float], result: Dict[str, Any]):
        """
        Add result from a single experiment
        
        Args:
            param_value: Parameter value for this experiment
            result: Results dictionary from the experiment
        """
        self.results[param_value] = result
    
    def finalize_results(self, total_time: float, individual_times: List[float]) -> Dict[str, Any]:
        """
        Finalize and aggregate all results
        
        Args:
            total_time: Total execution time for the sweep
            individual_times: List of individual experiment times
            
        Returns:
            Complete aggregated results dictionary
        """
        # Extract RMSPE values in order
        rmspe_values = []
        valid_results = []
        
        for param_val in self.sweep_config["values"]:
            if param_val in self.results:
                result = self.results[param_val]
                if 'error' not in result:
                    rmspe_values.append(result.get('rmspe', float('inf')))
                    valid_results.append(result)
                else:
                    rmspe_values.append(float('inf'))
        
        # Calculate aggregated metrics
        valid_rmspe = [r for r in rmspe_values if r != float('inf')]
        aggregated_metrics = {
            'rmspe_values': rmspe_values,
            'mean_rmspe': np.mean(valid_rmspe) if valid_rmspe else float('inf'),
            'std_rmspe': np.std(valid_rmspe) if valid_rmspe else 0,
            'min_rmspe': np.min(valid_rmspe) if valid_rmspe else float('inf'),
            'max_rmspe': np.max(valid_rmspe) if valid_rmspe else float('inf'),
            'total_execution_time': total_time,
            'individual_times': individual_times,
            'average_time_per_experiment': np.mean(individual_times) if individual_times else 0,
            'num_successful_experiments': len(valid_results),
            'num_failed_experiments': len(self.results) - len(valid_results)
        }
        
        # Compile final results
        final_results = {
            'sweep_type': self.sweep_config.get('sweep_type', 'parameter_sweep'),
            'sweep_parameter': self.sweep_config['parameter'],
            'sweep_values': self.sweep_config['values'],
            'fixed_params': self.sweep_config.get('fixed_params', {}),
            'plot_config': self.sweep_config.get('plot_config', {}),
            'results': self.results,
            'aggregated_metrics': aggregated_metrics,
            'timestamp': time.strftime("%d_%m_%Y_%H_%M")
        }
        
        return final_results


class MultiLossSweepPlotter:
    """
    Handles plotting of multi-loss sweep results
    """
    
    def __init__(self, sweep_results: Dict[str, Any], plots_path: Path):
        self.sweep_results = sweep_results
        self.plots_path = plots_path
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
    def plot_multi_loss_comparison(self):
        """Plot comparison of all loss functions on the same graph"""
        import matplotlib.pyplot as plt
        
        # Extract data
        param_values = self.sweep_results['sweep_values']
        loss_functions = self.sweep_results['loss_functions']
        param_name = self.sweep_results['sweep_parameter']
        plot_config = self.sweep_results.get('plot_config', {})
        
        plt.figure(figsize=(12, 8))
        
        # Plot each loss function
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        markers = ['o', 's', '^', 'D', 'v', '<']
        
        for i, loss_function in enumerate(loss_functions):
            rmspe_values = self.sweep_results['aggregated_metrics'][loss_function]['rmspe_values']
            
            # Filter out infinite values for plotting
            valid_indices = [j for j, r in enumerate(rmspe_values) if r != float('inf')]
            valid_param_values = [param_values[j] for j in valid_indices]
            valid_rmspe_values = [rmspe_values[j] for j in valid_indices]
            
            if valid_param_values:
                color = colors[i % len(colors)]
                marker = markers[i % len(markers)]
                
                plt.plot(valid_param_values, valid_rmspe_values, 
                        color=color, marker=marker, linewidth=2, markersize=6,
                        label=f'{loss_function} (mean: {self.sweep_results["aggregated_metrics"][loss_function]["mean_rmspe"]:.4f}°)')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel(plot_config.get('x_label', param_name))
        plt.ylabel(plot_config.get('y_label', 'RMSPE (degrees)'))
        plt.title(plot_config.get('title', f'RMSPE vs {param_name} - Loss Function Comparison'))
        plt.legend()
        
        # Save plot
        save_name = f'multi_loss_comparison_{param_name}'
        plt.tight_layout()
        plt.savefig(self.plots_path / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_path / f'{save_name}.pdf', bbox_inches='tight')
        plt.show()
        
        print(f"Multi-loss comparison plot saved to: {self.plots_path / f'{save_name}.png'}")
    
    def plot_individual_loss_functions(self):
        """Plot individual graphs for each loss function"""
        for loss_function in self.sweep_results['loss_functions']:
            # Create individual plot for this loss function
            single_loss_results = {
                'sweep_parameter': self.sweep_results['sweep_parameter'],
                'sweep_values': self.sweep_results['sweep_values'],
                'plot_config': self.sweep_results['plot_config'],
                'aggregated_metrics': {
                    'rmspe_values': self.sweep_results['aggregated_metrics'][loss_function]['rmspe_values'],
                    'mean_rmspe': self.sweep_results['aggregated_metrics'][loss_function]['mean_rmspe'],
                    'std_rmspe': self.sweep_results['aggregated_metrics'][loss_function]['std_rmspe']
                }
            }
            
            # Use the single loss plotter
            individual_plotter = SweepPlotter(single_loss_results, self.plots_path)
            individual_plotter.plot_rmspe_vs_parameter(suffix=f'_{loss_function}')


class SweepPlotter:
    """
    Handles plotting of sweep results
    """
    
    def __init__(self, sweep_results: Dict[str, Any], plots_path: Path):
        self.sweep_results = sweep_results
        self.plots_path = plots_path
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
    def plot_rmspe_vs_parameter(self, suffix=''):
        """Plot RMSPE vs the sweep parameter"""
        import matplotlib.pyplot as plt
        
        # Extract data
        param_values = self.sweep_results['sweep_values']
        rmspe_values = self.sweep_results['aggregated_metrics']['rmspe_values']
        param_name = self.sweep_results['sweep_parameter']
        plot_config = self.sweep_results.get('plot_config', {})
        
        # Filter out infinite values for plotting
        valid_indices = [i for i, r in enumerate(rmspe_values) if r != float('inf')]
        valid_param_values = [param_values[i] for i in valid_indices]
        valid_rmspe_values = [rmspe_values[i] for i in valid_indices]
        
        if not valid_param_values:
            print("No valid results to plot")
            return
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(valid_param_values, valid_rmspe_values, 'b-o', linewidth=2, markersize=6)
        plt.grid(True, alpha=0.3)
        
        # Customize plot
        plt.xlabel(plot_config.get('x_label', param_name))
        plt.ylabel(plot_config.get('y_label', 'RMSPE (degrees)'))
        plt.title(plot_config.get('title', f'RMSPE vs {param_name}'))
        
        # Add statistics text
        mean_rmspe = self.sweep_results['aggregated_metrics']['mean_rmspe']
        std_rmspe = self.sweep_results['aggregated_metrics']['std_rmspe']
        plt.text(0.02, 0.98, f'Mean RMSPE: {mean_rmspe:.4f}°\nStd RMSPE: {std_rmspe:.4f}°', 
                 transform=plt.gca().transAxes, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot
        save_name = plot_config.get('save_name', f'rmspe_vs_{param_name}') + suffix
        plt.tight_layout()
        plt.savefig(self.plots_path / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.plots_path / f'{save_name}.pdf', bbox_inches='tight')
        
        # Show plot
        plt.show()
        
        print(f"Plot saved to: {self.plots_path / f'{save_name}.png'}")