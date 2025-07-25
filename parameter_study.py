#!/usr/bin/env python3
"""
Enhanced Parameter Study for TinyMPC Controller - Rewritten based on generate.py model
Study the effect of planning horizon, control frequency, and noise on trajectory tracking performance
Ensures complete consistency with generate.py implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, Tuple, List
import argparse
import os
from dataclasses import dataclass

# Import from the main tinympc_generator module  
from tinympc_generator import TinyMPCGenerator, TrajectoryType, ControlMode, create_simulator

# Suppress solver output
class SuppressOutput:
    def __init__(self):
        self.null_fd = os.open(os.devnull, os.O_WRONLY)
        self.save_fd = os.dup(1)
        
    def __enter__(self):
        os.dup2(self.null_fd, 1)
        
    def __exit__(self, *_):
        os.dup2(self.save_fd, 1)
        os.close(self.null_fd)
        os.close(self.save_fd)

class ParameterStudy:
    """Enhanced class to perform parameter studies on TinyMPC controller"""
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        
        # Trajectory mapping - consistent with generate.py
        self.traj_map = {
            'hover': (TrajectoryType.HOVER, {'position': [0, 0, 1.5]}),
            'circle': (TrajectoryType.CIRCLE, {'radius': 1.2, 'center': [0, 0, 1.5]}),
            'figure8': (TrajectoryType.FIGURE8, {'scale': 1.0, 'center': [0, 0, 1.5]}),
            'line': (TrajectoryType.LINE, {'start_pos': [0, 0, 1], 'end_pos': [2, 1, 2]}),
            'spiral': (TrajectoryType.SPIRAL, {'radius_max': 1.5, 'center': [0, 0, 1]}),
            'landing': (TrajectoryType.LANDING, {'start_height': 3.0, 'land_pos': [0, 0, 0.1]}),
            'step': (TrajectoryType.STEP, {'step_size': 0.8, 'step_duration': 2.0}),
            'zigzag': (TrajectoryType.ZIGZAG, {'amplitude': 1.0, 'frequency': 0.5}),
            'waypoints': (TrajectoryType.WAYPOINTS, {'waypoints': [[0, 0, 1], [1, 0, 1.5], [1, 1, 1.5], [0, 1, 1]]}),
            'acceleration': (TrajectoryType.ACCELERATION, {'max_acceleration': 2.0}),
            'constrained': (TrajectoryType.CONSTRAINED, {'obstacle_center': [0.5, 0.5, 1.0], 'obstacle_radius': 0.4}),
            'rapid_zigzag': (TrajectoryType.RAPID_ZIGZAG, {'amplitude': 0.3, 'segment_duration': 2.0, 'height': 1.5, 'forward_speed': 1.0})
        }
        
    def study_trajectory_tracking(self, 
                                 traj_name: str, 
                                 horizon_range: Tuple[int, int] = (5, 100),
                                 freq_range: Tuple[int, int] = (10, 1000),
                                 n_horizon_points: int = 10,
                                 n_freq_points: int = 12,
                                 duration: float = 8.0,
                                 control_mode: ControlMode = ControlMode.TRACKING) -> Dict:
        """
        Study the effect of planning horizon and control frequency on tracking error
        Uses the same model as generate.py for complete consistency
        
        Args:
            traj_name: Name of trajectory to study (must be in traj_map)
            horizon_range: Range of planning horizons (min, max)
            freq_range: Range of control frequencies (min, max)
            n_horizon_points: Number of horizon points to test
            n_freq_points: Number of frequency points to test
            duration: Simulation duration in seconds (consistent with generate.py)
            control_mode: Control mode (tracking or regulator)
            
        Returns:
            Dictionary containing study results
        """
        print(f"\nParameter Study for {traj_name.title()} Trajectory")
        print("=" * 60)
        
        # Validate trajectory name
        if traj_name not in self.traj_map:
            raise ValueError(f"Unknown trajectory '{traj_name}'. Available: {list(self.traj_map.keys())}")
            
        traj_type, traj_kwargs = self.traj_map[traj_name]
        
        # Generate parameter ranges
        horizons = np.linspace(horizon_range[0], horizon_range[1], n_horizon_points, dtype=int)
        horizons = np.unique(horizons)  # Remove duplicates
        
        # Use linear spacing for frequencies
        frequencies = np.linspace(freq_range[0], freq_range[1], n_freq_points, dtype=int)
        frequencies = np.unique(frequencies)  # Remove duplicates
        
        # Initialize results
        rmse_matrix = np.zeros((len(horizons), len(frequencies)))
        success_matrix = np.zeros((len(horizons), len(frequencies)), dtype=bool)
        
        # Progress tracking
        total_tests = len(horizons) * len(frequencies)
        current_test = 0
        
        print(f"Using generate.py model with fixed seed: {self.random_seed}")
        print(f"Testing {len(horizons)} horizons × {len(frequencies)} frequencies = {total_tests} combinations")
        print(f"Horizon range: {horizons[0]} to {horizons[-1]}")
        print(f"Frequency range: {frequencies[0]} to {frequencies[-1]} Hz")
        print(f"Duration: {duration}s (consistent with generate.py)")
        print("Progress: ", end="", flush=True)
        
        # Grid search
        for i, horizon in enumerate(horizons):
            for j, freq in enumerate(frequencies):
                current_test += 1
                if current_test % 10 == 0:
                    print(f"{current_test}/{total_tests}...", end="", flush=True)
                
                try:
                    # Use the same generator setup as generate.py
                    generator = TinyMPCGenerator(random_seed=self.random_seed)
                    
                    # Generate problem using the same method as generate.py
                    problem = generator.generate_problem(
                        control_freq=freq,
                        horizon=horizon,
                        traj_type=traj_type,
                        traj_duration=duration,
                        control_mode=control_mode,
                        **traj_kwargs
                    )
                    
                    # Run simulation with suppressed output
                    simulator = create_simulator(problem)
                    sim_steps = int(duration * freq)
                    with SuppressOutput():
                        simulator.simulate(steps=sim_steps, verbose=False)
                    
                    # Calculate RMSE using the same method as generate.py
                    rmse = self._calculate_rmse_consistent_with_generate(simulator, problem)
                    
                    rmse_matrix[i, j] = rmse
                    success_matrix[i, j] = True
                    
                except Exception as e:
                    print(f"\nWarning: Failed at horizon={horizon}, freq={freq}: {e}")
                    rmse_matrix[i, j] = np.nan
                    success_matrix[i, j] = False
        
        print(" Done!")
        
        # Calculate success rate
        success_rate = np.mean(success_matrix) * 100
        print(f"Success rate: {success_rate:.1f}%")
        
        # Return results
        return {
            'horizons': horizons,
            'frequencies': frequencies,
            'rmse_matrix': rmse_matrix,
            'success_matrix': success_matrix,
            'traj_type': traj_type,
            'traj_name': traj_name,
            'traj_kwargs': traj_kwargs,
            'duration': duration,
            'control_mode': control_mode,
            'random_seed': self.random_seed
        }
    
    def _calculate_rmse_consistent_with_generate(self, simulator, problem) -> float:
        """Calculate RMSE using the exact same method as generate.py"""
        x_history = np.array(simulator.x_history)
        X_ref = problem['trajectory']['X_ref']
        
        # Store component-wise errors for RMSE calculation (consistent with generate.py)
        position_error_components = []
        
        for i in range(len(x_history)):
            if i < X_ref.shape[1]:
                ref_pos = X_ref[:3, i]
            else:
                ref_pos = X_ref[:3, -1]  # Use last reference point
            
            actual_pos = x_history[i, :3]
            error_components = actual_pos - ref_pos
            position_error_components.append(error_components)
        
        # Calculate RMSE consistent with generate.py (component-wise)
        position_error_components = np.array(position_error_components)
        rmse_error = np.sqrt(np.mean(position_error_components**2))
        
        return rmse_error
    
    def visualize_results(self, results: Dict, save_figure: bool = False, 
                         figure_name: Optional[str] = None) -> None:
        """
        Create four-subplot visualization of parameter study results
        
        Args:
            results: Results dictionary from study_trajectory_tracking
            save_figure: Whether to save the figure
            figure_name: Name for saved figure
        """
        horizons = results['horizons']
        frequencies = results['frequencies']
        rmse_matrix = results['rmse_matrix']
        traj_name = results['traj_name']
        traj_type = results['traj_type']
        traj_kwargs = results['traj_kwargs']
        duration = results['duration']
        control_mode = results.get('control_mode', ControlMode.TRACKING)
        
        # Create visualization
        _, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Trajectory plot (use middle parameters) - consistent with generate.py style
        ax = axes[0]
        mid_horizon = horizons[len(horizons)//2]
        mid_freq = frequencies[len(frequencies)//2]
        
        # Generate the trajectory for visualization using the same method as generate.py
        generator = TinyMPCGenerator(random_seed=results['random_seed'])
        problem = generator.generate_problem(
            control_freq=mid_freq,
            horizon=mid_horizon,
            traj_type=traj_type,
            traj_duration=duration,
            control_mode=control_mode,
            **traj_kwargs
        )
        X_ref = problem['trajectory']['X_ref']
        
        # Plot using the same style as generate.py static plots
        ax.plot(X_ref[0, :], X_ref[1, :], 'r--', linewidth=2, label='Reference', alpha=0.8)
        ax.scatter(X_ref[0, 0], X_ref[1, 0], c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(X_ref[0, -1], X_ref[1, -1], c='red', s=100, marker='s', label='End', zorder=5)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{traj_name.title()} Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # 2. Heatmap
        ax = axes[1]
        # Mask NaN values for better visualization
        masked_rmse = np.ma.masked_invalid(rmse_matrix)
        im = ax.imshow(masked_rmse, aspect='auto', origin='lower', cmap='viridis')
        
        # Set ticks and labels
        n_freq_ticks = min(8, len(frequencies))
        n_horizon_ticks = min(8, len(horizons))
        
        freq_tick_indices = np.linspace(0, len(frequencies)-1, n_freq_ticks, dtype=int)
        horizon_tick_indices = np.linspace(0, len(horizons)-1, n_horizon_ticks, dtype=int)
        
        ax.set_xticks(freq_tick_indices)
        ax.set_xticklabels([str(frequencies[i]) for i in freq_tick_indices], rotation=45)
        ax.set_yticks(horizon_tick_indices)
        ax.set_yticklabels([str(horizons[i]) for i in horizon_tick_indices])
        
        ax.set_xlabel('Control Frequency (Hz)')
        ax.set_ylabel('Planning Horizon')
        ax.set_title('RMSE Heatmap')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSE (m)')
        
        # 3. RMSE vs Horizon (for different frequencies)
        ax = axes[2]
        freq_indices = [0, len(frequencies)//4, len(frequencies)//2, 3*len(frequencies)//4, -1]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, color in zip(freq_indices, colors):
            if idx == -1:
                idx = len(frequencies) - 1
            # Only plot if we have valid data
            valid_mask = ~np.isnan(rmse_matrix[:, idx])
            if np.any(valid_mask):
                ax.plot(horizons[valid_mask], rmse_matrix[valid_mask, idx], 'o-', 
                        color=color, label=f'{frequencies[idx]} Hz', linewidth=2)
        
        ax.set_xlabel('Planning Horizon')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('RMSE vs Planning Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([horizons[0]-2, horizons[-1]+2])
        
        # 4. RMSE vs Frequency (for different horizons)
        ax = axes[3]
        horizon_indices = [0, len(horizons)//4, len(horizons)//2, 3*len(horizons)//4, -1]
        
        for idx, color in zip(horizon_indices, colors):
            if idx == -1:
                idx = len(horizons) - 1
            # Only plot if we have valid data
            valid_mask = ~np.isnan(rmse_matrix[idx, :])
            if np.any(valid_mask):
                ax.plot(frequencies[valid_mask], rmse_matrix[idx, valid_mask], 'o-', 
                       color=color, label=f'N={horizons[idx]}', linewidth=2)
        
        ax.set_xlabel('Control Frequency (Hz)')
        ax.set_ylabel('RMSE (m)')
        ax.set_title('RMSE vs Control Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim([frequencies[0]*0.8, frequencies[-1]*1.2])
        
        plt.suptitle(f'Parameter Study: {traj_name.title()} Trajectory ({control_mode.value} mode, seed={results["random_seed"]})', fontsize=16)
        plt.tight_layout()
        
        if save_figure:
            if figure_name is None:
                figure_name = f'parameter_study_{traj_name.lower()}.png'
            plt.savefig(figure_name, dpi=300, bbox_inches='tight')
            print(f"Figure saved as: {figure_name}")
        
        plt.show()
    
    def analyze_frequency_effects(self, results: Dict) -> None:
        """Analyze frequency effects and detect positive cases"""
        rmse_matrix = results['rmse_matrix']
        horizons = results['horizons']
        frequencies = results['frequencies']
        traj_name = results['traj_name']
        
        print(f"\n🔍 Frequency Effect Analysis for {traj_name.title()}:")
        print("-" * 50)
        
        positive_cases = []
        
        for i, horizon in enumerate(horizons):
            horizon_rmse = rmse_matrix[i, :]
            valid_mask = ~np.isnan(horizon_rmse)
            
            if np.sum(valid_mask) >= 2:
                valid_freqs = frequencies[valid_mask]
                valid_rmse = horizon_rmse[valid_mask]
                
                # Check if higher frequency helps
                correlation = np.corrcoef(np.log(valid_freqs), valid_rmse)[0, 1]
                improvement = valid_rmse[0] / valid_rmse[-1]  # low freq / high freq
                
                print(f"  Horizon {horizon:2d}: ", end="")
                
                if improvement > 1.2 and correlation < -0.3:
                    positive_cases.append((horizon, improvement, correlation))
                    print(f"✅ Higher freq helps: {improvement:.2f}x (corr: {correlation:.3f})")
                elif improvement < 0.8:
                    print(f"❌ Higher freq hurts: {1/improvement:.2f}x (corr: {correlation:.3f})")
                else:
                    print(f"➖ Mixed/neutral: {improvement:.2f}x (corr: {correlation:.3f})")
        
        if positive_cases:
            print(f"\n🎯 FOUND POSITIVE FREQUENCY EFFECTS!")
            best_case = max(positive_cases, key=lambda x: x[1])
            print(f"   Best case: Horizon {best_case[0]} with {best_case[1]:.2f}x improvement")
            print(f"   Use these parameters for demo: N={best_case[0]}, compare low vs high frequency")
    
    def analyze_horizon_effects(self, results: Dict) -> None:
        """Analyze horizon effects"""
        rmse_matrix = results['rmse_matrix']
        horizons = results['horizons']
        frequencies = results['frequencies']
        traj_name = results['traj_name']
        
        print(f"\n📏 Horizon Effect Analysis for {traj_name.title()}:")
        print("-" * 50)
        
        positive_cases = []
        
        for j, freq in enumerate(frequencies):
            freq_rmse = rmse_matrix[:, j]
            valid_mask = ~np.isnan(freq_rmse)
            
            if np.sum(valid_mask) >= 2:
                valid_horizons = horizons[valid_mask]
                valid_rmse = freq_rmse[valid_mask]
                
                # Check if longer horizon helps
                correlation = np.corrcoef(valid_horizons, valid_rmse)[0, 1]
                improvement = valid_rmse[0] / valid_rmse[-1]  # short / long horizon
                
                print(f"  {freq:3.0f} Hz: ", end="")
                
                if improvement > 1.2 and correlation < -0.3:
                    positive_cases.append((freq, improvement, correlation))
                    print(f"✅ Longer horizon helps: {improvement:.2f}x (corr: {correlation:.3f})")
                elif improvement < 0.8:
                    print(f"❌ Longer horizon hurts: {1/improvement:.2f}x (corr: {correlation:.3f})")
                else:
                    print(f"➖ Mixed/neutral: {improvement:.2f}x (corr: {correlation:.3f})")
        
        if positive_cases:
            print(f"\n🎯 FOUND POSITIVE HORIZON EFFECTS!")
            best_case = max(positive_cases, key=lambda x: x[1])
            print(f"   Best case: {best_case[0]:.0f} Hz with {best_case[1]:.2f}x improvement")
            print(f"   Use these parameters for demo: {best_case[0]:.0f} Hz, compare short vs long horizon")
    
    def study_all_trajectories(self,
                                horizon_range: Tuple[int, int] = (5, 100),
                                freq_range: Tuple[int, int] = (10, 1000),
                                n_horizon_points: int = 10,
                                n_freq_points: int = 12,
                                duration: float = 8.0,
                                control_mode: ControlMode = ControlMode.TRACKING) -> Dict:
        """
        Study all trajectories and compute geometric mean of results
        
        Args:
            Same as study_trajectory_tracking but applies to all trajectories
            
        Returns:
            Dictionary containing results for all trajectories and geometric mean
        """
        print(f"\n🌍 COMPREHENSIVE STUDY: ALL TRAJECTORIES")
        print("=" * 80)
        print(f"Control mode: {control_mode.value}")
        print(f"Random seed: {self.random_seed}")
        print(f"Testing {len(self.traj_map)} different trajectory types")
        
        all_results = {}
        all_rmse_matrices = []
        trajectory_names = []
        
        # Test each trajectory
        for i, traj_name in enumerate(self.traj_map.keys()):
            print(f"\n📊 [{i+1}/{len(self.traj_map)}] Processing {traj_name.upper()} trajectory...")
            
            try:
                # Run parameter study for this trajectory
                results = self.study_trajectory_tracking(
                    traj_name=traj_name,
                    horizon_range=horizon_range,
                    freq_range=freq_range,
                    n_horizon_points=n_horizon_points,
                    n_freq_points=n_freq_points,
                    duration=duration,
                    control_mode=control_mode
                )
                
                all_results[traj_name] = results
                all_rmse_matrices.append(results['rmse_matrix'])
                trajectory_names.append(traj_name)
                
                # Quick summary for this trajectory
                valid_rmse = results['rmse_matrix'][~np.isnan(results['rmse_matrix'])]
                print(f"   ✅ {traj_name.title()}: Mean RMSE = {np.mean(valid_rmse):.4f} m ({len(valid_rmse)} valid tests)")
                
            except Exception as e:
                print(f"   ❌ {traj_name.title()}: Failed - {e}")
                continue
        
        if not all_rmse_matrices:
            print("\n❌ No successful trajectory studies!")
            return {}
        
        # Compute geometric mean across all trajectories
        print(f"\n🧮 COMPUTING GEOMETRIC MEAN ACROSS {len(all_rmse_matrices)} TRAJECTORIES...")
        
        # Stack all RMSE matrices
        all_rmse_stack = np.stack(all_rmse_matrices, axis=0)  # Shape: (n_trajectories, n_horizons, n_freqs)
        
        # Compute geometric mean (handling NaN values)
        # For each (horizon, frequency) combination, compute geometric mean across trajectories
        horizons = all_results[trajectory_names[0]]['horizons']
        frequencies = all_results[trajectory_names[0]]['frequencies']
        
        geometric_mean_matrix = np.full((len(horizons), len(frequencies)), np.nan)
        
        for i in range(len(horizons)):
            for j in range(len(frequencies)):
                # Get RMSE values across all trajectories for this (horizon, freq) combination
                rmse_values = all_rmse_stack[:, i, j]
                valid_values = rmse_values[~np.isnan(rmse_values)]
                
                if len(valid_values) > 0:
                    # Compute geometric mean: (a1 * a2 * ... * an)^(1/n)
                    geometric_mean_matrix[i, j] = np.exp(np.mean(np.log(valid_values)))
        
        # Create comprehensive results
        comprehensive_results = {
            'individual_results': all_results,
            'trajectory_names': trajectory_names,
            'horizons': horizons,
            'frequencies': frequencies,
            'geometric_mean_matrix': geometric_mean_matrix,
            'all_rmse_matrices': all_rmse_stack,
            'control_mode': control_mode,
            'random_seed': self.random_seed,
            'n_trajectories': len(trajectory_names)
        }
        
        return comprehensive_results
    
    def visualize_comprehensive_results(self, comprehensive_results: Dict, 
                                      save_figure: bool = False,
                                      figure_name: Optional[str] = None) -> None:
        """Visualize comprehensive results across all trajectories"""
        
        if not comprehensive_results:
            print("No results to visualize!")
            return
            
        trajectory_names = comprehensive_results['trajectory_names']
        horizons = comprehensive_results['horizons']
        frequencies = comprehensive_results['frequencies']
        geometric_mean_matrix = comprehensive_results['geometric_mean_matrix']
        all_rmse_matrices = comprehensive_results['all_rmse_matrices']
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(24, 12))
        
        # 1. Geometric Mean Heatmap
        ax = axes[0, 0]
        masked_rmse = np.ma.masked_invalid(geometric_mean_matrix)
        im = ax.imshow(masked_rmse, aspect='auto', origin='lower', cmap='viridis')
        
        # Set ticks and labels
        n_freq_ticks = min(8, len(frequencies))
        n_horizon_ticks = min(8, len(horizons))
        
        freq_tick_indices = np.linspace(0, len(frequencies)-1, n_freq_ticks, dtype=int)
        horizon_tick_indices = np.linspace(0, len(horizons)-1, n_horizon_ticks, dtype=int)
        
        ax.set_xticks(freq_tick_indices)
        ax.set_xticklabels([str(frequencies[i]) for i in freq_tick_indices], rotation=45)
        ax.set_yticks(horizon_tick_indices)
        ax.set_yticklabels([str(horizons[i]) for i in horizon_tick_indices])
        
        ax.set_xlabel('Control Frequency (Hz)')
        ax.set_ylabel('Planning Horizon')
        ax.set_title(f'Geometric Mean RMSE\n({len(trajectory_names)} trajectories)')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RMSE (m)')
        
        # 2. Individual trajectory comparison
        ax = axes[0, 1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_names)))
        
        for i, (traj_name, color) in enumerate(zip(trajectory_names, colors)):
            rmse_matrix = all_rmse_matrices[i]
            valid_rmse = rmse_matrix[~np.isnan(rmse_matrix)]
            
            if len(valid_rmse) > 0:
                ax.scatter([i], [np.mean(valid_rmse)], 
                          color=color, s=100, alpha=0.7, label=traj_name.title())
                ax.errorbar([i], [np.mean(valid_rmse)], 
                           yerr=[np.std(valid_rmse)], 
                           color=color, alpha=0.5)
        
        ax.set_xlabel('Trajectory Type')
        ax.set_ylabel('Mean RMSE (m)')
        ax.set_title('Mean RMSE by Trajectory Type')
        ax.set_xticks(range(len(trajectory_names)))
        ax.set_xticklabels([name.title() for name in trajectory_names], rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # 3. Geometric Mean vs Frequency (for different horizons)
        ax = axes[0, 2]
        horizon_indices = [0, len(horizons)//4, len(horizons)//2, 3*len(horizons)//4, -1]
        colors_lines = ['blue', 'green', 'red', 'purple', 'orange']
        
        for idx, color in zip(horizon_indices, colors_lines):
            if idx == -1:
                idx = len(horizons) - 1
            valid_mask = ~np.isnan(geometric_mean_matrix[idx, :])
            if np.any(valid_mask):
                ax.plot(frequencies[valid_mask], geometric_mean_matrix[idx, valid_mask], 'o-', 
                       color=color, label=f'N={horizons[idx]}', linewidth=2)
        
        ax.set_xlabel('Control Frequency (Hz)')
        ax.set_ylabel('Geometric Mean RMSE (m)')
        ax.set_title('Geometric Mean vs Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Geometric Mean vs Horizon (for different frequencies)
        ax = axes[1, 0]
        freq_indices = [0, len(frequencies)//4, len(frequencies)//2, 3*len(frequencies)//4, -1]
        
        for idx, color in zip(freq_indices, colors_lines):
            if idx == -1:
                idx = len(frequencies) - 1
            valid_mask = ~np.isnan(geometric_mean_matrix[:, idx])
            if np.any(valid_mask):
                ax.plot(horizons[valid_mask], geometric_mean_matrix[valid_mask, idx], 'o-', 
                       color=color, label=f'{frequencies[idx]} Hz', linewidth=2)
        
        ax.set_xlabel('Planning Horizon')
        ax.set_ylabel('Geometric Mean RMSE (m)')
        ax.set_title('Geometric Mean vs Horizon')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Trajectory ranking (best parameters for each trajectory)
        ax = axes[1, 1]
        best_rmse_per_traj = []
        traj_labels = []
        
        for traj_name in trajectory_names:
            rmse_matrix = comprehensive_results['individual_results'][traj_name]['rmse_matrix']
            best_rmse = np.nanmin(rmse_matrix)
            best_rmse_per_traj.append(best_rmse)
            traj_labels.append(traj_name.title())
        
        # Sort by performance
        sorted_indices = np.argsort(best_rmse_per_traj)
        sorted_rmse = [best_rmse_per_traj[i] for i in sorted_indices]
        sorted_labels = [traj_labels[i] for i in sorted_indices]
        
        bars = ax.barh(range(len(sorted_labels)), sorted_rmse, 
                      color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_labels))))
        ax.set_yticks(range(len(sorted_labels)))
        ax.set_yticklabels(sorted_labels)
        ax.set_xlabel('Best RMSE (m)')
        ax.set_title('Trajectory Ranking\n(Best Achievable RMSE)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, sorted_rmse)):
            ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', ha='left', va='center', fontsize=9)
        
        # 6. Success rate analysis
        ax = axes[1, 2]
        success_rates = []
        
        for traj_name in trajectory_names:
            success_matrix = comprehensive_results['individual_results'][traj_name]['success_matrix']
            success_rate = np.mean(success_matrix) * 100
            success_rates.append(success_rate)
        
        bars = ax.bar(range(len(trajectory_names)), success_rates, 
                     color=plt.cm.viridis(np.array(success_rates)/100))
        ax.set_xlabel('Trajectory Type')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Simulation Success Rate')
        ax.set_xticks(range(len(trajectory_names)))
        ax.set_xticklabels([name.title() for name in trajectory_names], rotation=45, ha='right')
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, value + 1, 
                   f'{value:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Comprehensive Parameter Study: All Trajectories\n'
                    f'({comprehensive_results["control_mode"].value} mode, '
                    f'seed={comprehensive_results["random_seed"]}, '
                    f'{comprehensive_results["n_trajectories"]} trajectories)', 
                    fontsize=16)
        plt.tight_layout()
        
        if save_figure:
            if figure_name is None:
                figure_name = f'comprehensive_parameter_study_all_trajectories.png'
            plt.savefig(figure_name, dpi=300, bbox_inches='tight')
            print(f"Comprehensive figure saved as: {figure_name}")
        
        plt.show()
    
    def print_comprehensive_summary(self, comprehensive_results: Dict) -> None:
        """Print comprehensive summary across all trajectories"""
        
        if not comprehensive_results:
            print("No results to summarize!")
            return
            
        trajectory_names = comprehensive_results['trajectory_names']
        geometric_mean_matrix = comprehensive_results['geometric_mean_matrix']
        horizons = comprehensive_results['horizons']
        frequencies = comprehensive_results['frequencies']
        
        print(f"\n" + "=" * 80)
        print(f"🌍 COMPREHENSIVE ANALYSIS: ALL TRAJECTORIES")
        print("=" * 80)
        print(f"Model: TinyMPCGenerator (consistent with generate.py)")
        print(f"Control mode: {comprehensive_results['control_mode'].value}")
        print(f"Random seed: {comprehensive_results['random_seed']}")
        print(f"Number of trajectories: {comprehensive_results['n_trajectories']}")
        print(f"Tested trajectories: {', '.join([name.title() for name in trajectory_names])}")
        
        # Geometric mean statistics
        valid_geometric_mean = geometric_mean_matrix[~np.isnan(geometric_mean_matrix)]
        if len(valid_geometric_mean) > 0:
            print(f"\n📊 GEOMETRIC MEAN STATISTICS:")
            print(f"  Valid parameter combinations: {len(valid_geometric_mean)}/{geometric_mean_matrix.size}")
            print(f"  Geometric mean RMSE: {np.mean(valid_geometric_mean):.4f} m")
            print(f"  Std of geometric means: {np.std(valid_geometric_mean):.4f} m")
            print(f"  Best geometric mean: {np.nanmin(geometric_mean_matrix):.4f} m")
            
            best_idx = np.unravel_index(np.nanargmin(geometric_mean_matrix), geometric_mean_matrix.shape)
            print(f"  Best parameters: horizon={horizons[best_idx[0]]}, freq={frequencies[best_idx[1]]} Hz")
            
            print(f"  Worst geometric mean: {np.nanmax(geometric_mean_matrix):.4f} m")
            worst_idx = np.unravel_index(np.nanargmax(geometric_mean_matrix), geometric_mean_matrix.shape)
            print(f"  Worst parameters: horizon={horizons[worst_idx[0]]}, freq={frequencies[worst_idx[1]]} Hz")
            
            performance_range = np.nanmax(geometric_mean_matrix) / np.nanmin(geometric_mean_matrix)
            print(f"  Performance range: {performance_range:.2f}x")
        
        # Individual trajectory performance
        print(f"\n🎯 INDIVIDUAL TRAJECTORY PERFORMANCE:")
        print("-" * 60)
        trajectory_stats = []
        
        for traj_name in trajectory_names:
            results = comprehensive_results['individual_results'][traj_name]
            rmse_matrix = results['rmse_matrix']
            valid_rmse = rmse_matrix[~np.isnan(rmse_matrix)]
            
            if len(valid_rmse) > 0:
                mean_rmse = np.mean(valid_rmse)
                best_rmse = np.nanmin(rmse_matrix)
                worst_rmse = np.nanmax(rmse_matrix)
                success_rate = np.mean(results['success_matrix']) * 100
                
                trajectory_stats.append((traj_name, mean_rmse, best_rmse, worst_rmse, success_rate))
                
                print(f"  {traj_name.title():>12}: "
                      f"Mean={mean_rmse:.4f}m, "
                      f"Best={best_rmse:.4f}m, "
                      f"Range={worst_rmse/best_rmse:.1f}x, "
                      f"Success={success_rate:.1f}%")
        
        # Ranking
        if trajectory_stats:
            print(f"\n🏆 TRAJECTORY RANKING (by best achievable RMSE):")
            print("-" * 60)
            trajectory_stats.sort(key=lambda x: x[2])  # Sort by best RMSE
            
            for i, (traj_name, mean_rmse, best_rmse, worst_rmse, success_rate) in enumerate(trajectory_stats):
                medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1:2d}."
                print(f"  {medal} {traj_name.title():>12}: {best_rmse:.4f}m")
        
        # Overall frequency and horizon effects
        print(f"\n📈 OVERALL PARAMETER EFFECTS (Geometric Mean):")
        print("-" * 60)
        
        # Frequency effect analysis
        freq_improvements = []
        for i in range(len(horizons)):
            horizon_rmse = geometric_mean_matrix[i, :]
            valid_mask = ~np.isnan(horizon_rmse)
            
            if np.sum(valid_mask) >= 2:
                valid_freqs = frequencies[valid_mask]
                valid_rmse = horizon_rmse[valid_mask]
                improvement = valid_rmse[0] / valid_rmse[-1]  # low freq / high freq
                freq_improvements.append(improvement)
        
        if freq_improvements:
            avg_freq_improvement = np.mean(freq_improvements)
            print(f"  Average frequency effect: {avg_freq_improvement:.2f}x improvement from low to high freq")
            
            if avg_freq_improvement > 1.2:
                print(f"  ✅ Higher frequencies are generally beneficial!")
            elif avg_freq_improvement < 0.8:
                print(f"  ❌ Higher frequencies are generally harmful!")
            else:
                print(f"  ➖ Frequency effects are mixed/neutral")
        
        # Horizon effect analysis
        horizon_improvements = []
        for j in range(len(frequencies)):
            freq_rmse = geometric_mean_matrix[:, j]
            valid_mask = ~np.isnan(freq_rmse)
            
            if np.sum(valid_mask) >= 2:
                valid_horizons = horizons[valid_mask]
                valid_rmse = freq_rmse[valid_mask]
                improvement = valid_rmse[0] / valid_rmse[-1]  # short / long horizon
                horizon_improvements.append(improvement)
        
        if horizon_improvements:
            avg_horizon_improvement = np.mean(horizon_improvements)
            print(f"  Average horizon effect: {avg_horizon_improvement:.2f}x improvement from short to long horizon")
            
            if avg_horizon_improvement > 1.2:
                print(f"  ✅ Longer horizons are generally beneficial!")
            elif avg_horizon_improvement < 0.8:
                print(f"  ❌ Longer horizons are generally harmful!")
            else:
                print(f"  ➖ Horizon effects are mixed/neutral")

    def print_summary(self, results: Dict) -> None:
        """Print comprehensive summary statistics of the parameter study"""
        rmse_matrix = results['rmse_matrix']
        horizons = results['horizons']
        frequencies = results['frequencies']
        traj_name = results['traj_name']
        
        # Remove NaN values for statistics
        valid_rmse = rmse_matrix[~np.isnan(rmse_matrix)]
        
        if len(valid_rmse) == 0:
            print("No valid results to summarize!")
            return
        
        print(f"\n{traj_name.title()} Trajectory - Summary Statistics:")
        print("-" * 50)
        print(f"  Model: TinyMPCGenerator (consistent with generate.py)")
        print(f"  Random seed: {results['random_seed']}")
        print(f"  Valid simulations: {len(valid_rmse)}/{rmse_matrix.size}")
        print(f"  Mean RMSE: {np.mean(valid_rmse):.4f} m")
        print(f"  Std RMSE: {np.std(valid_rmse):.4f} m")
        print(f"  Best RMSE: {np.nanmin(rmse_matrix):.4f} m")
        
        best_idx = np.unravel_index(np.nanargmin(rmse_matrix), rmse_matrix.shape)
        print(f"  Best parameters: horizon={horizons[best_idx[0]]}, freq={frequencies[best_idx[1]]} Hz")
        
        print(f"  Worst RMSE: {np.nanmax(rmse_matrix):.4f} m")
        worst_idx = np.unravel_index(np.nanargmax(rmse_matrix), rmse_matrix.shape)
        print(f"  Worst parameters: horizon={horizons[worst_idx[0]]}, freq={frequencies[worst_idx[1]]} Hz")
        
        # Performance range analysis
        best_rmse = np.nanmin(rmse_matrix)
        worst_rmse = np.nanmax(rmse_matrix)
        performance_range = worst_rmse / best_rmse
        print(f"  Performance range: {performance_range:.2f}x (shows parameter sensitivity)")
        
        # Run effect analyses
        self.analyze_frequency_effects(results)
        self.analyze_horizon_effects(results)


def main():
    """Main function for parameter study based on generate.py model"""
    parser = argparse.ArgumentParser(description='Parameter Study using generate.py model for complete consistency')
    
    # Trajectory and range parameters
    parser.add_argument('--trajectory', '-t', 
                       choices=['hover', 'circle', 'figure8', 'line', 'spiral', 'landing', 
                               'step', 'zigzag', 'waypoints', 'acceleration', 'constrained', 'rapid_zigzag', 'all'], 
                       default='hover', help='Trajectory type to study (use "all" for comprehensive study)')
    parser.add_argument('--horizon-min', type=int, default=5, 
                       help='Minimum planning horizon')
    parser.add_argument('--horizon-max', type=int, default=100, 
                       help='Maximum planning horizon')
    parser.add_argument('--freq-min', type=int, default=10, 
                       help='Minimum control frequency (Hz)')
    parser.add_argument('--freq-max', type=int, default=1000, 
                       help='Maximum control frequency (Hz)')
    parser.add_argument('--n-horizon', type=int, default=10, 
                       help='Number of horizon points to test')
    parser.add_argument('--n-freq', type=int, default=12, 
                       help='Number of frequency points to test')
    parser.add_argument('--duration', type=float, default=8.0, 
                       help='Simulation duration (seconds) - consistent with generate.py')
    
    # Output parameters
    parser.add_argument('--save-figure', action='store_true', 
                       help='Save the figure')
    parser.add_argument('--figure-name', type=str, 
                       help='Name for saved figure')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (consistent with generate.py)')
    
    # Control mode
    parser.add_argument('--control-mode', choices=['tracking', 'regulator'], default='tracking',
                       help='Control mode: tracking (default) or regulator')
    
    args = parser.parse_args()
    
    # Create study with fixed seed for consistency
    study = ParameterStudy(random_seed=args.seed)
    
    traj_name = args.trajectory
    control_mode = ControlMode.REGULATOR if args.control_mode == 'regulator' else ControlMode.TRACKING
    
    print(f"🔬 Parameter Study using generate.py model")
    print(f"📊 Trajectory: {traj_name}")
    print(f"🎛️ Control mode: {control_mode.value}")
    print(f"🎲 Random seed: {args.seed}")
    
    if traj_name == 'all':
        # Run comprehensive study on all trajectories
        print("\n🌍 Running comprehensive study on ALL trajectories...")
        results = study.study_all_trajectories(
            horizon_range=(args.horizon_min, args.horizon_max),
            freq_range=(args.freq_min, args.freq_max),
            n_horizon_points=args.n_horizon,
            n_freq_points=args.n_freq,
            duration=args.duration,
            control_mode=control_mode
        )
        
        # Print comprehensive summary
        study.print_comprehensive_summary(results)
        
        # Visualize comprehensive results
        study.visualize_comprehensive_results(results, save_figure=args.save_figure, 
                                            figure_name=args.figure_name)
    else:
        # Run single trajectory study
        results = study.study_trajectory_tracking(
            traj_name=traj_name,
            horizon_range=(args.horizon_min, args.horizon_max),
            freq_range=(args.freq_min, args.freq_max),
            n_horizon_points=args.n_horizon,
            n_freq_points=args.n_freq,
            duration=args.duration,
            control_mode=control_mode
        )
        
        # Print summary
        study.print_summary(results)
        
        # Visualize results
        study.visualize_results(results, save_figure=args.save_figure, 
                               figure_name=args.figure_name)


if __name__ == "__main__":
    main()