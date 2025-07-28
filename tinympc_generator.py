#!/usr/bin/env python3
"""
TinyMPC Parameter Generator for Crazyflie - Refactored Modular Version
Generates system matrices, trajectories, and constraints for TinyMPC controller
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from enum import Enum

# Import new modular components
from dynamics import (
    DynamicsModel, LinearizedQuadcopterDynamics, ScalableQuadcopterDynamics,
    CrazyflieParams, QuadcopterParams, NoiseModel, create_dynamics_model
)
from trajectory import (
    TrajectoryType, TrajectoryFactory, create_trajectory
)
from simulator import (
    ControlMode, MPCSimulator, TinyMPCSimulator, create_simulator,
    SimpleMPCSimulator, RegulatorMPCSimulator  # Legacy compatibility
)

class TinyMPCGenerator:
    """TinyMPC Parameter Generator - Refactored with modular architecture"""
    
    def __init__(self, platform: str = "crazyflie",
                 scale_factor: float = 1.0,
                 custom_params: Optional[QuadcopterParams] = None,
                 noise_model: Optional[NoiseModel] = None,
                 random_seed: int = 42):
        """Initialize TinyMPC Generator with modular architecture
        
        Args:
            platform: Platform type ("crazyflie", "scaled_crazyflie", "custom")
            scale_factor: Scaling factor for platforms
            custom_params: Custom parameters for "custom" platform
            noise_model: Custom noise model
            random_seed: Random seed for reproducibility
        """
        self.platform = platform
        self.scale_factor = scale_factor
        self.random_seed = random_seed
        self.set_random_seed(random_seed)
        
        # Create dynamics model using factory
        self.dynamics_model = create_dynamics_model(
            platform=platform,
            scale_factor=scale_factor,
            custom_params=custom_params,
            noise_model=noise_model
        )
        
        # Extract properties for backward compatibility
        self.params = self.dynamics_model.params
        self.noise_model = self.dynamics_model.noise_model
        self.nstates = self.dynamics_model.nstates
        self.ninputs = self.dynamics_model.ninputs
        
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        self.random_seed = seed
    
    def generate_system_matrices(self, control_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate discrete-time system matrices A and B"""
        return self.dynamics_model.generate_system_matrices(control_freq)
    
    def generate_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LQR cost matrices Q and R"""
        return self.dynamics_model.generate_cost_matrices()
    
    def generate_constraints(self) -> Dict:
        """Generate constraint parameters"""
        return self.dynamics_model.generate_constraints()
    
    def generate_trajectory(self, traj_type: TrajectoryType, 
                          duration: float, 
                          control_freq: float,
                          **kwargs) -> np.ndarray:
        """Generate reference trajectory using modular trajectory generators"""
        return create_trajectory(
            traj_type=traj_type,
            duration=duration,
            control_freq=control_freq,
            nstates=self.nstates,
            **kwargs
        )
    
    def generate_problem(self, 
                        control_freq: float,
                        horizon: int,
                        traj_type: TrajectoryType,
                        traj_duration: float,
                        control_mode: ControlMode = ControlMode.TRACKING,
                        **traj_kwargs) -> Dict:
        """Generate complete MPC problem"""
        # Generate system matrices
        A, B = self.generate_system_matrices(control_freq)
        Q, R = self.generate_cost_matrices()
        
        # Generate trajectory
        X_ref = self.generate_trajectory(traj_type, traj_duration, control_freq, **traj_kwargs)
        
        # Generate constraints
        constraints = self.generate_constraints()
        
        # Create problem dictionary for backward compatibility
        problem = {
            'system': {
                'A': A,
                'B': B,
                'nstates': self.nstates,
                'ninputs': self.ninputs,
                'dt': 1.0 / control_freq,
                'control_freq': control_freq,
                'gravity_disturbance': self.dynamics_model.gravity_disturbance
            },
            'cost': {
                'Q': Q,
                'R': R
            },
            'trajectory': {
                'X_ref': X_ref,
                'type': traj_type.value,
                'duration': traj_duration
            },
            'constraints': constraints,
            'horizon': horizon,
            'noise_model': self.noise_model,
            'control_mode': control_mode,
            'params': self.params,
            'dynamics_model': self.dynamics_model  # Add dynamics model for new API
        }
        
        return problem
    
    def create_simulator(self, problem: Dict, 
                        solver_type: str = "tinympc",
                        mpc_solver_type: str = "auto") -> MPCSimulator:
        """Create simulator instance using modular architecture"""
        dynamics_model = problem.get('dynamics_model', self.dynamics_model)
        X_ref = problem['trajectory']['X_ref']
        horizon = problem['horizon']
        control_mode = problem.get('control_mode', ControlMode.TRACKING)
        
        simulator = create_simulator(
            dynamics_model=dynamics_model,
            X_ref=X_ref,
            horizon=horizon,
            control_mode=control_mode,
            solver_type=solver_type,
            mpc_solver_type=mpc_solver_type
        )
        
        # Set control frequency
        control_freq = problem['system']['control_freq']
        simulator.set_control_frequency(control_freq)
        
        return simulator
    
    def print_summary(self, problem: Dict):
        """Print problem summary"""
        print("=" * 60)
        print("TinyMPC Problem Summary (Modular Architecture)")
        print("=" * 60)
        
        # System parameters
        system = problem['system']
        print(f"System Parameters:")
        print(f"  Platform: {self.platform}")
        if self.platform == "scaled_crazyflie":
            print(f"  Scale factor: {self.scale_factor}x")
        print(f"  States: {system['nstates']}")
        print(f"  Inputs: {system['ninputs']}")
        print(f"  Sample time: {system['dt']:.4f} s")
        print(f"  Control frequency: {system['control_freq']:.0f} Hz")
        print(f"  Horizon: {problem['horizon']}")
        
        # System matrices
        A, B = system['A'], system['B']
        print(f"\nSystem Matrices:")
        print(f"  A matrix condition number: {np.linalg.cond(A):.2f}")
        print(f"  B matrix norm: {np.linalg.norm(B):.4f}")
        eigenvals = np.linalg.eigvals(A)
        print(f"  A eigenvalue range: [{np.min(np.real(eigenvals)):.3f}, {np.max(np.real(eigenvals)):.3f}]")
        
        # Cost matrices
        Q, R = problem['cost']['Q'], problem['cost']['R']
        print(f"\nCost Matrices:")
        print(f"  Q diagonal range: [{np.min(np.diag(Q)):.1f}, {np.max(np.diag(Q)):.1f}]")
        print(f"  R diagonal: {np.diag(R)[0]:.1f} (all equal)")
        
        # Trajectory info
        traj = problem['trajectory']
        X_ref = traj['X_ref']
        print(f"\nTrajectory:")
        print(f"  Type: {traj['type']}")
        print(f"  Duration: {traj['duration']:.1f} s")
        print(f"  Points: {X_ref.shape[1]}")
        print(f"  Position range: [{np.min(X_ref[:3,:]):.2f}, {np.max(X_ref[:3,:]):.2f}] m")
        print(f"  Max velocity: {np.max(np.abs(X_ref[6:9,:])):.2f} m/s")
        
        # Constraints
        constraints = problem['constraints']
        print(f"\nConstraints:")
        print(f"  Input range: [{constraints['u_min'][0]:.2f}, {constraints['u_max'][0]:.2f}]")
        print(f"  Position range: [{constraints['x_min'][0]:.1f}, {constraints['x_max'][0]:.1f}] m")
        print(f"  Height range: [{constraints['x_min'][2]:.1f}, {constraints['x_max'][2]:.1f}] m")
        
        print("=" * 60)
    
    def verify_system(self, problem: Dict):
        """Verify system properties"""
        print("System Verification:")
        print("-" * 30)
        
        A = problem['system']['A']
        B = problem['system']['B']
        Q = problem['cost']['Q']
        R = problem['cost']['R']
        
        # Stability check
        eigenvals = np.linalg.eigvals(A)
        max_eigenval = np.max(np.abs(eigenvals))
        print(f"Max eigenvalue magnitude: {max_eigenval:.4f}")
        if max_eigenval < 1.0:
            print("✓ System is stable")
        else:
            print("✗ System may be unstable")
            
        # Controllability check
        controllability_matrix = B
        for i in range(1, 12):
            controllability_matrix = np.hstack([controllability_matrix, np.linalg.matrix_power(A, i) @ B])
        rank = np.linalg.matrix_rank(controllability_matrix)
        print(f"Controllability matrix rank: {rank}/12")
        if rank == 12:
            print("✓ System is controllable")
        else:
            print("✗ System is not fully controllable")
            
        # Cost matrix check
        Q_eigenvals = np.linalg.eigvals(Q)
        R_eigenvals = np.linalg.eigvals(R)
        if np.min(Q_eigenvals) >= 0 and np.min(R_eigenvals) > 0:
            print("✓ Cost matrices are positive definite")
        else:
            print("✗ Cost matrices are not positive definite")
            
        print("-" * 30)
    
    def plot_trajectory(self, X_ref: np.ndarray, title: str = "Trajectory"):
        """Plot trajectory"""
        _, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 3D trajectory
        ax = axes[0, 0]
        ax.plot(X_ref[0, :], X_ref[1, :], 'b-', linewidth=2)
        ax.scatter(X_ref[0, 0], X_ref[1, 0], c='g', s=100, label='Start')
        ax.scatter(X_ref[0, -1], X_ref[1, -1], c='r', s=100, label='End')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'{title} - XY View')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        # Position vs time
        ax = axes[0, 1]
        t = np.linspace(0, X_ref.shape[1]/50, X_ref.shape[1])
        ax.plot(t, X_ref[0, :], 'r-', label='X')
        ax.plot(t, X_ref[1, :], 'g-', label='Y')
        ax.plot(t, X_ref[2, :], 'b-', label='Z')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (m)')
        ax.set_title('Position vs Time')
        ax.legend()
        ax.grid(True)
        
        # Velocity vs time
        ax = axes[1, 0]
        ax.plot(t, X_ref[6, :], 'r-', label='Vx')
        ax.plot(t, X_ref[7, :], 'g-', label='Vy')
        ax.plot(t, X_ref[8, :], 'b-', label='Vz')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity vs Time')
        ax.legend()
        ax.grid(True)
        
        # Attitude vs time
        ax = axes[1, 1]
        ax.plot(t, X_ref[3, :], 'r-', label='Roll')
        ax.plot(t, X_ref[4, :], 'g-', label='Pitch')
        ax.plot(t, X_ref[5, :], 'b-', label='Yaw')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Attitude (rad)')
        ax.set_title('Attitude vs Time')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_noise_characteristics(self, control_freq: float = 50.0):
        """Visualize noise characteristics for different control frequencies"""
        # Create figure
        _, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Test different control frequencies
        frequencies = [10, 25, 50, 100, 200]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        # Plot 1: Process noise vs control frequency
        ax = axes[0, 0]
        state_names = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz']
        
        for i, freq in enumerate(frequencies):
            dt_test = 1.0 / freq
            noise_std = self.noise_model.get_state_noise_std(dt_test)
            ax.bar(np.arange(12) + i*0.15, noise_std * 1000, width=0.15, 
                   label=f'{freq} Hz', color=colors[i], alpha=0.7)
        
        ax.set_xlabel('State Variable')
        ax.set_ylabel('Process Noise Std (×10³)')
        ax.set_title('Process Noise vs Control Frequency')
        ax.set_xticks(np.arange(12) + 0.3)
        ax.set_xticklabels(state_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Initial state uncertainty
        ax = axes[0, 1]
        initial_noise = self.noise_model.get_initial_state_noise_std()
        ax.bar(range(12), initial_noise, color='darkblue', alpha=0.7)
        ax.set_xlabel('State Variable')
        ax.set_ylabel('Initial Uncertainty')
        ax.set_title('Initial State Uncertainty')
        ax.set_xticks(range(12))
        ax.set_xticklabels(state_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Noise power spectral density
        ax = axes[1, 0]
        freqs = np.array(frequencies)
        position_noise_power = []
        angle_noise_power = []
        
        for freq in freqs:
            dt_test = 1.0 / freq
            noise_std = self.noise_model.get_state_noise_std(dt_test)
            position_noise_power.append(np.mean(noise_std[0:3]**2) / dt_test)
            angle_noise_power.append(np.mean(noise_std[3:6]**2) / dt_test)
        
        ax.loglog(freqs, position_noise_power, 'o-', label='Position noise PSD', linewidth=2)
        ax.loglog(freqs, angle_noise_power, 's-', label='Angle noise PSD', linewidth=2)
        ax.set_xlabel('Control Frequency (Hz)')
        ax.set_ylabel('Noise Power Spectral Density')
        ax.set_title('Noise Power vs Control Frequency')
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        
        # Plot 4: Measurement noise
        ax = axes[1, 1]
        measurement_noise = self.noise_model.get_measurement_noise_std()
        ax.bar(range(12), measurement_noise, color='darkred', alpha=0.7)
        ax.set_xlabel('State Variable')
        ax.set_ylabel('Measurement Noise Std')
        ax.set_title('Sensor Measurement Noise')
        ax.set_xticks(range(12))
        ax.set_xticklabels(state_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Noise Characteristics - {self.platform} (Reference: {control_freq} Hz)', fontsize=14)
        plt.tight_layout()
        plt.show()

def main():
    """Main function - demo usage with modular architecture"""
    print("TinyMPC Parameter Generator - Modular Architecture")
    print("=" * 50)
    
    # Demo different platform types
    platforms = [
        ("crazyflie", 1.0, "Standard Crazyflie"),
        ("scaled_crazyflie", 2.0, "2x Scaled Crazyflie"),
        ("scaled_crazyflie", 0.5, "0.5x Scaled Crazyflie")
    ]
    
    for platform, scale, description in platforms:
        print(f"\n{description} Demo")
        print("-" * 30)
        
        # Create generator with different platforms
        generator = TinyMPCGenerator(
            platform=platform,
            scale_factor=scale,
            random_seed=42
        )
        
        # Generate test problem
        problem = generator.generate_problem(
            control_freq=100.0,
            horizon=30,
            traj_type=TrajectoryType.CIRCLE,
            traj_duration=10.0,
            radius=1.0,
            center=[0, 0, 1.5]
        )
        
        # Print summary
        generator.print_summary(problem)
        
        # Verify system
        generator.verify_system(problem)
        
        # Create and test simulator with software solver
        print("\nTesting Software Simulator:")
        simulator = generator.create_simulator(problem, mpc_solver_type="software")
        simulator.simulate(steps=50, verbose=False)
        
        results = simulator.get_results()
        print(f"  Final position error: {results['final_position_error']:.4f} m")
        print(f"  Mean position error: {results['mean_position_error']:.4f} m")
        
        # Optionally test hardware simulator if available
        print("\nTesting Hardware Simulator (if available):")
        try:
            hw_simulator = generator.create_simulator(problem, mpc_solver_type="hardware")
            hw_simulator.simulate(steps=50, verbose=False)
            
            hw_results = hw_simulator.get_results()
            print(f"  Hardware - Final position error: {hw_results['final_position_error']:.4f} m")
            print(f"  Hardware - Mean position error: {hw_results['mean_position_error']:.4f} m")
        except Exception as e:
            print(f"  Hardware solver not available: {str(e)[:60]}...")
        
        # Ask to continue
        try:
            if input("\nContinue to next platform demo? (y/n): ").lower() != 'y':
                break
        except (EOFError, KeyboardInterrupt):
            break
    
    print("\nModular architecture demo complete!")

if __name__ == "__main__":
    main()