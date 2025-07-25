#!/usr/bin/env python3
"""
TinyMPC Parameter Generator for Crazyflie
Generates system matrices, trajectories, and constraints for TinyMPC controller
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass
import tinympc

class TrajectoryType(Enum):
    HOVER = "hover"
    CIRCLE = "circle"
    FIGURE8 = "figure8"
    LINE = "line"
    SPIRAL = "spiral"
    LANDING = "landing"
    STEP = "step"
    ZIGZAG = "zigzag"
    WAYPOINTS = "waypoints"
    ACCELERATION = "acceleration"
    CONSTRAINED = "constrained"
    RAPID_ZIGZAG = "rapid_zigzag"

class ControlMode(Enum):
    """MPC Control Mode"""
    TRACKING = "tracking"     # Traditional trajectory tracking
    REGULATOR = "regulator"   # Regulator mode - each step tracks back to origin

@dataclass
class CrazyflieParams:
    """Physical parameters of Crazyflie"""
    mass: float = 0.036  # kg
    gravity: float = 9.81  # m/s^2
    arm_length: float = 0.046  # m
    thrust_to_torque: float = 0.005964552
    Ixx: float = 1.43e-5  # kg*m^2
    Iyy: float = 1.43e-5  # kg*m^2
    Izz: float = 2.89e-5  # kg*m^2

@dataclass
class NoiseModel:
    """Realistic noise model for Crazyflie simulation"""
    # Position noise (m) - GPS/optical flow uncertainty
    position_std: float = 0.005  
    
    # Velocity noise (m/s) - Velocity estimation uncertainty
    velocity_std: float = 0.02
    
    # Angle noise (rad) - IMU measurement noise
    angle_std: float = 0.002  # ~0.1 degrees
    
    # Angular velocity noise (rad/s) - Gyroscope noise
    angular_velocity_std: float = 0.01  # ~0.6 deg/s
    
    # Process noise scaling factors
    # These represent unmodeled dynamics and disturbances
    position_process_noise: float = 0.0001
    velocity_process_noise: float = 0.001
    angle_process_noise: float = 0.0001
    angular_velocity_process_noise: float = 0.005
    
    # Actuator noise - thrust variations
    thrust_noise_std: float = 0.02  # 2% of nominal thrust
    
    def get_state_noise_std(self, dt: float) -> np.ndarray:
        """Get noise standard deviations for all state variables, scaled by timestep"""
        # Process noise scales with sqrt(dt) (Brownian motion)
        sqrt_dt = np.sqrt(dt)
        
        # State vector: [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        noise_std = np.array([
            # Position noise (increases with time)
            self.position_process_noise * sqrt_dt,  # x
            self.position_process_noise * sqrt_dt,  # y
            self.position_process_noise * sqrt_dt * 1.5,  # z (more uncertainty in vertical)
            # Angle noise
            self.angle_process_noise * sqrt_dt,     # roll
            self.angle_process_noise * sqrt_dt,     # pitch
            self.angle_process_noise * sqrt_dt * 2, # yaw (less controlled)
            # Velocity noise
            self.velocity_process_noise * sqrt_dt,  # vx
            self.velocity_process_noise * sqrt_dt,  # vy
            self.velocity_process_noise * sqrt_dt * 1.2,  # vz
            # Angular velocity noise
            self.angular_velocity_process_noise * sqrt_dt,  # wx
            self.angular_velocity_process_noise * sqrt_dt,  # wy
            self.angular_velocity_process_noise * sqrt_dt * 1.5,  # wz
        ])
        
        return noise_std
    
    def get_measurement_noise_std(self) -> np.ndarray:
        """Get measurement noise standard deviations"""
        return np.array([
            self.position_std, self.position_std, self.position_std * 1.5,  # position
            self.angle_std, self.angle_std, self.angle_std * 2,  # angles
            self.velocity_std, self.velocity_std, self.velocity_std * 1.2,  # velocity
            self.angular_velocity_std, self.angular_velocity_std, self.angular_velocity_std * 1.5  # angular velocity
        ])
    
    def get_initial_state_noise_std(self) -> np.ndarray:
        """Get initial state uncertainty"""
        return np.array([
            0.05, 0.05, 0.1,  # Initial position uncertainty (m)
            0.05, 0.05, 0.1,  # Initial angle uncertainty (rad) ~3-6 degrees
            0.01, 0.01, 0.02,  # Initial velocity uncertainty (m/s)
            0.02, 0.02, 0.05   # Initial angular velocity uncertainty (rad/s)
        ])

class TinyMPCGenerator:
    """TinyMPC Parameter Generator for Crazyflie"""
    
    def __init__(self, params: Optional[CrazyflieParams] = None, 
                 noise_model: Optional[NoiseModel] = None,
                 random_seed: int = 42):
        self.params = params or CrazyflieParams()
        self.noise_model = noise_model or NoiseModel()
        self.nstates = 12  # [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        self.ninputs = 4   # [u1, u2, u3, u4]
        self.random_seed = random_seed
        self.set_random_seed(random_seed)
        
    def set_random_seed(self, seed: int):
        """Set random seed for reproducibility"""
        np.random.seed(seed)
        self.random_seed = seed
        
    def generate_system_matrices(self, control_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate discrete-time system matrices A and B"""
        dt = 1.0 / control_freq
        g = self.params.gravity
        
        # Continuous-time system matrix
        A_cont = np.zeros((12, 12))
        
        # Position dynamics: x_dot = v + gravity_coupling
        A_cont[0, 6] = 1.0    # dx/dt = vx
        A_cont[1, 7] = 1.0    # dy/dt = vy  
        A_cont[2, 8] = 1.0    # dz/dt = vz
        A_cont[0, 4] = g      # dx/dt += g*phi_y (gravity coupling)
        A_cont[1, 3] = -g     # dy/dt += -g*phi_x (gravity coupling)
        
        # Attitude dynamics: phi_dot = omega
        A_cont[3, 9] = 1.0    # dphi_x/dt = wx
        A_cont[4, 10] = 1.0   # dphi_y/dt = wy
        A_cont[5, 11] = 1.0   # dphi_z/dt = wz
        
        # Velocity dynamics with damping
        drag_coeff = 0.1
        A_cont[6, 6] = -drag_coeff   # dvx/dt = -drag*vx
        A_cont[7, 7] = -drag_coeff   # dvy/dt = -drag*vy
        A_cont[8, 8] = -drag_coeff   # dvz/dt = -drag*vz
        
        ang_damping = 0.5
        A_cont[9, 9] = -ang_damping    # dwx/dt = -damping*wx
        A_cont[10, 10] = -ang_damping  # dwy/dt = -damping*wy
        A_cont[11, 11] = -ang_damping  # dwz/dt = -damping*wz
        
        # Discretize
        A = np.eye(12) + A_cont * dt
        
        # Add gravity as a constant disturbance in the discrete model
        # This represents the effect of gravity on z-velocity between time steps
        gravity_effect = np.zeros(12)
        gravity_effect[8] = -g * dt  # dvz due to gravity over dt
        
        # Store gravity effect for use in simulation
        self._gravity_disturbance = gravity_effect
        
        # Control matrix B
        B = np.zeros((12, 4))
        
        # Thrust affects vertical acceleration
        # For small deviations around hover: dvz/dt = (thrust_total - mg)/m
        # Linearized: dvz/dt = thrust_deviation/m 
        thrust_gain = 1.0 / self.params.mass  # Proper thrust-to-acceleration conversion
        B[8, :] = thrust_gain
        
        # Moments affect angular accelerations
        arm = 0.707 * self.params.arm_length
        
        # Roll moment: tau_x = arm * (u3 + u4 - u1 - u2) / 4
        roll_gain = arm / (4 * self.params.Ixx) * 100
        B[9, 0] = -roll_gain
        B[9, 1] = -roll_gain
        B[9, 2] = roll_gain
        B[9, 3] = roll_gain
        
        # Pitch moment: tau_y = arm * (u1 + u4 - u2 - u3) / 4
        pitch_gain = arm / (4 * self.params.Iyy) * 100
        B[10, 0] = pitch_gain
        B[10, 1] = -pitch_gain
        B[10, 2] = -pitch_gain
        B[10, 3] = pitch_gain
        
        # Yaw moment: tau_z = k * (u1 + u3 - u2 - u4) / 4
        yaw_gain = self.params.thrust_to_torque / (4 * self.params.Izz) * 100
        B[11, 0] = yaw_gain
        B[11, 1] = -yaw_gain
        B[11, 2] = yaw_gain
        B[11, 3] = -yaw_gain
        
        # Discretize
        B = B * dt
        
        return A, B
        
    def generate_cost_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate LQR cost matrices Q and R"""
        # State weights: [x, y, z, phi_x, phi_y, phi_z, vx, vy, vz, wx, wy, wz]
        q_diag = np.array([
            100.0,   # x position
            100.0,   # y position
            400.0,   # z position (higher weight)
            4.0,     # roll angle
            4.0,     # pitch angle
            1111.0,  # yaw angle (very high weight)
            4.0,     # x velocity
            4.0,     # y velocity
            100.0,   # z velocity (higher weight for gravity compensation)
            2.0,     # roll rate
            2.0,     # pitch rate
            25.0     # yaw rate
        ])
        
        Q = np.diag(q_diag)
        R = np.diag([144.0] * 4)  # Control weights
        
        return Q, R
        
    def generate_trajectory(self, traj_type: TrajectoryType, 
                          duration: float, 
                          control_freq: float,
                          **kwargs) -> np.ndarray:
        """Generate reference trajectory"""
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        
        X_ref = np.zeros((12, N))
        
        if traj_type == TrajectoryType.HOVER:
            hover_pos = kwargs.get('position', [0, 0, 1])
            hover_yaw = kwargs.get('yaw', 0)
            
            X_ref[0, :] = hover_pos[0]  # x
            X_ref[1, :] = hover_pos[1]  # y
            X_ref[2, :] = hover_pos[2]  # z
            X_ref[5, :] = hover_yaw     # yaw
            
        elif traj_type == TrajectoryType.CIRCLE:
            radius = kwargs.get('radius', 1.0)
            center = kwargs.get('center', [0, 0, 1])
            omega = kwargs.get('omega', 2*np.pi/duration)
            
            X_ref[0, :] = center[0] + radius * np.cos(omega * t)
            X_ref[1, :] = center[1] + radius * np.sin(omega * t)
            X_ref[2, :] = center[2]
            
            # Velocities
            X_ref[6, :] = -radius * omega * np.sin(omega * t)
            X_ref[7, :] = radius * omega * np.cos(omega * t)
            X_ref[8, :] = 0
            
        elif traj_type == TrajectoryType.FIGURE8:
            scale = kwargs.get('scale', 1.0)
            center = kwargs.get('center', [0, 0, 1])
            omega = kwargs.get('omega', 2*np.pi/duration)
            
            X_ref[0, :] = center[0] + scale * np.sin(omega * t)
            X_ref[1, :] = center[1] + scale * np.sin(2*omega * t)/2
            X_ref[2, :] = center[2]
            
            # Velocities
            X_ref[6, :] = scale * omega * np.cos(omega * t)
            X_ref[7, :] = scale * omega * np.cos(2*omega * t)
            X_ref[8, :] = 0
            
        elif traj_type == TrajectoryType.LINE:
            start_pos = kwargs.get('start_pos', [0, 0, 1])
            end_pos = kwargs.get('end_pos', [2, 0, 1])
            
            for i in range(3):
                X_ref[i, :] = np.linspace(start_pos[i], end_pos[i], N)
                
            # Constant velocity
            vel = np.array(end_pos) - np.array(start_pos)
            vel = vel / duration
            X_ref[6:9, :] = vel[:, np.newaxis]
            
        elif traj_type == TrajectoryType.SPIRAL:
            radius_max = kwargs.get('radius_max', 1.0)
            center = kwargs.get('center', [0, 0, 1])
            height_gain = kwargs.get('height_gain', 0.5)
            omega = kwargs.get('omega', 4*np.pi/duration)
            
            r_t = radius_max * t / duration
            
            X_ref[0, :] = center[0] + r_t * np.cos(omega * t)
            X_ref[1, :] = center[1] + r_t * np.sin(omega * t)
            X_ref[2, :] = center[2] + height_gain * t / duration
            
        elif traj_type == TrajectoryType.LANDING:
            start_height = kwargs.get('start_height', 2.0)
            land_pos = kwargs.get('land_pos', [0, 0, 0])
            
            X_ref[0, :] = land_pos[0]
            X_ref[1, :] = land_pos[1]
            X_ref[2, :] = np.linspace(start_height, land_pos[2], N)
            
            # Descending velocity
            X_ref[8, :] = -(start_height - land_pos[2]) / duration
            
        elif traj_type == TrajectoryType.STEP:
            # Step trajectory - sudden position changes
            step_size = kwargs.get('step_size', 1.0)
            step_duration = kwargs.get('step_duration', 2.0)
            step_points = int(step_duration * control_freq)
            
            # Initial position
            X_ref[0:3, :] = np.array([[0], [0], [1.0]])
            
            # Add steps every step_duration seconds
            for i in range(1, int(duration / step_duration)):
                start_idx = i * step_points
                if start_idx < N:
                    X_ref[0, start_idx:] += step_size * ((-1) ** i)
                    X_ref[1, start_idx:] += step_size * 0.5 * ((-1) ** (i+1))
                    
        elif traj_type == TrajectoryType.ZIGZAG:
            # Zigzag trajectory - sharp direction changes
            amplitude = kwargs.get('amplitude', 1.0)
            frequency = kwargs.get('frequency', 0.5)
            
            # Create zigzag pattern
            X_ref[0, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
            X_ref[1, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + np.pi/2))
            X_ref[2, :] = 1.0  # Constant height
            
        elif traj_type == TrajectoryType.WAYPOINTS:
            # Multi-waypoint trajectory
            waypoints = kwargs.get('waypoints', [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
            waypoints = np.array(waypoints)
            n_waypoints = len(waypoints)
            
            # Create smooth trajectory through waypoints
            for i in range(n_waypoints - 1):
                start_idx = int(i * N / (n_waypoints - 1))
                end_idx = int((i + 1) * N / (n_waypoints - 1))
                
                if end_idx <= N:
                    for j in range(3):  # x, y, z
                        X_ref[j, start_idx:end_idx] = np.linspace(
                            waypoints[i, j], waypoints[i+1, j], 
                            end_idx - start_idx
                        )
                        
        elif traj_type == TrajectoryType.ACCELERATION:
            # High acceleration trajectory
            max_accel = kwargs.get('max_acceleration', 2.0)
            
            # Sinusoidal trajectory with high acceleration
            freq = 2.0  # High frequency for high acceleration
            X_ref[0, :] = (max_accel / (2 * np.pi * freq)**2) * np.sin(2 * np.pi * freq * t)
            X_ref[1, :] = (max_accel / (2 * np.pi * freq)**2) * np.cos(2 * np.pi * freq * t)
            X_ref[2, :] = 1.0 + 0.5 * np.sin(np.pi * freq * t)
            
            # Add velocities (derivatives)
            X_ref[6, :] = (max_accel / (2 * np.pi * freq)) * np.cos(2 * np.pi * freq * t)
            X_ref[7, :] = -(max_accel / (2 * np.pi * freq)) * np.sin(2 * np.pi * freq * t)
            X_ref[8, :] = 0.5 * np.pi * freq * np.cos(np.pi * freq * t)
            
        elif traj_type == TrajectoryType.CONSTRAINED:
            # Obstacle avoidance trajectory
            obstacle_center = kwargs.get('obstacle_center', [0.5, 0.5, 1.0])
            obstacle_radius = kwargs.get('obstacle_radius', 0.5)
            
            # Create trajectory that goes around obstacle
            center_x, center_y, center_z = obstacle_center
            radius = obstacle_radius + 0.3  # Safety margin
            
            # Circular path around obstacle
            X_ref[0, :] = center_x + radius * np.cos(2 * np.pi * t / duration)
            X_ref[1, :] = center_y + radius * np.sin(2 * np.pi * t / duration)
            X_ref[2, :] = center_z
            
        elif traj_type == TrajectoryType.RAPID_ZIGZAG:
            # Rapid zigzag trajectory with sharp turns for testing high-frequency control
            amplitude = kwargs.get('amplitude', 0.3)  # Very small swings for max 10 m/s velocity
            segment_duration = kwargs.get('segment_duration', 2.0)  # Much longer segments for controlled velocity
            height = kwargs.get('height', 1.5)
            forward_speed = kwargs.get('forward_speed', 1.0)
            
            # Number of segments (sharp turns)
            n_segments = int(duration / segment_duration)
            
            # Calculate bounds to stay within constraints
            x_constraint = 4.0  # Stay within [-5, 5] with some margin
            y_constraint = 1.0  # Stay within [-5, 5] with very conservative margin
            
            # Create a single pass trajectory that goes forward then backward
            # Total trajectory has one forward pass and one backward pass
            waypoints_x = []
            waypoints_y = []
            
            # Determine the midpoint for the turnaround
            mid_segment = n_segments // 2
            
            for i in range(n_segments + 1):
                if i <= mid_segment:
                    # Forward pass: 0 to x_constraint
                    progress = i / mid_segment if mid_segment > 0 else 0
                    x = progress * x_constraint
                else:
                    # Backward pass: x_constraint back to 0
                    progress = (i - mid_segment) / (n_segments - mid_segment) if (n_segments - mid_segment) > 0 else 0
                    x = x_constraint * (1 - progress)
                
                # Create dramatic Y oscillations that get more intense at turns
                # Use a combination of alternating pattern and sinusoidal variation
                base_oscillation = amplitude * ((-1) ** i)
                
                # Add intensity near the turnaround point
                intensity_factor = 1.0
                if abs(i - mid_segment) <= 2:  # Near turnaround
                    intensity_factor = 1.5
                
                # Add some variation to make it more chaotic
                variation = 0.3 * np.sin(i * 1.3) + 0.2 * np.cos(i * 2.1)
                
                y = base_oscillation * intensity_factor + variation * amplitude * 0.4
                y = np.clip(y, -y_constraint, y_constraint)  # Ensure Y stays within bounds
                
                waypoints_x.append(x)
                waypoints_y.append(y)
            
            # Interpolate between waypoints with sharp transitions
            for i in range(len(t)):
                segment_idx = min(int(t[i] / segment_duration), n_segments - 1)
                segment_progress = (t[i] % segment_duration) / segment_duration
                
                # Use a very smooth transition function for low velocity
                smooth_factor = 1.5  # Very low value = very smooth transitions for max 10 m/s
                transition = 0.5 * (1 + np.tanh(smooth_factor * (segment_progress - 0.5)))
                
                # Interpolate position
                if segment_idx < len(waypoints_x) - 1:
                    X_ref[0, i] = waypoints_x[segment_idx] + \
                                  (waypoints_x[segment_idx + 1] - waypoints_x[segment_idx]) * transition
                    X_ref[1, i] = waypoints_y[segment_idx] + \
                                  (waypoints_y[segment_idx + 1] - waypoints_y[segment_idx]) * transition
                else:
                    X_ref[0, i] = waypoints_x[-1]
                    X_ref[1, i] = waypoints_y[-1]
                
                # Height with very small oscillations for low velocity
                X_ref[2, i] = height + 0.05 * np.sin(2 * t[i])
                
                # Calculate velocities analytically for controlled speed
                # Limit maximum velocity to 8 m/s (with safety margin under 10 m/s)
                max_allowed_velocity = 8.0
                
                if i > 0:
                    dt_local = t[i] - t[i-1]
                    vx_raw = (X_ref[0, i] - X_ref[0, i-1]) / dt_local
                    vy_raw = (X_ref[1, i] - X_ref[1, i-1]) / dt_local
                    vz_raw = (X_ref[2, i] - X_ref[2, i-1]) / dt_local
                    
                    # Limit velocity magnitude
                    v_magnitude = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
                    if v_magnitude > max_allowed_velocity:
                        scale_factor = max_allowed_velocity / v_magnitude
                        X_ref[6, i] = vx_raw * scale_factor
                        X_ref[7, i] = vy_raw * scale_factor
                        X_ref[8, i] = vz_raw * scale_factor
                    else:
                        X_ref[6, i] = vx_raw
                        X_ref[7, i] = vy_raw
                        X_ref[8, i] = vz_raw
                else:
                    # For first point, use forward difference
                    if len(t) > 1:
                        dt_local = t[i+1] - t[i]
                        vx_raw = (X_ref[0, i+1] - X_ref[0, i]) / dt_local
                        vy_raw = (X_ref[1, i+1] - X_ref[1, i]) / dt_local
                        vz_raw = (X_ref[2, i+1] - X_ref[2, i]) / dt_local
                        
                        # Limit velocity magnitude
                        v_magnitude = np.sqrt(vx_raw**2 + vy_raw**2 + vz_raw**2)
                        if v_magnitude > max_allowed_velocity:
                            scale_factor = max_allowed_velocity / v_magnitude
                            X_ref[6, i] = vx_raw * scale_factor
                            X_ref[7, i] = vy_raw * scale_factor
                            X_ref[8, i] = vz_raw * scale_factor
                        else:
                            X_ref[6, i] = vx_raw
                            X_ref[7, i] = vy_raw
                            X_ref[8, i] = vz_raw
                    else:
                        X_ref[6, i] = 0
                        X_ref[7, i] = 0
                        X_ref[8, i] = 0
            
        return X_ref
        
    def generate_constraints(self) -> Dict:
        """Generate constraint parameters (box constraints only)"""
        constraints = {
            'u_min': np.array([-0.5, -0.5, -0.5, -0.5]),
            'u_max': np.array([0.5, 0.5, 0.5, 0.5]),
            'x_min': np.array([-5, -5, 0, -0.5, -0.5, -np.pi, -3, -3, -3, -2*np.pi, -2*np.pi, -2*np.pi]),
            'x_max': np.array([5, 5, 5, 0.5, 0.5, np.pi, 3, 3, 3, 2*np.pi, 2*np.pi, 2*np.pi])
        }
        return constraints
        
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
        
        return {
            'system': {
                'A': A,
                'B': B,
                'nstates': self.nstates,
                'ninputs': self.ninputs,
                'dt': 1.0 / control_freq,
                'control_freq': control_freq,
                'gravity_disturbance': getattr(self, '_gravity_disturbance', np.zeros(12))
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
            'params': self.params
        }
        
    def print_summary(self, problem: Dict):
        """Print problem summary"""
        print("=" * 60)
        print("TinyMPC Problem Summary")
        print("=" * 60)
        
        # System parameters
        system = problem['system']
        print(f"System Parameters:")
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
            # Convert to power (variance per unit time)
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
        
        plt.suptitle(f'Noise Characteristics (Reference: {control_freq} Hz)', fontsize=14)
        plt.tight_layout()
        plt.show()

class SimpleMPCSimulator:
    """MPC simulator using TinyMPC solver"""
    
    def __init__(self, problem: Dict):
        self.problem = problem
        self.A = problem['system']['A']
        self.B = problem['system']['B']
        self.Q = problem['cost']['Q']
        self.R = problem['cost']['R']
        self.X_ref = problem['trajectory']['X_ref']
        self.constraints = problem['constraints']
        self.dt = problem['system']['dt']
        self.noise_model = problem.get('noise_model', NoiseModel())
        self.horizon = problem['horizon']
        self._gravity_disturbance = problem['system'].get('gravity_disturbance', np.zeros(12))
        self.params = problem.get('params', CrazyflieParams())
        
        # Setup TinyMPC solver
        self.mpc = tinympc.TinyMPC()
        self.mpc.setup(self.A, self.B, self.Q, self.R, self.horizon)
        
        # Set bounds if available
        if 'u_min' in self.constraints and 'u_max' in self.constraints:
            self.mpc.u_min = self.constraints['u_min']
            self.mpc.u_max = self.constraints['u_max']
        
        # Set state bounds if available
        if 'x_min' in self.constraints and 'x_max' in self.constraints:
            self.mpc.x_min = self.constraints['x_min']
            self.mpc.x_max = self.constraints['x_max']
        
    def simulate(self, steps: int = 200, initial_state: Optional[np.ndarray] = None, 
                 verbose: bool = True):
        """Run simulation"""
        if initial_state is None:
            # Use realistic initial state noise
            initial_noise_std = self.noise_model.get_initial_state_noise_std()
            state = self.X_ref[:, 0] + np.random.normal(0, initial_noise_std, 12)
        else:
            state = initial_state.copy()
        
        state[2] = max(state[2], 0.1)  # Keep above ground
        self.x_current = state
        self.x_history = [self.x_current.copy()]
        self.u_history = []
        self.cost_history = []
        
        for step in range(steps):
            # Get reference trajectory for the horizon
            ref_start_idx = min(step, self.X_ref.shape[1] - 1)
            
            # Create reference trajectory for the horizon
            # TinyMPC expects references for N-1 steps (not N)
            X_ref_horizon = np.zeros((self.A.shape[0], self.horizon))
            U_ref_horizon = np.zeros((self.B.shape[1], self.horizon - 1))
            
            for i in range(self.horizon):
                ref_idx = min(ref_start_idx + i, self.X_ref.shape[1] - 1)
                X_ref_horizon[:, i] = self.X_ref[:, ref_idx]
            
            for i in range(self.horizon - 1):
                # Zero reference for control inputs
                U_ref_horizon[:, i] = np.zeros(self.B.shape[1])
            
            # Set current state and references
            self.mpc.set_x0(self.x_current)
            self.mpc.set_x_ref(X_ref_horizon)
            self.mpc.set_u_ref(U_ref_horizon)
            
            # Solve MPC problem
            try:
                solution = self.mpc.solve()
                if solution is not None and 'controls' in solution:
                    # Extract first control input from solution
                    u_control = solution['controls'].flatten()
                else:
                    # Fallback to zero control if solver fails
                    u_control = np.zeros(self.B.shape[1])
                    print(f"Warning: MPC solver failed at step {step}, using zero control")
            except Exception as e:
                print(f"Warning: MPC solver error at step {step}: {e}")
                u_control = np.zeros(self.B.shape[1])
            
            # Compute cost for tracking
            x_error = self.x_current - X_ref_horizon[:, 0]
            cost = x_error.T @ self.Q @ x_error + u_control.T @ self.R @ u_control
            self.cost_history.append(cost)
            
            # Add actuator noise to control inputs
            actuator_noise = np.random.normal(0, self.noise_model.thrust_noise_std, len(u_control))
            u_noisy = u_control * (1 + actuator_noise)
            
            # Simulate forward with realistic process noise
            process_noise_std = self.noise_model.get_state_noise_std(self.dt)
            process_noise = np.random.normal(0, process_noise_std, len(self.x_current))
            # Add gravity disturbance if available
            gravity_disturbance = getattr(self, '_gravity_disturbance', np.zeros(12))
            self.x_current = self.A @ self.x_current + self.B @ u_noisy + process_noise + gravity_disturbance
            
            # Apply state constraints
            self.x_current = np.clip(self.x_current, self.constraints['x_min'], self.constraints['x_max'])
            
            # Record
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_control.copy())
            
        if verbose:
            self._print_results()
        
    def _print_results(self):
        """Print simulation results and generate plots"""
        x_history = np.array(self.x_history)
        u_history = np.array(self.u_history)
        
        # Final position error
        final_error = np.linalg.norm(x_history[-1, :3] - self.X_ref[:3, min(len(x_history)-1, self.X_ref.shape[1]-1)])
        
        # Average cost
        avg_cost = np.mean(self.cost_history)
        
        # Max control
        max_control = np.max(np.abs(u_history))
        
        print("Simulation Results:")
        print(f"  Final position error: {final_error:.4f} m")
        print(f"  Average cost: {avg_cost:.3f}")
        print(f"  Max control input: {max_control:.3f}")
        
        # Check constraint violations
        u_min = self.constraints['u_min']
        u_max = self.constraints['u_max']
        violations = np.sum((u_history < u_min) | (u_history > u_max))
        if violations == 0:
            print("  ✓ No constraint violations")
        else:
            print(f"  ✗ {violations} constraint violations")
        
        # Generate plots
        self._plot_simulation_results(x_history)
        
    def _plot_simulation_results(self, x_history):
        """Plot simulation results with reference trajectory and position error"""
        # Create figure with subplots
        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time arrays
        t_sim = np.arange(len(x_history)) * self.dt
        t_ref = np.arange(min(len(x_history), self.X_ref.shape[1])) * self.dt
        
        # Plot 1: XY trajectory comparison
        ax1.plot(self.X_ref[0, :len(t_ref)], self.X_ref[1, :len(t_ref)], 
                'r--', linewidth=2, label='Reference', alpha=0.8)
        ax1.plot(x_history[:, 0], x_history[:, 1], 
                'b-', linewidth=2, label='Actual')
        ax1.scatter(x_history[0, 0], x_history[0, 1], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        ax1.scatter(x_history[-1, 0], x_history[-1, 1], 
                   c='red', s=100, marker='s', label='End', zorder=5)
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Trajectory Tracking (XY View)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Position error over time
        position_errors = []
        position_error_components = []
        
        for i in range(len(x_history)):
            if i < self.X_ref.shape[1]:
                ref_pos = self.X_ref[:3, i]
            else:
                ref_pos = self.X_ref[:3, -1]  # Use last reference point
            
            actual_pos = x_history[i, :3]
            error = np.linalg.norm(actual_pos - ref_pos)  # 3D distance for plotting
            position_errors.append(error)
            
            # Store component-wise errors for RMSE calculation (consistent with parameter_study.py)
            error_components = actual_pos - ref_pos
            position_error_components.append(error_components)
        
        ax2.plot(t_sim, position_errors, 'b-', linewidth=2, label='Position Error')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Tracking Error vs Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add statistics to the error plot  
        mean_error = np.mean(position_errors)
        max_error = np.max(position_errors)
        
        # Calculate RMSE consistent with parameter_study.py (component-wise)
        position_error_components = np.array(position_error_components)
        rmse_error = np.sqrt(np.mean(position_error_components**2))
        ax2.axhline(y=mean_error, color='orange', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_error:.3f} m')
        ax2.axhline(y=max_error, color='red', linestyle='--', alpha=0.7, 
                   label=f'Max: {max_error:.3f} m')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"  Plot saved as 'simulation_results.png'")
        print(f"  Mean position error: {mean_error:.4f} m")
        print(f"  RMSE position error: {rmse_error:.4f} m")
        print(f"  Max position error: {max_error:.4f} m")

class RegulatorMPCSimulator(SimpleMPCSimulator):
    """MPC simulator in regulator mode - treats each step as regulation to origin"""
    
    def __init__(self, problem: Dict):
        """Initialize regulator MPC simulator"""
        super().__init__(problem)
        self.control_mode = problem.get('control_mode', ControlMode.TRACKING)
        
    def simulate(self, steps: int = 200, initial_state: Optional[np.ndarray] = None, 
                 verbose: bool = True):
        """Run simulation with regulator mode logic"""
        if initial_state is None:
            # Use realistic initial state noise
            initial_noise_std = self.noise_model.get_initial_state_noise_std()
            state = self.X_ref[:, 0] + np.random.normal(0, initial_noise_std, 12)
        else:
            state = initial_state.copy()
        
        state[2] = max(state[2], 0.1)  # Keep above ground
        self.x_current = state
        self.x_history = [self.x_current.copy()]
        self.u_history = []
        self.cost_history = []
        
        for step in range(steps):
            # Get reference trajectory for the horizon based on control mode
            if self.control_mode == ControlMode.REGULATOR:
                # Regulator mode: each step aims to reach the next trajectory point from current position
                X_ref_horizon, U_ref_horizon = self._generate_regulator_references(step)
            else:
                # Traditional tracking mode
                X_ref_horizon, U_ref_horizon = self._generate_tracking_references(step)
            
            # Set current state and references
            self.mpc.set_x0(self.x_current)
            self.mpc.set_x_ref(X_ref_horizon)
            self.mpc.set_u_ref(U_ref_horizon)
            
            # Solve MPC problem
            try:
                solution = self.mpc.solve()
                if solution is not None and 'controls' in solution:
                    # Extract first control input from solution
                    u_control = solution['controls'].flatten()
                else:
                    # Fallback to zero control if solver fails
                    u_control = np.zeros(self.B.shape[1])
                    if verbose and step < 10:  # Only warn for first few steps
                        print(f"Warning: MPC solver failed at step {step}, using zero control")
            except Exception as e:
                if verbose and step < 10:  # Only warn for first few steps
                    print(f"Warning: MPC solver error at step {step}: {e}")
                u_control = np.zeros(self.B.shape[1])
            
            # Compute cost for tracking
            if self.control_mode == ControlMode.REGULATOR:
                # In regulator mode, cost is relative to the target point
                ref_point = self._get_target_reference_point(step)
                x_error = self.x_current - ref_point
            else:
                # Traditional tracking mode
                ref_start_idx = min(step, self.X_ref.shape[1] - 1)
                x_error = self.x_current - self.X_ref[:, ref_start_idx]
            
            cost = x_error.T @ self.Q @ x_error + u_control.T @ self.R @ u_control
            self.cost_history.append(cost)
            
            # Add actuator noise to control inputs
            actuator_noise = np.random.normal(0, self.noise_model.thrust_noise_std, len(u_control))
            u_noisy = u_control * (1 + actuator_noise)
            
            # Simulate forward with realistic process noise
            process_noise_std = self.noise_model.get_state_noise_std(self.dt)
            process_noise = np.random.normal(0, process_noise_std, len(self.x_current))
            # Add gravity disturbance if available
            gravity_disturbance = getattr(self, '_gravity_disturbance', np.zeros(12))
            self.x_current = self.A @ self.x_current + self.B @ u_noisy + process_noise + gravity_disturbance
            
            # Apply state constraints
            self.x_current = np.clip(self.x_current, self.constraints['x_min'], self.constraints['x_max'])
            
            # Record
            self.x_history.append(self.x_current.copy())
            self.u_history.append(u_control.copy())
            
        if verbose:
            self._print_results()
    
    def _generate_regulator_references(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate references for regulator mode - each step tracks back to origin"""
        # Get the target reference point for this step
        ref_start_idx = min(step, self.X_ref.shape[1] - 1)
        
        # Create reference trajectory for the horizon
        X_ref_horizon = np.zeros((self.A.shape[0], self.horizon))
        U_ref_horizon = np.zeros((self.B.shape[1], self.horizon - 1))
        
        # In regulator mode, we create a trajectory from current state to target
        target_state = self.X_ref[:, min(ref_start_idx, self.X_ref.shape[1] - 1)]
        
        # Generate a smooth trajectory from current state to target over the horizon
        for i in range(self.horizon):
            # Linear interpolation from current state to target
            alpha = min(i / (self.horizon - 1), 1.0) if self.horizon > 1 else 1.0
            X_ref_horizon[:, i] = (1 - alpha) * self.x_current + alpha * target_state
        
        # Zero reference for control inputs (regulator characteristic)
        for i in range(self.horizon - 1):
            U_ref_horizon[:, i] = np.zeros(self.B.shape[1])
        
        return X_ref_horizon, U_ref_horizon
    
    def _generate_tracking_references(self, step: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate references for traditional tracking mode"""
        ref_start_idx = min(step, self.X_ref.shape[1] - 1)
        
        # Create reference trajectory for the horizon
        X_ref_horizon = np.zeros((self.A.shape[0], self.horizon))
        U_ref_horizon = np.zeros((self.B.shape[1], self.horizon - 1))
        
        for i in range(self.horizon):
            ref_idx = min(ref_start_idx + i, self.X_ref.shape[1] - 1)
            X_ref_horizon[:, i] = self.X_ref[:, ref_idx]
        
        # Calculate hover thrust needed for each motor to counteract gravity
        hover_thrust_per_motor = self.params.mass * self.params.gravity / 4.0  # mg/4 per motor
        # Convert to control deviation (thrust above/below hover)
        hover_thrust_deviation = 0.0  # In linearized model, hover = equilibrium = 0 deviation
        
        for i in range(self.horizon - 1):
            # Set control reference to hover equilibrium
            U_ref_horizon[:, i] = hover_thrust_deviation
        
        return X_ref_horizon, U_ref_horizon
    
    def _get_target_reference_point(self, step: int) -> np.ndarray:
        """Get the target reference point for current step"""
        ref_idx = min(step, self.X_ref.shape[1] - 1)
        return self.X_ref[:, ref_idx]

def create_simulator(problem: Dict):
    """Factory function to create the appropriate simulator based on control mode"""
    control_mode = problem.get('control_mode', ControlMode.TRACKING)
    
    if control_mode == ControlMode.REGULATOR:
        return RegulatorMPCSimulator(problem)
    else:
        return SimpleMPCSimulator(problem)

def test_reproducibility():
    """Test that results are reproducible with the same seed"""
    print("\nTesting Reproducibility with Random Seeds")
    print("=" * 40)
    
    # Test 1: Same seed should produce same results
    print("\nTest 1: Same seed (42) should produce identical results")
    print("Parameters: freq=50Hz, horizon=10, duration=5s")
    
    # First run with seed 42
    generator1 = TinyMPCGenerator(random_seed=42)
    problem1 = generator1.generate_problem(
        control_freq=50.0,
        horizon=10,
        traj_type=TrajectoryType.HOVER,
        traj_duration=5.0,
        position=[0, 0, 1.0]
    )
    simulator1 = SimpleMPCSimulator(problem1)
    simulator1.simulate(steps=50, verbose=False)
    history1 = np.array(simulator1.x_history)
    
    # Second run with same seed 42
    generator2 = TinyMPCGenerator(random_seed=42)
    problem2 = generator2.generate_problem(
        control_freq=50.0,
        horizon=10,
        traj_type=TrajectoryType.HOVER,
        traj_duration=5.0,
        position=[0, 0, 1.0]
    )
    simulator2 = SimpleMPCSimulator(problem2)
    simulator2.simulate(steps=50, verbose=False)
    history2 = np.array(simulator2.x_history)
    
    # Check if results are identical
    are_identical = np.allclose(history1, history2, rtol=1e-10)
    print(f"Results identical: {are_identical}")
    if are_identical:
        print("✓ Success: Same seed produces identical results")
    else:
        print("✗ Failed: Results differ despite same seed")
        print(f"Max difference: {np.max(np.abs(history1 - history2))}")
    
    # Test 2: Different seeds should produce different results
    print("\nTest 2: Different seed (123) should produce different results")
    
    generator3 = TinyMPCGenerator(random_seed=123)
    problem3 = generator3.generate_problem(
        control_freq=50.0,
        horizon=10,
        traj_type=TrajectoryType.HOVER,
        traj_duration=5.0,
        position=[0, 0, 1.0]
    )
    simulator3 = SimpleMPCSimulator(problem3)
    simulator3.simulate(steps=50, verbose=False)
    history3 = np.array(simulator3.x_history)
    
    # Check if results are different
    are_different = not np.allclose(history1, history3, rtol=1e-10)
    print(f"Results different: {are_different}")
    if are_different:
        print("✓ Success: Different seeds produce different results")
        print(f"Mean difference: {np.mean(np.abs(history1 - history3))}")
    else:
        print("✗ Failed: Results are identical despite different seeds")
    
    print("\nReproducibility test complete!")
    return are_identical and are_different

def main():
    """Main function - demo usage"""
    print("TinyMPC Parameter Generator")
    print("=" * 30)
    
    # Option to test reproducibility
    try:
        test_repro = input("Test reproducibility first? (y/n): ").lower() == 'y'
        if test_repro:
            test_reproducibility()
            if input("\nContinue with trajectory tests? (y/n): ").lower() != 'y':
                return
    except (EOFError, KeyboardInterrupt):
        pass
    
    generator = TinyMPCGenerator()
    
    # Option to visualize noise characteristics
    try:
        visualize_noise = input("\nVisualize noise characteristics? (y/n): ").lower() == 'y'
        if visualize_noise:
            generator.visualize_noise_characteristics()
            if input("\nContinue with trajectory tests? (y/n): ").lower() != 'y':
                return
    except (EOFError, KeyboardInterrupt):
        pass
    
    # Option for custom parameters
    try:
        custom_params = input("\nUse custom parameters? (y/n): ").lower() == 'y'
        if custom_params:
            try:
                freq = float(input("Control frequency (Hz) [default: 50]: ") or 50)
                horizon = int(input("Planning horizon [default: 20]: ") or 20)
                duration = float(input("Trajectory duration (s) [default: 20]: ") or 20)
                
                # Create custom test case
                test_cases = [{
                    'name': 'Custom',
                    'type': TrajectoryType.HOVER,
                    'freq': freq,
                    'horizon': horizon,
                    'duration': duration,
                    'kwargs': {'position': [0, 0, 1.5]}
                }]
                
                # Ask for trajectory type
                print("\nTrajectory types: hover, circle, figure8, line, spiral")
                traj_type = input("Trajectory type [default: hover]: ").lower() or 'hover'
                
                if traj_type == 'circle':
                    test_cases[0]['type'] = TrajectoryType.CIRCLE
                    test_cases[0]['kwargs'] = {
                        'radius': float(input("Circle radius (m) [default: 1.0]: ") or 1.0),
                        'center': [0, 0, 1.5]
                    }
                elif traj_type == 'figure8':
                    test_cases[0]['type'] = TrajectoryType.FIGURE8
                    test_cases[0]['kwargs'] = {
                        'scale': float(input("Figure-8 scale [default: 1.0]: ") or 1.0),
                        'center': [0, 0, 1.5]
                    }
                elif traj_type == 'line':
                    test_cases[0]['type'] = TrajectoryType.LINE
                    test_cases[0]['kwargs'] = {
                        'start': [0, 0, 1.0],
                        'end': [2, 2, 1.5]
                    }
                elif traj_type == 'spiral':
                    test_cases[0]['type'] = TrajectoryType.SPIRAL
                    test_cases[0]['kwargs'] = {
                        'radius': float(input("Spiral radius (m) [default: 1.0]: ") or 1.0),
                        'height': float(input("Spiral height (m) [default: 2.0]: ") or 2.0),
                        'center': [0, 0, 1.0]
                    }
            except ValueError:
                print("Invalid input, using default test cases")
                custom_params = False
    except (EOFError, KeyboardInterrupt):
        custom_params = False
    
    if not custom_params:
        # Default test cases with consistent duration
        fixed_duration = 20.0  # 20 seconds for all trajectories
        
        test_cases = [
            {
                'name': 'Hover',
                'type': TrajectoryType.HOVER,
                'freq': 50.0,
                'horizon': 20,
                'duration': fixed_duration,
                'kwargs': {'position': [0, 0, 1.5]}
            },
            {
                'name': 'Circle',
                'type': TrajectoryType.CIRCLE,
                'freq': 50.0,
                'horizon': 30,
                'duration': fixed_duration,
                'kwargs': {'radius': 1.2, 'center': [0, 0, 1.5]}
            },
            {
                'name': 'Figure8',
                'type': TrajectoryType.FIGURE8,
                'freq': 100.0,
                'horizon': 50,
                'duration': fixed_duration,
                'kwargs': {'scale': 1.0, 'center': [0, 0, 1.5]}
            }
        ]
    
    for test in test_cases:
        print(f"\n{test['name']} Trajectory Test (Horizon: {test['horizon']})")
        print("-" * 50)
        
        # Generate problem
        problem = generator.generate_problem(
            control_freq=test['freq'],
            horizon=test['horizon'],
            traj_type=test['type'],
            traj_duration=test['duration'],
            **test['kwargs']
        )
        
        # Print summary
        generator.print_summary(problem)
        
        # Verify system
        generator.verify_system(problem)
        
        # Plot trajectory
        generator.plot_trajectory(problem['trajectory']['X_ref'], test['name'])
        
        # Run simulation with consistent length
        print("\nRunning simulation...")
        simulator = SimpleMPCSimulator(problem)
        sim_steps = int(fixed_duration * test['freq'])  # trajectory duration
        simulator.simulate(steps=sim_steps)
        
        # Ask to continue
        try:
            if input("\nContinue to next test? (y/n): ").lower() != 'y':
                break
        except (EOFError, KeyboardInterrupt):
            break
    
    print("\nDone!")

if __name__ == "__main__":
    main()