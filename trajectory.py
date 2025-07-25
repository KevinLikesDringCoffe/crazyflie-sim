#!/usr/bin/env python3
"""
Trajectory Generation Module
Provides extensible trajectory generators for different flight patterns
"""

import numpy as np
from typing import Dict, List, Optional, Union
from enum import Enum
from abc import ABC, abstractmethod

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

class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators"""
    
    def __init__(self, nstates: int = 12):
        self.nstates = nstates
    
    @abstractmethod
    def generate(self, duration: float, control_freq: float, **kwargs) -> np.ndarray:
        """Generate trajectory
        
        Args:
            duration: Trajectory duration in seconds
            control_freq: Control frequency in Hz
            **kwargs: Trajectory-specific parameters
        
        Returns:
            X_ref: Reference trajectory array of shape (nstates, N)
        """
        pass

class HoverTrajectoryGenerator(TrajectoryGenerator):
    """Hover at a fixed position"""
    
    def generate(self, duration: float, control_freq: float, 
                 position: List[float] = [0, 0, 1], 
                 yaw: float = 0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        X_ref[0, :] = position[0]  # x
        X_ref[1, :] = position[1]  # y
        X_ref[2, :] = position[2]  # z
        X_ref[5, :] = yaw          # yaw
        
        return X_ref

class CircleTrajectoryGenerator(TrajectoryGenerator):
    """Circular trajectory in horizontal plane"""
    
    def generate(self, duration: float, control_freq: float,
                 radius: float = 1.0, 
                 center: List[float] = [0, 0, 1],
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 2*np.pi/duration
        
        X_ref[0, :] = center[0] + radius * np.cos(omega * t)
        X_ref[1, :] = center[1] + radius * np.sin(omega * t)
        X_ref[2, :] = center[2]
        
        # Velocities
        X_ref[6, :] = -radius * omega * np.sin(omega * t)
        X_ref[7, :] = radius * omega * np.cos(omega * t)
        X_ref[8, :] = 0
        
        return X_ref

class Figure8TrajectoryGenerator(TrajectoryGenerator):
    """Figure-8 trajectory pattern"""
    
    def generate(self, duration: float, control_freq: float,
                 scale: float = 1.0,
                 center: List[float] = [0, 0, 1],
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 2*np.pi/duration
        
        X_ref[0, :] = center[0] + scale * np.sin(omega * t)
        X_ref[1, :] = center[1] + scale * np.sin(2*omega * t)/2
        X_ref[2, :] = center[2]
        
        # Velocities
        X_ref[6, :] = scale * omega * np.cos(omega * t)
        X_ref[7, :] = scale * omega * np.cos(2*omega * t)
        X_ref[8, :] = 0
        
        return X_ref

class LineTrajectoryGenerator(TrajectoryGenerator):
    """Straight line trajectory between two points"""
    
    def generate(self, duration: float, control_freq: float,
                 start_pos: List[float] = [0, 0, 1],
                 end_pos: List[float] = [2, 0, 1], **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        for i in range(3):
            X_ref[i, :] = np.linspace(start_pos[i], end_pos[i], N)
        
        # Constant velocity
        vel = np.array(end_pos) - np.array(start_pos)
        vel = vel / duration
        X_ref[6:9, :] = vel[:, np.newaxis]
        
        return X_ref

class SpiralTrajectoryGenerator(TrajectoryGenerator):
    """Spiral trajectory with expanding radius"""
    
    def generate(self, duration: float, control_freq: float,
                 radius_max: float = 1.0,
                 center: List[float] = [0, 0, 1],
                 height_gain: float = 0.5,
                 omega: Optional[float] = None, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        if omega is None:
            omega = 4*np.pi/duration
        
        r_t = radius_max * t / duration
        
        X_ref[0, :] = center[0] + r_t * np.cos(omega * t)
        X_ref[1, :] = center[1] + r_t * np.sin(omega * t)
        X_ref[2, :] = center[2] + height_gain * t / duration
        
        return X_ref

class LandingTrajectoryGenerator(TrajectoryGenerator):
    """Landing trajectory from height to ground"""
    
    def generate(self, duration: float, control_freq: float,
                 start_height: float = 2.0,
                 land_pos: List[float] = [0, 0, 0], **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        X_ref[0, :] = land_pos[0]
        X_ref[1, :] = land_pos[1]
        X_ref[2, :] = np.linspace(start_height, land_pos[2], N)
        
        # Descending velocity
        X_ref[8, :] = -(start_height - land_pos[2]) / duration
        
        return X_ref

class StepTrajectoryGenerator(TrajectoryGenerator):
    """Step response trajectory with sudden position changes"""
    
    def generate(self, duration: float, control_freq: float,
                 step_size: float = 1.0,
                 step_duration: float = 2.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
        step_points = int(step_duration * control_freq)
        
        # Initial position
        X_ref[0:3, :] = np.array([[0], [0], [1.0]])
        
        # Add steps every step_duration seconds
        for i in range(1, int(duration / step_duration)):
            start_idx = i * step_points
            if start_idx < N:
                X_ref[0, start_idx:] += step_size * ((-1) ** i)
                X_ref[1, start_idx:] += step_size * 0.5 * ((-1) ** (i+1))
        
        return X_ref

class ZigzagTrajectoryGenerator(TrajectoryGenerator):
    """Zigzag trajectory with sharp direction changes"""
    
    def generate(self, duration: float, control_freq: float,
                 amplitude: float = 1.0,
                 frequency: float = 0.5, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Create zigzag pattern
        X_ref[0, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        X_ref[1, :] = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + np.pi/2))
        X_ref[2, :] = 1.0  # Constant height
        
        return X_ref

class WaypointsTrajectoryGenerator(TrajectoryGenerator):
    """Multi-waypoint trajectory"""
    
    def generate(self, duration: float, control_freq: float,
                 waypoints: List[List[float]] = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], 
                 **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        X_ref = np.zeros((self.nstates, N))
        
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
        
        return X_ref

class AccelerationTrajectoryGenerator(TrajectoryGenerator):
    """High acceleration trajectory for testing control limits"""
    
    def generate(self, duration: float, control_freq: float,
                 max_acceleration: float = 2.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Sinusoidal trajectory with high acceleration
        freq = 2.0  # High frequency for high acceleration
        X_ref[0, :] = (max_acceleration / (2 * np.pi * freq)**2) * np.sin(2 * np.pi * freq * t)
        X_ref[1, :] = (max_acceleration / (2 * np.pi * freq)**2) * np.cos(2 * np.pi * freq * t)
        X_ref[2, :] = 1.0 + 0.5 * np.sin(np.pi * freq * t)
        
        # Add velocities (derivatives)
        X_ref[6, :] = (max_acceleration / (2 * np.pi * freq)) * np.cos(2 * np.pi * freq * t)
        X_ref[7, :] = -(max_acceleration / (2 * np.pi * freq)) * np.sin(2 * np.pi * freq * t)
        X_ref[8, :] = 0.5 * np.pi * freq * np.cos(np.pi * freq * t)
        
        return X_ref

class ConstrainedTrajectoryGenerator(TrajectoryGenerator):
    """Constrained trajectory for obstacle avoidance testing"""
    
    def generate(self, duration: float, control_freq: float,
                 obstacle_center: List[float] = [0.5, 0.5, 1.0],
                 obstacle_radius: float = 0.5, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Create trajectory that goes around obstacle
        center_x, center_y, center_z = obstacle_center
        radius = obstacle_radius + 0.3  # Safety margin
        
        # Circular path around obstacle
        X_ref[0, :] = center_x + radius * np.cos(2 * np.pi * t / duration)
        X_ref[1, :] = center_y + radius * np.sin(2 * np.pi * t / duration)
        X_ref[2, :] = center_z
        
        return X_ref

class RapidZigzagTrajectoryGenerator(TrajectoryGenerator):
    """Rapid zigzag trajectory with controlled maximum velocity"""
    
    def generate(self, duration: float, control_freq: float,
                 amplitude: float = 0.3,
                 segment_duration: float = 2.0,
                 height: float = 1.5,
                 forward_speed: float = 1.0, **kwargs) -> np.ndarray:
        N = int(duration * control_freq)
        t = np.linspace(0, duration, N)
        X_ref = np.zeros((self.nstates, N))
        
        # Number of segments
        n_segments = int(duration / segment_duration)
        
        # Calculate bounds
        x_constraint = 4.0
        y_constraint = 1.0
        
        # Create waypoints
        waypoints_x = []
        waypoints_y = []
        
        mid_segment = n_segments // 2
        
        for i in range(n_segments + 1):
            if i <= mid_segment:
                progress = i / mid_segment if mid_segment > 0 else 0
                x = progress * x_constraint
            else:
                progress = (i - mid_segment) / (n_segments - mid_segment) if (n_segments - mid_segment) > 0 else 0
                x = x_constraint * (1 - progress)
            
            # Create Y oscillations
            base_oscillation = amplitude * ((-1) ** i)
            intensity_factor = 1.5 if abs(i - mid_segment) <= 2 else 1.0
            variation = 0.3 * np.sin(i * 1.3) + 0.2 * np.cos(i * 2.1)
            y = base_oscillation * intensity_factor + variation * amplitude * 0.4
            y = np.clip(y, -y_constraint, y_constraint)
            
            waypoints_x.append(x)
            waypoints_y.append(y)
        
        # Interpolate between waypoints with velocity limiting
        max_allowed_velocity = 8.0
        
        for i in range(len(t)):
            segment_idx = min(int(t[i] / segment_duration), n_segments - 1)
            segment_progress = (t[i] % segment_duration) / segment_duration
            
            smooth_factor = 1.5
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
            
            X_ref[2, i] = height + 0.05 * np.sin(2 * t[i])
            
            # Calculate and limit velocities
            if i > 0:
                dt_local = t[i] - t[i-1]
                vx_raw = (X_ref[0, i] - X_ref[0, i-1]) / dt_local
                vy_raw = (X_ref[1, i] - X_ref[1, i-1]) / dt_local
                vz_raw = (X_ref[2, i] - X_ref[2, i-1]) / dt_local
                
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
        
        return X_ref

class TrajectoryFactory:
    """Factory class for creating trajectory generators"""
    
    _generators = {
        TrajectoryType.HOVER: HoverTrajectoryGenerator,
        TrajectoryType.CIRCLE: CircleTrajectoryGenerator,
        TrajectoryType.FIGURE8: Figure8TrajectoryGenerator,
        TrajectoryType.LINE: LineTrajectoryGenerator,
        TrajectoryType.SPIRAL: SpiralTrajectoryGenerator,
        TrajectoryType.LANDING: LandingTrajectoryGenerator,
        TrajectoryType.STEP: StepTrajectoryGenerator,
        TrajectoryType.ZIGZAG: ZigzagTrajectoryGenerator,
        TrajectoryType.WAYPOINTS: WaypointsTrajectoryGenerator,
        TrajectoryType.ACCELERATION: AccelerationTrajectoryGenerator,
        TrajectoryType.CONSTRAINED: ConstrainedTrajectoryGenerator,
        TrajectoryType.RAPID_ZIGZAG: RapidZigzagTrajectoryGenerator,
    }
    
    @classmethod
    def create_generator(cls, traj_type: TrajectoryType, nstates: int = 12) -> TrajectoryGenerator:
        """Create a trajectory generator instance
        
        Args:
            traj_type: Type of trajectory to generate
            nstates: Number of state variables
        
        Returns:
            TrajectoryGenerator instance
        """
        if traj_type not in cls._generators:
            raise ValueError(f"Unknown trajectory type: {traj_type}")
        
        return cls._generators[traj_type](nstates)
    
    @classmethod
    def generate_trajectory(cls, traj_type: TrajectoryType, 
                           duration: float, 
                           control_freq: float,
                           nstates: int = 12,
                           **kwargs) -> np.ndarray:
        """Generate trajectory directly without creating generator instance
        
        Args:
            traj_type: Type of trajectory to generate
            duration: Trajectory duration in seconds
            control_freq: Control frequency in Hz
            nstates: Number of state variables
            **kwargs: Trajectory-specific parameters
        
        Returns:
            X_ref: Reference trajectory array of shape (nstates, N)
        """
        generator = cls.create_generator(traj_type, nstates)
        return generator.generate(duration, control_freq, **kwargs)
    
    @classmethod
    def register_generator(cls, traj_type: TrajectoryType, 
                          generator_class: type):
        """Register a new trajectory generator
        
        Args:
            traj_type: Trajectory type enum value
            generator_class: Generator class inheriting from TrajectoryGenerator
        """
        cls._generators[traj_type] = generator_class
    
    @classmethod
    def get_available_types(cls) -> List[TrajectoryType]:
        """Get list of available trajectory types"""
        return list(cls._generators.keys())

def create_trajectory(traj_type: Union[str, TrajectoryType], 
                     duration: float, 
                     control_freq: float,
                     nstates: int = 12,
                     **kwargs) -> np.ndarray:
    """Convenience function to create trajectories
    
    Args:
        traj_type: Type of trajectory (string or enum)
        duration: Trajectory duration in seconds
        control_freq: Control frequency in Hz
        nstates: Number of state variables
        **kwargs: Trajectory-specific parameters
    
    Returns:
        X_ref: Reference trajectory array of shape (nstates, N)
    """
    if isinstance(traj_type, str):
        traj_type = TrajectoryType(traj_type)
    
    return TrajectoryFactory.generate_trajectory(
        traj_type, duration, control_freq, nstates, **kwargs
    )