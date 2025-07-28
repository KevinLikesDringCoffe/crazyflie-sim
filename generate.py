#!/usr/bin/env python3
"""
Quick TinyMPC parameter generation script with animated trajectory visualization
Usage: python generate.py [trajectory] [options]

Animation colors match static plots:
  - Reference trajectory: Red dashed line (r--) with red markers (ro)
  - Actual trajectory: Blue solid line (b-) with blue markers (bo)
  - Start point: Green circle, End point: Red square

Examples:
  # Basic parameter generation with static plot
  python generate.py figure8 --frequency 100 --horizon 30
  
  # Show live animation with consistent styling
  python generate.py circle --animate --frequency 50 --horizon 30
  
  # Generate GIF with default sampling (150 frames max)
  python generate.py rapid_zigzag --save-gif --frequency 100
  
  # Generate GIF with custom frame limit and fixed seed
  python generate.py figure8 --save-gif --max-frames 60 --frequency 100 --seed 42
  
  # Generate GIF with custom sampling rate (every 10th frame)
  python generate.py spiral --save-gif --sample-rate 10 --frequency 50
  
  # Use hardware solver for acceleration (if available)
  python generate.py figure8 --solver-type hardware --frequency 200
  
  # Compare software vs hardware solver performance
  python generate.py circle --solver-type software --frequency 100
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Import from refactored modular architecture
from tinympc_generator import TinyMPCGenerator
from trajectory import TrajectoryType
from simulator import ControlMode, SimpleMPCSimulator

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

def generate_and_verify(traj_name: str, freq: float, horizon: int = 50, control_mode: ControlMode = ControlMode.TRACKING, random_seed: int = 42, solver_type: str = "auto"):
    """Generate parameters and verify"""
    
    # Trajectory mapping
    traj_map = {
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
    
    if traj_name not in traj_map:
        print(f"Error: Unknown trajectory '{traj_name}'")
        print(f"Available: {list(traj_map.keys())}")
        return
    
    traj_type, kwargs = traj_map[traj_name]
    
    # Generate problem with consistent duration and seed
    generator = TinyMPCGenerator(random_seed=random_seed)
    
    # Fixed duration for all trajectories regardless of frequency
    fixed_duration = 8.0  # 8 seconds for all trajectories (shorter for faster animation)
    
    print(f"Generating {traj_name} trajectory at {freq} Hz (duration: {fixed_duration}s, horizon: {horizon})...")
    
    problem = generator.generate_problem(
        control_freq=freq,
        horizon=horizon,
        traj_type=traj_type,
        traj_duration=fixed_duration,
        control_mode=control_mode,
        **kwargs
    )
    
    # Print summary
    generator.print_summary(problem)
    
    # Verify system
    generator.verify_system(problem)
    
    # Quick simulation test
    print("\nSimulation Test:")
    print("-" * 20)
    
    simulator = SimpleMPCSimulator(problem, solver_type)
    # Use fixed simulation steps to ensure consistent simulation length
    sim_steps = int(fixed_duration * freq)
    
    # Suppress solver output during simulation
    with SuppressOutput():
        simulator.simulate(steps=sim_steps)
    
    # Extract matrices for user
    A = problem['system']['A']
    B = problem['system']['B']
    Q = problem['cost']['Q']
    R = problem['cost']['R']
    X_ref = problem['trajectory']['X_ref']
    constraints = problem['constraints']
    
    print(f"\nGenerated Parameters:")
    print(f"  A matrix: {A.shape}")
    print(f"  B matrix: {B.shape}")
    print(f"  Q matrix: {Q.shape}")
    print(f"  R matrix: {R.shape}")
    print(f"  Reference trajectory: {X_ref.shape}")
    print(f"  Constraints: u_min={constraints['u_min']}, u_max={constraints['u_max']}")
    
    return problem, simulator

def create_trajectory_animation(problem, simulator, save_gif=False, filename="trajectory_animation.gif", 
                              max_frames=150, sample_rate=None):
    """Create animated visualization of reference vs actual trajectory
    
    Args:
        max_frames: Maximum number of frames for GIF (default: 150)
        sample_rate: Custom sampling rate (if None, auto-calculated)
    """
    
    # Get data
    X_ref = problem['trajectory']['X_ref']
    x_history = np.array(simulator.x_history)
    
    # Align lengths
    min_len = min(len(x_history), X_ref.shape[1])
    
    # Calculate sampling for GIF export
    if sample_rate is None:
        if min_len <= max_frames:
            sample_rate = 1  # Use all frames
        else:
            sample_rate = max(1, min_len // max_frames)  # Sample to get ~max_frames
    
    # Create frame indices for sampling
    if save_gif:
        frame_indices = np.arange(0, min_len, sample_rate)
        animation_frames = len(frame_indices)
        print(f"🎬 Sampling {animation_frames} frames from {min_len} total frames (every {sample_rate} frames)")
    else:
        frame_indices = np.arange(min_len)
        animation_frames = min_len
    
    ref_pos = X_ref[:3, :min_len]
    actual_pos = x_history[:min_len, :3].T
    
    # Setup figure
    fig = plt.figure(figsize=(14, 6))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D Trajectory Comparison (Reference vs Actual)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # 2D XY trajectory plot
    ax2 = fig.add_subplot(122)
    ax2.set_title('XY Plane Trajectory Comparison')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot full reference trajectories (static) - consistent with static plot style
    ax1.plot(ref_pos[0, :], ref_pos[1, :], ref_pos[2, :], 'r--', alpha=0.3, linewidth=1, label='Reference Path')
    ax1.plot(actual_pos[0, :], actual_pos[1, :], actual_pos[2, :], 'b-', alpha=0.3, linewidth=1, label='Actual Path')
    ax2.plot(ref_pos[0, :], ref_pos[1, :], 'r--', alpha=0.3, linewidth=1, label='Reference Path')
    ax2.plot(actual_pos[0, :], actual_pos[1, :], 'b-', alpha=0.3, linewidth=1, label='Actual Path')
    
    # Add start and end markers consistent with static plot
    # Start point (green circle)
    ax1.scatter(actual_pos[0, 0], actual_pos[1, 0], actual_pos[2, 0], 
               c='green', s=100, marker='o', label='Start', zorder=5, alpha=0.7)
    ax2.scatter(actual_pos[0, 0], actual_pos[1, 0], 
               c='green', s=100, marker='o', label='Start', zorder=5, alpha=0.7)
    
    # End point (red square)
    ax1.scatter(actual_pos[0, -1], actual_pos[1, -1], actual_pos[2, -1], 
               c='red', s=100, marker='s', label='End', zorder=5, alpha=0.7)
    ax2.scatter(actual_pos[0, -1], actual_pos[1, -1], 
               c='red', s=100, marker='s', label='End', zorder=5, alpha=0.7)
    
    # Initialize animated elements - consistent with static plot style
    # 3D plot elements
    ref_point_3d, = ax1.plot([], [], [], 'ro', markersize=8, label='Reference Position')
    actual_point_3d, = ax1.plot([], [], [], 'bo', markersize=8, label='Actual Position')
    ref_trail_3d, = ax1.plot([], [], [], 'r-', linewidth=2, alpha=0.8)
    actual_trail_3d, = ax1.plot([], [], [], 'b-', linewidth=2, alpha=0.8)
    
    # 2D plot elements
    ref_point_2d, = ax2.plot([], [], 'ro', markersize=8, label='Reference Position')
    actual_point_2d, = ax2.plot([], [], 'bo', markersize=8, label='Actual Position')
    ref_trail_2d, = ax2.plot([], [], 'r-', linewidth=2, alpha=0.8)
    actual_trail_2d, = ax2.plot([], [], 'b-', linewidth=2, alpha=0.8)
    
    # Error line connecting ref and actual
    error_line_3d, = ax1.plot([], [], [], 'k--', alpha=0.5, linewidth=1)
    error_line_2d, = ax2.plot([], [], 'k--', alpha=0.5, linewidth=1)
    
    # Time text
    time_text = fig.text(0.02, 0.98, '', transform=fig.transFigure, fontsize=12, 
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Set axis limits
    all_pos = np.concatenate([ref_pos, actual_pos], axis=1)
    margin = 0.5
    ax1.set_xlim(np.min(all_pos[0, :]) - margin, np.max(all_pos[0, :]) + margin)
    ax1.set_ylim(np.min(all_pos[1, :]) - margin, np.max(all_pos[1, :]) + margin)
    ax1.set_zlim(np.min(all_pos[2, :]) - margin, np.max(all_pos[2, :]) + margin)
    
    ax2.set_xlim(np.min(all_pos[0, :]) - margin, np.max(all_pos[0, :]) + margin)
    ax2.set_ylim(np.min(all_pos[1, :]) - margin, np.max(all_pos[1, :]) + margin)
    
    # Add legends
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper right')
    
    # Trail length for visualization
    trail_length = min(50, min_len // 4)  # Show recent trail
    
    def animate(anim_frame):
        # Get actual frame index from sampling
        if save_gif and anim_frame < len(frame_indices):
            frame = frame_indices[anim_frame]
        else:
            frame = anim_frame
            
        # Ensure frame is within bounds
        frame = min(frame, min_len - 1)
        
        # Current positions
        ref_x, ref_y, ref_z = ref_pos[0, frame], ref_pos[1, frame], ref_pos[2, frame]
        act_x, act_y, act_z = actual_pos[0, frame], actual_pos[1, frame], actual_pos[2, frame]
        
        # Update current position points
        ref_point_3d.set_data([ref_x], [ref_y])
        ref_point_3d.set_3d_properties([ref_z])
        actual_point_3d.set_data([act_x], [act_y])
        actual_point_3d.set_3d_properties([act_z])
        
        ref_point_2d.set_data([ref_x], [ref_y])
        actual_point_2d.set_data([act_x], [act_y])
        
        # Update trails (adaptive trail length based on sampling)
        trail_length_adjusted = trail_length if not save_gif else max(10, trail_length // sample_rate)
        trail_start = max(0, frame - trail_length_adjusted)
        trail_ref_x = ref_pos[0, trail_start:frame+1]
        trail_ref_y = ref_pos[1, trail_start:frame+1]
        trail_ref_z = ref_pos[2, trail_start:frame+1]
        trail_act_x = actual_pos[0, trail_start:frame+1]
        trail_act_y = actual_pos[1, trail_start:frame+1]
        trail_act_z = actual_pos[2, trail_start:frame+1]
        
        ref_trail_3d.set_data(trail_ref_x, trail_ref_y)
        ref_trail_3d.set_3d_properties(trail_ref_z)
        actual_trail_3d.set_data(trail_act_x, trail_act_y)
        actual_trail_3d.set_3d_properties(trail_act_z)
        
        ref_trail_2d.set_data(trail_ref_x, trail_ref_y)
        actual_trail_2d.set_data(trail_act_x, trail_act_y)
        
        # Update error lines
        error_line_3d.set_data([ref_x, act_x], [ref_y, act_y])
        error_line_3d.set_3d_properties([ref_z, act_z])
        error_line_2d.set_data([ref_x, act_x], [ref_y, act_y])
        
        # Update time and error info
        dt = problem['trajectory'].get('dt', 1.0 / 50.0)  # Default 50Hz if not specified
        time_step = frame * dt
        error_magnitude = np.sqrt((ref_x - act_x)**2 + (ref_y - act_y)**2 + (ref_z - act_z)**2)
        
        if save_gif:
            time_text.set_text(f'Time: {time_step:.2f}s\nError: {error_magnitude:.3f}m\nFrame: {anim_frame+1}/{animation_frames}')
        else:
            time_text.set_text(f'Time: {time_step:.2f}s\nError: {error_magnitude:.3f}m\nFrame: {frame+1}/{min_len}')
        
        return (ref_point_3d, actual_point_3d, ref_trail_3d, actual_trail_3d,
                ref_point_2d, actual_point_2d, ref_trail_2d, actual_trail_2d,
                error_line_3d, error_line_2d, time_text)
    
    if save_gif:
        print(f"\n🎬 Creating animation ({animation_frames} sampled frames for GIF)...")
    else:
        print(f"\n🎬 Creating animation ({min_len} frames)...")
    
    # Create animation
    ani = animation.FuncAnimation(fig, animate, frames=animation_frames, interval=50, blit=False, repeat=True)
    
    plt.tight_layout()
    
    if save_gif:
        print(f"💾 Saving animation as {filename}...")
        ani.save(filename, writer='pillow', fps=20)
        print(f"✅ Animation saved: {filename}")
    
    plt.show()
    
    return ani

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate TinyMPC parameters for different trajectories')
    parser.add_argument('trajectory', 
                       choices=['hover', 'circle', 'figure8', 'line', 'spiral', 'landing', 
                               'step', 'zigzag', 'waypoints', 'acceleration', 'constrained', 'rapid_zigzag'],
                       help='Trajectory type to generate')
    parser.add_argument('--frequency', '-f', type=float, default=50.0,
                       help='Control frequency in Hz (default: 50.0)')
    parser.add_argument('--horizon', '-H', type=int, default=50,
                       help='Planning horizon (default: 50)')
    parser.add_argument('--mode', '-m', choices=['tracking', 'regulator'], default='tracking',
                       help='Control mode: tracking (default) or regulator')
    parser.add_argument('--animate', '-a', action='store_true',
                       help='Show animated trajectory comparison')
    parser.add_argument('--save-gif', action='store_true',
                       help='Save animation as GIF file')
    parser.add_argument('--gif-name', type=str, default='trajectory_animation.gif',
                       help='Name for saved GIF file')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames for GIF export (default: 150)')
    parser.add_argument('--sample-rate', type=int, default=None,
                       help='Custom frame sampling rate (if not specified, auto-calculated)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--solver-type', choices=['auto', 'software', 'hardware'], default='auto',
                       help='MPC solver type: auto (default), software, or hardware')
    
    args = parser.parse_args()
    
    traj_name = args.trajectory.lower()
    freq = args.frequency
    horizon = args.horizon
    control_mode = ControlMode.REGULATOR if args.mode == 'regulator' else ControlMode.TRACKING
    
    problem, simulator = generate_and_verify(traj_name, freq, horizon, control_mode, args.seed, args.solver_type)
    
    if problem and simulator:
        print(f"\nSuccess! Generated {traj_name} trajectory parameters at {freq} Hz with horizon {horizon}")
        print(f"Control mode: {control_mode.value}")
        print(f"Solver type: {args.solver_type}")
        print("Use the returned matrices in your TinyMPC solver.")
        
        # Show animation if requested
        if args.animate:
            ani = create_trajectory_animation(problem, simulator, args.save_gif, args.gif_name, 
                                            args.max_frames, args.sample_rate)
        elif args.save_gif:
            # Save GIF without showing animation
            print("\n🎬 Generating animation GIF...")
            ani = create_trajectory_animation(problem, simulator, True, args.gif_name, 
                                            args.max_frames, args.sample_rate)

if __name__ == "__main__":
    main()