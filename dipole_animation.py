import numpy as np
import matplotlib.pyplot as plt
import drawing_ops as ops
from solver import solve_laplace
from utils import *
from manim import *

N = 256 #todo resolve dependency of stream lines on grid size
charges = ops.composition(
    ops.circle(center=(N/2-N/5, N/2), radius=N/16, filled=True, value=1),
    ops.circle(center=(N/2+N/5, N/2), radius=N/16, filled=True, value=-1)
)
Rho = charges(np.zeros((N, N)))

phi = solve_laplace(U=np.zeros((N, N)), Rho=Rho)

def make_function_from_potential(potential, scale_arrows=1.2):
    """
    Create a function that returns the electric field vector at a given position
    based on the potential.
    Args:
        potential (np.ndarray): The potential array.
    Returns:
        func (callable): A function that takes a position (x, y) and returns the electric field vector (Ex, Ey).
        funcnorm (callable): A function that takes a position (x, y) and returns the normalized electric field vector (Ex_norm, Ey_norm).
    """
    Ey, Ex = electric_field_from_potential(potential)
    E = amplitude(Ex, Ey)
    # shift the field to center
    N = len(potential) 
    Ex_centered = np.roll(Ex, N//2, axis=0)
    Ey_centered = np.roll(Ey, N//2, axis=0)
    E = np.roll(E, N//2, axis=0)
    # Normalize the field for visualization
    Ex_norm = scale_arrows * Ex_centered / (E + 1e-8)
    Ey_norm = scale_arrows * Ey_centered / (E + 1e-8)
    
    def func(pos):
        # Define a scaling factor to map positions to array indices
        scale_factor = N / config.frame_width
        return np.array([
            Ex_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            Ey_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            0
        ])
    def funcnorm(pos):
        # Define a scaling factor to map positions to array indices
        scale_factor = N / config.frame_width
        return np.array([
            Ex_centered[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            Ey_centered[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            0
        ])
    # Return the function that computes the electric field vector    
    return func, funcnorm

# Create a function maakes a ArrowVectorField from the potential
def make_arrow_vector_field_E_from_potential(potential, scale_arrows=1.2):
    """
    Create a function that returns a Manim ArrowVectorField
    based on the potential.
    Args:
        potential (np.ndarray): The potential array.
    Returns:
        func (callable): Manim ArrowVectorField
    """
    Ey, Ex = electric_field_from_potential(potential)
    E = amplitude(Ex, Ey)
    # Normalize the field for visualization
    Ex_norm = scale_arrows * Ex / (E + 1e-8)
    Ey_norm = scale_arrows * Ey / (E + 1e-8)
    # shift the field to center
    N = len(potential)
    Ex_centered = np.roll(Ex, (N//2, N//2), axis=(0, 1))
    Ey_centered = np.roll(Ey, (N//2, N//2), axis=(0, 1))
    # E_centered = amplitude(Ex_centered, Ey_centered)

    # Define a scaling factor to map positions to array indices
    scale_factor = N / config.frame_width
    funcnorm = lambda pos: np.array([
            Ex_centered[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            Ey_centered[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            0
        ])
    
    # Create a vector field
    vector_field = ArrowVectorField(
        funcnorm,
        x_range=[-config.frame_width / 2, config.frame_width / 2, 0.8],
        y_range=[-config.frame_height / 2, config.frame_height / 2, 0.8]
    )
    
    return vector_field

# Create a function that returns the streamlines based on the potential
def make_stream_lines_from_potential(potential, scale_speed=1.2, stroke_width=0.7, max_anchors_per_line=100, virtual_time=0.5, color=BLUE,):
    """
    Create a function that returns the streamlines based on the potential.
    Args:
        potential (np.ndarray): The potential array.
    Returns:
        func (callable): A function that takes a position (x, y) and returns the electric field vector (Ex, Ey).
    """
    Ey, Ex = electric_field_from_potential(potential)
    E = amplitude(Ex, Ey)
    # shift the field to center
    N = len(potential)
    Ex_centered = np.roll(Ex, (N//2, N//2), axis=(0, 1))
    Ey_centered = np.roll(Ey, (N//2, N//2), axis=(0, 1))
    E_centered = amplitude(Ex_centered, Ey_centered)
    # Normalize the field for visualization
    Ex_norm = scale_speed * Ex_centered / (E_centered + 1e-8)
    Ey_norm = scale_speed * Ey_centered / (E_centered + 1e-8)
    
    # Define a scaling factor to map positions to array indices
    scale_factor = N / config.frame_width
    func = lambda pos: np.array([
            Ex_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            Ey_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
            0
        ])
    
    # Create streamlines
    stream_lines = StreamLines(func, stroke_width=stroke_width, max_anchors_per_line=max_anchors_per_line, virtual_time=virtual_time, color=color,)
    return stream_lines

class Dipole_animation(Scene):
    
    def construct(self):
        # Create the vector field and streamlines
        vector_field = make_arrow_vector_field_E_from_potential(phi, scale_arrows=1)
        stream_lines = make_stream_lines_from_potential(phi,)

        # Create the charges
        positive_charge = Circle(radius=0.4, color=RED, fill_opacity=1).move_to([-config.frame_width * 2.5 / 10, 0, 0])
        negative_charge = Circle(radius=0.4, color=BLUE, fill_opacity=1).move_to([config.frame_width * 2.5 / 10, 0, 0])

        # Animate the vector field, streamlines, and charges
        self.add(vector_field)
        # Add charges to the animation
        self.add(stream_lines)
        self.add(positive_charge, negative_charge)
        
        stream_lines.start_animation(warm_up=False, flow_speed=1, time_width=0.5,)
        self.wait(4)
        self.play(stream_lines.end_animation())
        self.wait(2)    
        

     
