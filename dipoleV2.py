import numpy as np
import matplotlib.pyplot as plt
import drawing_ops as ops
from solver import solve_laplace
from utils import *
from manim import *

Nvert = 9 * 20
Nhoriz = 16 * 20
charges = ops.composition(
    ops.circle(center=(Nhoriz/2-Nhoriz/5, Nvert/2), radius=Nvert/16, filled=True, value=1),
    ops.circle(center=(Nhoriz/2+Nhoriz/5, Nvert/2), radius=Nvert/16, filled=True, value=-1)
)
Rho = charges(np.zeros((Nvert, Nhoriz)))

phi = solve_laplace(U=np.zeros((Nvert, Nhoriz)), Rho=Rho)

def make_function_from_potential(potential, scale_arrows=1.2):
    """
    Create a function that returns the electric field vector at a given position"""
    # Let's make the result ready to be used in Manim
    Ey, Ex = electric_field_from_potential(potential)
    # shift the field to center
    Nvert, Nhoriz = Ex.shape
    Ex_centered = np.roll(Ex, (Nhoriz//2, Nvert//2), axis=(1, 0))
    Ey_centered = np.roll(Ey, (Nhoriz//2, Nvert//2), axis=(1, 0))
    Ex_centered = np.roll(Ex_centered, Nvert//2, axis=0)
    Ey_centered = np.roll(Ey_centered, Nvert//2, axis=0)
    Ex_centered = np.roll(Ex_centered, Nhoriz//2, axis=1)
    Ey_centered = np.roll(Ey_centered, Nhoriz//2, axis=1)
    E = np.sqrt(Ex_centered**2 + Ey_centered**2)
    # plot_E_from_ExEy(Ex_centered, Ey_centered)
    
    # Define a function that returns the electric field vector at a given position if the value exists if not it interpolates
    def func(pos):
        # Define scaling factors
        # scale_factor_x = Nhoriz / config.frame_width
        # scale_factor_y = Nvert / config.frame_height
        # Calculate indices in the array
        scale_factor_x = 1
        scale_factor_y = 1
        x_index = int(pos[0] * scale_factor_x)
        y_index = int(pos[1] * scale_factor_y)
        # Ensure indices are within bounds
        x_index = np.clip(x_index, 0, Nhoriz - 1)
        y_index = np.clip(y_index, 0, Nvert - 1)
        # Return the electric field vector at the position
        return np.array([
            Ex_centered[y_index, x_index] * scale_arrows,
            Ey_centered[y_index, x_index] * scale_arrows,
            0
        ])
    def funcnorm(pos):
        Ex_centered = Ex_centered / (E + 1e-8)
        Ey_centered = Ey_centered / (E + 1e-8)
        # Define scaling factors
        scale_factor_x = Nhoriz / config.frame_width
        scale_factor_y = Nvert / config.frame_height
        # Calculate indices in the array
        x_index = int(pos[0] * scale_factor_x)
        y_index = int(pos[1] * scale_factor_y)
        # Ensure indices are within bounds
        x_index = np.clip(x_index, 0, Nhoriz - 1)
        y_index = np.clip(y_index, 0, Nvert - 1)
        # Return the normalized electric field vector at the position
        return np.array([
            Ex_centered[y_index, x_index],
            Ey_centered[y_index, x_index],
            0
        ])
    return func, funcnorm

# Create a function that returns the electric field vector based on the potential
def make_arrow_vector_field_E_from_potential(potential, scale_arrows=1.2, No_of_points=(16, 9)):
    """
    Create a function that returns a Manim ArrowVectorField
    based on the potential.
    This function generates a vector field using the electric field derived from the given potential.
    Args:
        potential (np.ndarray): The potential array.
    Returns:
        func (callable): Manim ArrowVectorField
    """
    
    Ey, Ex = electric_field_from_potential(potential)
    # shift the field to center
    Nvert, Nhoriz = Ex.shape
    Ex_centered = np.roll(Ex, (Nhoriz//2, Nvert//2), axis=(1, 0))
    Ey_centered = np.roll(Ey, (Nhoriz//2, Nvert//2), axis=(1, 0))
    Ex_centered = np.roll(Ex_centered, Nvert//2, axis=0)
    Ey_centered = np.roll(Ey_centered, Nvert//2, axis=0)
    Ex_centered = np.roll(Ex_centered, Nhoriz//2, axis=1)
    Ey_centered = np.roll(Ey_centered, Nhoriz//2, axis=1)
    E = np.sqrt(Ex_centered**2 + Ey_centered**2)
    
    scale_factor_x = Nhoriz / config.frame_width
    scale_factor_y = Nvert / config.frame_height
    no_x = No_of_points[0]
    no_y = No_of_points[1]
    arrow_field = ArrowVectorField(
        lambda pos: np.array([
            Ex_centered[int(pos[1] * scale_factor_y), int(pos[0] * scale_factor_x)] ,
            Ey_centered[int(pos[1] * scale_factor_y), int(pos[0] * scale_factor_x)] ,
            0
        ]),
        x_range=(-config.frame_width / 2, config.frame_width / 2, config.frame_width / no_x),
        y_range=(-config.frame_height / 2, config.frame_height / 2, config.frame_height / no_y),
        length_func=lambda x: E[int(x[1] * scale_factor_y), int(x[0] * scale_factor_x)],
        color=WHITE
    )
    
    # Return the function that computes the electric field vector
    return arrow_field

class DipoleScene(Scene):
    def construct(self):
        # Create the vector field and streamlines
        vector_field = make_arrow_vector_field_E_from_potential(phi, scale_arrows=1)


        # Create the charges
        positive_charge = Circle(radius=0.4, color=RED, fill_opacity=1).move_to([-config.frame_width * 3 / 10, 0, 0])
        negative_charge = Circle(radius=0.4, color=BLUE, fill_opacity=1).move_to([config.frame_width * 3 / 10, 0, 0])

        # Animate the vector field, streamlines, and charges
        self.add(vector_field)
        self.add(positive_charge, negative_charge)
        