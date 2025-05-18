import numpy as np
import matplotlib.pyplot as plt
import drawing_ops as ops
from solver import solve_poisson
from utils import *
from manim import *

N = 256 #todo resolve dependency of stream lines on grid size
charges = ops.composition(
    ops.circle(center=(N/2-N/5, N/2), radius=N/16, filled=True, value=1),
    ops.circle(center=(N/2+N/5, N/2), radius=N/16, filled=True, value=-1)
)
Rho = charges(np.zeros((N, N)))

phi = solve_poisson(U=np.zeros((N, N)), Rho=Rho)
Ey, Ex = electric_field_from_potential(phi)
E = amplitude(Ex, Ey)

# shifting the field to center
Ex = np.roll(Ex, N//2, axis=0)
Ey = np.roll(Ey, N//2, axis=0)
# shifting the field to center
E = np.roll(E, N//2, axis=0)

def make_function_from_vector_field(vector_field):
    """Create a function that returns the vector field at a given position. and is able to interpolate
    the field values."""
    def func(pos):
        x, y = pos
        # Convert to array indices
        i = int((x + config.frame_width / 2) / config.frame_width * N)
        j = int((y + config.frame_height / 2) / config.frame_height * N)
        # Ensure indices are within bounds
        i = np.clip(i, 0, N - 1)
        j = np.clip(j, 0, N - 1)
        return np.array([vector_field[0][i, j], vector_field[1][i, j], 0])
    return func

class ElectricFieldVisualization(Scene):
    
    def construct(self):
        # Normalize the field for visualization
        Ex_norm = 1.2*Ex / (E + 1e-8)
        Ey_norm = 1.2*Ey / (E + 1e-8)

        # Define a scaling factor to map positions to array indices
        scale_factor = N / config.frame_width
        funcnorm = lambda pos: np.array([
                Ex_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
                Ey_norm[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
                0
            ])
        func = lambda pos: np.array([
                Ex[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
                Ey[int(pos[1] * scale_factor), int(pos[0] * scale_factor)],
                0
            ])
        # Create a vector field
        vector_field = ArrowVectorField(
            funcnorm,
            x_range=[-config.frame_width / 2, config.frame_width / 2, 1],
            y_range=[-config.frame_height / 2, config.frame_height / 2, 1]
        )
        # Create streamlines
        stream_lines = StreamLines(func, stroke_width=0.7, max_anchors_per_line=500, virtual_time=0.5, color=BLUE,)

       
        # Adjust charges to ensure visibility
        positive_charge = Circle(radius=0.3, color=RED, fill_opacity=1).move_to([-config.frame_width * 3 / 10, 0, 0])
        negative_charge = Circle(radius=0.3, color=BLUE, fill_opacity=1).move_to([config.frame_width * 3 / 10, 0, 0])


        # Animate the vector field, streamlines, and charges
        self.add(vector_field)
        # Add charges to the animation
        self.add(stream_lines)
        self.add(positive_charge, negative_charge)
        
        stream_lines.start_animation(warm_up=False, flow_speed=1, time_width=0.5,)
        self.wait(4)
        self.play(stream_lines.end_animation())
        self.wait(2)    
        

     
