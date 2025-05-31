import numpy as np
import drawing_ops as ops
from solver import solve_laplace
from utils import *
from manim import *

N = 256
charges = ops.composition(
    ops.circle(center=(N/2-N/5, N/2), radius=N/16, filled=True, value=1),
    ops.circle(center=(N/2+N/5, N/2), radius=N/16, filled=True, value=-1)
)
Rho = charges(np.zeros((N, N)))
phi = solve_laplace(U=np.zeros((N, N)), Rho=Rho)
class Dipole_animation(Scene):
    
    def construct(self):
        # Create the vector field and streamlines
        vector_field = make_arrow_vector_field_E_from_potential(phi, scale_arrows=1)
        stream_lines = make_stream_lines_from_potential(phi,)

        # Create the charges
        charge1 = Charge(5 , LEFT*3)
        charge2 = Charge(-5, RIGHT*3)

        # Animate the vector field, streamlines, and charges
        self.add(vector_field)
        # Add charges to the animation
        self.add(stream_lines)
        self.add(charge1, charge2)
        
        stream_lines.start_animation(warm_up=False, flow_speed=1, time_width=0.5,)
        self.wait(4)
        self.play(stream_lines.end_animation())
        self.wait(2)    
        

     
