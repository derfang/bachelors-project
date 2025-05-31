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


# Let's make the result ready to be used in Manim
Ey, Ex = electric_field_from_potential(phi)
# shift the field to center
Nvert, Nhoriz = Ex.shape
Ex_centered = np.roll(Ex, (Nhoriz//2, Nvert//2), axis=(1, 0))
Ey_centered = np.roll(Ey, (Nhoriz//2, Nvert//2), axis=(1, 0))
Ex_centered = np.roll(Ex_centered, Nvert//2, axis=0)
Ey_centered = np.roll(Ey_centered, Nvert//2, axis=0)
Ex_centered = np.roll(Ex_centered, Nhoriz//2, axis=1)
Ey_centered = np.roll(Ey_centered, Nhoriz//2, axis=1)







# lets plot the field up until here
plot_E_from_ExEy(Ex_centered, Ey_centered, field_points= 1000)