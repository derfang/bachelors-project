import sys
# import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from solver import solve_laplace
from matplotlib.colors import Normalize
from matplotlib.animation import FuncAnimation
from manim import *

# ----------------------------------------------------------------------------

###########################
######### GENERAL #########
###########################

def progbar(iterable, length=30, symbol='='):
    """Wrapper generator function for an iterable. 
       Prints a progressbar when yielding an item. \\
       Args:
          iterable: an object supporting iteration
          length: length of the progressbar
    """
    n = len(iterable)
    for i, item in enumerate(iterable):
        steps = int(length/n*(i+1))
        sys.stdout.write('\r')        
        sys.stdout.write(f"[{symbol*steps:{length}}] {(100/n*(i+1)):.1f}%")
        sys.stdout.flush()
        yield item

# ----------------------------------------------------------------------------

###########################
###### VISUALIZATION ######
###########################

def plot_images(images, labels=[], nrow=3, cmap='viridis'):
    n_images = len(images)
    for i, image in enumerate(images):
        plt.subplot(-(n_images//-nrow), nrow, i+1)
        plt.imshow(image, cmap=cmap)
        if labels:
            plt.title(labels[i], fontdict={'fontsize': 22})
        plt.xlabel('x', fontdict={'fontsize': 18})
        plt.ylabel('y', fontdict={'fontsize': 18})
        plt.colorbar(shrink=0.35)

# ----------------------------------------------------------------------------

def make_colored_plot(A, num_levels=None, s=None, mul=1., cmap='afmhot'):  
    s = A.max()-A.min() if s is None else s
    
    cms = cm.get_cmap(cmap) 
    if num_levels is None:
        return cms(mul*(A - A.min()) / s)    

    Ac = np.floor((A - A.min())/s*num_levels) / num_levels
    return cms(mul*Ac)

# ----------------------------------------------------------------------------

def make_frames(values, 
                U, configU=None, configU_t=None,
                Rho=None, configRho=None, configRho_t=None, 
                Eps=None, configEps=None, configEps_t=None, 
                out=['phi', 'E']
                ):
    out_dict = {k: [] for k in out}

    if configU is not None:
        configU_t = lambda _: configU
    if configRho is not None:
        configRho_t = lambda _: configRho
    if configEps is not None:
        configEps_t = lambda _: configEps
    
    Uc, RhoC, EpsC = U, Rho, Eps
    for val in progbar(values):
        if configU_t is not None:
            Uc = configU_t(val)(U.copy())
        if configRho_t is not None:
            RhoC = configRho_t(val)(Rho.copy())
        if configEps_t is not None:
            EpsC = configEps_t(val)(Eps.copy())

        if 'U' in out: 
            out_dict['U'].append(Uc)

        if 'phi' in out:
            phi = solve_laplace(Uc, RhoC, EpsC)
            out_dict['phi'].append(phi)

        if 'E' in out:
            E = get_E_abs(phi)
            out_dict['E'].append(E)
    
    return out_dict

# ----------------------------------------------------------------------------

def colorize_frames(frames, num_levels=None, mul=1.):
    max_val = np.max([frame - np.min(frame) for frame in frames]) 
    frames_c = [make_colored_plot(frame, 
                                  num_levels=num_levels, 
                                  s=max_val, 
                                  mul=mul,
                                  cmap='afmhot')[:,:,:3] 
                                  for frame in frames]
    return frames_c

# ----------------------------------------------------------------------------

def plot_frames(frames, num=20, nrow=5):
    for i, frame in enumerate(frames[::len(frames)//num]):
        plt.subplot(-(num//-nrow), nrow, i+1)
        plt.imshow(frame)

# ----------------------------------------------------------------------------

def electric_field_from_potential(phi, scale=1.):
    """Returns a field vector of the electric field E from the potential phi.
       The field vector is scaled by the given scale factor.
    """
    Ey, Ex = np.gradient(phi)
    Ex, Ey = -Ex, -Ey
    return Ey*scale, Ex*scale

# ----------------------------------------------------------------------------

def plot_field_vector_of_E(phi, scale=1., nrow=5, field_points=100):
    """Plots the field vector of the electric field E from the potential phi.
       The field vector is scaled by the given scale factor. The field is visualized with a specific number of points.
    """
    Ey, Ex = electric_field_from_potential(phi, scale=scale)
    
    # Downsample the field to contain only a specific number of points
    x_indices = np.linspace(0, phi.shape[1] - 1, int(np.sqrt(field_points))).astype(int)
    y_indices = np.linspace(0, phi.shape[0] - 1, int(np.sqrt(field_points))).astype(int)
    Ex_downsampled = Ex[np.ix_(y_indices, x_indices)]
    Ey_downsampled = Ey[np.ix_(y_indices, x_indices)]
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Compute the magnitude of the field vectors
    E_magnitude = np.sqrt(Ex_downsampled**2 + Ey_downsampled**2)
    
    # normalize the magnitude of vectors so thier length is normalized
    Ex_normalized = 10 * Ex_downsampled / E_magnitude
    Ey_normalized = 10 * Ey_downsampled / E_magnitude
    
    
    # Set up the figure and axis
    
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, phi.shape[1])
    ax.set_ylim(0, phi.shape[0])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Electric Field Vector', fontdict={'fontsize': 22})
    ax.set_xlabel('x', fontdict={'fontsize': 18})
    ax.set_ylabel('y', fontdict={'fontsize': 18})
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Plot the field vectors with color representing magnitude
    quiver = ax.quiver(X, Y, Ex_normalized, Ey_normalized, E_magnitude, angles='xy', scale_units='xy', scale=1, cmap='afmhot', norm=Normalize())
    cbar = fig.colorbar(quiver, ax=ax, orientation='vertical')
    cbar.set_label('Field Magnitude', fontsize=14)
    
    plt.show()
    
# ----------------------------------------------------------------------------

def plot_field_vector_of_E_animated_with_rays(phi, scale=1., nrow=5, number_of_rays=100, field_points=100, number_of_particles=100):
    """Plots the field vector of the electric field E from the potential phi.
       The field vector is scaled by the given scale factor. Rays move along the vector field, with their speed based on the field's magnitude.
       The vectors are log-normalized in length but their original magnitude is shown in color (from blue to red).
       Rays have a short lifetime (~1 second) and are replaced by new random rays afterward.
    """
    import matplotlib.pyplot as plt

    Ey, Ex = electric_field_from_potential(phi, scale=scale)
    
    # Downsample the field to contain only a specific number of points
    x_indices = np.linspace(0, phi.shape[1] - 1, int(np.sqrt(field_points))).astype(int)
    y_indices = np.linspace(0, phi.shape[0] - 1, int(np.sqrt(field_points))).astype(int)
    Ex_downsampled = Ex[np.ix_(y_indices, x_indices)]
    Ey_downsampled = Ey[np.ix_(y_indices, x_indices)]
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Compute the magnitude of the field vectors
    E_magnitude = np.sqrt(Ex_downsampled**2 + Ey_downsampled**2)
    
    # Create random particles
    particles = np.random.rand(number_of_particles, 2) * np.array([phi.shape[1], phi.shape[0]])
    
    # Set up the figure and axis
    fig, ax = plt.subplots()
    ax.set_xlim(0, phi.shape[1])
    ax.set_ylim(0, phi.shape[0])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Electric Field Vector with Particles', fontdict={'fontsize': 22})
    ax.set_xlabel('x', fontdict={'fontsize': 18})
    ax.set_ylabel('y', fontdict={'fontsize': 18})
    
    # Plot the field vectors with color representing magnitude
    quiver = ax.quiver(X, Y, Ex_downsampled, Ey_downsampled, E_magnitude, angles='xy', scale_units='xy', scale=1, cmap='viridis', norm=Normalize())
    cbar = fig.colorbar(quiver, ax=ax, orientation='vertical')
    cbar.set_label('Field Magnitude', fontsize=14)
    
    # Scatter plot for particles
    scatter = ax.scatter(particles[:, 0], particles[:, 1], color='red', s=5)
    
    def update(frame):
        nonlocal particles
        # Update particles positions according to the field
        new_particles = []
        for particle in particles:
            x, y = particle
            if 0 <= int(y) < phi.shape[0] and 0 <= int(x) < phi.shape[1]:
                x_new = x + Ex[int(y), int(x)] * 0.1
                y_new = y + Ey[int(y), int(x)] * 0.1
                new_particles.append([x_new, y_new])
            else:
                # Reset particle if it goes out of bounds
                new_particles.append(np.random.rand(2) * np.array([phi.shape[1], phi.shape[0]]))
        particles = np.array(new_particles)
        scatter.set_offsets(particles)
        return scatter,
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=True)
    plt.show()
    
# ----------------------------------------------------------------------------
def plot_E_intensity (phi, shrink=0.6, levels=15 , field_vector=False):
    Ey, Ex = electric_field_from_potential(phi)
    E = amplitude(Ex, Ey)
    plt.figure(figsize=(8, 10))
    plt.imshow(E, cmap='afmhot')
    plt.colorbar(shrink=shrink)
    plt.contour(E, levels=levels)
    if field_vector == True:
        s = 6
        ymax = phi.shape[0]
        xmax = phi.shape[1]
        yy, xx = np.mgrid[0:ymax:s, 0:xmax:s]
        plt.quiver(xx, yy, Ex[::s, ::s], -Ey[::s, ::s], color='lightgrey')
    plt.title("E", fontdict={'fontsize': 22})
    plt.xlabel('x', fontdict={'fontsize': 18})
    plt.ylabel('y', fontdict={'fontsize': 18})
    plt.show()
    
# ----------------------------------------------------------------------------
def plot_rho_eps_phi(rho, eps, phi): #!change this so it can take any number of arguments
    plt.figure(figsize=(15, 10))
    plot_images([rho, eps, phi], labels=['ρ', 'ε', 'φ'], cmap='afmhot')
        
###########################
######### PHYSICS #########
###########################

def get_E_abs(phi):
    Ey, Ex = np.gradient(phi)
    E = np.sqrt(Ex**2 + Ey**2)
    return E

def amplitude(a, b):
    return np.sqrt(a**2 + b**2) 

# ----------------------------------------------------------------------------
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
    N = len(potential) #todo the grid may not be square
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
    Ex_centered = np.roll(Ex, N//2, axis=0)
    Ey_centered = np.roll(Ey, N//2, axis=0)
    # E_centered = np.roll(E, N//2, axis=0)

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
        x_range=[-config.frame_width / 2, config.frame_width / 2, 1],
        y_range=[-config.frame_height / 2, config.frame_height / 2, 1]
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
    Ex_centered = np.roll(Ex, N//2, axis=0)
    Ey_centered = np.roll(Ey, N//2, axis=0)
    E_centered = np.roll(E, N//2, axis=0)
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

class Charge(VGroup):
    def __init__(
        self,
        magnitude: float = 1,
        point: np.ndarray = ORIGIN,
        add_glow: bool = True,
        **kwargs,
    ) -> None:
        """An electrostatic charge object to produce an :class:`~ElectricField`.

        Parameters
        ----------
        magnitude
            The strength of the electrostatic charge.
        point
            The position of the charge.
        add_glow
            Whether to add a glowing effect. Adds rings of
            varying opacities to simulate glowing effect.
        kwargs
            Additional parameters to be passed to ``VGroup``.
        """
        VGroup.__init__(self, **kwargs)
        self.magnitude = magnitude
        self.point = point
        self.radius = (abs(magnitude) * 0.4 if abs(magnitude) < 2 else 0.8) * 0.3

        if magnitude > 0:
            label = VGroup(
                Rectangle(width=0.32 * 1.1, height=0.006 * 1.1).set_z_index(1),
                Rectangle(width=0.006 * 1.1, height=0.32 * 1.1).set_z_index(1),
            )
            color = RED
            layer_colors = [RED_D, RED_A]
            layer_radius = 4
        else:
            label = Rectangle(width=0.27, height=0.003)
            color = BLUE
            layer_colors = ["#3399FF", "#66B2FF"]
            layer_radius = 2

        if add_glow:  # use many arcs to simulate glowing
            layer_num = 80
            color_list = color_gradient(layer_colors, layer_num)
            opacity_func = lambda t: 1500 * (1 - abs(t - 0.009) ** 0.0001)
            rate_func = lambda t: t**2

            for i in range(layer_num):
                self.add(
                    Arc(
                        radius=layer_radius * rate_func((0.5 + i) / layer_num),
                        angle=TAU,
                        color=color_list[i],
                        stroke_width=101
                        * (rate_func((i + 1) / layer_num) - rate_func(i / layer_num))
                        * layer_radius,
                        stroke_opacity=opacity_func(rate_func(i / layer_num)),
                    ).shift(point)
                )

        self.add(Dot(point=self.point, radius=self.radius, color=color))
        self.add(label.scale(self.radius / 0.3).shift(point))
        for mob in self:
            mob.set_z_index(1)


class ElectricField(ArrowVectorField):
    def __init__(self, *charges: Charge, **kwargs) -> None:
        """An electric field.

        Parameters
        ----------
        charges
            The charges affecting the electric field.
        kwargs
            Additional parameters to be passed to ``ArrowVectorField``.

        Examples
        --------
        .. manim:: ElectricFieldExampleScene
            :save_last_frame:


            class ElectricFieldExampleScene(Scene):
                def construct(self):
                    charge1 = Charge(-1, LEFT + DOWN)
                    charge2 = Charge(2, RIGHT + DOWN)
                    charge3 = Charge(-1, UP)
                    field = ElectricField(charge1, charge2, charge3)
                    self.add(charge1, charge2, charge3)
                    self.add(field)
        """
        self.charges = charges
        positions = []
        magnitudes = []
        for charge in charges:
            positions.append(charge.get_center())
            magnitudes.append(charge.magnitude)
        super().__init__(lambda p: self._field_func(p, positions, magnitudes), **kwargs)

    def _field_func(
        self,
        p: np.ndarray,
        positions: list[np.ndarray],
        magnitudes: list[float],
    ) -> np.ndarray:
        field_vect = np.zeros(3)
        for p0, mag in zip(positions, magnitudes):
            r = p - p0
            dist = np.linalg.norm(r)
            if dist < 0.1:
                return np.zeros(3)
            field_vect += mag / dist**2 * normalize(r)
        return field_vect
    
    
def plot_E_from_ExEy(Ex, Ey, scale=1. , field_points=100):
    """Plots the electric field vector from the Ex and Ey components."""
    
    # Downsample the field to contain only a specific number of points
    x_indices = np.linspace(0, Ex.shape[1] - 1, int(np.sqrt(field_points))).astype(int)
    y_indices = np.linspace(0, Ex.shape[0] - 1, int(np.sqrt(field_points))).astype(int)
    Ex_downsampled = Ex[np.ix_(y_indices, x_indices)]
    Ey_downsampled = Ey[np.ix_(y_indices, x_indices)]
    X, Y = np.meshgrid(x_indices, y_indices)
    
    # Compute the magnitude of the field vectors
    E_magnitude = np.sqrt(Ex_downsampled**2 + Ey_downsampled**2)
    
    # normalize the magnitude of vectors so thier length is normalized
    Ex_normalized = 10 * Ex_downsampled / E_magnitude
    Ey_normalized = 10 * Ey_downsampled / E_magnitude
    
    
    # Set up the figure and axis
    
    
    fig, ax = plt.subplots()
    ax.set_xlim(0, Ex.shape[1])
    ax.set_ylim(0, Ex.shape[0])
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Electric Field Vector', fontdict={'fontsize': 22})
    ax.set_xlabel('x', fontdict={'fontsize': 18})
    ax.set_ylabel('y', fontdict={'fontsize': 18})
    ax.set_facecolor('black')
    fig.patch.set_facecolor('black')
    
    # Plot the field vectors with color representing magnitude
    quiver = ax.quiver(X, Y, Ex_normalized, Ey_normalized, E_magnitude, angles='xy', scale_units='xy', scale=1, cmap='afmhot', norm=Normalize())
    cbar = fig.colorbar(quiver, ax=ax, orientation='vertical')
    cbar.set_label('Field Magnitude', fontsize=14)
    
    plt.show()