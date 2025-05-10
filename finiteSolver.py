import numpy as np
import matplotlib.pyplot as plt
from matplotlib.quiver import Quiver
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as patches

class ElectrostaticFDSolver:
    """
    A finite difference solver for electrostatic fields from charge distributions.
    
    Attributes:
        grid_size (tuple): Number of grid points in (x, y) directions
        extent (tuple): Physical size of the domain in meters (x_max, y_max)
        dx (float): Grid spacing in x-direction
        dy (float): Grid spacing in y-direction
        potential (ndarray): Electric potential grid
        field (ndarray): Electric field vector grid (Ex, Ey)
        charges (ndarray): Charge density grid
    """
    
    def __init__(self, grid_size=(100, 100), extent=(1.0, 1.0)):
        """
        Initialize the solver with grid parameters.
        
        Args:
            grid_size: Tuple of (nx, ny) grid points
            extent: Tuple of (x_max, y_max) physical domain size in meters
        """
        self.grid_size = grid_size
        self.extent = extent
        self.dx = extent[0] / (grid_size[0] - 1)
        self.dy = extent[1] / (grid_size[1] - 1)
        
        # Initialize grids
        self.potential = np.zeros(grid_size)
        self.field = np.zeros((2, *grid_size))  # 2 components (Ex, Ey) for each grid point
        self.charges = np.zeros(grid_size)
        self.normalization_factor = 4  # Default normalization factor

    def set_normalization_factor(self, factor):
        """
        Set the normalization factor for the field vectors.

        Args:
            factor: The new normalization factor to be applied.
        """
        self.normalization_factor = factor
        
    def add_point_charge(self, q, position):
        """
        Add a point charge to the system.
        
        Args:
            q: Charge value in Coulombs
            position: Tuple of (x, y) position in meters
        """
        # Find nearest grid point
        i = int(round(position[0] / self.extent[0] * (self.grid_size[0] - 1)))
        j = int(round(position[1] / self.extent[1] * (self.grid_size[1] - 1)))
        
        # Ensure indices are within bounds
        i = max(0, min(i, self.grid_size[0] - 1))
        j = max(0, min(j, self.grid_size[1] - 1))
        
        # Add charge (distributed over the grid cell area)
        self.charges[i, j] += q / (self.dx * self.dy)
        
    def solve(self, max_iter=10000, tolerance=1e-6):
        """
        Solve for the electric potential using finite difference method.
        
        Args:
            max_iter: Maximum number of iterations
            tolerance: Convergence tolerance
        """
        # Initialize potential with Dirichlet boundary conditions (zero at boundaries)
        self.potential = np.zeros(self.grid_size)
        
        # Precompute coefficients
        dx2, dy2 = self.dx**2, self.dy**2
        denominator = 2 * (dx2 + dy2)
        eps0 = 8.8541878128e-12  # Vacuum permittivity
        
        # Successive Over-Relaxation (SOR) parameter
        omega = 1.9  # Optimal for many problems
        
        for iteration in range(max_iter):
            max_error = 0.0
            
            # Update interior points using finite difference
            for i in range(1, self.grid_size[0]-1):
                for j in range(1, self.grid_size[1]-1):
                    # Poisson equation: ∇²φ = -ρ/ε₀
                    new_val = ((dy2 * (self.potential[i+1, j] + self.potential[i-1, j]) +
                               dx2 * (self.potential[i, j+1] + self.potential[i, j-1]) +
                               dx2 * dy2 * self.charges[i, j] / eps0) / denominator)
                    
                    # SOR update
                    delta = omega * (new_val - self.potential[i, j])
                    self.potential[i, j] += delta
                    max_error = max(max_error, abs(delta))
            
            # Check for convergence
            if max_error < tolerance:
                print(f"Converged after {iteration} iterations")
                break
        else:
            print(f"Warning: Solution did not converge after {max_iter} iterations")
        
        # Calculate electric field as negative gradient of potential
        self._calculate_field()
        
    def _calculate_field(self):
        """Calculate electric field from potential using central differences."""
        # Ex = -dφ/dx
        self.field[0, 1:-1, :] = -(self.potential[2:, :] - self.potential[:-2, :]) / (2 * self.dx)
        # Ey = -dφ/dy
        self.field[1, :, 1:-1] = -(self.potential[:, 2:] - self.potential[:, :-2]) / (2 * self.dy)
        
        # Handle boundaries with forward/backward differences
        # Left boundary
        self.field[0, 0, :] = -(self.potential[1, :] - self.potential[0, :]) / self.dx
        # Right boundary
        self.field[0, -1, :] = -(self.potential[-1, :] - self.potential[-2, :]) / self.dx
        # Bottom boundary
        self.field[1, :, 0] = -(self.potential[:, 1] - self.potential[:, 0]) / self.dy
        # Top boundary
        self.field[1, :, -1] = -(self.potential[:, -1] - self.potential[:, -2]) / self.dy
        
    def get_field_at_point(self, x, y):
        """
        Get electric field vector at a specific point using bilinear interpolation.
        
        Args:
            x, y: Coordinates in meters
            
        Returns:
            Tuple of (Ex, Ey) at the point
        """
        # Normalize coordinates to grid indices
        xi = x / self.extent[0] * (self.grid_size[0] - 1)
        yi = y / self.extent[1] * (self.grid_size[1] - 1)
        
        # Get the four surrounding grid points
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = min(x0 + 1, self.grid_size[0] - 1), min(y0 + 1, self.grid_size[1] - 1)
        
        # Bilinear interpolation weights
        xd = xi - x0
        yd = yi - y0
        
        # Interpolate Ex and Ey components
        ex00 = self.field[0, x0, y0]
        ex10 = self.field[0, x1, y0]
        ex01 = self.field[0, x0, y1]
        ex11 = self.field[0, x1, y1]
        
        ey00 = self.field[1, x0, y0]
        ey10 = self.field[1, x1, y0]
        ey01 = self.field[1, x0, y1]
        ey11 = self.field[1, x1, y1]
        
        ex0 = ex00 * (1 - xd) + ex10 * xd
        ex1 = ex01 * (1 - xd) + ex11 * xd
        ex = ex0 * (1 - yd) + ex1 * yd
        
        ey0 = ey00 * (1 - xd) + ey10 * xd
        ey1 = ey01 * (1 - xd) + ey11 * xd
        ey = ey0 * (1 - yd) + ey1 * yd
        
        return (ex, ey)
        
    def plot_potential(self):
        """Plot the electric potential as a colormap."""
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, self.extent[0], self.grid_size[0])
        y = np.linspace(0, self.extent[1], self.grid_size[1])
        plt.contourf(x, y, self.potential.T, levels=50, cmap='viridis')
        plt.colorbar(label='Electric Potential (V)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Electric Potential')
        plt.show()
        
    def plot_field(self, stride=5, scale=20):
        """
        Plot the electric field as a quiver plot with normalized vectors and real amplitude in color.

        Args:
            stride: Plot every nth arrow for clarity
            scale: Scaling factor for arrow lengths
        """
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, self.extent[0], self.grid_size[0])
        y = np.linspace(0, self.extent[1], self.grid_size[1])
        X, Y = np.meshgrid(x, y)

        # Subsample the field for clearer visualization
        Ex = self.field[0, ::stride, ::stride]
        Ey = self.field[1, ::stride, ::stride]
        X_sub = X[::stride, ::stride]
        Y_sub = Y[::stride, ::stride]

        # Calculate the amplitude of the field
        amplitude = np.sqrt(Ex**2 + Ey**2)

        # Normalize the field vectors using np.linalg.norm and the normalization factor
        norms = np.linalg.norm(np.stack((Ex, Ey), axis=-1), axis=-1)
        Ex_normalized = Ex * self.normalization_factor / norms
        Ey_normalized = Ey * self.normalization_factor / norms

        # Plot the quiver with normalized vectors and color by amplitude
        plt.quiver(X_sub, Y_sub, Ex_normalized.T, Ey_normalized.T, amplitude.T, scale=scale, scale_units='inches', cmap='viridis')
        plt.colorbar(label='Electric Field Amplitude (V/m)')
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Electric Field (Normalized Vectors, Amplitude in Color)')
        plt.show()
        
    def get_field_vector(self, x, y):
        """
        Get the electric field vector (Ex, Ey) at a specific point (x, y).

        Parameters:
        - x, y: Coordinates of the point where the field vector is needed.

        Returns:
        - A tuple (Ex, Ey) representing the electric field vector at (x, y).
        """
        if self.E_x is None or self.E_y is None:
            raise ValueError("Solve for electric field first")

        # Interpolate the field components
        Ex = np.interp(x, self.x, self.E_x[:, 0])
        Ey = np.interp(y, self.y, self.E_y[0, :])
        return Ex, Ey

    def get_potential_data(self):
        """
        Get the potential data as a 2D array.

        Returns:
        - A 2D numpy array representing the electric potential.
        """
        if self.phi is None:
            raise ValueError("Solve for potential first")
        return self.phi

    def get_field_vector_interpolated(self, x, y):
        """
        Get the electric field vector (Ex, Ey) at a specific point (x, y) using interpolation.

        Parameters:
        - x, y: Coordinates of the point where the field vector is needed.

        Returns:
        - A tuple (Ex, Ey) representing the electric field vector at (x, y).
        """
        if self.field is None:
            raise ValueError("Solve for electric field first")

        # Normalize coordinates to grid indices
        xi = (x + self.domain_size[0] / 2) / self.domain_size[0] * (self.grid_size[0] - 1)
        yi = (y + self.domain_size[1] / 2) / self.domain_size[1] * (self.grid_size[1] - 1)

        # Get the four surrounding grid points
        x0, y0 = int(np.floor(xi)), int(np.floor(yi))
        x1, y1 = min(x0 + 1, self.grid_size[0] - 1), min(y0 + 1, self.grid_size[1] - 1)

        # Bilinear interpolation weights
        xd = xi - x0
        yd = yi - y0

        # Interpolate Ex and Ey components
        ex00 = self.field[0, x0, y0]
        ex10 = self.field[0, x1, y0]
        ex01 = self.field[0, x0, y1]
        ex11 = self.field[0, x1, y1]

        ey00 = self.field[1, x0, y0]
        ey10 = self.field[1, x1, y0]
        ey01 = self.field[1, x0, y1]
        ey11 = self.field[1, x1, y1]

        ex0 = ex00 * (1 - xd) + ex10 * xd
        ex1 = ex01 * (1 - xd) + ex11 * xd
        ex = ex0 * (1 - yd) + ex1 * yd

        ey0 = ey00 * (1 - xd) + ey10 * xd
        ey1 = ey01 * (1 - xd) + ey11 * xd
        ey = ey0 * (1 - yd) + ey1 * yd

        return (ex, ey)

class ElectrostaticFDSolverWithGeometry(ElectrostaticFDSolver):
    """
    Extended solver with geometric drawing capabilities and visualization of shapes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = []  # Store shapes for visualization

    def draw_circle(self, center, radius, value, color='red'):
        """
        Draw a circle on the charge density grid and store it for visualization.

        Args:
            center: Tuple of (x, y) coordinates of the circle center in meters.
            radius: Radius of the circle in meters.
            value: Charge density value inside the circle.
            color: Color to fill the circle in the plot.
        """
        cx, cy = center
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x = i * self.dx
                y = j * self.dy
                if (x - cx)**2 + (y - cy)**2 <= radius**2:
                    self.charges[i, j] = value
        # Store the circle for visualization
        self.shapes.append(patches.Circle(center, radius, fill=True, edgecolor='black', facecolor=color, linewidth=2))

    def draw_rectangle(self, bottom_left, width, height, value, color='blue'):
        """
        Draw a rectangle on the charge density grid and store it for visualization.

        Args:
            bottom_left: Tuple of (x, y) coordinates of the bottom-left corner in meters.
            width: Width of the rectangle in meters.
            height: Height of the rectangle in meters.
            value: Charge density value inside the rectangle.
            color: Color to fill the rectangle in the plot.
        """
        x0, y0 = bottom_left
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                x = i * self.dx
                y = j * self.dy
                if x0 <= x <= x0 + width and y0 <= y <= y0 + height:
                    self.charges[i, j] = value
        # Store the rectangle for visualization
        self.shapes.append(patches.Rectangle(bottom_left, width, height, fill=True, edgecolor='black', facecolor=color, linewidth=2))

    def plot_field(self, stride=5, scale=20):
        """
        Plot the electric field as a quiver plot with shapes overlaid.

        Args:
            stride: Plot every nth arrow for clarity
            scale: Scaling factor for arrow lengths
        """
        plt.figure(figsize=(8, 6))
        x = np.linspace(0, self.extent[0], self.grid_size[0])
        y = np.linspace(0, self.extent[1], self.grid_size[1])
        X, Y = np.meshgrid(x, y)

        # Subsample the field for clearer visualization
        Ex = self.field[0, ::stride, ::stride]
        Ey = self.field[1, ::stride, ::stride]
        X_sub = X[::stride, ::stride]
        Y_sub = Y[::stride, ::stride]

        # Calculate the amplitude of the field
        amplitude = np.sqrt(Ex**2 + Ey**2)

        # Normalize the field vectors
        norms = np.linalg.norm(np.stack((Ex, Ey), axis=-1), axis=-1)
        Ex_normalized = Ex * self.normalization_factor / norms
        Ey_normalized = Ey * self.normalization_factor / norms

        # Plot the quiver with normalized vectors and color by amplitude
        plt.quiver(X_sub, Y_sub, Ex_normalized.T, Ey_normalized.T, amplitude.T, scale=scale, scale_units='inches', cmap='viridis')
        plt.colorbar(label='Electric Field Amplitude (V/m)')

        # Overlay shapes
        ax = plt.gca()
        for shape in self.shapes:
            ax.add_patch(shape)

        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Electric Field with Shapes')
        plt.axis('equal')
        plt.show()

# Example usage
if __name__ == "__main__":
    # Create an extended solver with 100x100 grid over a 1m x 1m domain
    solver = ElectrostaticFDSolverWithGeometry(grid_size=(100, 100), extent=(1.0, 1.0))

    # Draw a circle with positive charge density
    solver.draw_circle(center=(0.5, 0.5), radius=0.1, value=1e-9)

    # Draw a rectangle with negative charge density
    solver.draw_rectangle(bottom_left=(0.2, 0.2), width=0.2, height=0.1, value=-1e-9)
    
    # Solve the system
    solver.solve()

    # Visualize results
    solver.plot_potential()
    solver.plot_field()

