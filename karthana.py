import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        """
        Initialize the Mobius strip with radius R, width w, and resolution n.
        R: Radius (distance from center to midline of the strip)
        w: Width of the strip
        n: Number of sample points for mesh grid
        """
        self.R = R
        self.w = w
        self.n = n

        # Create meshgrid over u ∈ [0, 2π], v ∈ [-w/2, w/2]
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(-w / 2, w / 2, n)
        self.u, self.v = np.meshgrid(u, v)

        # Compute 3D coordinates of the strip surface
        self.x, self.y, self.z = self._compute_coordinates()

    def _compute_coordinates(self):
        """
        Compute the (x, y, z) coordinates using the parametric equations of a Möbius strip:
        x(u,v) = (R + v*cos(u/2)) * cos(u)
        y(u,v) = (R + v*cos(u/2)) * sin(u)
        z(u,v) = v * sin(u/2)
        """
        u, v = self.u, self.v
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        """
        Approximate the surface area of the Möbius strip using numerical integration.
        Integration is done using the magnitude of the cross product of partial derivatives.
        """
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)

        # Compute partial derivatives
        xu = np.gradient(self.x, du, axis=1)
        xv = np.gradient(self.x, dv, axis=0)
        yu = np.gradient(self.y, du, axis=1)
        yv = np.gradient(self.y, dv, axis=0)
        zu = np.gradient(self.z, du, axis=1)
        zv = np.gradient(self.z, dv, axis=0)

        # Compute normal vector using cross product of partials
        nx = yu * zv - zu * yv
        ny = zu * xv - xu * zv
        nz = xu * yv - yu * xv

        # Surface area element
        dA = np.sqrt(nx**2 + ny**2 + nz**2)

        # Double integration over the surface
        area = simpson(simpson(dA, self.v[:, 0]), self.u[0])
        return area

    def compute_edge_length(self):
        """
        Compute the total edge length of the Möbius strip by summing the distances
        along the two boundary edges (v = ±w/2).
        """
        u_vals = np.linspace(0, 2 * np.pi, self.n)

        # Edge curve at v = +w/2
        edge_points_1 = np.array([
            (self.R + self.w / 2 * np.cos(u_vals / 2)) * np.cos(u_vals),
            (self.R + self.w / 2 * np.cos(u_vals / 2)) * np.sin(u_vals),
            self.w / 2 * np.sin(u_vals / 2)
        ]).T

        # Edge curve at v = -w/2
        edge_points_2 = np.array([
            (self.R - self.w / 2 * np.cos(u_vals / 2)) * np.cos(u_vals),
            (self.R - self.w / 2 * np.cos(u_vals / 2)) * np.sin(u_vals),
            -self.w / 2 * np.sin(u_vals / 2)
        ]).T

        # Numerical arc length computation
        def length(points):
            return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        total_edge_length = length(edge_points_1) + length(edge_points_2)
        return total_edge_length

    def plot(self):
        """
        Plot the 3D Möbius strip using matplotlib.
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Instantiate the Möbius strip with desired parameters
    mobius = MobiusStrip(R=1.0, w=0.4, n=200)

    # Compute and print surface area and edge length
    area = mobius.compute_surface_area()
    edge_len = mobius.compute_edge_length()
    print(f"Surface Area: {area:.4f}")
    print(f"Edge Length: {edge_len:.4f}")

    # Visualize the Möbius strip
    mobius.plot()
