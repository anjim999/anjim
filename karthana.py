import streamlit as st
import numpy as np
from scipy.integrate import simpson
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class MobiusStrip:
    def __init__(self, R=1.0, w=0.2, n=100):
        self.R = R
        self.w = w
        self.n = n
        u = np.linspace(0, 2 * np.pi, n)
        v = np.linspace(-w / 2, w / 2, n)
        self.u, self.v = np.meshgrid(u, v)
        self.x, self.y, self.z = self._compute_coordinates()

    def _compute_coordinates(self):
        u, v = self.u, self.v
        x = (self.R + v * np.cos(u / 2)) * np.cos(u)
        y = (self.R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def compute_surface_area(self):
        du = 2 * np.pi / (self.n - 1)
        dv = self.w / (self.n - 1)
        xu = np.gradient(self.x, du, axis=1)
        xv = np.gradient(self.x, dv, axis=0)
        yu = np.gradient(self.y, du, axis=1)
        yv = np.gradient(self.y, dv, axis=0)
        zu = np.gradient(self.z, du, axis=1)
        zv = np.gradient(self.z, dv, axis=0)
        nx = yu * zv - zu * yv
        ny = zu * xv - xu * zv
        nz = xu * yv - yu * xv
        dA = np.sqrt(nx**2 + ny**2 + nz**2)
        area = simpson(simpson(dA, self.v[:, 0]), self.u[0])
        return area

    def compute_edge_length(self):
        u_vals = np.linspace(0, 2 * np.pi, self.n)
        edge_points_1 = np.array([
            (self.R + self.w / 2 * np.cos(u_vals / 2)) * np.cos(u_vals),
            (self.R + self.w / 2 * np.cos(u_vals / 2)) * np.sin(u_vals),
            self.w / 2 * np.sin(u_vals / 2)
        ]).T
        edge_points_2 = np.array([
            (self.R - self.w / 2 * np.cos(u_vals / 2)) * np.cos(u_vals),
            (self.R - self.w / 2 * np.cos(u_vals / 2)) * np.sin(u_vals),
            -self.w / 2 * np.sin(u_vals / 2)
        ]).T

        def length(points):
            return np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))

        return length(edge_points_1) + length(edge_points_2)

    def plot(self):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.x, self.y, self.z, cmap='viridis', edgecolor='none')
        ax.set_title("Möbius Strip")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        return fig

# Streamlit UI
st.title("🌀 Möbius Strip Visualizer")

R = st.slider("Radius (R)", 0.5, 2.0, 1.0, 0.1)
w = st.slider("Width (w)", 0.1, 1.0, 0.4, 0.05)
n = st.slider("Resolution (n)", 50, 300, 200, 10)

mobius = MobiusStrip(R=R, w=w, n=n)
area = mobius.compute_surface_area()
edge_len = mobius.compute_edge_length()

st.write(f"**Surface Area:** {area:.4f}")
st.write(f"**Edge Length:** {edge_len:.4f}")

fig = mobius.plot()
st.pyplot(fig)
