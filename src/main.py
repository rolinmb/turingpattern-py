import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define parameters for the Gray-Scott model
width, height = 100, 100  # Grid size
Du, Dv = 0.16, 0.08       # Diffusion rates for u and v
F, k = 0.035, 0.065       # Feed and kill rates

# Initialize concentrations
u = np.ones((width, height))  # Uniform concentration of u
v = np.zeros((width, height)) # No v initially

# Seed an initial disturbance
r = 20  # Radius of disturbance
center = (width // 2, height // 2)
y, x = np.ogrid[:width, :height]
mask = (x - center[1])**2 + (y - center[0])**2 <= r**2
u[mask] = 0.5
v[mask] = 0.25

# Update function for reaction-diffusion system
def update(u, v, Du, Dv, F, k, dt):
    laplacian_u = (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
    )
    laplacian_v = (
        np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
        np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
    )
    reaction = u * v**2
    u += (Du * laplacian_u - reaction + F * (1 - u)) * dt
    v += (Dv * laplacian_v + reaction - (F + k) * v) * dt
    return u, v

if __name__ == "__main__":
    # Create the figure
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    im = ax.imshow(v, cmap='inferno', interpolation='bilinear')
    # Animation function
    def animate(frame):
        global u, v
        u, v = update(u, v, Du, Dv, F, k, dt=1.0)
        im.set_data(v)
        return [im]
    # Run the animation
    ani = FuncAnimation(fig, animate, frames=200, interval=50, blit=True)
    plt.show()
