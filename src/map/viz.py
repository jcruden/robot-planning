import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# 1. Load the data
# Make sure this matches the path you used in Blender!
file_path = r"C:\Users\lucas\OneDrive\Documents\133\final_square_map.csv"
grid = np.genfromtxt(file_path, delimiter=",")

# 2. Setup the Plotting Area
fig = plt.figure(figsize=(12, 5))

# --- LEFT: 2D Heatmap (Top-Down View) ---
ax1 = fig.add_subplot(1, 2, 1)
im = ax1.imshow(grid, cmap='terrain', origin='lower')
plt.colorbar(im, ax=ax1, label='Elevation (m)')
ax1.set_title("Top-Down Elevation Grid")
ax1.set_xlabel("Grid X")
ax1.set_ylabel("Grid Y")

# --- RIGHT: 3D Surface Plot ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

# Create X and Y coordinates for the 3D plot
rows, cols = grid.shape
x = np.linspace(0, cols, cols)
y = np.linspace(0, rows, rows)
X, Y = np.meshgrid(x, y)

# Plot the surface
surf = ax2.plot_surface(X, Y, grid, cmap='terrain', 
                       linewidth=0, antialiased=False)
ax2.set_title("3D Terrain Reconstruction")
ax2.set_zlabel("Height (m)")

plt.tight_layout()
plt.show()