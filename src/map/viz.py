import numpy as np
import matplotlib.pyplot as plt

# 1. Load your map
grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')

# 2. Define your scale (Must match your Blender RESOLUTION)
resolution = 0.05  # Each cell is 0.1 meters (10cm)
rows, cols = grid.shape
width_m = cols * resolution
height_m = rows * resolution

# 3. Setup the figure
fig = plt.figure(figsize=(14, 6))

# --- LEFT: 2D Heatmap To Scale ---
ax1 = fig.add_subplot(1, 2, 1)
# 'extent' defines the [left, right, bottom, top] in meters
extent = [0, width_m, 0, height_m]
im = ax1.imshow(grid, cmap='terrain', origin='lower', extent=extent)

ax1.set_title(f"2D Map ({width_m:.1f}m x {height_m:.1f}m)")
ax1.set_xlabel("Meters (X)")
ax1.set_ylabel("Meters (Y)")
plt.colorbar(im, ax=ax1, label='Elevation (m)')

# --- RIGHT: 3D Surface To Scale ---
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
x = np.linspace(0, width_m, cols)
y = np.linspace(0, height_m, rows)
X, Y = np.meshgrid(x, y)

surf = ax2.plot_surface(X, Y, grid, cmap='terrain', antialiased=True)

# set_box_aspect ensures the physical proportions look correct.
z_range = np.ptp(grid) # The difference between max and min height
ax2.set_box_aspect((width_m, height_m, z_range))

ax2.set_title("3D Proportional Reconstruction")
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Y (m)")
ax2.set_zlabel("Z (m)")

plt.tight_layout()
plt.show()