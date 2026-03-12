import numpy as np
import matplotlib.pyplot as plt
import pygame
import matplotlib.backends.backend_agg as agg
from exploration import robot

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
#ax1.axis('off')
#ax1.set_position([0, 0, 1, 1])
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

def draw_robot(surf, rob):
    surface = surf.copy()
    # Get the bounding box of ax1 in pixels
    bbox = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax1_x, ax1_y, ax1_width, ax1_height = (
        int(bbox.x0 * fig.dpi),
        int(bbox.y0 * fig.dpi),
        int(bbox.width * fig.dpi),
        int(bbox.height * fig.dpi),
    )

    # Calculate scaling factors between ax1 and the full pygame surface
    scale_x = ax1_width / (width_m / resolution)
    scale_y = ax1_height / (height_m / resolution)

    # Convert robot's position from meters to pixels relative to ax1
    robot_x_px = int(rob.x / resolution * scale_x)
    robot_y_px = int(rob.y / resolution * scale_y)

    # Invert the Y-axis for pygame rendering
    robot_y_px = ax1_height - robot_y_px

    # Adjust the robot's position to the ax1 region within the full surface
    robot_x_px += ax1_x
    robot_y_px += ax1_y

    # Draw the robot as a red circle on the surface
    pygame.draw.circle(surface, (255, 0, 0), (robot_x_px, robot_y_px), 5)  # Radius of 5 pixels

    return surface

def viz_surface():
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()

    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf
