import numpy as np
import matplotlib.pyplot as plt
import pygame
import matplotlib.backends.backend_agg as agg
from matplotlib.gridspec import GridSpec
from exploration import robot

# 1. Load your map
grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')

# 2. Define your scale (Must match your Blender RESOLUTION)
resolution = 0.05  # Each cell is 0.1 meters (10cm)
rows, cols = grid.shape
width_m = cols * resolution
height_m = rows * resolution

# 3. Setup the figure
fig = plt.figure(figsize=(10, 10))
gs = GridSpec(2, 2, figure=fig)

# --- LEFT: 2D Heatmap To Scale ---
ax1 = fig.add_subplot(gs[0,0])
# 'extent' defines the [left, right, bottom, top] in meters
extent = [0, width_m, 0, height_m]
im = ax1.imshow(grid, cmap='terrain', origin='lower', extent=extent)
ax1.set_title(f"2D Map ({width_m:.1f}m x {height_m:.1f}m)")
ax1.set_xlabel("Meters (X)")
ax1.set_ylabel("Meters (Y)")
plt.colorbar(im, ax=ax1, label='Elevation (m)')

# --- RIGHT: 3D Surface To Scale ---
ax2 = fig.add_subplot(gs[0,1], projection='3d')
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

# Generated map 
ax3 = fig.add_subplot(gs[1,0])  # Add ax3 in a 2x2 grid layout
ax3.set_xlim(0, width_m)
ax3.set_ylim(0, height_m)
ax3.grid(True)
ax3.set_title("2D Map")

# Crater map 
ax4 = fig.add_subplot(gs[1,1])  # Add ax3 in a 2x2 grid layout
ax4.set_xlim(0, width_m)
ax4.set_ylim(0, height_m)
ax4.grid(True)
ax4.set_title("Voxel Map")

plt.tight_layout()
#plt.show()

def get_points_on_ax(ax, x, y, surface):
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    ax_x, ax_y, ax_width, ax_height = (
        int(bbox.x0 * fig.dpi),
        int(bbox.y0 * fig.dpi),
        int(bbox.width * fig.dpi),
        int(bbox.height * fig.dpi),
    )
    surf_w, surf_h = surface.get_size()
    bottom_left = (ax_x,           surf_h - ax_y)
    top_right = (ax_x + ax_width,  surf_h - (ax_y + ax_height))

    scale_x = ax_width / width_m
    scale_y = ax_height / height_m

    # Convert to px relative to axis
    x_px = bottom_left[0] + int(x * scale_x)
    y_px = bottom_left[1] -int(y * scale_y)

    return x_px, y_px

def draw_rob_ax(ax, rob, surface):
    x, y = get_points_on_ax(ax, rob.x, rob.y, surface)
    pygame.draw.circle(surface, (255, 0, 0), (x, y), 5)


def _draw_points_on_ax(ax, points, surface, color=(0, 255, 0), radius=2):
    if points is None:
        return

    for hx, hy in points:
        if np.isnan(hx) or np.isnan(hy):
            continue
        
        px, py = get_points_on_ax(ax, hx, hy, surface)

        pygame.draw.circle(surface, color, (px, py), radius)


def draw_robot(surf, rob):
    surface = surf.copy()
    draw_rob_ax(ax1, rob, surface)
    draw_rob_ax(ax3, rob, surface)
    draw_rob_ax(ax4, rob, surface)

    # Draw recent lidar scan
    scan = getattr(rob, "last_scan", None)
    hit_points = scan.hit_points if scan is not None else None
    _draw_points_on_ax(ax1, hit_points, surface)
    _draw_points_on_ax(ax3, hit_points, surface)
    _draw_points_on_ax(ax4, hit_points, surface)
    return surface

def viz_surface():
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.buffer_rgba()

    size = canvas.get_width_height()
    surf = pygame.image.frombuffer(raw_data, size, "RGBA")
    return surf
