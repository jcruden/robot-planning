import numpy as np

from map import generated_map
from map import viz
from map.improvedLidar import Lidar
from map.voxelMap import VoxelMap, VOXEL_RESOLUTION
from exploration.VoxelRobot import VoxelRobot
import time
import pyvista as pv
import random



if __name__== "__main__":

    m = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')

    lidar = Lidar(m, world_resolution=viz.resolution, grid_resolution=VOXEL_RESOLUTION)
    log_odds = VoxelMap()
    rob = VoxelRobot(1,1,log_odds,lidar,m)
    plotter = pv.Plotter()
    
    truex = np.arange(m.shape[1])*.5
    truey = np.arange(m.shape[0])*.5
    x_grid, y_grid = np.meshgrid(truex, truey)
    z_grid = m/VOXEL_RESOLUTION  # The heights from your CSV

    # Create a StructuredGrid (This is a surface mesh, not voxels)
    true_surface = pv.StructuredGrid(x_grid, y_grid, z_grid)

    # Add it once before the loop starts
    plotter.add_mesh(
        true_surface, 
        color="white", 
        opacity=0.2,      # Faint overlay
        style="surface",  # Try "wireframe" for a technical look
        label="Ground Truth",
        show_scalar_bar=False,
        name="truth"      # Give it a name so it stays static
    )
    
    # Define grid structure
    grid = pv.ImageData()
    grid.dimensions = np.array(log_odds.logodds.shape) + 1
    grid.spacing = (1, 1, 1)

    grid.cell_data["values"] = log_odds.logodds.flatten(order="F")
    
    # Try a very low threshold initially to make sure SOMETHING shows up
    # or just show the whole grid once to fix the camera
    initial_mesh = grid.threshold(-10.0)
    
    plotter.add_mesh(initial_mesh, scalars="values", cmap="viridis", name="voxel_map")
    plotter.reset_camera()
    # Open the window
    plotter.show(interactive_update=True)
    
    init_time = time.time()
    rob.sensor_update()
    
    # 4. Scanning Loop
    running = True
    while running:
        if (rob.fuel > 0):
            rob.sensor_update()
            rob.move_path()
        else:
            running = False

        
        # Update grid data
        grid.cell_data["values"] = rob.voxelmap.logodds.flatten(order="F")
        
        # Calculate new occupied volume
        # IMPORTANT: If you shifted your data, change 0.05 to ~2.21
        threshed = grid.threshold(0.5) 
        
        scanner_mesh = pv.Sphere(radius=1, center=(rob.x/VOXEL_RESOLUTION, rob.y/VOXEL_RESOLUTION, rob.get_elevation()/VOXEL_RESOLUTION + 2))
        
        plotter.add_mesh(
            scanner_mesh, 
            color="red", 
            name="scanner_pos",  # Using a unique name keeps only the latest position
            render_points_as_spheres=True
        )
        
        # Only update if the threshold actually found voxels
        if threshed.n_cells > 0:
            plotter.add_mesh(
                threshed, 
                scalars="values", 
                cmap="viridis", 
                show_edges=True, 
                name="voxel_map",
                reset_camera=False # Don't jump the camera around every frame
            )
        
        # Force the UI to refresh
        plotter.update()
    
        
        if plotter.render_window is None:
            break
    print("Robot 1 (random): ", int(100 * rob.calculate_score()), " / 100")
    plotter.show()