import numpy as np

from map import generated_map
from map import viz
from map.improvedLidar import Lidar
from map.voxelMap import VoxelMap
import time
import pyvista as pv
import random



if __name__== "__main__":
    # 1. Setup Data and Lidar
    m = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    # Make sure these resolutions match your data
    lidar = Lidar(m, world_resolution=viz.resolution, grid_resolution=0.1)
    log_odds = VoxelMap()
    
    # 2. Setup the Plotter
    plotter = pv.Plotter()
    
    # Define grid structure
    grid = pv.ImageData()
    grid.dimensions = np.array(log_odds.logodds.shape) + 1
    grid.spacing = (1, 1, 1)

    # 3. Handle the first frame carefully
    # We do a dummy scan or use the empty map to initialize the camera
    grid.cell_data["values"] = log_odds.logodds.flatten(order="F")
    
    # Try a very low threshold initially to make sure SOMETHING shows up
    # or just show the whole grid once to fix the camera
    initial_mesh = grid.threshold(-10.0) # Show everything to start
    
    plotter.add_mesh(initial_mesh, scalars="values", cmap="viridis", name="voxel_map")
    plotter.reset_camera() # This points the camera at your data
    
    # Open the window
    plotter.show(interactive_update=True)

    # 4. Scanning Loop
    for i in range(100):
        x = random.randint(1, 9)
        y = random.randint(1, 9)
        
        scan = lidar.scan(x, y)
        log_odds.scan_update(scan)
        
        # Update grid data
        grid.cell_data["values"] = log_odds.logodds.flatten(order="F")
        
        # Calculate new occupied volume
        # IMPORTANT: If you shifted your data, change 0.05 to ~2.21
        threshed = grid.threshold(0.05) 
        
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

    plotter.show()