import numpy as np

from map import generated_map
from map import viz
from map.Lidar import Lidar
from map.voxelMap import VoxelMap
import time
import pyvista as pv
import random

if __name__== "__main__":
    m = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    lidar = Lidar(m, world_resolution=viz.resolution, grid_resolution=0.1)
    log_odds = VoxelMap()
    for i in range(10):
        x = random.randint(1,9)
        y = random.randint(1,9)
        scan = lidar.scan(x,y)
        log_odds.scan_update(scan)
    grid = pv.ImageData()
    grid.dimensions = np.array(log_odds.logodds.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.cell_data["values"] = log_odds.logodds.flatten(order="F")
    grid
    threshed = grid.threshold(0.05) # Only show occupied
    threshed.plot(show_edges=True, cmap="viridis")