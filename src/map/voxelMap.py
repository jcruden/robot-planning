import numpy as np
import math
from map.improvedLidar import LaserScan

L_FREE = 0.1
L_OCCUPIED = 0.2
VOXEL_RESOLUTION = 0.1

class VoxelMap:
    def __init__(self,width = 9.85,height = 5.0,resolution = VOXEL_RESOLUTION):
        self.width = width
        self.height = height
        self.resolution = resolution

        self.x = int(np.ceil(width/resolution))
        self.y = self.x
        self.z = int(height/resolution)

        self.logodds = np.ndarray((self.x,self.y,self.z))
        
        
    def scan_update(self, scan: LaserScan):
        """
        Updates 3d occupancy grid with Lidar scan data

        Args:
            scan (LaserScan) Provides information about the lidar scan
        """
        #Arrange scan data into arrays for indexing
        hitxy = np.floor(scan.hit_points/self.resolution)
        heights = np.floor(scan.elevations/self.resolution)
        free = np.array(list(scan.free))
        hits = np.hstack((hitxy, heights))
        mask_keep = ~np.isnan(hits).any(axis=1)

        min_vals = np.array([0, 0, 0])
        max_vals = np.array([self.x - 1, self.y - 1, self.z - 1])
        hits = hits[mask_keep]
        hits = hits.astype(int)
        mask = np.array([True, True, True])

        # Clip based on max indexes
        clipped_hits = np.clip(hits, min_vals, max_vals)
        clipped_free = np.clip(free, min_vals, max_vals)
        # Apply the mask
        hits = np.where(mask, clipped_hits, hits)
        free = np.where(mask, clipped_free, free)
     
        #update logodds
        np.add.at(self.logodds, tuple(free.T), -L_FREE)
        np.add.at(self.logodds, tuple(hits.T), L_OCCUPIED)
    
    def get_heightmap(self):
        # 1. Create a boolean mask of where cells are "occupied"
        occupied_mask = self.logodds > 0.5
        
        # 2. Create an array of Z-indices [0, 1, 2, ..., depth]
        # We broadcast this to match the shape of the log_odds
        z_indices = np.arange(self.logodds.shape[2])
        z_grid = np.broadcast_to(z_indices, self.logodds.shape)
        
        # 3. Mask the Z-indices: Keep the index if occupied, else -1
        # Note: We use -1 so that empty columns are easily identifiable
        occupied_z_values = np.where(occupied_mask, z_grid.astype(float), -1.0)
        
        # 4. Find the maximum Z index for each (x, y) column
        heightmap_2d = np.max(occupied_z_values, axis=2)*self.resolution
        
        # 5. Optional: Clean up columns that had NO occupied voxels
        # Replace -1 with your desired ground level (e.g., 0)
        heightmap_2d[heightmap_2d == -1.0*self.resolution] = np.nan
        
        return heightmap_2d
        
    def get_max_elev(self,x,y, heights):
            (cols, rows, _ ) = self.logodds.shape
            col = x / self.resolution
            row = y / self.resolution

            c0 = int(np.floor(col))
            r0 = int(np.floor(row))
            c1 = c0 + 1
            r1 = r0 + 1

            if c0 < 0 or r0 < 0 or c1 >= cols or r1 >= rows:
                return 0.0

            dc = col - c0
            dr = row - r0
         
            z = (heights[r0, c0] * (1 - dc) * (1 - dr) +
            heights[r0, c1] * dc * (1 - dr) +
            heights[r1, c0] * (1 - dc) * dr +
            heights[r1, c1] * dc * dr)