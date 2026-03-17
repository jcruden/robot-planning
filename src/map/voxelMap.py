import numpy as np
import math
from map.Lidar import LaserScan

L_FREE = 0.1
L_OCCUPIED = 0.1

class VoxelMap:
    def __init__(self,width = 9.85,height = 5.0,resolution = 0.1):
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
        hitxy = np.floor(scan.hit_points/self.resolution)
        heights = np.floor(scan.elevations/self.resolution)
        free = np.array(list(scan.free))
        hits = np.hstack((hitxy, heights))
        mask_keep = ~np.isnan(hits).any(axis=1)

        # Apply the mask to the array to get the cleaned data
        hits = hits[mask_keep]
        hits = hits.astype(int)
        hits = np.unique(hits, axis=0)
        
        self.logodds[free[:,0], free[:,1], free[:,2]] -= L_FREE
        self.logodds[hits[:,0], hits[:,1], hits[:,2]] += L_OCCUPIED