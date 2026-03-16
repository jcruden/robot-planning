import numpy as np
import math
from Lidar import LaserScan

L_FREE = 0.1
L_OCCUPIED = 0.1

class VoxelMap:
    def init(self,
             width = 9.85,
             height = 5.0,
             resolution = 0.2
             ):
        
        self.width = width
        self.height = height
        self.resolution = resolution

        self.x = int(np.ceil(width/resolution))
        self.y = self.x
        self.z = int(height/resolution)

        self.map = np.ndarray((self.x,self.y,self.z))
        
        
    def scan_update(self, scan: LaserScan):
        """
        Updates 3d occupancy grid with Lidar scan data

        Args:
            scan (LaserScan) Provides information about the lidar scan
        """
        hitxy = np.floor(scan.hit_points/self.resolution)
        heights = np.floor(scan.elevations/self.resolution)
        free = np.array(list(scan.free)).T
        hits = np.hstack(hitxy, heights)
        
        self.map[free] -= L_FREE
        self.map[hits[:,0], hits[:,1], hits[:,2]] += L_OCCUPIED