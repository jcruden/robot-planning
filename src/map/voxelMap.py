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
        
        
    def scan_update(self, start: tuple[float, float, float], rays: list[tuple[float, float, float]], range_max: float):
        """
        Updates 3d occupancy grid with Lidar scan data

        Args:
            start (tuple[float, float, float]): coordinates of the lidar position (above robot)
            rays (list[tuple[float, float, float]]): A list of all rays in the form (theta_horizontal,      theta_vertical, distance), distance is np.nan if did not hit anything
        """
        # 1. Convert to NumPy array
        # ray_data shape: (N, 3) -> [theta_h, theta_v, dist]
        ray_data = np.array(rays)
        theta_h, theta_v = ray_data[:, 0], ray_data[:, 1]
        dist = ray_data[:, 2]

        # 2. Identify Hits vs Misses
        is_hit = ~np.isnan(dist)
        # For calculation, treat NaNs as range_max to clear free space
        clean_dist = np.where(is_hit, dist, range_max)

        # 3. Spherical to Cartesian (Relative to start)
        # Using cos(theta_v) as the horizontal projection multiplier
        cos_tv = np.cos(theta_v)
        dx = clean_dist * cos_tv * np.cos(theta_h)
        dy = clean_dist * cos_tv * np.sin(theta_h)
        dz = clean_dist * np.sin(theta_v)
        
        # stack into (N, 3) relative vectors
        ray_vectors = np.column_stack((dx, dy, dz))
        
        # 4. Generate Sampling Points (Free Space)
        # We sample every half-resolution to ensure we don't skip voxels
        num_samples = int((range_max / self.resolution) * 2)
        # t moves from 0.0 (start) to ~0.95 (just before the hit/max)
        t_vals = np.linspace(0.0, 0.95, num_samples)

        # Broadcast: (num_samples, 1) * (N, 3) -> (num_samples, N, 3)
        # Then add the start position
        free_points = np.array(start) + (t_vals[:, np.newaxis, np.newaxis] * ray_vectors)
        
        # 5. Convert to Voxel Indices
        free_voxels = np.floor(free_points / self.resolution).astype(np.int32)
        # Flatten and keep unique indices
        free_voxels = np.unique(free_voxels.reshape(-1, 3), axis=0)

        # 6. Extract Occupied Voxels
        # Only for rays that actually hit (is_hit)
        hit_points = np.array(start) + ray_vectors[is_hit]
        occupied_voxels = np.floor(hit_points / self.resolution).astype(np.int32)
        occupied_voxels = np.unique(occupied_voxels, axis=0)
        
        self.map[free_voxels[:,0], free_voxels[:,1], free_voxels[:,2]] -= L_FREE
        self.map[occupied_voxels[:,0], occupied_voxels[:,1], occupied_voxels[:,2]] += L_OCCUPIED
        