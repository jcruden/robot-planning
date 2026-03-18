import numpy as np
from dataclasses import dataclass
from math import pi
from numba import njit

@dataclass
class LaserScan:
    """Scan result with range and elevation data."""
    angle_min:       float          
    angle_max:       float
    vertical_min:    float
    vertical_max:    float          
    angle_increment: float          
    range_min:       float          
    range_max:       float          
    ranges:          np.ndarray     
    thetas:          np.ndarray 
    deltas:          np.ndarray   
    elevations:      np.ndarray                                           
    hit_points:      np.ndarray
    free:            np.ndarray # Changed from set to a NumPy array of unique (x,y,z) coordinates


# -----------------------------------------------------------------------------
# JIT-Compiled Core Functions (Must be outside the class to avoid 'self')
# -----------------------------------------------------------------------------

@njit(fastmath=True)
def _get_elevation_fast(x, y, ground_truth, world_res, cols, rows, world_height, origin_lower):
    """JIT-compiled bilinear interpolation."""
    col = x / world_res
    row = y / world_res if origin_lower else (world_height - y) / world_res

    c0 = int(np.floor(col))
    r0 = int(np.floor(row))
    c1 = c0 + 1
    r1 = r0 + 1

    if c0 < 0 or r0 < 0 or c1 >= cols or r1 >= rows:
        return 0.0

    dc = col - c0
    dr = row - r0

    z = (ground_truth[r0, c0] * (1 - dc) * (1 - dr) +
         ground_truth[r0, c1] * dc * (1 - dr) +
         ground_truth[r1, c0] * (1 - dc) * dr +
         ground_truth[r1, c1] * dc * dr)

    return float(z)


@njit(fastmath=True)
def _sweep_fast(
    x0, y0, z0, theta, thetas, deltas, ground_truth, world_res, cols, rows,
    world_width, world_height, max_elev, origin_lower, grid_res, 
    range_min, range_max, ray_step, noise_std
):
    """JIT-compiled sweep separating physical collision from perceived free space."""
    n = len(thetas)
    k = len(deltas)
    
    ranges = np.full(n * k, range_max)
    elevations = np.full(n * k, np.nan)
    hit_points = np.full((n * k, 2), np.nan)
    
    start_step = max(1, int(range_min / ray_step))
    max_steps = int(range_max / ray_step) + 1
    
    max_possible_free = n * k * max_steps
    free_spaces = np.zeros((max_possible_free, 3), dtype=np.int32)
    free_count = 0

    for i in range(n):
        world_azimuth = thetas[i] + theta
        cos_az = np.cos(world_azimuth)
        sin_az = np.sin(world_azimuth)

        for j in range(k):
            idx = i * k + j
            elevation = deltas[j]
            cos_el = np.cos(elevation)
            sin_el = np.sin(elevation)

            dx = cos_el * cos_az
            dy = cos_el * sin_az
            dz = sin_el

            # ==========================================
            # PHASE 1: Find Ground Truth Collision
            # ==========================================
            true_range = range_max
            hit_terrain = False

            for s in range(start_step, max_steps + 1):
                dist = s * ray_step
                rx = x0 + dx * dist
                ry = y0 + dy * dist
                rz = z0 + dz * dist

                # Stop if the ray leaves the simulation boundaries
                if rx < 0 or rx >= world_width or ry < 0 or ry >= world_height or rz < 0 or rz > max_elev - grid_res:
                    break 

                terrain_z = _get_elevation_fast(rx, ry, ground_truth, world_res, cols, rows, world_height, origin_lower)

                if rz <= terrain_z:
                    hit_terrain = True
                    
                    # Refine true hit with linear interpolation
                    prev_dist = max((s - 1) * ray_step, range_min)
                    prev_rz = z0 + dz * prev_dist
                    prev_terrain = _get_elevation_fast(x0 + dx * prev_dist, y0 + dy * prev_dist, 
                                                       ground_truth, world_res, cols, rows, world_height, origin_lower)
                    
                    gap_prev = prev_rz - prev_terrain
                    gap_curr = terrain_z - rz
                    denom = gap_prev + gap_curr

                    if denom > 0:
                        true_range = prev_dist + (gap_prev / denom) * ray_step
                    else:
                        true_range = dist
                    
                    true_range = max(true_range, range_min)
                    break

            # ==========================================
            # PHASE 2: Apply Sensor Noise & Record Hits
            # ==========================================
            if hit_terrain:
                if noise_std > 0.0:
                    measured_range = true_range + np.random.normal(0.0, noise_std)
                    measured_range = min(max(measured_range, range_min), range_max)
                else:
                    measured_range = true_range

                ranges[idx] = measured_range
                hit_points[idx, 0] = x0 + dx * measured_range
                hit_points[idx, 1] = y0 + dy * measured_range
                elevations[idx] = z0 + dz * measured_range
            else:
                # NO HIT: The ray went into the sky or off the map.
                measured_range = range_max # We still march free space up to max range
                ranges[idx] = np.inf       # Standard ROS representation for "no return"
                hit_points[idx, 0] = np.nan
                hit_points[idx, 1] = np.nan
                elevations[idx] = np.nan

            # ==========================================
            # PHASE 3: Generate Perceived Free Space
            # ==========================================
            # (This phase remains exactly the same as before)
            perceived_steps = int(measured_range / ray_step)
            
            for s in range(start_step, perceived_steps):
                dist = s * ray_step
                rx = x0 + dx * dist
                ry = y0 + dy * dist
                rz = z0 + dz * dist
                
                free_spaces[free_count, 0] = int(np.floor(rx / grid_res))
                free_spaces[free_count, 1] = int(np.floor(ry / grid_res))
                free_spaces[free_count, 2] = int(np.ceil(rz / grid_res))
                free_count += 1

    return ranges, elevations, hit_points, free_spaces[:free_count]


# -----------------------------------------------------------------------------
# Main LiDAR Class
# -----------------------------------------------------------------------------

class Lidar:
    """Simulated 3D LiDAR that raycasts against an elevation grid."""

    def __init__(
        self,
        ground_truth:     np.ndarray,
        world_resolution: float = 0.05,
        grid_resolution:  float = 0.1,
        robot_height:     float = 0.1,
        angle_min:        float = 0.0,
        angle_max:        float = 2 * pi,
        angle_increment:  float = None,
        vertical_min:     float = -pi/6,
        vertical_max:     float = pi/6,
        range_min:        float = 0.12,
        range_max:        float = 2.5,
        ray_step:         float = None,
        noise_std:        float = 0.1,
        seed:             int   = None,
        origin_lower:     bool  = True,
    ):
        self.ground_truth = ground_truth
        self.world_resolution = world_resolution
        self.grid_resolution = grid_resolution
        self.robot_height = robot_height
        self.rows, self.cols = ground_truth.shape

        self.world_width  = self.cols * world_resolution
        self.world_height = self.rows * world_resolution
        self.max_elev = ground_truth.max()

        self.angle_min       = angle_min
        self.angle_max       = angle_max
        self.angle_increment = 2 * pi / (360 * .05 / grid_resolution) if angle_increment is None else angle_increment
        self.vertical_min    = vertical_min
        self.vertical_max    = vertical_max
        self.range_min       = range_min
        self.range_max       = range_max
        self.ray_step        = ray_step if ray_step else grid_resolution/2
        self.noise_std       = noise_std
        self.origin_lower    = origin_lower

        self._rng = np.random.default_rng(seed)
        self.thetas = np.arange(angle_min, angle_max, self.angle_increment)
        self.deltas = np.arange(vertical_min, vertical_max, self.angle_increment)

    
    def scan(self, x: float, y: float, theta: float = 0.0) -> LaserScan:
        """Perform a full lidar sweep."""
        
        # Get robot Z height
        robot_z = _get_elevation_fast(
            x, y, self.ground_truth, self.world_resolution, 
            self.cols, self.rows, self.world_height, self.origin_lower
        ) + self.robot_height

        # Run the blazing fast JIT compiled sweep (now handling noise internally)
        ranges, elevations, hit_points, raw_free = _sweep_fast(
            x, y, robot_z, theta, self.thetas, self.deltas,
            self.ground_truth, self.world_resolution, self.cols, self.rows,
            self.world_width, self.world_height, self.max_elev, self.origin_lower,
            self.grid_resolution, self.range_min, self.range_max, self.ray_step, 
            self.noise_std # Passed in here
        )
        

        return LaserScan(
            angle_min       = self.angle_min,
            angle_max       = self.angle_max,
            vertical_min    = self.vertical_min,
            vertical_max    = self.vertical_max,
            angle_increment = self.angle_increment,
            range_min       = self.range_min,
            range_max       = self.range_max,
            ranges          = ranges.reshape(-1, 1),
            thetas          = self.thetas,
            deltas          = self.deltas,
            elevations      = elevations.reshape(-1, 1),
            hit_points      = hit_points.reshape(-1, 2),
            free            = raw_free
        )

    def set_ground_truth(self, ground_truth: np.ndarray) -> None:
        """Replace the ground truth elevation grid at runtime."""
        self.ground_truth = ground_truth
        self.rows, self.cols = ground_truth.shape
        self.world_width  = self.cols * self.world_resolution
        self.world_height = self.rows * self.world_resolution
        self.max_elev = ground_truth.max()