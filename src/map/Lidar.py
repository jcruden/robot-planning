import numpy as np
from dataclasses import dataclass
from math import pi



@dataclass
class LaserScan:
    """Scan result with range and elevation data."""
    angle_min:       float          
    angle_max:       float          
    angle_increment: float          
    range_min:       float          
    range_max:       float          
    ranges:          np.ndarray     
    thetas:          np.ndarray    
    elevations:      np.ndarray     
                                    
    hit_points:      np.ndarray     
                                    


# ───────────────────────────────────────────────────────────────────
#  Lidar class
# ───────────────────────────────────────────────────────────────────
class Lidar:
    """Simulated 3D LiDAR that raycasts against an elevation grid."""

    

    def __init__(
        self,
        ground_truth:     np.ndarray,
        resolution:       float = 0.05,
        robot_height:     float = 0.0,
        angle_min:        float = 0.0,
        angle_max:        float = 2 * pi,
        angle_increment:  float = 2 * pi / 360,
        elevation_angles: list  = None,
        range_min:        float = 0.12,
        range_max:        float = 3.5,
        ray_step:         float = None,
        noise_std:        float = 0.01,
        seed:             int   = None,
        origin_lower:     bool  = True,
    ):
        self.ground_truth = ground_truth
        self.resolution   = resolution
        self.robot_height = robot_height
        self.rows, self.cols = ground_truth.shape

        self.world_width  = self.cols * resolution
        self.world_height = self.rows * resolution

        self.angle_min       = angle_min
        self.angle_max       = angle_max
        self.angle_increment = angle_increment
        self.range_min       = range_min
        self.range_max       = range_max
        self.ray_step        = ray_step if ray_step else resolution
        self.noise_std       = noise_std
        self.origin_lower    = origin_lower

        if elevation_angles is None:
            self.elevation_angles = [-0.3, -0.15, -0.05, 0.0, 0.05, 0.15]
        else:
            self.elevation_angles = list(elevation_angles)

        self._rng = np.random.default_rng(seed)
        self.thetas = np.arange(angle_min, angle_max, angle_increment)

    
    def scan(self, x: float, y: float, theta: float = 0.0) -> LaserScan:
        """Perform a full lidar sweep."""

        robot_z = self._get_elevation(x, y) + self.robot_height
        n = len(self.thetas)

        ranges     = np.full(n, self.range_max)
        elevations = np.full(n, np.nan)
        hit_points = np.full((n, 2), np.nan)

        for i, azimuth in enumerate(self.thetas):
            world_azimuth = azimuth + theta
            best_range    = self.range_max
            highest_elev  = -np.inf
            highest_hit   = None
            any_hit       = False

            for elev in self.elevation_angles:
                r, hx, hy = self._cast_ray(x, y, robot_z, world_azimuth, elev)
                if r < self.range_max and not np.isnan(hx):
                    any_hit = True
                    # Track closest hit for range
                    if r < best_range:
                        best_range = r
                    # Track highest elevation across all hits
                    hit_z = self._get_elevation(hx, hy)
                    if hit_z > highest_elev:
                        highest_elev = hit_z
                        highest_hit  = (hx, hy)

            ranges[i] = best_range
            if any_hit:
                elevations[i] = highest_elev
                hit_points[i] = highest_hit

        # Add noise to ranges
        if self.noise_std > 0.0:
            noise = self._rng.normal(0.0, self.noise_std, size=ranges.shape)
            ranges = np.clip(ranges + noise, 0.0, self.range_max)

        # Add noise to elevation readings
        valid = ~np.isnan(elevations)
        if self.noise_std > 0.0 and np.any(valid):
            z_noise = self._rng.normal(0.0, self.noise_std, size=elevations[valid].shape)
            elevations[valid] += z_noise

        return LaserScan(
            angle_min       = self.angle_min,
            angle_max       = self.angle_max,
            angle_increment = self.angle_increment,
            range_min       = self.range_min,
            range_max       = self.range_max,
            ranges          = ranges,
            thetas          = self.thetas,
            elevations      = elevations,
            hit_points      = hit_points,
        )

    def set_ground_truth(self, ground_truth: np.ndarray) -> None:
        """Replace the ground truth elevation grid at runtime."""
        self.ground_truth = ground_truth
        self.rows, self.cols = ground_truth.shape
        self.world_width  = self.cols * self.resolution
        self.world_height = self.rows * self.resolution

    
    def _get_elevation(self, x: float, y: float) -> float:
        """interpolated elevation at world coords (x, y)."""
        col = x / self.resolution
        row = y / self.resolution if self.origin_lower else (self.world_height - y) / self.resolution

        c0 = int(np.floor(col))
        r0 = int(np.floor(row))
        c1 = c0 + 1
        r1 = r0 + 1

        if c0 < 0 or r0 < 0 or c1 >= self.cols or r1 >= self.rows:
            return 0.0

        dc = col - c0
        dr = row - r0

        z = (self.ground_truth[r0, c0] * (1 - dc) * (1 - dr) +
             self.ground_truth[r0, c1] * dc * (1 - dr) +
             self.ground_truth[r1, c0] * (1 - dc) * dr +
             self.ground_truth[r1, c1] * dc * dr)

        return float(z)

    def _cast_ray(self, x0, y0, z0, azimuth, elevation):
        """March a ray and return (range, hit_x, hit_y)"""

      
        cos_az = np.cos(azimuth)
        sin_az = np.sin(azimuth)
        cos_el = np.cos(elevation)
        sin_el = np.sin(elevation)

        dx = cos_el * cos_az
        dy = cos_el * sin_az
        dz = sin_el

        step = self.ray_step
        start_step = max(1, int(self.range_min / step))
        max_steps  = int(self.range_max / step) + 1

        for s in range(start_step, max_steps + 1):
            dist = s * step
            rx = x0 + dx * dist
            ry = y0 + dy * dist
            rz = z0 + dz * dist

            if rx < 0 or rx >= self.world_width or ry < 0 or ry >= self.world_height:
                return (self.range_max, np.nan, np.nan)

            terrain_z = self._get_elevation(rx, ry)

            if rz <= terrain_z:
                # Refine hit with linear interpolation
                prev_dist    = max((s - 1) * step, self.range_min)
                prev_rz      = z0 + dz * prev_dist
                prev_terrain = self._get_elevation(x0 + dx * prev_dist,
                                                    y0 + dy * prev_dist)
                gap_prev = prev_rz - prev_terrain
                gap_curr = terrain_z - rz
                denom    = gap_prev + gap_curr

                if denom > 0:
                    t = gap_prev / denom
                    refined = prev_dist + t * step
                else:
                    refined = dist

                refined = max(refined, self.range_min)
                hx = x0 + dx * refined
                hy = y0 + dy * refined
                return (refined, hx, hy)

        return (self.range_max, np.nan, np.nan)
