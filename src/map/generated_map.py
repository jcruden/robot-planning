import numpy as np

# Update
RESOLUTION = 0.05
LFREE     = -0.03
LOCCUPIED =  0.3
VAR = 0.1

class Generated_Map():
    #################
    # Initialization in m and resolution in m/cell i.e. 0.1:
    def __init__(self, width, length, resolution):
        # Define
        self.width = width
        self.length = length
        self.height = length
        self.resolution = resolution
        self.logoddsratio = np.zeros((int(length / resolution), int(width / resolution)))

        self.elevationmean = np.full((int(length/resolution), int(width / resolution)),np.nan)
        self.elevationvar = np.full((int(length/resolution), int(width / resolution)),np.nan)

    def get_elevation(self):
        return self.elevationmean

    def get_elevation_var(self):
        return self.elevationvar

    def set(self, u, v, value):
        # Update only if legal.
        if (u>=0) and (u<self.width) and (v>=0) and (v<self.height):
            self.logoddsratio[v,u] = value
        else:
            print("Out of bounds (%d, %d)" % (u,v))

    # Adjust the log odds ratio value
    def adjust(self, u, v, delta):
        # Update only if legal.
        if (u>=0) and (u<self.width) and (v>=0) and (v<self.height):
            self.logoddsratio[v,u] += delta
        else:
            print("Out of bounds (%d, %d)" % (u,v))

    # Return a list of all intermediate (integer) pixel coordinates
    # from (start) to (end) pixel coordinates (possibly non-integer).
    # In classic Python fashion, this excludes the end coordinates.
    def bresenham(self, start, end):
        # Extract the coordinates
        (us, vs) = start
        (ue, ve) = end

        # Move along ray (excluding endpoint).
        if (np.abs(ue-us) >= np.abs(ve-vs)):
            return[(u, int(vs + (ve-vs)/(ue-us) * (u+0.5-us)))
                   for u in range(int(us), int(ue), int(np.sign(ue-us)))]
        else:
            return[(int(us + (ue-us)/(ve-vs) * (v+0.5-vs)), v)
                   for v in range(int(vs), int(ve), int(np.sign(ve-vs)))]
        
    # Update the occupancy grid (log odds ratio).
    def updateoccupancy(self, xc, yc, thetac, thetas, ranges, rmin, rmax):
        # Process each ray (at a different angle)
        for (theta,r) in zip(thetas, ranges):
            # Make sure the ray is valid (greater than the minimum).
            if (r <= rmin):
                continue

            # Grab the slope.
            s  = np.sin(theta + thetac)
            c  = np.cos(theta + thetac)

            # Start/End points, shifted/scaled into the grid coordinates.
            rend = min(r, rmax)
            us = (xc + rmin*c) / RESOLUTION
            vs = (yc + rmin*s) / RESOLUTION
            ue = (xc + rend*c) / RESOLUTION
            ve = (yc + rend*s) / RESOLUTION

            # Move along ray (excluding endpoint) to release the free space.
            for (u,v) in self.bresenham((us,vs), (ue,ve)):
                self.adjust(u, v, LFREE)

            # Update the endpoint, either free space or occupied.
            if (r >= rmax):
                self.adjust(int(ue), int(ve), LFREE)
            else:
                self.adjust(int(ue), int(ve), LOCCUPIED)

    def updateelevation(self, x, y, elevations, hit_points, lidar_var = None):
        max_heights = {}

        for i in range(len(elevations)):
            elev = elevations[i]
            if np.isnan(elev):
                continue

            hx, hy = hit_points[i]
            if np.isnan(hx) or np.isnan(hy):
                continue

            # Convert world coords to grid cell
            u = int(hx / RESOLUTION)
            v = int(hy / RESOLUTION)

            # Bounds check
            if u < 0 or u >= self.elevationmean.shape[1]:
                continue
            if v < 0 or v >= self.elevationmean.shape[0]:
                continue

            # Add highest elev for cell
            if (u, v) not in max_heights or elev > max_heights[(u, v)]:
                max_heights[(u, v)] = elev
        
        for (u, v), elev in max_heights.items():
            if np.isnan(self.elevationmean[v, u]):
                self.elevationmean[v, u] = elev
                self.elevationvar[v, u] = VAR
            else:
                if lidar_var is None:
                    lidar_var = VAR
                # Update mean and stdev with Kalman filter
                old_mean = self.elevationmean[v, u]
                old_var = self.elevationvar[v, u]
                k = old_var / (old_var + lidar_var)
                new_mean = old_mean + k * (elev - old_mean)
                new_var = (1 - k) * old_var
                self.elevationmean[v, u] = new_mean
                self.elevationvar[v, u] = new_var