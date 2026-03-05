import numpy as np

# Update
RESOLUTION = 0.05
LFREE     = -0.03
LOCCUPIED =  0.3

class Generated_Map():
    #################
    # Initialization in m and resolution in m/cell i.e. 0.1:
    def __init__(self, width, length, resolution):
        # Define
        self.width = width
        self.length = length
        self.logoddsratio = np.zeros((int(length / resolution), int(width / resolution)))

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