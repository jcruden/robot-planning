# Update with velocity etc.
import pygame
import numpy as np

class Robot():
    #################
    def __init__(self, x, y, generated_map, lidar):
        # Define
        self.x = x
        self.y = y
        self.generated_map = generated_map
        self.lidar = lidar
        self.last_scan = None

    def sensor_update(self):
        if self.lidar is None or self.generated_map is None:
            return

        scan = self.lidar.scan(self.x, self.y)
        self.last_scan = scan
        self.generated_map.updateelevation(self.x, self.y, scan.elevations, scan.hit_points)
    
    def moveTo(self, x, y):
        self.x = x
        self.y = y
    
    def move(self, vx, vy):
        self.x += vx
        self.y += vy