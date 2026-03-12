# Update with velocity etc.
import pygame

class Robot():
    #################
    def __init__(self, x, y, generated_map, lidar):
        # Define
        self.x = x
        self.y = y
        self.generated_map = generated_map
        self.lidar = lidar

    def sensor_update(self, x, y):
        # Update generated map with lidar data
        return
    
    def move(self, x, y):
        self.x = x
        self.y = y

    def draw(self, screen):
        pygame.draw.circle(screen, (255, 0, 0), (self.x, self.y), 5)