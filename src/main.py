'''prmtriangles_updatedsolution.py

   This is the updated PRM solution code for the 2D triangular problem
   with post-processing and alternate connections.

   Use the PRM algorithm to find a path around polygonal obstacles.

'''

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame.locals import *
import sys
import random

from map import generated_map
from map import viz
from map.Lidar import Lidar
from exploration import robot

WIDTH, HEIGHT = 1000, 1000

    
def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    gen_map = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    lidar = Lidar(grid, world_resolution=viz.resolution)
    rob = robot.Robot(1, 1, gen_map, lidar)
    map_surf = viz.viz_surface()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        x, y = random.uniform(1, 9), random.uniform(1, 9)
        rob.moveTo(x, y)

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(map_surf, rob)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        input("Hit enter for new position: ")
    
    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
