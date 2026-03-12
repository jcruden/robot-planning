'''prmtriangles_updatedsolution.py

   This is the updated PRM solution code for the 2D triangular problem
   with post-processing and alternate connections.

   Use the PRM algorithm to find a path around polygonal obstacles.

'''

import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import sys
import random

from map import generated_map
from map import viz
from exploration import robot

WIDTH, HEIGHT = 800, 800

def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    clock = pygame.time.Clock()
    x, y = 1, 1
    vx, vy = 0.1, 0.09

    rob = robot.Robot(x, y, generated_map, None)
    map_surf = viz.viz_surface()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        x += vx
        y += vy
        if (x >= viz.width_m or x <= 0):
            vx = -vx
        if (y >= viz.height_m or y <= 0):
            vy = -vy

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(map_surf, rob)
        screen.blit(surf, (0, 0))
        rob.move(x, y)
        #rob.draw(screen)

        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
