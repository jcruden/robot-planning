'''prmtriangles_updatedsolution.py

   This is the updated PRM solution code for the 2D triangular problem
   with post-processing and alternate connections.

   Use the PRM algorithm to find a path around polygonal obstacles.

'''

import bisect
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import time
import pygame
import sys

from math               import inf, pi, sin, cos, sqrt, ceil, dist

from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep

from map import generated_map
from exploration import robot

xmin = 0
xmax = 10
ymin = 0
ymax = 10

WIDTH, HEIGHT = 800, 800

def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    clock = pygame.time.Clock()
    x, y = 100, 100
    vx, vy = 2, 1

    rob = robot.Robot(x, y, generated_map, None)

    running = True
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        x += vx
        y += vy
        if (x > WIDTH or y > HEIGHT or x < 0 or y < 0):
            vx = -vx
            vy = -vy

        screen.fill((30, 30, 30))
        #screen.blit(surf, (0, 0))
        rob.move(x, y)
        rob.draw(screen)

        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
