import matplotlib
matplotlib.use("Agg")
import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt
import pygame
from pygame.locals import *
import sys
import numpy as np

from map import generated_map
from map import viz
from map.improvedLidar import Lidar
from exploration import robot

WIDTH, HEIGHT = 600, 900

def main():
    pygame.init()

    window = pygame.display.set_mode((WIDTH, HEIGHT))
    screen = pygame.display.get_surface()
    pygame.display.set_caption("Robot Planning")

    clock = pygame.time.Clock()

    grid = np.genfromtxt('src/map/final_square_map.csv', delimiter=',')
    gen_map = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    gen_map2 = generated_map.Generated_Map(viz.width_m, viz.height_m, viz.resolution)
    lidar = Lidar(grid, world_resolution=viz.resolution, noise_std=0.1)
    rob = robot.Robot(1, 1, gen_map, lidar, grid, random = True)
    lidar2 = Lidar(grid, world_resolution=viz.resolution, noise_std=0.1)
    rob2 = robot.Robot(1, 1, gen_map2, lidar2, grid, random=False)
    interval = 20
    last_time = pygame.time.get_ticks()
    rob.sensor_update()
    rob2.sensor_update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        current_time = pygame.time.get_ticks()
        if current_time - last_time >= interval:
            if (rob.fuel > 0):
                rob.sensor_update()
                rob.move_path()
            if (rob2.fuel > 0):
                rob2.sensor_update()
                rob2.move_path()
            if (rob.fuel <= 0 and rob2.fuel <= 0):
                running = False
            last_time = current_time

        screen.fill((30, 30, 30))
        surf = viz.draw_robot(rob, rob2)
        screen.blit(surf, (0, 0))

        pygame.display.flip()
        clock.tick(30)
    
    print("Robot 1 (random): ", int(100 * rob.calculate_score()), " / 100")
    print("Robot 2 (closest): ", int(100 * rob2.calculate_score()), " / 100")
    input("Hit enter to end:")

    pygame.quit()
    sys.exit()

if __name__== "__main__":
    main()
