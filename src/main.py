'''prmtriangles_updatedsolution.py

   This is the updated PRM solution code for the 2D triangular problem
   with post-processing and alternate connections.

   Use the PRM algorithm to find a path around polygonal obstacles.

'''

import bisect
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from math               import inf, pi, sin, cos, sqrt, ceil, dist

from shapely.geometry   import Point, LineString, Polygon, MultiPolygon
from shapely.prepared   import prep

from map import generated_map
from exploration import robot

xmin = 0
xmax = 10
ymin = 0
ymax = 10

######################################################################
#
#   Visualization Class
#
#   This renders the world.  In particular it provides the methods:
#     show(text = '')                   Show the current figure
#     drawNode(node,         **kwargs)  Draw a single node
#     drawEdge(node1, node2, **kwargs)  Draw an edge between nodes
#     drawPath(path,         **kwargs)  Draw a path (list of nodes)
#
class Visualization:
    def __init__(self):
        # Clear the current, or create a new figure.
        plt.clf()

        # Create a new axes, enable the grid, and set axis limits.
        plt.axes()
        plt.grid(True)
        plt.gca().axis('on')
        plt.gca().set_xlim(xmin, xmax)
        plt.gca().set_ylim(ymin, ymax)
        plt.gca().set_aspect('equal')

        # Show immediately.
        self.show()

    def show(self, text = ''):
        # Show the plot.
        plt.pause(0.001)
        # If text is specified, print and wait for confirmation.
        if len(text)>0:
            input(text + ' (hit return to continue)')

    def drawRobot(self, robot, **kwargs):
        self.circle = plt.Circle((robot.x, robot.y), 0.05, color='r', fill=True)
        plt.gca().add_patch(self.circle)
        plt.ion()

    def updateRobot(self, robot, **kwargs):
        self.circle.center = (robot.x, robot.y)

    #def drawMap(self, generated_map, **kwargs):
        # draw heat map

    def drawNode(self, node, **kwargs):
        plt.plot(node.x, node.y, **kwargs)

    def drawEdge(self, head, tail, **kwargs):
        plt.plot([head.x, tail.x], [head.y, tail.y], **kwargs)

    def drawPath(self, path, **kwargs):
        for i in range(len(path)-1):
            self.drawEdge(path[i], path[i+1], **kwargs)

######################################################################
#
#  Main Code
#
def main():

    # Create the figure.  Some computers seem to need an additional show()?
    visual = Visualization()
    visual.show()

    # Create map and robot
    gen_map = generated_map.Generated_Map(10, 10, 0.1)
    rob = robot.Robot(1, 1, gen_map, None)

    # Show robot and map
    visual.drawRobot(rob)

    start = time.time()
    t = time.time()
    while t - start < 10:
        # Update robot position
        rob.move(rob.x + 0.1, rob.y + 0.1)

        # Update map with lidar
        rob.sensor_update(rob.x, rob.y)

        # Show robot and map
        visual.updateRobot(rob)
        visual.show()
        time.sleep(1)
        t = time.time()


if __name__== "__main__":
    main()
