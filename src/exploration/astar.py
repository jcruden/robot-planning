'''gridplanner_solution.py

   This is the solution code for the grid planner, using Dijstra's and
   the Astar algorithms.  Simply change the variable cost flag and
   cost-to-go factor below.

   This finds a path in a simple 2D grid.

'''

import bisect
import heapq

from math       import inf, sqrt

#
#   Define the CONFIGURATION
#
VARIABLECOST = False    # False (P1, P3) or True (P2)


#
#   Colors
#
WHITE  = [1.000, 1.000, 1.000]
BLACK  = [0.000, 0.000, 0.000]
RED    = [1.000, 0.000, 0.000]
BARK   = [0.325, 0.192, 0.094]  # bark brown
GREEN  = [0.133, 0.545, 0.133]  # forrest green
SKY    = [0.816, 0.925, 0.992]  # light blue


#
#   Node Class
#
#   We create one node to match each valid state being the robot
#   occupying a valid space in the grid.  That is, one node per
#   unblocked grid element (with a row/column).
#
#   To encode the graph, we also note a list of accessible neighbors.
#   And, as part of the search, we store the parent (in the tree), the
#   cost to reach this node (via the tree), and the status flags.
#
class Node:
    # Initialization
    def __init__(self, row, col):
        # Save the matching state.
        self.row = row
        self.col = col

        # Clear the list of neighbors (used for the full graph).
        self.neighbors = []

        # Clear the parent (used for the search tree), as well as the
        # actual cost to reach (via the parent) and the estimated
        # total path cost including the estimated cost to the goal.
        self.parent = None      # No parent
        self.creach = inf       # Unable to reach = infinite cost
        self.cost   = inf       # Estimated total path cost

        # State of the node during the search algorithm.
        self.seen = False
        self.done = False


    # Define the Manhattan distance to another node.
    def distance(self, other):
        return abs(self.row - other.row) + abs(self.col - other.col)

    # Define the "less-than" to enable sorting by cost.
    def __lt__(self, other):
        return self.cost < other.cost


    # Print (for debugging).
    def __str__(self):
        return("(%2d,%2d)" % (self.row, self.col))
    def __repr__(self):
        return("<Node %s, %7s, cost %f>" %
               (str(self),
                "done" if self.done else "seen" if self.seen else "unknown",
                self.cost))


#
#   Search/Planner Algorithm
#
#   This is the core algorithm.  It builds a search tree inside the
#   node graph, transfering nodes from air (not seen) to leaf (seen,
#   but not done) to trunk (done).
#
# Compute a delta cost between two nodes.
def deltacost(node1, node2):
    return sqrt((node1.row-node2.row)**2 + (node1.col-node2.col)**2)

# Actual cost from node to it's neighbor.
def costtoneighbor(node, neighbor, elevation_map):
    elevation_cost = abs(elevation_map[neighbor.row, neighbor.col] - elevation_map[node.row, node.col])
    return sqrt((node.row-neighbor.row)**2 + (node.col-neighbor.col)**2 + elevation_cost**2)

# Estimate the cost to go from state to goal.
def costtogoest(node, goal):
    return deltacost(node, goal)

# Run the planner.
def planner(start, goal, elevation_map, generated_map, show=None):
    # Create nodes for the grid
    rows, cols = elevation_map.shape
    nodes = [[Node(row, col) for col in range(cols)] for row in range(rows)]

    # Set up neighbors for each node
    for row in range(rows):
        for col in range(cols):
            node = nodes[row][col]
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = row + dr, col + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    node.neighbors.append(nodes[nr][nc])

    # Convert start and goal to Node objects
    start_node = nodes[start[1]][start[0]]
    goal_node = nodes[goal[1]][goal[0]]

    # Use the start node to initialize the on-deck priority queue
    start_node.seen = True
    start_node.creach = 0
    start_node.cost = 0 + costtogoest(start_node, goal_node)
    start_node.parent = None
    onDeck = []
    heapq.heappush(onDeck, (start_node.cost, start_node))

    while onDeck:
        # Grab the next state (smallest cost from the priority queue)
        _, node = heapq.heappop(onDeck)
        node.done = True

        if node == goal_node:
            break

        for neighbor in node.neighbors:
            if neighbor.done:
                continue

            creach = node.creach + costtoneighbor(node, neighbor, elevation_map)

            if neighbor.seen:
                if neighbor.creach <= creach:
                    continue
                else:
                    # Remove the neighbor from the priority queue
                    onDeck = [(c, n) for c, n in onDeck if n != neighbor]
                    heapq.heapify(onDeck)

            neighbor.seen = True
            neighbor.creach = creach
            neighbor.cost = creach + costtogoest(neighbor, goal_node)
            neighbor.parent = node
            heapq.heappush(onDeck, (neighbor.cost, neighbor))

    path = []
    current = goal_node
    while current.parent:
        path.insert(0, (current.col, current.row))
        current = current.parent
    path.insert(0, (start_node.col, start_node.row))
    return path