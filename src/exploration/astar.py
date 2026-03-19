import heapq
from math       import inf, sqrt, isfinite

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
    return abs(node1.row-node2.row) + abs(node1.col-node2.col)

# Compute the absolute slope between two neighboring cells.
def slope_between(node, neighbor, elevation_map, resolution):
    dz = elevation_map[neighbor.row, neighbor.col] - elevation_map[node.row, node.col]
    return abs(dz) / resolution

# Check whether the robot can legally move between two neighboring cells.
def traversable(node, neighbor, elevation_map, resolution, max_slope):
    src_height = elevation_map[node.row, node.col]
    dst_height = elevation_map[neighbor.row, neighbor.col]

    if not (isfinite(src_height) and isfinite(dst_height)):
        return True

    return slope_between(node, neighbor, elevation_map, resolution) <= max_slope

# Actual cost from node to it's neighbor.
def costtoneighbor(node, neighbor, elevation_map):
    elevation_cost = (elevation_map[neighbor.row, neighbor.col] - elevation_map[node.row, node.col])
    if (elevation_cost < 0):
        return 1 # no elev cost for downhill
    return sqrt(1 + (20 * elevation_cost)**2)

# Estimate the cost to go from state to goal.
def costtogoest(node, goal):
    return deltacost(node, goal)

# Run the planner.
def planner(start, goal, elevation_map, generated_map, show=None):
    resolution = getattr(generated_map, "resolution", 1.0)
    max_slope = getattr(generated_map, "MAX_SLOPE", None)
    if max_slope is None:
        max_slope = getattr(generated_map, "max_slope", 10)

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
            if not traversable(node, neighbor, elevation_map, resolution, max_slope):
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

    if start_node != goal_node and goal_node.parent is None:
        return []

    path = []
    current = goal_node
    while current.parent:
        path.insert(0, (current.col, current.row))
        current = current.parent
    path.insert(0, (start_node.col, start_node.row))
    return path
