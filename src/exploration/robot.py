# Update with velocity etc.

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