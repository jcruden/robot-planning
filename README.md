# robot-planning

Simple robot exploration + planning demos on a 2.5D elevation grid (pygame + numpy). Includes a simulated 3D LiDAR, frontier selection, and an A* path planner that accounts for uphill cost.

## Run

From the repo root:

```bash
python3 -m pip install numpy pygame matplotlib numba
python3 src/robotnav.py
```

- `src/robotnav.py` runs **two robots** on the same map:
  - Robot 1: `random=True` (selects the best-scoring frontier under the “random” scoring mode)
  - Robot 2: `random=False` (selects the best frontier using the weighted elevation score)

There are other entrypoints in `archive/` (for older/alternate demos), but `robotnav.py` is the most up-to-date runner.

## Key files

- `src/exploration/robot.py`: robot behavior, frontier detection/scoring, path following, and scoring.
- `src/exploration/astar.py`: A* grid planner with an uphill movement penalty.
- `src/map/improvedLidar.py`: simulated 3D LiDAR raycasting against an elevation grid.
- `src/map/generated_map.py`: “known” map built incrementally from LiDAR returns.
- `src/map/viz.py`: pygame/matplotlib visualization helpers.
- `src/map/final_square_map.csv`: ground-truth elevation grid used by the simulator.

## Useful knobs

- **LiDAR noise**: `noise_std` in `src/map/improvedLidar.py` is the standard deviation (σ) of Gaussian noise added to each measured range, then clamped to `[range_min, range_max]`. Example usage is in `src/robotnav.py`:
  - `Lidar(grid, world_resolution=viz.resolution, noise_std=0.2)`

- **Frontier policy**: in `src/exploration/robot.py`, `Robot(..., random=True/False)` toggles which frontier scoring rule is used.
