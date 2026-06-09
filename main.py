import argparse

import boids
from config import Settings


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Boid simulation")

    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--screen-size", type=int, nargs=2, metavar=("W", "H"), default=None)
    parser.add_argument("--grid-tile-size", type=int, default=None)
    parser.add_argument("--border-walls", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--obstacle-move-time", type=int, default=None)
    parser.add_argument("--n-obstacles", type=int, default=None)
    parser.add_argument("--n-boids", type=int, default=None)
    parser.add_argument("--boid-color-rot-rate", type=float, default=None)
    parser.add_argument("--boid-separation-distance", type=int, default=None)
    parser.add_argument("--boid-size", type=int, default=None)
    parser.add_argument("--boid-sight", type=int, default=None)
    parser.add_argument("--boid-speed", type=int, default=None)
    parser.add_argument("--boid-obst-avoid-arc", type=float, default=None)
    parser.add_argument("--boid-path-trace", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--boid-path-trace-segments", type=int, default=None)
    parser.add_argument("--w-vector", type=float, default=None)
    parser.add_argument("--w-separation", type=float, default=None)
    parser.add_argument("--w-alignment", type=float, default=None)
    parser.add_argument("--w-cohesion", type=float, default=None)
    parser.add_argument("--w-obstacle", type=float, default=None)
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None)

    a = parser.parse_args()

    return Settings(
        fps=a.fps,
        screen_size=tuple(a.screen_size) if a.screen_size else None,
        grid_tile_size=a.grid_tile_size,
        border_walls=a.border_walls,
        obstacle_move_time=a.obstacle_move_time,
        n_obstacles=a.n_obstacles,
        n_boids=a.n_boids,
        boid_color_rot_rate=a.boid_color_rot_rate,
        boid_seperation_distance=a.boid_separation_distance,
        boid_size=a.boid_size,
        boid_sight=a.boid_sight,
        boid_speed=a.boid_speed,
        boid_obst_avoid_arc=a.boid_obst_avoid_arc,
        boid_path_trace=a.boid_path_trace,
        boid_path_trace_segments=a.boid_path_trace_segments,
        w_vector=a.w_vector,
        w_separation=a.w_separation,
        w_alignment=a.w_alignment,
        w_cohesion=a.w_cohesion,
        w_obstacle=a.w_obstacle,
        debug=a.debug,
    )


if __name__ == "__main__":
    settings = parse_args()
    boids.run(settings)
