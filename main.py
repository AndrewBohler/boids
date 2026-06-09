import argparse

import boids
from config import Settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boid simulation")

    # get defaults for help text
    s = Settings()

    parser.add_argument("--fps", type=int, default=None,
        help=f"Target frames per second (default: {s.fps})")
    parser.add_argument("--screen-size", type=int, nargs=2, metavar=("W", "H"), default=None,
        help=f"Window size in pixels (default: {s.screen_size[0]} {s.screen_size[1]})")
    parser.add_argument("--grid-tile-size", type=int, default=None,
        help=f"Spatial grid tile size in pixels (default: {s.grid_tile_size})")
    parser.add_argument("--border-walls", action=argparse.BooleanOptionalAction, default=None,
        help=f"Enable/disable border walls (default: {s.border_walls})")
    parser.add_argument("--obstacle-move-time", type=int, default=None,
        help=f"Seconds between obstacle relocations, 0 to disable (default: {s.obstacle_move_time})")
    parser.add_argument("--n-obstacles", type=int, default=None,
        help=f"Number of obstacles (default: {s.n_obstacles})")
    parser.add_argument("--n-boids", type=int, default=None,
        help=f"Number of boids (default: {s.n_boids})")
    parser.add_argument("--boid-color-rot-rate", type=float, default=None,
        help=f"Color rotation rate in radians per frame (default: {s.boid_color_rot_rate:.6f})")
    parser.add_argument("--boid-separation-distance", type=int, default=None,
        help=f"Distance at which boids repel each other (default: {s.boid_separation_dist})")
    parser.add_argument("--boid-size", type=int, default=None,
        help=f"Boid size in pixels (default: {s.boid_size})")
    parser.add_argument("--boid-sight", type=int, default=None,
        help=f"Boid sight radius in pixels (default: {s.boid_sight})")
    parser.add_argument("--boid-speed", type=int, default=None,
        help=f"Boid speed in pixels per frame (default: {s.boid_speed})")
    parser.add_argument("--boid-obst-avoid-arc", type=float, default=None,
        help=f"Obstacle avoidance arc in radians (default: {s.boid_obst_avoid_arc:.6f})")
    parser.add_argument("--boid-path-trace", action=argparse.BooleanOptionalAction, default=None,
        help=f"Enable/disable path trail rendering (default: {s.boid_path_trace})")
    parser.add_argument("--boid-path-segments", type=int, default=None,
        help=f"Number of path trail segments (default: {s.boid_path_segments})")
    parser.add_argument("--w-vector", type=float, default=None,
        help=f"Rule weight: current heading (default: {s.w_vector})")
    parser.add_argument("--w-separation", type=float, default=None,
        help=f"Rule weight: separation from nearby boids (default: {s.w_separation})")
    parser.add_argument("--w-alignment", type=float, default=None,
        help=f"Rule weight: alignment with nearby boids (default: {s.w_alignment})")
    parser.add_argument("--w-cohesion", type=float, default=None,
        help=f"Rule weight: cohesion toward nearby boids (default: {s.w_cohesion})")
    parser.add_argument("--w-obstacle", type=float, default=None,
        help=f"Rule weight: obstacle avoidance (default: {s.w_obstacle})")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=None,
        help=f"Enable debug overlay and timing output (default: {s.debug})")

    ret = parser.parse_args()

    return ret


if __name__ == "__main__":
    a = parse_args()
    settings = Settings(
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
        boid_path_segments=a.boid_path_segments,
        w_vector=a.w_vector,
        w_separation=a.w_separation,
        w_alignment=a.w_alignment,
        w_cohesion=a.w_cohesion,
        w_obstacle=a.w_obstacle,
        debug=a.debug,
    )
    boids.run(settings)
