import numpy as np


class Settings:
    def __init__(
        self,
        fps: int | None = None,
        screen_size: tuple[int, int] | None = None,
        grid_tile_size: int | None = None,
        border_walls: bool | None = None,
        obstacle_move_time: int | None = None,
        n_obstacles: int | None = None,
        n_boids: int | None = None,
        boid_color_rot_rate: float | None = None,
        boid_seperation_distance: int | None = None,
        boid_size: int | None = None,
        boid_sight: int | None = None,
        boid_speed: int | None = None,
        boid_obst_avoid_arc: float | None = None,
        boid_path_trace: bool | None = None,
        boid_path_trace_segments: int | None = None,
        w_vector: float | None = None,
        w_separation: float | None = None,
        w_alignment: float | None = None,
        w_cohesion: float | None = None,
        w_obstacle: float | None = None,
        debug: bool | None = None,
    ) -> None:
        self.FPS = fps or 30
        self.SCREENSIZE = screen_size or (500, 500)
        self.GRID_TILE_SIZE = grid_tile_size or 25 # pixels
        if border_walls is None:
            self.BORDER_WALLS = True
        else:
            self.BORDER_WALLS = border_walls
        # wont move when 0
        if obstacle_move_time is None:
            self.OBSTACLE_MOVE_TIME = 10
        else:
            self.OBSTACLE_MOVE_TIME = obstacle_move_time
        if n_obstacles is None:
            self.N_OBSTACLES = 15
        else:
            self.N_OBSTACLES = n_obstacles
        self.N_BOID = n_boids or 100
        if boid_color_rot_rate is None:
            self.BOID_COLOR_ROT_RATE = np.pi/128
        else:
            self.BOID_COLOR_ROT_RATE = boid_color_rot_rate
        self.BOID_SEPERATION_DISTANCE = boid_seperation_distance or 7
        self.BOID_SIZE = boid_size or 5
        self.BOID_SIGHT = boid_sight or 25
        self.BOID_SPEED = boid_speed or 3
        self.BOID_OBST_AVOID_ARC = boid_obst_avoid_arc or np.pi/16 # radians
        if boid_path_trace is None:
            self.BOID_PATH_TRACE = True
        else:
            self.BOID_PATH_TRACE = boid_path_trace
        if boid_path_trace_segments is None:
            self.BOID_PATH_TRACE_SEGMENTS = 50
        else:
            self.BOID_PATH_TRACE_SEGMENTS = boid_path_trace_segments

        # rule weights
        if w_vector is None:
            self.w_vector = 1.0
        else:
            self.w_vector = w_vector
        if w_separation is None:
            self.w_separation = 0.15
        else:
            self.w_separation = w_separation
        if w_alignment is None:
            self.w_alignment = 0.1
        else:
            self.w_alignment = w_alignment
        if w_cohesion is None:
            self.w_cohesion = 0.1
        else:
            self.w_cohesion = w_cohesion
        if w_obstacle is None:
            self.w_obstacle = 0.8
        else:
            self.w_obstacle = w_obstacle

        if debug is None:
            self.debug = False
        else:
            self.debug = debug
