from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import chain
import math
import numpy as np
import pygame
import pygame.gfxdraw
from pygame.locals import *
import random
import time
from typing import List, Tuple, Iterable


FPS = 60
SCREENSIZE = (500, 500)
GRID_TILE_SIZE = 25 # pixels
BORDER_WALLS = True
OBSTACLE_MOVE_TIME = 10 # wont move when 0
N_OBSTACLES = 15
N_BOID = 100
BOID_COLOR_ROT = 0
BOID_COLOR_ROT_RATE = np.pi/128
BOID_SEPERATION_DISTANCE = 7
BOID_SIZE = 5
BOID_SIGHT = 25
BOID_SPEED = 3
BOID_OBST_AVOID_ARC = np.pi/16 # radians
BOID_PATH_TRACE = True
BOID_PATH_TRACE_SEGMENTS = 50
RULE_WEIGHTS = np.array([
    [1.  ], # boid.vector
    [ .15], # seperation
    [ .1 ], # alignment
    [ .1 ], # cohesion
    [ .8 ]  # obstacle
], dtype=float)

DEBUG = False
DEBUG_SURF = pygame.Surface(SCREENSIZE)
DEBUG_SURF.set_colorkey((0, 0, 0))

GRID = np.empty([d // GRID_TILE_SIZE + 1 for d in SCREENSIZE], dtype=list)
for row in range(GRID.shape[0]):
    for col in range(GRID.shape[1]):
        GRID[row, col] = defaultdict(list)

TWO_PI = 2 * np.pi
TWO_THIRDS_PI = 2 * np.pi / 3
FIVE_SIXTHS_PI = 5/6 * np.pi


# rules that boids live by:
###########################
def rules(
    boid,
    group: Iterable[object],
    obstacles: Iterable[pygame.Rect],
    weights=RULE_WEIGHTS
    ) -> np.ndarray:

    def seperation() -> np.ndarray:
        # weight 1
        too_close = [tc.pos for tc in filter(
            lambda b: np.hypot(*(b.pos - boid.pos)) \
                < boid.seperation_distance,
            group)
        ]
        
        if too_close:
            average_vector = np.mean(too_close, axis=0) - boid.pos
            unit_vector = average_vector / np.linalg.norm(average_vector)
            # weights[1] *= boid.seperation_distance - np.linalg.norm(average_vector)
            return -unit_vector

        else:
            return np.array([np.nan, np.nan])

    def alignment() -> np.ndarray:
        # weight 2
        if group:
            average_vector = np.mean(np.stack([b.vector for b in group]), axis=0)
            
            # unit vector
            return average_vector / np.linalg.norm(average_vector)
        
        else:
            return np.array([np.nan, np.nan])

    def cohesion() -> np.ndarray:
        # weight 3
        if group:
            average_vector = np.mean([b.pos for b in group], axis=0)
            relative_vector = average_vector - boid.pos

            # unit vector
            return relative_vector / np.linalg.norm(relative_vector)

        else:
            return np.array([np.nan, np.nan])

    def obstacle() -> np.ndarray:
        # weight 4
        in_range = [
            o for o in obstacles if boid.sight.colliderect(o)
        ]

        if not in_range:
            return np.array([np.nan, np.nan])

        can_see = []
        # can_see_distance = []
        for obst in in_range:
            relative_vector = obst.center - boid.pos
            relative_norm = np.linalg.norm(relative_vector)

            # boid.vector obmitted because magnitude = 1
            relative_angle = np.arccos(
                boid.vector.dot(relative_vector) / relative_norm)
            
            if abs(relative_angle) < np.pi/4:
                can_see.append(obst.center)
                # can_see_distance.append(relative_norm)

        # size equals 0 when there are no obstacles
        if not can_see:
            return np.array([np.nan, np.nan])

        average_vector = np.mean(can_see, axis=0)
        relative_vector = average_vector - boid.pos
        unit_vector = relative_vector / np.linalg.norm(relative_vector)
        distance = np.linalg.norm(relative_vector)

        if DEBUG:
            for v in can_see:
                # from obstacle to avg vector
                pygame.draw.aaline(
                    DEBUG_SURF, (255, 100, 100),
                    v, average_vector.astype(int))
            # circle at avg vector
            pygame.gfxdraw.filled_circle(
                DEBUG_SURF,*average_vector.astype(int),
                5, (255, 100, 100))
            # line away from average vector
            pygame.draw.aaline(
                DEBUG_SURF, (150, 150, 255), 
                average_vector.astype(int), (boid.pos + (-unit_vector*25)).astype(int))
            # endcap for line
            pygame.gfxdraw.filled_circle(
                DEBUG_SURF, *(boid.pos -unit_vector*25).astype(int),
                5, (150, 150, 255))
            # highlight boid
            pygame.gfxdraw.filled_polygon(
                DEBUG_SURF, boid.poly, (255, 255, 255))

        return -unit_vector * (distance / boid.seperation_distance)
        
    ret = np.array([
        boid.vector,
        seperation(),
        alignment(),
        cohesion(),
        obstacle()
    ])

    return np.nanmean(ret * weights, axis=0)
############################

class Boid:

    instances = []
    inst_rects = []
    background_surface = pygame.Surface(SCREENSIZE)
    boid_surface = pygame.Surface(SCREENSIZE)
    boid_surface.set_colorkey((0, 0, 0))
    overlay_surface = pygame.Surface(SCREENSIZE)
    overlay_surface.set_colorkey((0, 0, 0))
    overlay_surface.set_alpha(50)
    game_border = background_surface.get_rect()
    border_walls = []
    obstacles = []
    wall_color = (150, 150, 150)
    obstacle_color = (100, 100, 100)
    
    if BORDER_WALLS:
        border_walls.extend([
            *[pygame.Rect(x*10, 0, 10, 10) \
                for x in range(SCREENSIZE[0] // 10 + 1)],
            *[pygame.Rect(x*10, SCREENSIZE[1]-10, 10, 10) \
                for x in range(SCREENSIZE[0] // 10 + 1)],
            *[pygame.Rect(0, y*10, 10, 10) \
                for y in range(SCREENSIZE[1] // 10 + 1)],
            *[pygame.Rect(SCREENSIZE[0]-10, y*10, 10, 10) \
                for y in range(SCREENSIZE[1] // 10 + 1)],
        ])

        for wall in border_walls:
            GRID[
                wall.center[0] // GRID_TILE_SIZE,
                wall.center[1] // GRID_TILE_SIZE
            ]['walls'].append(wall)
        
    if N_OBSTACLES:
        obstacles.extend([
            pygame.Rect(
                np.random.randint(SCREENSIZE[0]),
                np.random.randint(SCREENSIZE[1]),
                10,
                10
            ) for _ in range(N_OBSTACLES)
        ])
        for obst in obstacles:
            GRID[
                obst.center[0] // GRID_TILE_SIZE,
                obst.center[1] // GRID_TILE_SIZE
            ]['walls'].append(obst)    

    def __init__(self,
    pos: List[int],
    speed: float,
    sight: float,
    size: float,
    seperation_distance: float,
    angle: float = 0.,
    obst_avoidance_arc = np.pi/8
    ):
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.vector = np.array([np.cos(angle), np.sin(angle)])
        self.outline_color = (255, 255, 255)
        self.size = float(size)
        self.rect = pygame.Rect(*self.pos.astype(int), size, size)
        self.sight = pygame.Rect(*self.pos.astype(int), sight, sight)
        self.seperation_distance = seperation_distance
        self.obst_avoidance_arc = obst_avoidance_arc

        self._update_color()

        self.path_history = deque(
            # [self.pos] * BOID_PATH_TRACE_SEGMENTS,
            maxlen=BOID_PATH_TRACE_SEGMENTS
        )
        self.color_history = deque(
            # [self.color] * BOID_PATH_TRACE_SEGMENTS,
            maxlen=BOID_PATH_TRACE_SEGMENTS
        )

        self.instances.append(self)
        self.inst_rects.append(self.rect)
        self.grid_loc = tuple(self.pos.astype(int) // GRID_TILE_SIZE)
        GRID[self.grid_loc][self.__class__].append(self)

    def grid_adjacent(self, distance=1) -> np.array:
        x0 = max(self.grid_loc[0] - 1, 0)
        y0 = max(self.grid_loc[1] - 1, 0)
        x2 = min(
            (self.grid_loc[0] + 2),
            SCREENSIZE[0] // GRID_TILE_SIZE + 1
        )
        y2 = min(
            (self.grid_loc[1] + 2),
            SCREENSIZE[1] // GRID_TILE_SIZE + 1
        )
        return GRID[x0:x2, y0:y2]

    @property
    def angle(self) -> float:
        angle = np.arctan2(self.vector[1], self.vector[0])
        return (angle + TWO_PI) % TWO_PI

    def _update_color(self):
        # offset = 2 * np.pi / 3
        angle = self.angle + BOID_COLOR_ROT

        color = np.array([
            np.cos(angle                ) * 127 + 128,
            np.cos(angle + TWO_THIRDS_PI) * 127 + 128,
            np.cos(angle - TWO_THIRDS_PI) * 127 + 128
        ], dtype=int)

        self.color = color

    @staticmethod
    def angle_to_vector(angle):
        return np.array([np.cos(angle), np.sin(angle)])

    @property
    def poly(self) -> np.ndarray:
        angle = self.angle
        # offset = 5/6 * np.pi
        points = np.array([
            [np.cos(angle         ), np.sin(angle         )],
            [np.cos(angle + FIVE_SIXTHS_PI), np.sin(angle + FIVE_SIXTHS_PI)],
            [np.cos(angle - FIVE_SIXTHS_PI), np.sin(angle - FIVE_SIXTHS_PI)]
        ])
        return points * self.size + self.pos

    @classmethod
    def get_surfaces(cls):
        return [
            cls.background_surface,
            cls.boid_surface,
            cls.overlay_surface
        ]

    @classmethod
    def get_collisions(cls, boid, rect) -> list:
        return [b for b in filter(
            lambda x: x is not boid and rect.colliderect(x.rect),
            cls.instances)]

    @classmethod
    def move_obstacles(cls):
        for cell in GRID.flat:
            for wall in cell['walls']:
                if wall in cls.obstacles:
                    cell['walls'].remove(wall)

        for obst in cls.obstacles:
            old_loc = obst.center
            obst.center = (
                np.random.randint(SCREENSIZE[0]),
                np.random.randint(SCREENSIZE[1])
            )
            GRID[
                obst.centerx // GRID_TILE_SIZE,
                obst.centery // GRID_TILE_SIZE
            ]['walls'].append(obst)

        cls.draw_background()

    @classmethod
    def move_all(cls):
        for instance in cls.instances:
            instance.move()

    def move(self):
        self.pos += self.speed * self.angle_to_vector(self.angle)

        if self.pos[0] < 0:
            self.pos[0] += self.game_border.width

        elif self.pos[0] > self.game_border.width:
            self.pos[0] -= self.game_border.width

        if self.pos[1] < 0:
            self.pos[1] += self.game_border.height

        elif self.pos[1] > self.game_border.height:
            self.pos[1] -= self.game_border.height

        self.rect.center = self.pos.astype(int)
        self.sight.center = self.pos.astype(int)

        if BOID_PATH_TRACE:
            self.path_history.append(self.pos.astype(int))
            self.color_history.append(self.color)

        # keep track of grid location
        if tuple(self.pos.astype(int) // GRID_TILE_SIZE) != self.grid_loc:
            GRID[self.grid_loc][self.__class__].remove(self)
            self.grid_loc = tuple(self.pos.astype(int) // GRID_TILE_SIZE)
            GRID[self.grid_loc][self.__class__].append(self)

    @classmethod
    def update_all(cls):
        for b in cls.instances:
            b.update()
    
    def update(self):
        group = []
        walls = []
        for tile in self.grid_adjacent().flat:
            group.extend([
                b for b in tile[self.__class__] \
                    if not b is self and b.rect.colliderect(self.sight)
            ])
            walls.extend([
                w for w in tile['walls'] if self.sight.colliderect(w)])
        new_vector = rules(self, group, walls)
        # new_vector = rules(self, [b for b in self.instances if self.sight.colliderect(b.rect)], self.border_walls)
        self.vector = new_vector / np.linalg.norm(new_vector) # unit vector
        self._update_color()

    @classmethod
    def draw_background(cls):
        cls.background_surface.fill((0, 0, 0))
        for wall in Boid.border_walls:
                pygame.draw.rect(cls.background_surface, cls.wall_color, wall)

        for obst in cls.obstacles:
            pygame.draw.rect(cls.background_surface, cls.obstacle_color, obst)

        if DEBUG:
            for x in range(SCREENSIZE[0] // GRID_TILE_SIZE):
                pygame.gfxdraw.vline(
                    Boid.background_surface,
                    x * GRID_TILE_SIZE,
                    0,
                    SCREENSIZE[1],
                    (50, 50, 50))

            for y in range(SCREENSIZE[1] // GRID_TILE_SIZE):
                pygame.gfxdraw.hline(
                    Boid.background_surface,
                    0,
                    SCREENSIZE[0],
                    y * GRID_TILE_SIZE,
                    (50, 50, 50))

    @classmethod
    def draw_all(cls):
        cls.boid_surface.fill((0, 0, 0))
        cls.overlay_surface.fill((0, 0, 0))

        if BOID_PATH_TRACE:
            for instance in cls.instances:
                instance.draw_trace()

        for instance in cls.instances:
            instance.draw()

    def draw(self):
        pygame.gfxdraw.filled_polygon(self.boid_surface, self.poly.astype(int), self.color)
        # pygame.gfxdraw.aapolygon(self.surface, self.poly.astype(int), color)

    def draw_trace(self):
        for i in range(len(self.path_history)-1, 0, -1):
            pygame.draw.aaline(
                self.boid_surface,
                self.color_history[i] * (i/len(self.color_history)),
                self.path_history[i],
                self.path_history[i-1]
            )


@dataclass
class Mouse:
    button = defaultdict(lambda k: False)

@dataclass
class Keyboard:
    key = defaultdict(lambda k: False)


def main_loop(
    screen,
    clock,
    mouse=Mouse()
    ):
    global CURRENT_FRAME_TIME
    global LAST_FRAME_TIME
    global BOID_COLOR_ROT
    seconds_per_frame = 1. / FPS
    CURRENT_FRAME_TIME = time.time()
    LAST_FRAME_TIME = CURRENT_FRAME_TIME - seconds_per_frame
    obstacle_timer = time.time()
    done = False
    state = {}
    last_time = time.time()
    Boid.draw_background()
    times = defaultdict(list)

    while not done:
        frame_start = time.time()
        BOID_COLOR_ROT += BOID_COLOR_ROT_RATE

        for event in pygame.event.get():
            if event.type == QUIT:
                done = True
            
            if event.type == MOUSEBUTTONDOWN:
                mouse.button[event.button] = True

            if event.type == MOUSEBUTTONUP:
                mouse.button[event.button] = False

        # print(mouse.button)
        if DEBUG:
            now = time.time()
            Boid.move_all()
            move_time = time.time() - now

            now = time.time()
            Boid.update_all()
            update_time = time.time() - now

            now = time.time()
            Boid.draw_all()
            draw_time = time.time() - now

            now = time.time()
            for surface in Boid.get_surfaces():
                screen.blit(surface, (0, 0))
            blit_time = time.time() - now

            pygame.display.flip()
            frame_end = time.time()

            times['move_time'].append(move_time)
            times['update_time'].append(update_time)
            times['draw_time'].append(draw_time)
            times['blit_time'].append(blit_time)
            times['frame_time'].append(frame_end - frame_start)

            if time.time() - last_time > 1:
                print(f'''
                \r\t{N_BOID} boids | {len(Boid.border_walls)} walls | screen: {SCREENSIZE[0]} x {SCREENSIZE[1]}
                \r\tframes: {len(times["frame_time"])}
                \r\t--------------------------
                \r\tmove_time  : {np.mean(times["move_time"]) * 1e3:>10.6f} ms
                \r\tupdate_time: {np.mean(times["update_time"]) * 1e3:>10.6f} ms
                \r\tdraw_time  : {np.mean(times["draw_time"]) * 1e3:>10.6f} ms
                \r\tblit_time  : {np.mean(times["blit_time"]) * 1e3:>10.6f} ms
                \r\t--------------------------
                \r\tframe_time : {np.mean(times["frame_time"]) * 1e3:>10.6f}
                ''')
                for _, time_list in times.items():
                    time_list.clear()

                screen.blit(DEBUG_SURF, (0, 0))
                DEBUG_SURF.fill((0, 0, 0))

                pygame.display.set_caption(f'FPS: {clock.get_fps():1.2f}')
                last_time = time.time()
        
        else:
            Boid.move_all()
            Boid.update_all()
            Boid.draw_all()

            for surface in Boid.get_surfaces():
                screen.blit(surface, (0, 0))

            pygame.display.flip()
            frame_end = time.time()
        
        end_of_frame_time = time.time()

        if OBSTACLE_MOVE_TIME and end_of_frame_time - obstacle_timer > OBSTACLE_MOVE_TIME:
            Boid.move_obstacles()
            obstacle_timer = time.time()

        if time.time() - last_time > 1:
            pygame.display.set_caption(f'FPS: {clock.get_fps():1.2f}')
            last_time = time.time()
        
        clock.tick(FPS)
        

def main():
    pygame.init()
    clock = pygame.time.Clock()
    mouse = Mouse()
    keyboard = Keyboard()
    screen = pygame.display.set_mode(SCREENSIZE)
    screen.fill((255, 255, 255))
    pygame.display.flip()
    for _ in range(N_BOID):
        Boid(
            [random.randint(1, x-1) for x in SCREENSIZE],
            BOID_SPEED,
            BOID_SIGHT,
            BOID_SIZE,
            BOID_SEPERATION_DISTANCE,
            angle=(np.random.rand() * 2 * np.pi),
            obst_avoidance_arc=BOID_OBST_AVOID_ARC
        )
    screen.fill((0, 0, 0))
    pygame.display.flip()

    main_loop(screen, clock)

    pygame.quit()


if __name__ == '__main__':
    main()
