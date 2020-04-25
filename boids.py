from collections import defaultdict
from dataclasses import dataclass
import math
import numpy as np
import pygame
import pygame.gfxdraw
from pygame.locals import *
import random
import time
from typing import List, Tuple, Iterable


FPS = 60
SCREENSIZE = (800, 600)
ENTCOUNT = 100
DEBUG = False

DEBUG_SURF = pygame.Surface(SCREENSIZE)
DEBUG_SURF.set_colorkey((0, 0, 0))

# rules that boids live by:
###########################
def rules(
    boid,
    group: Iterable[object],
    obstacles: Iterable[pygame.Rect]
    ) -> np.ndarray:


    weights = np.array([
        [1],  # boid.vector
        [.15], # seperation
        [.1], # alignment
        [.1], # cohesion
        [.8]  # obstacle
    ], dtype=float)

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
            average_vector = np.mean(np.stack([b.pos for b in group]), axis=0)
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

        return -unit_vector * (distance/ boid.seperation_distance)
        
    
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
    surface = pygame.Surface(SCREENSIZE)
    overlay = pygame.Surface(SCREENSIZE)
    overlay.fill((0, 0, 0))
    overlay.set_colorkey((0, 0, 0))
    overlay.set_alpha(50)
    game_border = surface.get_rect()
    border_walls = [
        *[pygame.Rect(                            x*10,                                0, 10, 10) for x in range(SCREENSIZE[0]//10 + 1)],
        *[pygame.Rect(                            x*10,                 SCREENSIZE[1]-10, 10, 10) for x in range(SCREENSIZE[0]//10 + 1)],
        *[pygame.Rect(                               0,                             y*10, 10, 10) for y in range(SCREENSIZE[1]//10 + 1)],
        *[pygame.Rect(                SCREENSIZE[0]-10,                             y*10, 10, 10) for y in range(SCREENSIZE[1]//10 + 1)],
        *[pygame.Rect(np.random.randint(SCREENSIZE[0]), np.random.randint(SCREENSIZE[1]), 10, 10) for _ in range(                   50)],
    ]

    def __init__(self,
    pos: List[int],
    speed: float = 3.5,
    angle: float = np.random.rand(),
    sight = 100,
    size = 3,
    turn_rate = np.pi/16
    ):
        assert len(pos) == 2
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.vector = np.array([np.cos(angle), np.sin(angle)])
        self.outline_color = (255, 255, 255)
        self.size = float(size)
        self.rect = pygame.Rect(*self.pos.astype(int), size, size)
        self.sight = pygame.Rect(*self.pos.astype(int), sight, sight)
        self.seperation_distance = sight/2
        self.turn_rate = turn_rate

        self.instances.append(self)
        self.inst_rects.append(self.rect)
        self.focus = None

    @property
    def angle(self) -> float:
        angle = np.arctan2(self.vector[1], self.vector[0])
        return (angle + 2*np.pi) % (2*np.pi)

    @property
    def color(self) -> np.ndarray:
        offset = 2 * np.pi / 3
        angle = self.angle

        color = np.array([
            np.cos(angle) * 127 + 127,
            np.cos(angle + offset) * 127 + 127,
            np.cos(angle - offset) * 127 + 127
        ], dtype=int)

        return color

    @staticmethod
    def angle_to_vector(angle):
        return np.array([np.cos(angle), np.sin(angle)])

    @property
    def poly(self) -> np.ndarray:
        angle = self.angle
        offset = 5/6 * np.pi
        points = np.array([
            [np.cos(angle         ), np.sin(angle         )],
            [np.cos(angle + offset), np.sin(angle + offset)],
            [np.cos(angle - offset), np.sin(angle - offset)]
        ])
        return points * self.size + self.pos

    @classmethod
    def get_surface(cls):
        cls.surface.blit(cls.overlay, (0, 0))
        return cls.surface

    @classmethod
    def get_collisions(cls, boid, rect) -> list:
        return [b for b in filter(
            lambda x: x is not boid and rect.colliderect(x.rect),
            cls.instances)]

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

    @classmethod
    def update_all(cls):
        for b in cls.instances:
            b.update()
    
    def update(self):
        group = self.get_collisions(self, self.sight)
        new_vector = rules(self, group, self.border_walls)
        new_angle = np.arctan2(*new_vector[-1::-1])
        new_angle = (new_angle + 2*np.pi) % (2*np.pi)
        rot_angle = new_angle - self.angle
        
        if abs(rot_angle) > self.turn_rate:
            # multiply by -1 if rot_angle is negative
            rot_angle = self.turn_rate * (rot_angle / abs(rot_angle))

        rot_matrix = np.array([
            [np.cos(rot_angle), -np.sin(rot_angle)],
            [np.sin(rot_angle),  np.cos(rot_angle)]
        ])

        # self.vector = rot_matrix.dot(self.vector)
        self.vector = new_vector / np.linalg.norm(new_vector)

        if self.angle < 0:
            self.angle += 2 * np.pi

        if self.angle > 2 * np.pi:
            self.angle %= 2 * np.pi

    @classmethod
    def draw_all(cls):
        for instance in cls.instances:
            instance.draw()

    def draw(self):
        pygame.gfxdraw.filled_polygon(self.surface, self.poly.astype(int), self.color)
        pygame.gfxdraw.aapolygon(self.surface, self.poly.astype(int), self.color)
        # pygame.draw.aalines(self.surface, self.outline_color, True, self.poly)
        # pygame.draw.rect(self.surface, self.color, self.sight)
        # pygame.draw.rect(self.overlay, (255, 255, 255), self.sight)


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
    done = False
    state = {}
    last_time = time.time()

    while not done:
        screen.fill((0, 0, 0))
        if DEBUG:
            DEBUG_SURF.fill((0, 0, 0))
        Boid.surface.fill((0, 0, 0))
        Boid.overlay.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            
            if event.type == MOUSEBUTTONDOWN:
                mouse.button[event.button] = True

            if event.type == MOUSEBUTTONUP:
                mouse.button[event.button] = False

        # print(mouse.button)
        Boid.move_all()
        Boid.update_all()
        Boid.draw_all()

        for wall in Boid.border_walls:
            pygame.draw.rect(screen, (150, 150, 150), wall)

        screen.blit(Boid.get_surface(), (0, 0), special_flags=BLEND_MAX)
        screen.blit(DEBUG_SURF, (0, 0))
        if time.time() - last_time > 1:
            pygame.display.set_caption(f'FPS: {clock.get_fps():1.2f}')
            last_time = time.time()
        pygame.display.flip()
        clock.tick(FPS)
        

def main():
    pygame.init()
    clock = pygame.time.Clock()
    mouse = Mouse()
    keyboard = Keyboard()
    screen = pygame.display.set_mode(SCREENSIZE)
    screen.fill((255, 255, 255))
    pygame.display.flip()
    for _ in range(ENTCOUNT):
        Boid([random.randint(1, x-1) for x in SCREENSIZE])
    screen.fill((0, 0, 0))
    pygame.display.flip()
    # pygame.display.update([
    #     *[b.rect for b in Boid.instances],
    #     *[w for w in Boid.border_walls]
    # ])

    main_loop(screen, clock)

    pygame.quit()


if __name__ == '__main__':
    main()

