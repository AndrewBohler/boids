from dataclasses import dataclass
import math
import numpy as np
import pygame
from pygame.locals import *
import random
import time
from typing import List, Tuple, Iterable


FPS = 30
SCREENSIZE = (800, 600)
ENTCOUNT = 200

# rules that boids live by:
###########################
def rules(boid, group: Iterable[object]):
    
    def seperation() -> float:
        too_close = filter(
            lambda b: (np.hypot(*(b.pos - boid.pos)) < boid.seperation_distance),
            group
        )
        point = np.mean([b.pos for b in too_close], axis=0)
        return np.arctan2(*(point - boid.pos)) - boid.angle + np.pi

    def alignment() -> float:
        return np.mean([b.angle for b in group], axis=0) - boid.angle

    def cohesion() -> float:
        vector = np.mean([b.pos for b in group], axis=0)
        return np.arctan2(*(vector - boid.pos)) - boid.angle
    angles = np.array([[seperation(), alignment(), cohesion()]])
    weights = np.array([0.5, 1e6, 1e4])
    return np.mean(angles * weights)
############################


@dataclass
class Target:
    """Object for holding target variables"""
    obj: object

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
        pygame.Rect(0, 0, SCREENSIZE[0], 10),
        pygame.Rect(0, SCREENSIZE[1]-10, SCREENSIZE[0], 10),
        pygame.Rect(0, 0, 10, SCREENSIZE[1]),
        pygame.Rect(SCREENSIZE[0]-10, 0, 10, SCREENSIZE[1])
    ]
    seperation_distance = 20


    def __init__(self,
    pos: List[int],
    speed: float = 4,
    angle: float = 0.,
    color = [0, 20, 200],
    sight = 5,
    size = 5,
    turn_rate = np.pi/64
    ):
        assert len(pos) == 2
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.angle = 2 / np.random.randint(1, 10) * np.pi
        self.vector = None
        self.calc_vector()
        self.color = np.random.randint(0, 255, 3)
        self.outline_color = (255, 255, 255)
        self.size = float(size)
        self.rect = pygame.Rect(*self.pos.astype(int), size, size)
        self.sight = pygame.Rect(*self.pos.astype(int), size*sight, size*sight)
        self.seperation_distance = sight*size / 2
        self.turn_rate = turn_rate

        self.instances.append(self)
        self.inst_rects.append(self.rect)
        self.focus = None

    def calc_vector(self):
        self.vector = np.array([np.sin(self.angle), np.cos(self.angle)])

    @property
    def poly(self) -> List[Tuple[int]]:
        points = np.array([
            [np.sin(self.angle), np.cos(self.angle)],
            [np.sin(self.angle + np.pi / 6 * 5), np.cos(self.angle + np.pi / 6 * 5)],
            [np.sin(self.angle - np.pi / 6 * 5), np.cos(self.angle - np.pi / 6 * 5)]
        ])
        return points * self.size + self.pos

    @classmethod
    def get_surface(cls):
        cls.surface.blit(cls.overlay, (0, 0))
        return cls.surface

    @classmethod
    def get_collisions(cls, boid, rect) -> list:
        return [b for b in filter(
            lambda x: x is not boid and rect.colliderect(x.rect), cls.instances)
        ]

    @staticmethod
    def calculate_distance(x1, y1, x2, y2):
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    @classmethod
    def update_all(cls):
        for b in cls.instances:
            b.update()

    @classmethod
    def move_all(cls):
        for instance in cls.instances:
            instance.move()

    def move(self):
        self.pos += self.speed * np.array([np.sin(self.angle), np.cos(self.angle)])

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

    def update(self):
        group = self.get_collisions(self, self.sight)
        new_angle = rules(self, group)
        if new_angle is not (np.nan or 0):
            if new_angle > 0:
                self.angle += min(self.turn_rate, new_angle)
            elif new_angle < 0:
                self.angle -= min(self.turn_rate, -new_angle)

        if self.angle < -(2*np.pi):
            self.angle += 2 * np.pi
        elif self.angle > (2*np.pi):
            self.angle -= 2 * np.pi

        pygame.draw.polygon(self.surface, self.color, self.poly.astype(int))
        # pygame.draw.aalines(self.surface, self.outline_color, True, self.poly)
        # pygame.draw.circle(self.overlay, (255, 255, 255), self.sight.center, self.sight.width//3)


def main_loop(
    screen,
    clock,
    ):
    done = False
    while not done:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill((0, 0, 0))
        Boid.surface.fill((0, 0, 0))
        Boid.overlay.fill((0, 0, 0))
        

        Boid.move_all()
        Boid.update_all()

        for wall in Boid.border_walls:
            pygame.draw.rect(screen, (150, 150, 150), wall)
        screen.blit(Boid.get_surface(), (0, 0), special_flags=BLEND_MAX)

        pygame.display.flip()
        clock.tick(FPS)
        

def main():
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode(SCREENSIZE)
    screen.fill((255, 255, 255))
    pygame.display.flip()
    for _ in range(ENTCOUNT):
        Boid([random.randint(1, x-1) for x in SCREENSIZE])
    screen.fill((0, 0, 0))
    pygame.display.flip()
    main_loop(screen, clock)

    pygame.quit()


if __name__ == '__main__':
    main()

