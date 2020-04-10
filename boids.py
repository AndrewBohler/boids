from dataclasses import dataclass
import math
import numpy as np
import pygame
from pygame.locals import *
import random
import time
from typing import List, Tuple

FPS = 30
SCREENSIZE = (800, 600)
ENTCOUNT = 25

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


    def __init__(self,
    pos: List[int],
    speed: float = 4,
    angle: float = 0.,
    color = [0, 20, 200],
    sight = 10,
    size = 10
    ):
        assert len(pos) == 2
        self.pos = np.array(pos, dtype=float)
        self.speed = float(speed)
        self.angle = 2 / np.random.randint(1, 10) * np.pi
        self.vector = None
        self.calc_vector()
        self.color = np.random.randint(0, 255, 3)
        self.outline_color = np.random.randint(0, 255, 3)
        self.size = float(size)
        self.rect = pygame.Rect(*self.pos, size, size)
        self.sight = pygame.Rect(*self.pos, size*sight, size*sight)

        self.instances.append(self)
        self.inst_rects.append(self.rect)
        self.focus = None

    def calc_vector(self):
        self.vector = np.array([np.sin(self.angle), np.cos(self.angle)])

    def change_dir(self):
        self.angle += np.random.rand() / 4 * np.pi

    @property
    def poly(self) -> List[Tuple[int]]:
        points = np.array([
            [np.sin(self.angle), np.cos(self.angle)],
            [np.sin(self.angle + np.pi / 6 * 5), np.cos(self.angle + np.pi / 6 * 5)],
            [np.sin(self.angle - np.pi / 6 * 5), np.cos(self.angle - np.pi / 6 * 5)]
        ])
        return points * self.size + self.pos

    @classmethod
    def blit(cls):
        cls.surface.blit(cls.overlay, (0, 0))

    @classmethod
    def get_collisions(cls, rect) -> list:
        return [
            b for b in filter(
                lambda x: rect.colliderect(x.rect), cls.instances)]

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
        self.angle %= 2*np.pi
        # if self.angle > (2 * np.pi):
        #     self.angle -= (2 * np.pi)

        # elif self.angle < -(2 * np.pi):
        #     self.angle += (2 * np.pi)

        self.pos += self.speed * np.array([np.sin(self.angle), np.cos(self.angle)])

        if self.pos[0] < 0:
            self.pos[0] = self.game_border.width

        elif self.pos[0] > self.game_border.width:
            self.pos[0] = 0.

        if self.pos[1] < 0:
            self.pos[1] = self.game_border.height

        elif self.pos[1] > self.game_border.height:
            self.pos[1] = 0.

        self.rect.center = self.pos
        self.sight.center = self.pos

    def update(self):

        if self.sight.collidelist(self.inst_rects) != -1:
            collisions = self.get_collisions(self.sight)
            collisions.remove(self)
            # find closest object
            if collisions:
                target = Target(min(collisions, key=lambda b: self.calculate_distance(*self.pos, *b.pos)))
                target.pos = target.obj.pos - self.pos
                target.angle = target.pos[1]/target.pos[0]
                self.angle += target.angle / (np.hypot(target.pos[0], target.pos[1]) + self.size)
                self.focus = target
            else:
                self.focus = None
            
        else:
            self.focus = None
        try:
            pygame.draw.polygon(self.surface, self.color, self.poly)
        except TypeError:
            print(
                'TypeError: pygame.draw.polygon(self.surface, self.color, self.poly)',
                self.poly
            )
        pygame.draw.aalines(self.surface, self.outline_color, True, self.poly)
        # pygame.draw.circle(self.overlay, (255, 255, 255), self.sight.center, self.sight.width//3)
        if self.focus:
            pygame.draw.aaline(self.surface, self.color, self.pos, self.focus.obj.pos, False)


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

        Boid.blit()

        screen.blit(Boid.surface, (0, 0), special_flags=BLEND_MAX)

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

