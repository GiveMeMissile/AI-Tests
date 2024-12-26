import pygame
import torch
from torch import nn
import random

WINDOW_SIZE_X, WINDOW_SIZE_Y = 1100, 600
WINDOW = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
BACKGROUND = pygame.Rect(0, 0, WINDOW_SIZE_X, WINDOW_SIZE_Y)
FPS = 60

BLACK = [0, 0, 0]
YELLOW = [255, 255, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
LENIENCY = 25

INPUT_FEATURES = 2
OUTPUT_FEATURES = 2

GOAL_DIMS = 30
GOAL_TIME = 20000

AI_OBJECT_DIMS = 50


class Goal:
    def __init__(self, dims, timed_relocation, initial_location_y=None, initial_location_x=None):
        if initial_location_y is None:
            initial_location_y = random.randint(LENIENCY, WINDOW_SIZE_Y - LENIENCY)
        if initial_location_x is None:
            initial_location_x = random.randint(LENIENCY, WINDOW_SIZE_X - LENIENCY)
        self.goal_rect = pygame.Rect(initial_location_x, initial_location_y, dims, dims)
        self.timed_relocation = timed_relocation
        if self.timed_relocation:
            self.start_timer = 0

    def display_goal(self):
        pygame.draw.rect(WINDOW, YELLOW, self.goal_rect)

    def relocate_goal(self, timer):
        self.goal_rect.x = random.randint(LENIENCY, WINDOW_SIZE_X - LENIENCY)
        self.goal_rect.y = random.randint(LENIENCY, WINDOW_SIZE_Y - LENIENCY)
        if self.timed_relocation:
            self.start_timer = timer

    def get_goal_location(self):
        x_goal_center = self.goal_rect.x/2
        y_goal_center = self.goal_rect.y/2
        return x_goal_center, y_goal_center

    def check_for_relocation(self, timer):
        if self.timed_relocation:
            if timer - self.start_timer >= GOAL_TIME:
                self.relocate_goal(timer)


class AIModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_layers, hidden_features):
        super().__init__()
        self.input_layer = nn.Linear(in_features=input_features, out_features=hidden_features)
        self.hidden_layer = nn.Sequential(
            nn.Linear(in_features=hidden_features, out_features=hidden_features),
            nn.ReLU(),
            nn.Linear(in_features=hidden_features, out_features=hidden_features)
        )
        self.output_layer = nn.Linear(in_features=hidden_features, out_features=output_features)
        self.hidden_layers = hidden_layers

    def forward(self, x):
        x = self.input_layer(x)
        for i in range(self.hidden_layers):
            x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


class AIObject:
    def __init__(self, model, learning, dims, optimizer, loss_fn):
        self.model = model
        self.learning = learning
        self.ai_rect = ((WINDOW_SIZE_X/2-dims/2), (WINDOW_SIZE_Y/2-dims/2), dims, dims)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def predict(self, X):
        pass

    def display_ai_object(self):
        pygame.draw.rect(WINDOW, RED, self.ai_rect)


class TargetPerformance:
    def __init__(self, dims):
        self.target_rect = ((WINDOW_SIZE_X/2-dims/2), (WINDOW_SIZE_Y/2-dims/2), dims, dims)

    def determine_direction(self, goal_location_x, goal_location_y):
        pass


def simulation_display(goal):
    pygame.draw.rect(WINDOW, BLACK, BACKGROUND)
    goal.display_goal()
    pygame.display.update()


def main():
    running = True
    goal = Goal(dims=GOAL_DIMS, timed_relocation=True)
    clock = pygame.time.Clock()
    while running:
        current_time = pygame.time.get_ticks()
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        goal.check_for_relocation(timer=current_time)
        simulation_display(goal)


if __name__ == "__main__":
    main()
