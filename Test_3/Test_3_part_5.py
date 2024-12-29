import pygame
import torch
from torch import nn
import random
from pathlib import Path

# window
WINDOW_SIZE_X, WINDOW_SIZE_Y = 1100, 600
WINDOW = pygame.display.set_mode((WINDOW_SIZE_X, WINDOW_SIZE_Y))
BACKGROUND = pygame.Rect(0, 0, WINDOW_SIZE_X, WINDOW_SIZE_Y)
FPS = 60

# colors (wow)
BLACK = [0, 0, 0]
RED = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE = [0, 0, 255]
YELLOW = [255, 255, 0]
PURPLE = [255, 0, 255]
LIGHT_BLUE = [0, 255, 255]
WHITE = [255, 255, 255]

# AI info
LENIENCY = 25
INPUT_FEATURES = 2
OUTPUT_FEATURES = 2
DEFAULT_HIDDEN_FEATURES = 64
DEFAULT_HIDDEN_LAYERS = 1

GOAL_DIMS = 30
GOAL_TIME = 20000

AI_OBJECT_DIMS = 50

# Physics and friends
ACCELERATION = 1
FRICTION = 0.5
MAX_VELOCITY = 10


class Goal:
    def __init__(self, dims, timed_relocation, initial_location_y=None, initial_location_x=None):
        if initial_location_y is None:
            initial_location_y = random.randint(LENIENCY, WINDOW_SIZE_Y - LENIENCY)
        if initial_location_x is None:
            initial_location_x = random.randint(LENIENCY, WINDOW_SIZE_X - LENIENCY)
        self.goal_rect = pygame.Rect(initial_location_x, initial_location_y, dims, dims)
        self.timed_relocation = timed_relocation
        self.dims = dims
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
        x_goal_center = self.goal_rect.x + (self.dims/2)
        y_goal_center = self.goal_rect.y + (self.dims/2)
        return x_goal_center, y_goal_center

    def check_for_relocation(self, timer):
        if not self.timed_relocation:
            return
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
    def __init__(self, model, dims, optimizer, target, color, goal, physics_manager, name,
                 collision_relocate=False, learning=False, loss_fn=nn.BCEWithLogitsLoss()):
        self.model = model
        self.learning = learning
        self.ai_rect = pygame.Rect((WINDOW_SIZE_X/2-dims/2), (WINDOW_SIZE_Y/2-dims/2), dims, dims)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.color = color
        self.target = target
        self.goal = goal
        self.physics_manager = physics_manager
        self.dims = dims
        self.collision_relocate = collision_relocate
        self.total_collisions = 0
        self.name = name

    def predict_and_enact_movement(self):
        goal_x, goal_y = self.goal.get_goal_location()
        ai_x, ai_y = self.get_location()
        X = torch.tensor([[ai_x, ai_y], [goal_x, goal_y]], dtype=torch.float32)
        y_logits = torch.reshape(self.model.forward(X), (-1,))
        y = torch.sigmoid(y_logits).type(torch.float32)
        if self.learning:
            self.learn(y_logits)
        self.physics_manager.calculate_velocity(self.ai_rect, y, self.dims)

    def display_ai_object(self):
        pygame.draw.rect(WINDOW, self.color, self.ai_rect)

    def get_location(self):
        ai_x = self.ai_rect.x + (self.dims/2)
        ai_y = self.ai_rect.y + (self.dims/2)
        return ai_x, ai_y

    def learn(self, y_logits):
        y_target = self.target.determine_direction(object=self)
        loss = self.loss_fn(y_logits, y_target.type(torch.float32))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def check_for_collisions(self, timer):
        if self.ai_rect.colliderect(self.goal.goal_rect):
            self.total_collisions += 1
            if self.collision_relocate:
                self.goal.relocate_goal(timer)

    def print_results(self, timer):
        print(f"The AI model {self.name} collided with the goal object {self.total_collisions} times. \n"
              f"The AI did this within {timer/1000} seconds.")


class Target:
    def __init__(self, dims, goal, physics_manager):
        self.target_rect = pygame.Rect((WINDOW_SIZE_X/2-dims/2), (WINDOW_SIZE_Y/2-dims/2), dims, dims)
        self.goal = goal
        self.physics_manager = physics_manager
        self.dims = dims

    def determine_direction(self, object):
        goal_x, goal_y = self.goal.get_goal_location()
        object_x, object_y = object.get_location()
        right = 0
        left = 0
        down = 0
        up = 0
        if goal_x > object_x:
            right = 1
        elif goal_x < object_x:
            left = 1
        else:
            pass
        if goal_y < object_y:
            down = 1
        elif goal_y > object_y:
            up = 1
        desired_direction = torch.tensor([up, down, left, right])
        return desired_direction

    def get_location(self):
        target_x = self.target_rect.x + (self.dims/2)
        target_y = self.target_rect.y + (self.dims/2)
        return target_x, target_y

    def display_target(self):
        pygame.draw.rect(WINDOW, GREEN, self.target_rect)

    def move(self):
        direction = self.determine_direction(object=self)
        self.physics_manager.calculate_velocity(self.target_rect, direction, self.dims)


class PhysicsManager:
    def __init__(self):
        self.x_velocity = 0
        self.y_velocity = 0

    def calculate_velocity(self, rect, direction_tensor, dims):
        direction_list = direction_tensor.round().bool().tolist()
        up = False
        down = False
        right = False
        left = False
        x_acceleration = 0
        y_acceleration = 0

        for i, (direction) in enumerate(direction_list):
            match i:
                case 0:
                    up = direction
                case 1:
                    down = direction
                case 2:
                    left = direction
                case 3:
                    right = direction

        if up:
            y_acceleration += ACCELERATION
        if down:
            y_acceleration -= ACCELERATION
        if right:
            x_acceleration += ACCELERATION
        if left:
            x_acceleration -= ACCELERATION

        if self.x_velocity > 0:
            x_acceleration -= FRICTION
        if self.x_velocity < 0:
            x_acceleration += FRICTION
        if self.y_velocity > 0:
            self.y_velocity -= FRICTION
        if self.y_velocity < 0:
            self.y_velocity += FRICTION
        self.x_velocity += x_acceleration
        self.y_velocity += y_acceleration

        if self.x_velocity > MAX_VELOCITY:
            self.x_velocity = MAX_VELOCITY
        elif self.x_velocity < -MAX_VELOCITY:
            self.x_velocity = -MAX_VELOCITY
        if self.y_velocity > MAX_VELOCITY:
            self.y_velocity = MAX_VELOCITY
        elif self.y_velocity < -MAX_VELOCITY:
            self.y_velocity = -MAX_VELOCITY

        if rect.x > WINDOW_SIZE_X - dims or rect.x < 0:
            self.x_velocity = -self.x_velocity
        if rect.y > WINDOW_SIZE_Y - dims or rect.y < 0:
            self.y_velocity = -self.y_velocity
        if rect.y > WINDOW_SIZE_Y - dims + 5:
            rect.y -= 10
        if rect.y < -5:
            rect.y += 10
        if rect.x > WINDOW_SIZE_X - dims + 5:
            rect.x -= 10
        if rect.x < -5:
            rect.x += 10

        rect.x += self.x_velocity
        rect.y += self.y_velocity


def simulation_display(goal, ai, target, ai4):
    pygame.draw.rect(WINDOW, BLACK, BACKGROUND)
    target.display_target()
    goal.display_goal()
    ai.display_ai_object()
    ai4.display_ai_object()
    pygame.display.update()


def save_model(model):
    save = input("Do you want to save this model?: ")
    if not (save == "yes" or save == "Yes"):
        print("This model shall not be saved.")
        return 0
    model_path = Path(input("What file do you want to save your model in? \n"
                            "If you input a file that dose not exist. One will be created for you: "))
    model_path.mkdir(parents=True, exist_ok=True)
    model_name = input("What do you want your model dict's name to be?: ")+".pth"
    model_save_path = model_path/model_name
    print("Now downloading the model.....")
    # This will save the model dict. If you want to save the entire model then change this code to do so
    torch.save(obj=model.state_dict(), f=model_save_path)
    print("Model successfully saved! YIPPEE!!!")


def main():
    running = True
    model_0 = AIModel(input_features=INPUT_FEATURES,
                      output_features=OUTPUT_FEATURES,
                      hidden_layers=DEFAULT_HIDDEN_LAYERS,
                      hidden_features=DEFAULT_HIDDEN_FEATURES)
    model_4 = AIModel(input_features=INPUT_FEATURES,
                      output_features=OUTPUT_FEATURES,
                      hidden_layers=DEFAULT_HIDDEN_LAYERS,
                      hidden_features=DEFAULT_HIDDEN_FEATURES)
    optimizer_0 = torch.optim.SGD(params=model_0.parameters(), lr=0.001)
    optimizer_4 = torch.optim.SGD(params=model_4.parameters(), lr=0.0001)
    physics_manager_target = PhysicsManager()
    physics_manager_ai_0 = PhysicsManager()
    physics_manager_ai_4 = PhysicsManager()

    goal = Goal(dims=GOAL_DIMS, timed_relocation=True)
    target = Target(dims=AI_OBJECT_DIMS, goal=goal, physics_manager=physics_manager_target)
    ai_object_0 = AIObject(
        model=model_0,
        dims=AI_OBJECT_DIMS,
        optimizer=optimizer_0,
        target=target,
        color=RED,
        goal=goal,
        name="model_0",
        collision_relocate=True,
        physics_manager=physics_manager_ai_0,
        learning=True
    )
    ai_object_4 = AIObject(
        model=model_4,
        dims=AI_OBJECT_DIMS,
        optimizer=optimizer_4,
        target=target,
        color=WHITE,
        goal=goal,
        name="model 4",
        collision_relocate=True,
        physics_manager=physics_manager_ai_4,
        learning=True
    )
    current_time = pygame.time.get_ticks()
    clock = pygame.time.Clock()
    while running:
        current_time = pygame.time.get_ticks()
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        goal.check_for_relocation(timer=current_time)
        target.move()
        ai_object_0.predict_and_enact_movement()
        ai_object_0.check_for_collisions(current_time)
        ai_object_4.predict_and_enact_movement()
        ai_object_4.check_for_collisions(current_time)
        simulation_display(goal, ai_object_0, target, ai_object_4)
    pygame.display.quit()
    ai_object_0.print_results(current_time)
    ai_object_4.print_results(current_time)
    save_model(model_0)
    save_model(model_4)


if __name__ == "__main__":
    main()
