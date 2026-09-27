import os
import math
import random
import copy

import pygame
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TRACK_PATH = os.path.join(
    BASE_DIR,
    "track.png"
)

SAVE_PATH = os.path.join(
    BASE_DIR,
    "best_ai.pth"
)

CAR_PATH = os.path.join(
    BASE_DIR,
    "aicar.png"
)


# ============================================================
# SAVE / RESUME
# ============================================================

# Automatically load the previous AI when training starts.
AUTO_RESUME = True

# Currently unused, but available if we want to save every
# generation later.
SAVE_EVERY_GENERATION = False


# ============================================================
# TRAINING SETTINGS
# ============================================================

POPULATION_SIZE = 200

GENERATIONS = 100000

ELITE_COUNT = 16

PARENT_POOL_PERCENT = 0.40

RANDOM_IMMIGRANT_PERCENT = 0.10

MAX_STEPS = 60 * 30


# ============================================================
# MUTATION
# ============================================================

BASE_MUTATION_RATE = 0.08

BASE_MUTATION_STRENGTH = 0.12

HIGH_MUTATION_RATE = 0.18

HIGH_MUTATION_STRENGTH = 0.25

STAGNATION_LIMIT = 5


# ============================================================
# PHYSICS
# ============================================================

MAX_SPEED = 10.0

ENGINE_ACCELERATION = 0.32

DRAG = 0.992

MIN_START_SPEED = 0.2

MAX_STEERING_RATE = 4.5


# ============================================================
# DISPLAY
# ============================================================

WIDTH = 2000

HEIGHT = 1400

FPS = 60

SHOW_SENSORS = True


# ============================================================
# NETWORK
# ============================================================

# 9 wall sensors
# 1 speed
# 2 checkpoint direction
# 1 checkpoint distance
# 1 checkpoint index
# 1 current steering
# 1 lateral speed
#
# Total = 16

INPUT_SIZE = 16

HIDDEN_SIZE_1 = 64

HIDDEN_SIZE_2 = 64

OUTPUT_SIZE = 3


class RacingAI(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                INPUT_SIZE,
                HIDDEN_SIZE_1
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE_1,
                HIDDEN_SIZE_2
            ),

            nn.ReLU(),

            nn.Linear(
                HIDDEN_SIZE_2,
                OUTPUT_SIZE
            ),

            nn.Tanh()
        )


    def forward(self, inputs):

        return self.network(inputs)


# ============================================================
# SAVE AI
# ============================================================

def save_ai(
    network,
    fitness,
    generation,
    mutation_rate,
    mutation_strength,
    stagnation_count
):

    checkpoint = {

        "model_state_dict":
            copy.deepcopy(
                network.state_dict()
            ),

        "fitness":
            float(fitness),

        "generation":
            int(generation),

        "mutation_rate":
            float(mutation_rate),

        "mutation_strength":
            float(mutation_strength),

        "stagnation_count":
            int(stagnation_count)

    }

    torch.save(
        checkpoint,
        SAVE_PATH
    )

    print()
    print(">>> AI SAVED!")
    print(f">>> Fitness: {fitness:.2f}")
    print(f">>> Generation: {generation}")
    print(f">>> File: {SAVE_PATH}")
    print()


# ============================================================
# LOAD AI
# ============================================================

def load_ai():

    if not os.path.exists(SAVE_PATH):

        print()
        print("No previous AI save found.")
        print("Starting training from scratch.")
        print()

        return None


    try:

        checkpoint = torch.load(
            SAVE_PATH,
            map_location="cpu"
        )

        network = RacingAI()


        # ====================================================
        # NEW CHECKPOINT FORMAT
        # ====================================================

        if (
            isinstance(checkpoint, dict)
            and
            "model_state_dict" in checkpoint
        ):

            network.load_state_dict(
                checkpoint["model_state_dict"]
            )

            fitness = checkpoint.get(
                "fitness",
                -float("inf")
            )

            generation = checkpoint.get(
                "generation",
                0
            )

            mutation_rate = checkpoint.get(
                "mutation_rate",
                BASE_MUTATION_RATE
            )

            mutation_strength = checkpoint.get(
                "mutation_strength",
                BASE_MUTATION_STRENGTH
            )

            stagnation_count = checkpoint.get(
                "stagnation_count",
                0
            )


        # ====================================================
        # OLD SAVE FORMAT
        #
        # Your previous code saved only:
        #
        # network.state_dict()
        #
        # This keeps those saves compatible.
        # ====================================================

        else:

            network.load_state_dict(
                checkpoint
            )

            fitness = -float("inf")

            generation = 0

            mutation_rate = (
                BASE_MUTATION_RATE
            )

            mutation_strength = (
                BASE_MUTATION_STRENGTH
            )

            stagnation_count = 0


        print()
        print("==========================================")
        print("PREVIOUS AI FOUND")
        print("==========================================")
        print(
            f"Saved generation: {generation}"
        )
        print(
            f"Saved fitness:    {fitness:.2f}"
        )
        print("Loading previous AI...")
        print("==========================================")
        print()

        return {

            "network":
                network,

            "fitness":
                fitness,

            "generation":
                generation,

            "mutation_rate":
                mutation_rate,

            "mutation_strength":
                mutation_strength,

            "stagnation_count":
                stagnation_count

        }


    except Exception as error:

        print()
        print("==========================================")
        print("ERROR LOADING AI")
        print("==========================================")
        print(error)
        print()
        print("Starting training from scratch.")
        print("==========================================")
        print()

        return None


# ============================================================
# PYGAME INITIALIZATION
# ============================================================

pygame.init()

screen = pygame.display.set_mode(
    (
        WIDTH,
        HEIGHT
    )
)

pygame.display.set_caption(
    "AI Racing - Evolution Training"
)

clock = pygame.time.Clock()


# ============================================================
# LOAD TRACK
# ============================================================

try:

    track_original = pygame.image.load(
        TRACK_PATH
    ).convert_alpha()

except pygame.error as error:

    print("Could not load track.png")
    print(error)

    pygame.quit()

    raise SystemExit


# ============================================================
# SCALE TRACK
# ============================================================

scale = min(

    WIDTH
    / track_original.get_width(),

    HEIGHT
    / track_original.get_height()

)


track_width = int(
    track_original.get_width()
    * scale
)

track_height = int(
    track_original.get_height()
    * scale
)


track = pygame.transform.scale(
    track_original,
    (
        track_width,
        track_height
    )
)


track_x = (
    WIDTH
    - track_width
) // 2


track_y = (
    HEIGHT
    - track_height
) // 2


track_mask = pygame.mask.from_surface(
    track
)


# ============================================================
# LOAD AI CAR IMAGE
# ============================================================

try:

    car_image_original = pygame.image.load(
        CAR_PATH
    ).convert_alpha()

except pygame.error as error:

    print("Could not load aicar.png")
    print(error)

    pygame.quit()

    raise SystemExit


# Keep the car reasonably small relative to the track.
car_image_max_width = 14
car_image_max_height = 34

car_scale = min(
    car_image_max_width / car_image_original.get_width(),
    car_image_max_height / car_image_original.get_height(),
    1.0
)

car_image = pygame.transform.smoothscale(
    car_image_original,
    (
        max(
            1,
            int(
                car_image_original.get_width()
                * car_scale
            )
        ),

        max(
            1,
            int(
                car_image_original.get_height()
                * car_scale
            )
        )
    )
)


# ============================================================
# COLORS
# ============================================================

GRASS = (55, 125, 60)

GRASS_DARK = (48, 112, 53)

WHITE = (255, 255, 255)

BLACK = (20, 20, 20)

YELLOW = (255, 220, 50)

GREEN = (70, 240, 100)

RED = (240, 70, 70)

BLUE = (80, 170, 255)


# ============================================================
# CHECKPOINTS
# ============================================================

CHECKPOINTS = [

    # Start / Finish
    (15, 110, 60, 10),

    # Checkpoint 1
    (100, 10, 10, 30),

    # Checkpoint 2
    (400, 40, 10, 100),

    # Checkpoint 3
    (350, 200, 120, 10),

    # Checkpoint 4
    (200, 200, 10, 100),

]


# ============================================================
# START POSITION
# ============================================================

SPAWN_X = 30

SPAWN_Y = 125

SPAWN_ANGLE = 45


START_X = (
    track_x
    + SPAWN_X * scale
)

START_Y = (
    track_y
    + SPAWN_Y * scale
)


# ============================================================
# TRACK COLLISION
# ============================================================

def is_on_track(
    x,
    y
):

    local_x = int(
        x - track_x
    )

    local_y = int(
        y - track_y
    )

    if (
        local_x < 0
        or local_y < 0
        or local_x >= track_width
        or local_y >= track_height
    ):

        return False

    return track_mask.get_at(
        (
            local_x,
            local_y
        )
    )


# ============================================================
# CHECKPOINT CENTER
# ============================================================

def checkpoint_center(
    checkpoint
):

    x, y, width, height = checkpoint

    return (

        track_x
        + (x + width / 2)
        * scale,

        track_y
        + (y + height / 2)
        * scale

    )


# ============================================================
# TRAINING CAR
# ============================================================

class TrainingCar:

    def __init__(
        self,
        network
    ):

        self.network = network

        self.x = START_X
        self.y = START_Y
        self.angle = SPAWN_ANGLE

        # ----------------------------------------------------
        # Real-ish car physics
        # ----------------------------------------------------

        self.velocity_x = 0.0
        self.velocity_y = 0.0

        self.speed = 0.0
        self.forward_speed = 0.0
        self.lateral_speed = 0.0

        self.steering = 0.0
        self.throttle = 0.0
        self.brake = 0.0

        self.angular_velocity = 0.0

        # ----------------------------------------------------
        # Checkpoint state
        # ----------------------------------------------------

        self.next_checkpoint = 1
        self.checkpoints_reached = 0
        self.lap_completed = False

        self.alive = True
        self.steps_alive = 0
        self.distance_traveled = 0.0

        self.checkpoint_progress_reward = 0.0
        self.previous_checkpoint_distance = None
        self.closest_checkpoint_distance = None

        self.stuck_steps = 0
        self.total_reward = 0.0
        self.moving_steps = 0


# ============================================================
# RAYCAST
# ============================================================

def cast_ray(
    car,
    relative_angle,
    max_distance=180
):

    ray_angle = (
        car.angle
        + relative_angle
    )

    radians = math.radians(
        ray_angle
    )

    direction_x = math.sin(
        radians
    )

    direction_y = -math.cos(
        radians
    )

    distance = 0

    while distance < max_distance:

        test_x = (
            car.x
            + direction_x * distance
        )

        test_y = (
            car.y
            + direction_y * distance
        )

        if not is_on_track(
            test_x,
            test_y
        ):

            break

        distance += 4

    return (
        distance
        / max_distance
    )


# ============================================================
# SENSOR ANGLES
# ============================================================

SENSOR_ANGLES = [

    -90,
    -67.5,
    -45,
    -22.5,
    0,
    22.5,
    45,
    67.5,
    90

]


# ============================================================
# GET SENSORS
# ============================================================

def get_sensors(
    car
):

    sensors = []

    for relative_angle in SENSOR_ANGLES:

        sensors.append(
            cast_ray(
                car,
                relative_angle
            )
        )

    return sensors


# ============================================================
# CHECKPOINT DATA
# ============================================================

def get_checkpoint_data(
    car
):

    checkpoint = CHECKPOINTS[
        car.next_checkpoint
    ]

    target_x, target_y = checkpoint_center(
        checkpoint
    )

    dx = (
        target_x
        - car.x
    )

    dy = (
        target_y
        - car.y
    )

    distance = math.sqrt(
        dx * dx
        + dy * dy
    )

    target_angle = math.degrees(

        math.atan2(
            dx,
            -dy
        )

    )

    angle_difference = (
        target_angle
        - car.angle
    )

    while angle_difference > 180:

        angle_difference -= 360


    while angle_difference < -180:

        angle_difference += 360


    angle_normalized = (
        angle_difference
        / 180
    )

    distance_normalized = min(

        distance
        / 1000,

        1.0

    )

    return (
        angle_normalized,
        distance_normalized,
        distance
    )


# ============================================================
# NETWORK INPUTS
# ============================================================

def get_inputs(
    car
):

    sensors = get_sensors(car)

    (
        angle_to_checkpoint,
        distance_to_checkpoint,
        _
    ) = get_checkpoint_data(car)

    speed_normalized = max(
        0.0,
        min(
            car.forward_speed / MAX_SPEED,
            1.0
        )
    )

    lateral_speed_normalized = max(
        -1.0,
        min(
            car.lateral_speed / MAX_SPEED,
            1.0
        )
    )

    inputs = [

        # WALL SENSORS

        sensors[0],
        sensors[1],
        sensors[2],
        sensors[3],
        sensors[4],
        sensors[5],
        sensors[6],
        sensors[7],
        sensors[8],

        # FORWARD SPEED

        speed_normalized,

        # CHECKPOINT DIRECTION

        math.sin(
            angle_to_checkpoint * math.pi
        ),

        math.cos(
            angle_to_checkpoint * math.pi
        ),

        # CHECKPOINT DISTANCE

        distance_to_checkpoint,

        # CHECKPOINT INDEX

        car.next_checkpoint / max(
            1,
            len(CHECKPOINTS) - 1
        ),

        # CURRENT STEERING

        car.steering,

        # LATERAL SPEED / SLIP

        lateral_speed_normalized

    ]

    return torch.tensor(
        inputs,
        dtype=torch.float32
    )


# ============================================================
# CHECKPOINT DETECTION
# ============================================================

def inside_checkpoint(
    car,
    checkpoint
):

    x, y, width, height = checkpoint

    screen_x = (
        track_x
        + x * scale
    )

    screen_y = (
        track_y
        + y * scale
    )

    screen_width = (
        width * scale
    )

    screen_height = (
        height * scale
    )

    return (

        screen_x
        <= car.x
        <= screen_x + screen_width

        and

        screen_y
        <= car.y
        <= screen_y + screen_height

    )


# ============================================================
# CHECKPOINT REWARD
# ============================================================

def reward_checkpoint_progress(
    car,
    distance_to_checkpoint
):

    if car.previous_checkpoint_distance is None:

        car.previous_checkpoint_distance = (
            distance_to_checkpoint
        )

        car.closest_checkpoint_distance = (
            distance_to_checkpoint
        )

        return


    # ========================================================
    # DID WE GET CLOSER?
    # ========================================================

    if distance_to_checkpoint < car.closest_checkpoint_distance:

        improvement = (

            car.closest_checkpoint_distance
            - distance_to_checkpoint

        )

        car.checkpoint_progress_reward += (

            improvement
            * 10.0

        )

        car.closest_checkpoint_distance = (
            distance_to_checkpoint
        )


    # ========================================================
    # DID WE MOVE AWAY?
    # ========================================================

    elif distance_to_checkpoint > car.previous_checkpoint_distance:

        regression = (

            distance_to_checkpoint
            - car.previous_checkpoint_distance

        )

        car.checkpoint_progress_reward -= (

            regression
            * 3.2

        )


    car.previous_checkpoint_distance = (
        distance_to_checkpoint
    )


# ============================================================
# UPDATE CAR
# ============================================================

def update_car(
    car
):

    if not car.alive:

        return

    old_x = car.x
    old_y = car.y


    # ========================================================
    # NETWORK
    # ========================================================

    inputs = get_inputs(car)

    with torch.no_grad():

        outputs = car.network(inputs)


    steering_target = outputs[0].item()

    throttle_target = (
        outputs[1].item()
        + 1.0
    ) / 2.0

    brake_target = (
        outputs[2].item()
        + 1.0
    ) / 2.0


    car.steering = max(
        -1.0,
        min(
            1.0,
            steering_target
        )
    )

    car.throttle = max(
        0.0,
        min(
            1.0,
            throttle_target
        )
    )

    car.brake = max(
        0.0,
        min(
            1.0,
            brake_target
        )
    )


    # ========================================================
    # HEADING VECTORS
    # ========================================================

    radians = math.radians(
        car.angle
    )

    forward_x = math.sin(
        radians
    )

    forward_y = -math.cos(
        radians
    )

    right_x = math.cos(
        radians
    )

    right_y = math.sin(
        radians
    )


    # ========================================================
    # DECOMPOSE VELOCITY
    # ========================================================

    car.forward_speed = (

        car.velocity_x * forward_x
        + car.velocity_y * forward_y

    )

    car.lateral_speed = (

        car.velocity_x * right_x
        + car.velocity_y * right_y

    )


    # ========================================================
    # ENGINE
    # ========================================================

    speed_ratio = max(

        0.0,

        min(

            abs(car.forward_speed)
            / MAX_SPEED,

            1.0

        )

    )

    engine_effectiveness = max(

        0.0,

        1.0 - speed_ratio

    )

    engine_acceleration = (

        car.throttle
        * ENGINE_ACCELERATION
        * (
            0.25
            + 0.75
            * engine_effectiveness
        )

    )

    car.velocity_x += (
        forward_x
        * engine_acceleration
    )

    car.velocity_y += (
        forward_y
        * engine_acceleration
    )


    # ========================================================
    # BRAKES
    # ========================================================

    velocity_magnitude = math.sqrt(

        car.velocity_x ** 2
        + car.velocity_y ** 2

    )

    if (
        velocity_magnitude > 0.001
        and car.brake > 0.0
    ):

        brake_acceleration = (
            car.brake
            * 0.42
        )

        car.velocity_x -= (

            car.velocity_x
            / velocity_magnitude
            * brake_acceleration

        )

        car.velocity_y -= (

            car.velocity_y
            / velocity_magnitude
            * brake_acceleration

        )


    # ========================================================
    # DRAG
    # ========================================================

    car.velocity_x *= DRAG
    car.velocity_y *= DRAG


    # ========================================================
    # TIRE GRIP
    # ========================================================

    grip = 0.20

    lateral_velocity_x = (
        right_x
        * car.lateral_speed
    )

    lateral_velocity_y = (
        right_y
        * car.lateral_speed
    )

    car.velocity_x -= (
        lateral_velocity_x
        * grip
    )

    car.velocity_y -= (
        lateral_velocity_y
        * grip
    )


    # ========================================================
    # STEERING
    # ========================================================

    speed_for_steering = max(

        0.0,

        min(

            abs(car.forward_speed)
            / MAX_SPEED,

            1.0

        )

    )

    steering_strength = (

        0.20
        + 0.80
        * speed_for_steering

    )

    target_angular_velocity = (

        car.steering
        * MAX_STEERING_RATE
        * steering_strength

    )

    car.angular_velocity += (

        target_angular_velocity
        - car.angular_velocity

    ) * 0.22

    car.angle += (
        car.angular_velocity
    )


    # Keep angle manageable.

    if car.angle > 360:

        car.angle -= 360

    elif car.angle < -360:

        car.angle += 360


    # ========================================================
    # RECOMPUTE VELOCITY AFTER STEERING
    # ========================================================

    radians = math.radians(
        car.angle
    )

    forward_x = math.sin(
        radians
    )

    forward_y = -math.cos(
        radians
    )

    right_x = math.cos(
        radians
    )

    right_y = math.sin(
        radians
    )

    car.forward_speed = (

        car.velocity_x
        * forward_x

        +

        car.velocity_y
        * forward_y

    )

    car.lateral_speed = (

        car.velocity_x
        * right_x

        +

        car.velocity_y
        * right_y

    )


    # ========================================================
    # TOP SPEED
    # ========================================================

    velocity_magnitude = math.sqrt(

        car.velocity_x ** 2
        + car.velocity_y ** 2

    )

    if velocity_magnitude > MAX_SPEED:

        scale_down = (
            MAX_SPEED
            / velocity_magnitude
        )

        car.velocity_x *= (
            scale_down
        )

        car.velocity_y *= (
            scale_down
        )

        velocity_magnitude = MAX_SPEED

    car.speed = velocity_magnitude


    # ========================================================
    # MOVEMENT
    # ========================================================

    car.x += car.velocity_x
    car.y += car.velocity_y


    # ========================================================
    # COLLISION
    # ========================================================

    if not is_on_track(
        car.x,
        car.y
    ):

        car.x = old_x
        car.y = old_y

        car.velocity_x *= 0.15
        car.velocity_y *= 0.15

        car.speed *= 0.15

        car.alive = False

        return


    # ========================================================
    # DISTANCE TRAVELED
    # ========================================================

    movement = math.sqrt(

        (car.x - old_x) ** 2
        +

        (car.y - old_y) ** 2

    )

    car.distance_traveled += movement

    car.steps_alive += 1


    if movement > 0.1:

        car.moving_steps += 1


    # ========================================================
    # STUCK DETECTION
    # ========================================================

    if movement < 0.04:

        car.stuck_steps += 1

    else:

        car.stuck_steps = 0


    if car.stuck_steps > 180:

        car.alive = False

        return


    # ========================================================
    # CURRENT CHECKPOINT
    # ========================================================

    checkpoint = CHECKPOINTS[
        car.next_checkpoint
    ]

    target_x, target_y = checkpoint_center(
        checkpoint
    )

    distance_to_checkpoint = math.sqrt(

        (target_x - car.x) ** 2
        +

        (target_y - car.y) ** 2

    )


    # ========================================================
    # REWARD PROGRESS
    # ========================================================

    reward_checkpoint_progress(

        car,

        distance_to_checkpoint

    )


    # ========================================================
    # CHECKPOINT HIT
    # ========================================================

    if inside_checkpoint(
        car,
        checkpoint
    ):

        car.checkpoints_reached += 1

        checkpoint_number = (
            car.checkpoints_reached
        )

        car.checkpoint_progress_reward += (

            5000
            * checkpoint_number

        )


        if car.next_checkpoint == (

            len(CHECKPOINTS) - 1

        ):

            car.next_checkpoint = 0


        elif car.next_checkpoint == 0:

            car.lap_completed = True

            car.checkpoint_progress_reward += (
                50000
            )

            car.next_checkpoint = 1


        else:

            car.next_checkpoint += 1


        car.previous_checkpoint_distance = None

        car.closest_checkpoint_distance = None


# ============================================================
# FITNESS
# ============================================================

def calculate_fitness(
    car
):

    fitness = 0.0


    # ========================================================
    # CHECKPOINT PROGRESS
    # ========================================================

    fitness += (

        car.checkpoint_progress_reward

    )


    # ========================================================
    # CHECKPOINT COUNT
    # ========================================================

    for checkpoint_number in range(

        1,

        car.checkpoints_reached + 1

    ):

        fitness += (

            checkpoint_number
            * 1000

        )


    # ========================================================
    # DISTANCE
    # ========================================================

    fitness += (

        car.distance_traveled
        * 0.10

    )


    # ========================================================
    # SURVIVAL
    # ========================================================

    fitness += (

        car.steps_alive
        * 0.01

    )


    # ========================================================
    # LAP BONUS
    # ========================================================

    if car.lap_completed:

        fitness += 50000


    return fitness


# ============================================================
# CREATE POPULATION
# ============================================================

def create_population(
    previous_network=None
):

    population = []


    # ========================================================
    # NO PREVIOUS AI
    # ========================================================

    if previous_network is None:

        for _ in range(
            POPULATION_SIZE
        ):

            population.append(
                RacingAI()
            )

        return population


    # ========================================================
    # START FROM PREVIOUS BEST
    # ========================================================

    print(
        "Building new population around previous best..."
    )


    # --------------------------------------------------------
    # Keep previous best completely unchanged.
    # --------------------------------------------------------

    population.append(

        copy.deepcopy(
            previous_network
        )

    )


    # --------------------------------------------------------
    # Create mutated versions of previous best.
    # --------------------------------------------------------

    while len(
        population
    ) < POPULATION_SIZE:

        child = copy.deepcopy(
            previous_network
        )

        child = mutate(

            child,

            BASE_MUTATION_RATE,

            BASE_MUTATION_STRENGTH

        )

        population.append(
            child
        )


    return population


# ============================================================
# CROSSOVER
# ============================================================

def crossover(
    parent_a,
    parent_b
):

    child = RacingAI()


    with torch.no_grad():

        child_parameters = list(
            child.parameters()
        )

        a_parameters = list(
            parent_a.parameters()
        )

        b_parameters = list(
            parent_b.parameters()
        )


        for (
            child_param,
            a_param,
            b_param
        ) in zip(

            child_parameters,
            a_parameters,
            b_parameters

        ):

            mask = (
                torch.rand_like(
                    child_param
                )
                < 0.5
            )

            child_param.copy_(

                torch.where(

                    mask,

                    a_param,

                    b_param

                )

            )


    return child


# ============================================================
# MUTATION
# ============================================================

def mutate(
    network,
    mutation_rate,
    mutation_strength
):

    child = copy.deepcopy(
        network
    )


    with torch.no_grad():

        for parameter in child.parameters():

            mutation_mask = (

                torch.rand_like(
                    parameter
                )

                < mutation_rate

            )

            noise = (

                torch.randn_like(
                    parameter
                )

                * mutation_strength

            )

            parameter += (

                mutation_mask
                * noise

            )


    return child


# ============================================================
# EVALUATE NETWORK
# ============================================================

def evaluate(
    network
):

    car = TrainingCar(
        network
    )


    for _ in range(
        MAX_STEPS
    ):

        update_car(
            car
        )


        if not car.alive:

            break


        if car.lap_completed:

            break


    fitness = calculate_fitness(
        car
    )


    return (
        fitness,
        car
    )


# ============================================================
# BACKGROUND
# ============================================================

def draw_background():

    screen.fill(
        GRASS
    )

    spacing = 50


    for y in range(

        -HEIGHT,
        HEIGHT * 2,
        spacing

    ):

        pygame.draw.line(

            screen,
            GRASS_DARK,

            (0, y),

            (
                WIDTH,
                y + WIDTH
            ),

            1

        )


# ============================================================
# DRAW CHECKPOINTS
# ============================================================

def draw_checkpoints():

    for index, checkpoint in enumerate(
        CHECKPOINTS
    ):

        x, y, width, height = checkpoint


        screen_x = (
            track_x
            + x * scale
        )

        screen_y = (
            track_y
            + y * scale
        )


        rect = pygame.Rect(

            int(screen_x),
            int(screen_y),

            max(
                1,
                int(width * scale)
            ),

            max(
                1,
                int(height * scale)
            )

        )


        pygame.draw.rect(

            screen,
            YELLOW,
            rect,
            2

        )


        label = font.render(

            str(index),
            True,
            WHITE

        )


        screen.blit(

            label,

            label.get_rect(
                center=rect.center
            )

        )


# ============================================================
# DRAW AI CAR
# ============================================================

def draw_car(
    car,
    highlighted=False
):

    image = car_image


    if not highlighted:

        image = image.copy()

        image.set_alpha(
            205
        )


    rotated = pygame.transform.rotate(

        image,

        -car.angle

    )


    rect = rotated.get_rect(

        center=(

            int(car.x),
            int(car.y)

        )

    )


    screen.blit(

        rotated,

        rect

    )


    if highlighted:

        pygame.draw.circle(

            screen,
            WHITE,

            (
                int(car.x),
                int(car.y)
            ),

            max(

                rotated.get_width(),
                rotated.get_height()

            ) // 2 + 5,

            2

        )


# ============================================================
# DRAW SENSORS
# ============================================================

def draw_sensors(
    car
):

    sensors = get_sensors(
        car
    )


    for (
        relative_angle,
        value
    ) in zip(

        SENSOR_ANGLES,
        sensors

    ):

        ray_angle = (

            car.angle
            + relative_angle

        )


        radians = math.radians(
            ray_angle
        )


        distance = (
            value
            * 180
        )


        end_x = (

            car.x
            + math.sin(radians)
            * distance

        )


        end_y = (

            car.y
            - math.cos(radians)
            * distance

        )


        pygame.draw.line(

            screen,
            YELLOW,

            (
                int(car.x),
                int(car.y)
            ),

            (
                int(end_x),
                int(end_y)
            ),

            1

        )


# ============================================================
# FONT
# ============================================================

font = pygame.font.SysFont(
    "consolas",
    20
)


small_font = pygame.font.SysFont(
    "consolas",
    16
)


# ============================================================
# DRAW HUD
# ============================================================

def draw_hud(

    generation,
    generation_best,
    average_fitness,
    global_best,
    alive_count,
    best_car,
    mutation_rate,
    mutation_strength,
    parent_pool_size,
    stagnation_count

):

    panel_width = 520
    panel_height = 410


    panel = pygame.Surface(

        (
            panel_width,
            panel_height
        ),

        pygame.SRCALPHA

    )


    panel.fill(
        (0, 0, 0, 185)
    )


    # ========================================================
    # CENTER HUD
    # ========================================================

    panel_x = (
        WIDTH
        - panel_width
    ) // 2


    panel_y = (
        HEIGHT
        - panel_height
    ) // 2


    screen.blit(

        panel,

        (
            panel_x,
            panel_y
        )

    )


    # ========================================================
    # TARGET DISTANCE
    # ========================================================

    checkpoint_distance = 0


    if (
        best_car.closest_checkpoint_distance
        is not None
    ):

        checkpoint_distance = (

            best_car.closest_checkpoint_distance

        )


    # ========================================================
    # HUD TEXT
    # ========================================================

    lines = [

        f"GENERATION: {generation}/{GENERATIONS}",

        f"BEST: {generation_best:.2f}",

        f"AVERAGE: {average_fitness:.2f}",

        f"GLOBAL BEST: {global_best:.2f}",

        f"ALIVE: {alive_count}/{POPULATION_SIZE}",

        f"CHECKPOINTS: "
        f"{best_car.checkpoints_reached}",

        f"NEXT CHECKPOINT: "
        f"{best_car.next_checkpoint}",

        f"TARGET DISTANCE: "
        f"{checkpoint_distance:.1f}",

        f"PROGRESS REWARD: "
        f"{best_car.checkpoint_progress_reward:.1f}",

        f"DISTANCE: "
        f"{best_car.distance_traveled:.1f}",

        f"SPEED: "
        f"{best_car.speed:.2f}",

        f"THROTTLE: "
        f"{best_car.throttle:.2f}",

        f"BRAKE: "
        f"{best_car.brake:.2f}",

        f"STEERING: "
        f"{best_car.steering:.2f}",

        f"SLIP: "
        f"{best_car.lateral_speed:.2f}",

        f"MUTATION: "
        f"{mutation_rate:.2f}",

        f"MUTATION POWER: "
        f"{mutation_strength:.2f}",

        f"PARENT POOL: "
        f"{parent_pool_size}",

        f"STAGNATION: "
        f"{stagnation_count}",

        "ESC = stop training"

    ]


    # ========================================================
    # DRAW TEXT
    # ========================================================

    for index, text in enumerate(
        lines
    ):

        rendered = font.render(

            text,
            True,
            WHITE

        )


        screen.blit(

            rendered,

            (
                panel_x + 12,
                panel_y + 10 + index * 20
            )

        )


# ============================================================
# HANDLE EVENTS
# ============================================================

def handle_events():

    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            pygame.quit()

            raise SystemExit


        if event.type == pygame.KEYDOWN:

            if event.key == pygame.K_ESCAPE:

                pygame.quit()

                raise SystemExit


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("==========================================")
    print("AI RACING TRAINING")
    print("==========================================")


    # ========================================================
    # LOAD PREVIOUS SAVE
    # ========================================================

    previous_save = None


    if AUTO_RESUME:

        previous_save = load_ai()


    # ========================================================
    # INITIALIZE TRAINING STATE
    # ========================================================

    if previous_save is not None:

        previous_network = (
            previous_save["network"]
        )

        global_best_fitness = (
            previous_save["fitness"]
        )

        global_best_network = copy.deepcopy(
            previous_network
        )

        starting_generation = (

            previous_save["generation"]
            + 1

        )

        mutation_rate = (
            previous_save["mutation_rate"]
        )

        mutation_strength = (
            previous_save["mutation_strength"]
        )

        stagnation_count = (
            previous_save["stagnation_count"]
        )

        population = create_population(

            previous_network

        )


    else:

        previous_network = None

        global_best_fitness = -float(
            "inf"
        )

        global_best_network = None

        starting_generation = 1

        mutation_rate = (
            BASE_MUTATION_RATE
        )

        mutation_strength = (
            BASE_MUTATION_STRENGTH
        )

        stagnation_count = 0

        population = create_population()


    print("==========================================")

    if previous_save is not None:

        print(
            f"RESUMING FROM GENERATION "
            f"{starting_generation}"
        )

        print(
            f"PREVIOUS GLOBAL BEST: "
            f"{global_best_fitness:.2f}"
        )

    else:

        print(
            "STARTING NEW TRAINING RUN"
        )

    print("==========================================")
    print()


    # ========================================================
    # GENERATIONS
    # ========================================================

    for generation in range(

        starting_generation,

        GENERATIONS + 1

    ):

        handle_events()


        # ====================================================
        # CREATE CARS
        # ====================================================

        cars = [

            TrainingCar(
                network
            )

            for network in population

        ]


        # ====================================================
        # RUN GENERATION
        # ====================================================

        for step in range(
            MAX_STEPS
        ):

            handle_events()


            alive_count = 0


            for car in cars:

                if car.alive:

                    update_car(
                        car
                    )


                if car.alive:

                    alive_count += 1


            # =================================================
            # DRAW
            # =================================================

            draw_background()


            screen.blit(

                track,

                (
                    track_x,
                    track_y
                )

            )


            draw_checkpoints()


            # Find current best.

            current_best_car = max(

                cars,

                key=lambda c:
                calculate_fitness(c)

            )


            # Draw cars.

            for car in cars:

                if not car.alive:

                    continue


                highlighted = (

                    car
                    is current_best_car

                )


                draw_car(

                    car,

                    highlighted

                )


            # Draw sensors on best car.

            if SHOW_SENSORS:

                if current_best_car.alive:

                    draw_sensors(

                        current_best_car

                    )


            # =================================================
            # TEMPORARY STATISTICS
            # =================================================

            current_fitnesses = [

                calculate_fitness(
                    car
                )

                for car in cars

            ]


            temporary_best = max(
                current_fitnesses
            )


            temporary_average = (

                sum(
                    current_fitnesses
                )
                / len(
                    current_fitnesses
                )

            )


            parent_pool_size = max(

                ELITE_COUNT,

                int(

                    POPULATION_SIZE
                    * PARENT_POOL_PERCENT

                )

            )


            draw_hud(

                generation,

                temporary_best,

                temporary_average,

                max(

                    global_best_fitness,

                    temporary_best

                ),

                alive_count,

                current_best_car,

                mutation_rate,

                mutation_strength,

                parent_pool_size,

                stagnation_count

            )


            pygame.display.flip()


            clock.tick(
                FPS
            )


            # Everyone died.

            if alive_count == 0:

                break


            # Someone completed a lap.

            if any(

                car.lap_completed

                for car in cars

            ):

                break


        # ====================================================
        # SCORE GENERATION
        # ====================================================

        results = []


        for car in cars:

            fitness = calculate_fitness(
                car
            )


            results.append(

                (
                    fitness,
                    car.network,
                    car
                )

            )


        results.sort(

            key=lambda item:
            item[0],

            reverse=True

        )


        generation_best_fitness = (
            results[0][0]
        )


        average_fitness = (

            sum(

                result[0]

                for result in results

            )

            / len(results)

        )


        best_car = results[0][2]


        # ====================================================
        # GLOBAL BEST
        # ====================================================

        improved = (

            generation_best_fitness
            > global_best_fitness

        )


        if improved:

            global_best_fitness = (
                generation_best_fitness
            )


            global_best_network = copy.deepcopy(

                results[0][1]

            )


            # =================================================
            # SAVE NEW GLOBAL BEST
            # =================================================

            save_ai(

                global_best_network,

                global_best_fitness,

                generation,

                mutation_rate,

                mutation_strength,

                0

            )


            stagnation_count = 0


        else:

            stagnation_count += 1


        # ====================================================
        # ADAPTIVE MUTATION
        # ====================================================

        if stagnation_count >= STAGNATION_LIMIT:

            mutation_rate = (
                HIGH_MUTATION_RATE
            )

            mutation_strength = (
                HIGH_MUTATION_STRENGTH
            )

        else:

            mutation_rate = (
                BASE_MUTATION_RATE
            )

            mutation_strength = (
                BASE_MUTATION_STRENGTH
            )


        # ====================================================
        # CONSOLE OUTPUT
        # ====================================================

        print()

        print(
            "=========================================="
        )

        print(
            f"GENERATION {generation}"
        )

        print(
            "=========================================="
        )

        print(
            f"Best fitness:      "
            f"{generation_best_fitness:.2f}"
        )

        print(
            f"Average fitness:   "
            f"{average_fitness:.2f}"
        )

        print(
            f"Global best:       "
            f"{global_best_fitness:.2f}"
        )

        print(
            f"Checkpoints:       "
            f"{best_car.checkpoints_reached}"
        )

        print(
            f"Next checkpoint:   "
            f"{best_car.next_checkpoint}"
        )

        print(
            f"Target distance:   "
            f"{(
                best_car.closest_checkpoint_distance
                if best_car.closest_checkpoint_distance
                is not None
                else 0
            ):.2f}"
        )

        print(
            f"Progress reward:   "
            f"{best_car.checkpoint_progress_reward:.2f}"
        )

        print(
            f"Distance:          "
            f"{best_car.distance_traveled:.2f}"
        )

        print(
            f"Alive steps:       "
            f"{best_car.steps_alive}"
        )

        print(
            f"Average speed:     "
            f"{(
                best_car.distance_traveled
                / max(
                    1,
                    best_car.steps_alive
                )
            ):.2f}"
        )

        print(
            f"Stagnation:        "
            f"{stagnation_count}"
        )

        print(
            f"Mutation rate:     "
            f"{mutation_rate:.2f}"
        )

        print(
            f"Mutation strength: "
            f"{mutation_strength:.2f}"
        )


        if improved:

            print(
                ">>> NEW GLOBAL BEST!"
            )

            print(
                ">>> SAVE UPDATED!"
            )


        # ====================================================
        # CREATE NEXT GENERATION
        # ====================================================

        parent_pool_size = max(

            ELITE_COUNT,

            int(

                POPULATION_SIZE
                * PARENT_POOL_PERCENT

            )

        )


        parent_pool = [

            network

            for _, network, _

            in results[
                :parent_pool_size
            ]

        ]


        elites = [

            network

            for _, network, _

            in results[
                :ELITE_COUNT
            ]

        ]


        new_population = []


        # ====================================================
        # PRESERVE ELITES
        # ====================================================

        for elite in elites:

            new_population.append(

                copy.deepcopy(
                    elite
                )

            )


        # ====================================================
        # RANDOM IMMIGRANTS
        # ====================================================

        immigrant_count = int(

            POPULATION_SIZE
            * RANDOM_IMMIGRANT_PERCENT

        )


        for _ in range(
            immigrant_count
        ):

            if len(new_population) >= POPULATION_SIZE:

                break


            new_population.append(
                RacingAI()
            )


        # ====================================================
        # CREATE CHILDREN
        # ====================================================

        while len(
            new_population
        ) < POPULATION_SIZE:

            parent_a = random.choice(
                parent_pool
            )

            parent_b = random.choice(
                parent_pool
            )


            child = crossover(

                parent_a,

                parent_b

            )


            child = mutate(

                child,

                mutation_rate,

                mutation_strength

            )


            new_population.append(
                child
            )


        population = new_population


    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print()

    print(
        "=========================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=========================================="
    )

    print(
        f"Best fitness: "
        f"{global_best_fitness:.2f}"
    )

    print(
        "Saved to:"
    )

    print(
        SAVE_PATH
    )

    print(
        "=========================================="
    )


    # ========================================================
    # KEEP WINDOW OPEN
    # ========================================================

    while True:

        handle_events()

        pygame.display.flip()

        clock.tick(
            FPS
        )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    train()