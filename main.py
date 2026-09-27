import pygame
import sys
import math
import os

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


AI_SAVE_PATH = os.path.join(
    BASE_DIR,
    "best_ai.pth"
)


CAR_PATH = os.path.join(
    BASE_DIR,
    "car.png"
)


AI_CAR_PATH = os.path.join(
    BASE_DIR,
    "aicar.png"
)


# ============================================================
# DISPLAY
# ============================================================

WIDTH = 2000
HEIGHT = 1400

FPS = 60


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
    "AI Racing"
)


clock = pygame.time.Clock()


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
    sys.exit()


# ============================================================
# SCALE TRACK
# ============================================================

original_width = (
    track_original.get_width()
)

original_height = (
    track_original.get_height()
)


scale = min(

    WIDTH / original_width,

    HEIGHT / original_height

)


track_width = int(
    original_width * scale
)


track_height = int(
    original_height * scale
)


track = pygame.transform.scale(

    track_original,

    (
        track_width,
        track_height
    )

)


# ============================================================
# POSITION TRACK
# ============================================================

track_x = (
    WIDTH - track_width
) // 2


track_y = (
    HEIGHT - track_height
) // 2


# ============================================================
# TRACK COLLISION MASK
# ============================================================

track_mask = pygame.mask.from_surface(
    track
)


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
# SPAWN POSITION
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
# LOAD PLAYER CAR
# ============================================================

try:

    car_original = pygame.image.load(
        CAR_PATH
    ).convert_alpha()

except pygame.error as error:

    print("Could not load car.png")
    print(error)

    pygame.quit()
    sys.exit()


# ============================================================
# PLAYER CAR SIZE
# ============================================================

CAR_WIDTH = 32

CAR_HEIGHT = 26


car_image = pygame.transform.scale(

    car_original,

    (
        CAR_WIDTH,
        CAR_HEIGHT
    )

)


# ============================================================
# LOAD AI CAR
# ============================================================

try:

    ai_car_original = pygame.image.load(
        AI_CAR_PATH
    ).convert_alpha()

except pygame.error as error:

    print("Could not load aicar.png")
    print(error)

    pygame.quit()
    sys.exit()


# ============================================================
# AI CAR SIZE
# ============================================================

AI_CAR_WIDTH = 14

AI_CAR_HEIGHT = 34


ai_car_image = pygame.transform.smoothscale(

    ai_car_original,

    (
        AI_CAR_WIDTH,
        AI_CAR_HEIGHT
    )

)


# ============================================================
# NEURAL NETWORK
#
# THIS MUST MATCH THE TRAINING NETWORK EXACTLY.
# ============================================================

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


    def forward(
        self,
        inputs
    ):

        return self.network(
            inputs
        )


# ============================================================
# LOAD TRAINED AI
# ============================================================

ai_network = RacingAI()

ai_loaded = False


if os.path.exists(AI_SAVE_PATH):

    try:

        state_dict = torch.load(

            AI_SAVE_PATH,

            map_location="cpu",

            weights_only=True

        )

        ai_network.load_state_dict(
            state_dict
        )

        ai_network.eval()

        ai_loaded = True

        print()
        print("==========================================")
        print("TRAINED AI LOADED")
        print("==========================================")
        print(
            f"Loaded from: {AI_SAVE_PATH}"
        )
        print("==========================================")
        print()

    except Exception as error:

        print()
        print("==========================================")
        print("COULD NOT LOAD TRAINED AI")
        print("==========================================")
        print(error)
        print()
        print("A fresh untrained network will be used.")
        print()

else:

    print()
    print("==========================================")
    print("NO TRAINED AI FOUND")
    print("==========================================")
    print(
        f"Expected file: {AI_SAVE_PATH}"
    )
    print()
    print(
        "Train the AI first, then restart main.py."
    )
    print("==========================================")
    print()


# ============================================================
# PLAYER STATE
# ============================================================

player_x = START_X

player_y = START_Y

player_angle = SPAWN_ANGLE

player_speed = 0.0


# ============================================================
# PLAYER PHYSICS
# ============================================================

ACCELERATION = 0.16

BRAKE_POWER = 0.25

FRICTION = 0.97

PLAYER_MAX_SPEED = 6.0

MAX_REVERSE_SPEED = 2.0

STEERING_SPEED = 2.0


# ============================================================
# AI PHYSICS
#
# THESE MATCH THE TRAINING CODE.
# ============================================================

MAX_SPEED = 10.0

ENGINE_ACCELERATION = 0.32

DRAG = 0.992

MAX_STEERING_RATE = 4.5

AI_GRIP = 0.20

AI_BRAKE_POWER = 0.42


# ============================================================
# AI SENSORS
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


SENSOR_MAX_DISTANCE = 180


# ============================================================
# AI CAR CLASS
# ============================================================

class AICar:

    def __init__(self):

        self.reset()


    def reset(self):

        self.x = START_X

        self.y = START_Y

        self.angle = SPAWN_ANGLE


        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        self.velocity_x = 0.0

        self.velocity_y = 0.0

        self.speed = 0.0

        self.forward_speed = 0.0

        self.lateral_speed = 0.0

        self.angular_velocity = 0.0


        # ----------------------------------------------------
        # AI outputs
        # ----------------------------------------------------

        self.steering = 0.0

        self.throttle = 0.0

        self.brake = 0.0


        # ----------------------------------------------------
        # Checkpoint state
        # ----------------------------------------------------

        self.next_checkpoint = 1

        self.checkpoints_reached = 0

        self.lap_completed = False


        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        self.alive = True


# ============================================================
# CREATE AI CAR
# ============================================================

ai_car = AICar()


ai_mode = False


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
# PLAYER CAR COLLISION
# ============================================================

def car_is_on_track():

    angle = math.radians(
        player_angle
    )


    forward_x = math.sin(
        angle
    )

    forward_y = -math.cos(
        angle
    )


    right_x = math.cos(
        angle
    )

    right_y = math.sin(
        angle
    )


    half_width = (
        CAR_WIDTH / 2
    )

    half_height = (
        CAR_HEIGHT / 2
    )


    corners = [

        (
            player_x
            + forward_x * half_height
            + right_x * half_width,

            player_y
            + forward_y * half_height
            + right_y * half_width

        ),

        (
            player_x
            + forward_x * half_height
            - right_x * half_width,

            player_y
            + forward_y * half_height
            - right_y * half_width

        ),

        (
            player_x
            - forward_x * half_height
            + right_x * half_width,

            player_y
            - forward_y * half_height
            + right_y * half_width

        ),

        (
            player_x
            - forward_x * half_height
            - right_x * half_width,

            player_y
            - forward_y * half_height
            - right_y * half_width

        )

    ]


    for corner in corners:

        if not is_on_track(
            corner[0],
            corner[1]
        ):

            return False


    return True


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
# CHECKPOINT COLLISION
# ============================================================

def is_inside_checkpoint(
    x,
    y,
    checkpoint
):

    checkpoint_x, checkpoint_y, width, height = checkpoint


    screen_x = (
        track_x
        + checkpoint_x * scale
    )


    screen_y = (
        track_y
        + checkpoint_y * scale
    )


    screen_width = (
        width * scale
    )


    screen_height = (
        height * scale
    )


    rect = pygame.Rect(

        int(screen_x),
        int(screen_y),

        max(
            1,
            int(screen_width)
        ),

        max(
            1,
            int(screen_height)
        )

    )


    return rect.collidepoint(

        int(x),
        int(y)

    )


# ============================================================
# AI RAYCAST
# ============================================================

def cast_ray(
    car,
    relative_angle,
    max_distance=SENSOR_MAX_DISTANCE
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
# AI SENSORS
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
# AI CHECKPOINT DATA
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

        distance / 1000,

        1.0

    )


    return (

        angle_normalized,

        distance_normalized,

        distance

    )


# ============================================================
# AI NETWORK INPUTS
#
# MUST MATCH TRAINING EXACTLY.
# ============================================================

def get_ai_inputs(
    car
):

    sensors = get_sensors(
        car
    )


    (
        angle_to_checkpoint,
        distance_to_checkpoint,
        _
    ) = get_checkpoint_data(
        car
    )


    speed_normalized = max(

        0.0,

        min(

            car.forward_speed
            / MAX_SPEED,

            1.0

        )

    )


    lateral_speed_normalized = max(

        -1.0,

        min(

            car.lateral_speed
            / MAX_SPEED,

            1.0

        )

    )


    inputs = [

        # ----------------------------------------------------
        # 9 WALL SENSORS
        # ----------------------------------------------------

        sensors[0],
        sensors[1],
        sensors[2],
        sensors[3],
        sensors[4],
        sensors[5],
        sensors[6],
        sensors[7],
        sensors[8],


        # ----------------------------------------------------
        # SPEED
        # ----------------------------------------------------

        speed_normalized,


        # ----------------------------------------------------
        # CHECKPOINT DIRECTION
        # ----------------------------------------------------

        math.sin(
            angle_to_checkpoint * math.pi
        ),

        math.cos(
            angle_to_checkpoint * math.pi
        ),


        # ----------------------------------------------------
        # CHECKPOINT DISTANCE
        # ----------------------------------------------------

        distance_to_checkpoint,


        # ----------------------------------------------------
        # CHECKPOINT INDEX
        # ----------------------------------------------------

        car.next_checkpoint
        / max(
            1,
            len(CHECKPOINTS) - 1
        ),


        # ----------------------------------------------------
        # CURRENT STEERING
        # ----------------------------------------------------

        car.steering,


        # ----------------------------------------------------
        # LATERAL SPEED
        # ----------------------------------------------------

        lateral_speed_normalized

    ]


    return torch.tensor(

        inputs,

        dtype=torch.float32

    )


# ============================================================
# UPDATE AI
# ============================================================

def update_ai():

    if not ai_car.alive:

        return


    # ========================================================
    # NETWORK
    # ========================================================

    inputs = get_ai_inputs(
        ai_car
    )


    with torch.no_grad():

        outputs = ai_network(
            inputs
        )


    # --------------------------------------------------------
    # Output 0 = steering
    # --------------------------------------------------------

    steering_target = (
        outputs[0].item()
    )


    # --------------------------------------------------------
    # Output 1 = throttle
    #
    # Network output:
    #
    #     -1 to +1
    #
    # Convert to:
    #
    #      0 to 1
    # --------------------------------------------------------

    throttle_target = (

        outputs[1].item()
        + 1.0

    ) / 2.0


    # --------------------------------------------------------
    # Output 2 = brake
    # --------------------------------------------------------

    brake_target = (

        outputs[2].item()
        + 1.0

    ) / 2.0


    ai_car.steering = max(

        -1.0,

        min(
            1.0,
            steering_target
        )

    )


    ai_car.throttle = max(

        0.0,

        min(
            1.0,
            throttle_target
        )

    )


    ai_car.brake = max(

        0.0,

        min(
            1.0,
            brake_target
        )

    )


    # ========================================================
    # SAVE OLD POSITION
    # ========================================================

    old_x = ai_car.x

    old_y = ai_car.y


    # ========================================================
    # HEADING VECTORS
    # ========================================================

    radians = math.radians(
        ai_car.angle
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

    ai_car.forward_speed = (

        ai_car.velocity_x
        * forward_x

        +

        ai_car.velocity_y
        * forward_y

    )


    ai_car.lateral_speed = (

        ai_car.velocity_x
        * right_x

        +

        ai_car.velocity_y
        * right_y

    )


    # ========================================================
    # ENGINE
    # ========================================================

    speed_ratio = max(

        0.0,

        min(

            abs(
                ai_car.forward_speed
            )
            / MAX_SPEED,

            1.0

        )

    )


    engine_effectiveness = max(

        0.0,

        1.0
        - speed_ratio

    )


    engine_acceleration = (

        ai_car.throttle

        * ENGINE_ACCELERATION

        * (

            0.25
            + 0.75
            * engine_effectiveness

        )

    )


    ai_car.velocity_x += (

        forward_x
        * engine_acceleration

    )


    ai_car.velocity_y += (

        forward_y
        * engine_acceleration

    )


    # ========================================================
    # BRAKES
    # ========================================================

    velocity_magnitude = math.sqrt(

        ai_car.velocity_x ** 2
        + ai_car.velocity_y ** 2

    )


    if (

        velocity_magnitude > 0.001

        and ai_car.brake > 0.0

    ):

        brake_acceleration = (

            ai_car.brake
            * AI_BRAKE_POWER

        )


        ai_car.velocity_x -= (

            ai_car.velocity_x
            / velocity_magnitude
            * brake_acceleration

        )


        ai_car.velocity_y -= (

            ai_car.velocity_y
            / velocity_magnitude
            * brake_acceleration

        )


    # ========================================================
    # DRAG
    # ========================================================

    ai_car.velocity_x *= DRAG

    ai_car.velocity_y *= DRAG


    # ========================================================
    # TIRE GRIP
    # ========================================================

    lateral_velocity_x = (

        right_x
        * ai_car.lateral_speed

    )


    lateral_velocity_y = (

        right_y
        * ai_car.lateral_speed

    )


    ai_car.velocity_x -= (

        lateral_velocity_x
        * AI_GRIP

    )


    ai_car.velocity_y -= (

        lateral_velocity_y
        * AI_GRIP

    )


    # ========================================================
    # STEERING
    # ========================================================

    speed_for_steering = max(

        0.0,

        min(

            abs(
                ai_car.forward_speed
            )
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

        ai_car.steering

        * MAX_STEERING_RATE

        * steering_strength

    )


    ai_car.angular_velocity += (

        target_angular_velocity
        - ai_car.angular_velocity

    ) * 0.22


    ai_car.angle += (
        ai_car.angular_velocity
    )


    # ========================================================
    # KEEP ANGLE MANAGEABLE
    # ========================================================

    if ai_car.angle > 360:

        ai_car.angle -= 360


    elif ai_car.angle < -360:

        ai_car.angle += 360


    # ========================================================
    # RECOMPUTE VELOCITY
    # ========================================================

    radians = math.radians(
        ai_car.angle
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


    ai_car.forward_speed = (

        ai_car.velocity_x
        * forward_x

        +

        ai_car.velocity_y
        * forward_y

    )


    ai_car.lateral_speed = (

        ai_car.velocity_x
        * right_x

        +

        ai_car.velocity_y
        * right_y

    )


    # ========================================================
    # TOP SPEED
    # ========================================================

    velocity_magnitude = math.sqrt(

        ai_car.velocity_x ** 2
        + ai_car.velocity_y ** 2

    )


    if velocity_magnitude > MAX_SPEED:

        scale_down = (

            MAX_SPEED
            / velocity_magnitude

        )


        ai_car.velocity_x *= (
            scale_down
        )


        ai_car.velocity_y *= (
            scale_down
        )


        velocity_magnitude = MAX_SPEED


    ai_car.speed = (
        velocity_magnitude
    )


    # ========================================================
    # MOVEMENT
    # ========================================================

    ai_car.x += ai_car.velocity_x

    ai_car.y += ai_car.velocity_y


    # ========================================================
    # COLLISION
    # ========================================================

    if not is_on_track(

        ai_car.x,
        ai_car.y

    ):

        ai_car.x = old_x

        ai_car.y = old_y


        ai_car.velocity_x *= 0.15

        ai_car.velocity_y *= 0.15

        ai_car.speed *= 0.15


        ai_car.alive = False

        return


    # ========================================================
    # CHECKPOINT
    # ========================================================

    checkpoint = CHECKPOINTS[
        ai_car.next_checkpoint
    ]


    if is_inside_checkpoint(

        ai_car.x,
        ai_car.y,

        checkpoint

    ):

        ai_car.checkpoints_reached += 1


        if ai_car.next_checkpoint == (

            len(CHECKPOINTS) - 1

        ):

            ai_car.next_checkpoint = 0


        elif ai_car.next_checkpoint == 0:

            ai_car.lap_completed = True

            ai_car.next_checkpoint = 1


        else:

            ai_car.next_checkpoint += 1


# ============================================================
# DRAW AI SENSORS
# ============================================================

def draw_ai_sensors():

    if not ai_car.alive:

        return


    sensors = get_sensors(
        ai_car
    )


    for relative_angle, value in zip(

        SENSOR_ANGLES,

        sensors

    ):

        ray_angle = (

            ai_car.angle
            + relative_angle

        )


        radians = math.radians(
            ray_angle
        )


        distance = (

            value
            * SENSOR_MAX_DISTANCE

        )


        end_x = (

            ai_car.x

            +

            math.sin(radians)
            * distance

        )


        end_y = (

            ai_car.y

            -

            math.cos(radians)
            * distance

        )


        pygame.draw.line(

            screen,

            YELLOW,

            (
                int(ai_car.x),
                int(ai_car.y)
            ),

            (
                int(end_x),
                int(end_y)
            ),

            1

        )


# ============================================================
# DRAW AI CAR
# ============================================================

def draw_ai_car():

    if not ai_car.alive:

        return


    rotated = pygame.transform.rotate(

        ai_car_image,

        -ai_car.angle

    )


    rect = rotated.get_rect(

        center=(

            int(ai_car.x),
            int(ai_car.y)

        )

    )


    screen.blit(

        rotated,

        rect

    )


    # --------------------------------------------------------
    # Highlight ring
    # --------------------------------------------------------

    pygame.draw.circle(

        screen,

        WHITE,

        (
            int(ai_car.x),
            int(ai_car.y)
        ),

        max(

            rotated.get_width(),
            rotated.get_height()

        ) // 2 + 5,

        2

    )


# ============================================================
# RACE TIMING
# ============================================================

race_start_time = None

current_lap = 0

next_checkpoint = 0

lap_start_time = None

current_split_times = {}

best_split_times = {}

best_lap_time = None

last_split_time = None

last_split_delta = None

previous_checkpoint_position = (

    player_x,
    player_y

)


# ============================================================
# RACE TIME
# ============================================================

def get_race_time():

    if race_start_time is None:

        return 0.0


    return (

        pygame.time.get_ticks()
        - race_start_time

    ) / 1000.0


# ============================================================
# LAP TIME
# ============================================================

def get_lap_time():

    if lap_start_time is None:

        return 0.0


    return (

        pygame.time.get_ticks()
        - lap_start_time

    ) / 1000.0


# ============================================================
# PROCESS PLAYER CHECKPOINTS
# ============================================================

def process_checkpoints():

    global race_start_time

    global lap_start_time

    global current_lap

    global next_checkpoint

    global current_split_times

    global best_split_times

    global best_lap_time

    global last_split_time

    global last_split_delta

    global previous_checkpoint_position


    current_position = (

        player_x,
        player_y

    )


    previous_position = (

        previous_checkpoint_position[0],
        previous_checkpoint_position[1]

    )


    checkpoint = CHECKPOINTS[
        next_checkpoint
    ]


    was_inside = is_inside_checkpoint(

        previous_position[0],
        previous_position[1],

        checkpoint

    )


    is_inside = is_inside_checkpoint(

        current_position[0],
        current_position[1],

        checkpoint

    )


    previous_checkpoint_position = (
        current_position
    )


    if not is_inside:

        return


    if was_inside:

        return


    # ========================================================
    # START / FINISH
    # ========================================================

    if next_checkpoint == 0:

        if race_start_time is None:

            race_start_time = (
                pygame.time.get_ticks()
            )

            lap_start_time = (
                pygame.time.get_ticks()
            )

            current_lap = 1

            next_checkpoint = 1

            current_split_times = {}

            print("RACE STARTED")

            return


        completed_lap_time = (
            get_lap_time()
        )


        print()

        print(

            f"LAP {current_lap} COMPLETE: "
            f"{completed_lap_time:.3f}s"

        )


        if (

            best_lap_time is None

            or

            completed_lap_time
            < best_lap_time

        ):

            best_lap_time = (
                completed_lap_time
            )


            print(

                f"NEW BEST LAP: "
                f"{best_lap_time:.3f}s"

            )


        current_lap += 1

        lap_start_time = (
            pygame.time.get_ticks()
        )

        current_split_times = {}

        next_checkpoint = 1


        print(

            f"STARTING LAP {current_lap}"

        )


        return


    # ========================================================
    # NORMAL CHECKPOINT
    # ========================================================

    split_time = get_lap_time()


    current_split_times[
        next_checkpoint
    ] = split_time


    previous_best = best_split_times.get(
        next_checkpoint
    )


    if previous_best is None:

        best_split_times[
            next_checkpoint
        ] = split_time


        last_split_time = split_time

        last_split_delta = None


        print(

            f"CHECKPOINT {next_checkpoint}: "
            f"{split_time:.3f}s "
            f"(NEW BEST)"

        )


    else:

        delta = (

            split_time
            - previous_best

        )


        last_split_time = split_time

        last_split_delta = delta


        if split_time < previous_best:

            best_split_times[
                next_checkpoint
            ] = split_time


            print(

                f"CHECKPOINT {next_checkpoint}: "
                f"{split_time:.3f}s "
                f"({delta:+.3f}s) "
                f"NEW BEST"

            )


        else:

            print(

                f"CHECKPOINT {next_checkpoint}: "
                f"{split_time:.3f}s "
                f"({delta:+.3f}s)"

            )


    next_checkpoint += 1


    if next_checkpoint >= len(
        CHECKPOINTS
    ):

        next_checkpoint = 0


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
# DRAW PLAYER
# ============================================================

def draw_player():

    rotated_car = pygame.transform.rotate(

        car_image,

        -player_angle + 90

    )


    car_rect = rotated_car.get_rect(

        center=(

            int(player_x),
            int(player_y)

        )

    )


    screen.blit(

        rotated_car,

        car_rect

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


        screen_width = (

            width * scale

        )


        screen_height = (

            height * scale

        )


        rect = pygame.Rect(

            int(screen_x),
            int(screen_y),

            max(
                1,
                int(screen_width)
            ),

            max(
                1,
                int(screen_height)
            )

        )


        pygame.draw.rect(

            screen,

            YELLOW,

            rect,

            3

        )


        label = font.render(

            str(index),

            True,

            WHITE

        )


        label_rect = label.get_rect(

            center=rect.center

        )


        outline = font.render(

            str(index),

            True,

            BLACK

        )


        outline_rect = outline.get_rect(

            center=(

                label_rect.centerx + 2,

                label_rect.centery + 2

            )

        )


        screen.blit(

            outline,

            outline_rect

        )


        screen.blit(

            label,

            label_rect

        )


# ============================================================
# FORMAT TIME
# ============================================================

def format_time(
    seconds
):

    minutes = int(
        seconds // 60
    )


    remaining = seconds % 60


    return (

        f"{minutes}:{remaining:06.3f}"

    )


# ============================================================
# DRAW HUD
# ============================================================

def draw_debug():

    on_track = car_is_on_track()


    status = (

        "ON TRACK"

        if on_track

        else

        "OFF TRACK"

    )


    status_color = (

        GREEN

        if on_track

        else

        RED

    )


    lap_time = get_lap_time()


    # ========================================================
    # PANEL
    # ========================================================

    panel = pygame.Surface(

        (
            440,
            390
        ),

        pygame.SRCALPHA

    )


    panel.fill(

        (
            0,
            0,
            0,
            175
        )

    )


    screen.blit(

        panel,

        (
            12,
            12
        )

    )


    # ========================================================
    # PLAYER INFO
    # ========================================================

    lines = [

        (
            status,
            status_color
        ),

        (
            f"Lap: {current_lap}",
            WHITE
        ),

        (
            f"Time: {format_time(lap_time)}",
            WHITE
        ),

        (
            f"Speed: {player_speed:.2f}",
            WHITE
        ),

        (
            f"Next CP: {next_checkpoint}",
            WHITE
        ),

        (
            "",
            WHITE
        ),

        (
            "F1 = Toggle AI",
            YELLOW
        ),

        (
            "F2 = Reset AI",
            YELLOW
        ),

    ]


    for index, (
        text,
        color
    ) in enumerate(lines):

        if not text:

            continue


        rendered = font.render(

            text,

            True,

            color

        )


        screen.blit(

            rendered,

            (
                24,
                20 + index * 25
            )

        )


    # ========================================================
    # AI INFO
    # ========================================================

    ai_panel_y = 230


    ai_status = (

        "AI MODE: ON"

        if ai_mode

        else

        "AI MODE: OFF"

    )


    ai_status_color = (

        YELLOW

        if ai_mode

        else

        WHITE

    )


    ai_lines = [

        (
            ai_status,
            ai_status_color
        ),

        (
            (
                "AI SAVE: LOADED"
                if ai_loaded
                else
                "AI SAVE: NOT FOUND"
            ),
            GREEN if ai_loaded else RED
        ),

        (
            f"AI Speed: {ai_car.speed:.2f}",
            WHITE
        ),

        (
            f"AI Forward: "
            f"{ai_car.forward_speed:.2f}",
            WHITE
        ),

        (
            f"AI Steering: "
            f"{ai_car.steering:+.2f}",
            WHITE
        ),

        (
            f"AI Throttle: "
            f"{ai_car.throttle:.2f}",
            WHITE
        ),

        (
            f"AI Brake: "
            f"{ai_car.brake:.2f}",
            WHITE
        ),

        (
            f"AI Checkpoints: "
            f"{ai_car.checkpoints_reached}",
            WHITE
        ),

        (
            f"AI Next CP: "
            f"{ai_car.next_checkpoint}",
            WHITE
        ),

    ]


    for index, (
        text,
        color
    ) in enumerate(ai_lines):

        rendered = font.render(

            text,

            True,

            color

        )


        screen.blit(

            rendered,

            (
                24,
                ai_panel_y + index * 25
            )

        )


# ============================================================
# EVENT HANDLING
# ============================================================

def handle_events():

    global ai_mode


    for event in pygame.event.get():

        if event.type == pygame.QUIT:

            pygame.quit()
            sys.exit()


        if event.type == pygame.KEYDOWN:

            # ------------------------------------------------
            # ESC
            # ------------------------------------------------

            if event.key == pygame.K_ESCAPE:

                pygame.quit()
                sys.exit()


            # ------------------------------------------------
            # F1 = Toggle AI
            # ------------------------------------------------

            if event.key == pygame.K_F1:

                ai_mode = not ai_mode


                if ai_mode:

                    ai_car.reset()


                    print()
                    print("AI MODE: ON")


                    if ai_loaded:

                        print(
                            "Using saved trained network."
                        )

                    else:

                        print(
                            "WARNING: No trained network "
                            "was loaded."
                        )


                else:

                    print(
                        "AI MODE: OFF"
                    )


            # ------------------------------------------------
            # F2 = Reset AI
            # ------------------------------------------------

            if event.key == pygame.K_F2:

                ai_car.reset()

                print(
                    "AI RESET"
                )


# ============================================================
# MAIN LOOP
# ============================================================

running = True


while running:

    # ========================================================
    # EVENTS
    # ========================================================

    handle_events()


    # ========================================================
    # PLAYER INPUT
    # ========================================================

    keys = pygame.key.get_pressed()


    # ========================================================
    # PLAYER ACCELERATION
    # ========================================================

    if keys[pygame.K_w]:

        player_speed += ACCELERATION


        if player_speed > PLAYER_MAX_SPEED:

            player_speed = PLAYER_MAX_SPEED


    elif keys[pygame.K_s]:

        player_speed -= BRAKE_POWER


        if player_speed < -MAX_REVERSE_SPEED:

            player_speed = (
                -MAX_REVERSE_SPEED
            )


    else:

        player_speed *= FRICTION


    if abs(player_speed) < 0.01:

        player_speed = 0


    # ========================================================
    # PLAYER STEERING
    # ========================================================

    if abs(player_speed) > 0.05:

        steering_amount = STEERING_SPEED


        direction = (

            1

            if player_speed > 0

            else

            -1

        )


        if keys[pygame.K_a]:

            player_angle -= (

                steering_amount
                * direction

            )


        if keys[pygame.K_d]:

            player_angle += (

                steering_amount
                * direction

            )


    # ========================================================
    # PLAYER MOVEMENT
    # ========================================================

    angle = math.radians(
        player_angle
    )


    velocity_x = (

        math.sin(angle)
        * player_speed

    )


    velocity_y = (

        -math.cos(angle)
        * player_speed

    )


    new_x = (

        player_x
        + velocity_x

    )


    new_y = (

        player_y
        + velocity_y

    )


    old_x = player_x

    old_y = player_y


    player_x = new_x

    player_y = new_y


    # ========================================================
    # PLAYER COLLISION
    # ========================================================

    if not car_is_on_track():

        player_x = old_x

        player_y = old_y

        player_speed *= 0.35


    # ========================================================
    # PLAYER CHECKPOINTS
    # ========================================================

    process_checkpoints()


    # ========================================================
    # AI
    # ========================================================

    if ai_mode:

        update_ai()


    # ========================================================
    # DRAW BACKGROUND
    # ========================================================

    draw_background()


    # ========================================================
    # DRAW TRACK
    # ========================================================

    screen.blit(

        track,

        (
            track_x,
            track_y
        )

    )


    # ========================================================
    # DRAW CHECKPOINTS
    # ========================================================

    draw_checkpoints()


    # ========================================================
    # DRAW PLAYER
    # ========================================================

    draw_player()


    # ========================================================
    # DRAW AI
    # ========================================================

    if ai_mode:

        draw_ai_sensors()

        draw_ai_car()


    # ========================================================
    # DRAW HUD
    # ========================================================

    draw_debug()


    # ========================================================
    # DISPLAY
    # ========================================================

    pygame.display.flip()


    clock.tick(FPS)


# ============================================================
# CLEANUP
# ============================================================

pygame.quit()

sys.exit()