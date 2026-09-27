import pygame
import math
import os
import torch
import torch.nn as nn


# ============================================================
# NEURAL NETWORK CONFIGURATION
# ============================================================

INPUT_SIZE = 12

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64

OUTPUT_SIZE = 2


# ============================================================
# NEURAL NETWORK
# ============================================================

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
# AI CAR
# ============================================================

class AICar:

    def __init__(
        self,
        x,
        y,
        angle=0
    ):

        self.x = x
        self.y = y

        self.start_x = x
        self.start_y = y
        self.start_angle = angle

        self.angle = angle

        self.speed = 0.0

        self.width = 24
        self.height = 40

        # ----------------------------------------------------
        # Physics
        # ----------------------------------------------------

        self.acceleration = 0.16
        self.brake_power = 0.25
        self.friction = 0.97

        self.max_speed = 6.0
        self.max_reverse_speed = 2.0

        self.steering_speed = 2.0

        # ----------------------------------------------------
        # Race state
        # ----------------------------------------------------

        self.current_checkpoint = 1

        self.lap = 1

        self.alive = True

        self.total_reward = 0.0

        self.last_x = x
        self.last_y = y

        # ----------------------------------------------------
        # Sensors
        # ----------------------------------------------------

        self.sensor_values = [
            0,
            0,
            0,
            0,
            0
        ]

        # ----------------------------------------------------
        # Network outputs
        # ----------------------------------------------------

        self.steering = 0.0
        self.throttle = 0.0


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.x = self.start_x
        self.y = self.start_y

        self.angle = self.start_angle

        self.speed = 0.0

        self.current_checkpoint = 1

        self.lap = 1

        self.alive = True

        self.total_reward = 0.0

        self.last_x = self.x
        self.last_y = self.y

        self.sensor_values = [
            0,
            0,
            0,
            0,
            0
        ]

        self.steering = 0.0
        self.throttle = 0.0


# ============================================================
# AI CONTROLLER
# ============================================================

class AIController:

    def __init__(
        self,
        track,
        track_mask,
        track_x,
        track_y,
        scale,
        checkpoints,
        model_path="best_ai.pth"
    ):

        self.track = track

        self.track_mask = track_mask

        self.track_x = track_x
        self.track_y = track_y

        self.scale = scale

        self.checkpoints = checkpoints

        # ----------------------------------------------------
        # Create network
        # ----------------------------------------------------

        self.model = RacingAI()

        # ----------------------------------------------------
        # Find trained model
        # ----------------------------------------------------

        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        model_file = os.path.join(
            base_dir,
            model_path
        )

        # ----------------------------------------------------
        # Load trained network if it exists
        # ----------------------------------------------------

        if os.path.exists(model_file):

            try:

                self.model.load_state_dict(
                    torch.load(
                        model_file,
                        map_location="cpu"
                    )
                )

                print(
                    f"Loaded trained AI: {model_file}"
                )

            except Exception as error:

                print(
                    "Could not load trained AI."
                )

                print(error)

                print(
                    "Using a fresh random network."
                )

        else:

            print(
                "best_ai.pth was not found."
            )

            print(
                "AI will use a random network."
            )

        # ----------------------------------------------------
        # Evaluation mode
        # ----------------------------------------------------

        self.model.eval()


    # ========================================================
    # TRACK CHECK
    # ========================================================

    def is_on_track(
        self,
        x,
        y
    ):

        local_x = int(
            x - self.track_x
        )

        local_y = int(
            y - self.track_y
        )

        if (
            local_x < 0
            or local_y < 0
            or local_x >= self.track.get_width()
            or local_y >= self.track.get_height()
        ):

            return False

        return self.track_mask.get_at(
            (
                local_x,
                local_y
            )
        )


    # ========================================================
    # SENSOR
    # ========================================================

    def cast_ray(
        self,
        x,
        y,
        angle,
        max_distance=180
    ):

        radians = math.radians(
            angle
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
                x
                + direction_x * distance
            )

            test_y = (
                y
                + direction_y * distance
            )

            if not self.is_on_track(
                test_x,
                test_y
            ):

                break

            distance += 4

        return distance


    # ========================================================
    # GET SENSOR VALUES
    # ========================================================

    def get_sensors(
        self,
        car
    ):

        angles = [
            -70,
            -35,
            0,
            35,
            70
        ]

        sensors = []

        for relative_angle in angles:

            ray_angle = (
                car.angle
                + relative_angle
            )

            distance = self.cast_ray(
                car.x,
                car.y,
                ray_angle
            )

            normalized = (
                distance / 180.0
            )

            sensors.append(
                normalized
            )

        car.sensor_values = sensors

        return sensors


    # ========================================================
    # GET CHECKPOINT INFORMATION
    # ========================================================

    def get_checkpoint_data(
        self,
        car
    ):

        checkpoint = self.checkpoints[
            car.current_checkpoint
        ]

        # Use the CENTER of the checkpoint.
        checkpoint_x = (
            self.track_x
            + (
                checkpoint[0]
                + checkpoint[2] / 2
            )
            * self.scale
        )

        checkpoint_y = (
            self.track_y
            + (
                checkpoint[1]
                + checkpoint[3] / 2
            )
            * self.scale
        )

        dx = (
            checkpoint_x
            - car.x
        )

        dy = (
            checkpoint_y
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
            angle_difference / 180.0
        )

        distance_normalized = min(
            distance / 1000.0,
            1.0
        )

        return (
            angle_normalized,
            distance_normalized
        )


    # ========================================================
    # BUILD AI INPUT
    # ========================================================

    def get_inputs(
        self,
        car
    ):

        sensors = self.get_sensors(
            car
        )

        (
            angle_to_checkpoint,
            distance_to_checkpoint
        ) = self.get_checkpoint_data(
            car
        )

        speed = (
            car.speed
            / car.max_speed
        )

        inputs = [

            # ------------------------------------------------
            # 5 track sensors
            # ------------------------------------------------

            sensors[0],
            sensors[1],
            sensors[2],
            sensors[3],
            sensors[4],

            # ------------------------------------------------
            # Speed
            # ------------------------------------------------

            speed,

            # ------------------------------------------------
            # Checkpoint direction
            # ------------------------------------------------

            math.sin(
                angle_to_checkpoint
                * math.pi
            ),

            math.cos(
                angle_to_checkpoint
                * math.pi
            ),

            # ------------------------------------------------
            # Checkpoint distance
            # ------------------------------------------------

            distance_to_checkpoint,

            # ------------------------------------------------
            # Checkpoint number
            # ------------------------------------------------

            car.current_checkpoint
            / max(
                1,
                len(self.checkpoints) - 1
            ),

            # ------------------------------------------------
            # Current heading
            # ------------------------------------------------

            math.sin(
                math.radians(
                    car.angle
                )
            ),

            math.cos(
                math.radians(
                    car.angle
                )
            )
        ]

        return torch.tensor(
            inputs,
            dtype=torch.float32
        )


    # ========================================================
    # AI DECISION
    # ========================================================

    def decide(
        self,
        car
    ):

        inputs = self.get_inputs(
            car
        )

        with torch.no_grad():

            output = self.model(
                inputs
            )

        steering = output[0].item()

        throttle = output[1].item()

        car.steering = steering
        car.throttle = throttle

        return (
            steering,
            throttle
        )


    # ========================================================
    # UPDATE
    # ========================================================

    def update(
        self,
        car
    ):

        if not car.alive:

            return

        steering, throttle = self.decide(
            car
        )

        # ----------------------------------------------------
        # Steering
        # ----------------------------------------------------

        if abs(car.speed) > 0.05:

            direction = (
                1
                if car.speed > 0
                else -1
            )

            car.angle += (
                steering
                * car.steering_speed
                * direction
            )

        # ----------------------------------------------------
        # Throttle
        # ----------------------------------------------------

        throttle = max(
            0.0,
            throttle
        )

        car.speed += (
            car.acceleration
            * throttle
        )

        # Give the AI a small amount of initial movement.

        if car.speed < 1.0:

            car.speed += 0.05

        # ----------------------------------------------------
        # Speed limit
        # ----------------------------------------------------

        car.speed = max(
            -car.max_reverse_speed,
            min(
                car.speed,
                car.max_speed
            )
        )

        # ----------------------------------------------------
        # Friction
        # ----------------------------------------------------

        if throttle < 0.1:

            car.speed *= car.friction

        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

        radians = math.radians(
            car.angle
        )

        velocity_x = (
            math.sin(radians)
            * car.speed
        )

        velocity_y = (
            -math.cos(radians)
            * car.speed
        )

        old_x = car.x
        old_y = car.y

        car.x += velocity_x
        car.y += velocity_y

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if not self.is_on_track(
            car.x,
            car.y
        ):

            car.x = old_x
            car.y = old_y

            car.speed *= 0.35


    # ========================================================
    # DRAW AI CAR
    # ========================================================

    def draw_car(
        self,
        screen,
        car_image,
        car
    ):

        # aicar.png already faces UP.
        #
        # Our physics also considers 0 degrees to be UP.
        #
        # Therefore we rotate directly by -car.angle.

        rotated = pygame.transform.rotate(
            car_image,
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


    # ========================================================
    # DRAW SENSORS
    # ========================================================

    def draw_sensors(
        self,
        screen,
        car
    ):

        angles = [
            -70,
            -35,
            0,
            35,
            70
        ]

        for relative_angle, value in zip(
            angles,
            car.sensor_values
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
                (255, 220, 50),
                (
                    int(car.x),
                    int(car.y)
                ),
                (
                    int(end_x),
                    int(end_y)
                ),
                2
            )