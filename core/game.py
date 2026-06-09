# core/game.py

from dataclasses import dataclass, asdict

import random
import numpy as np
import pygame

from core.constants import *
from core.assets import (
    RUNNING,
    DUCKING,
    JUMPING,
    SMALL_CACTUS,
    LARGE_CACTUS,
    BIRD,
    CLOUD,
    BG,
)

# ══════════════════════════════════════════════════════════════════════════════
# GAME STATE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GameState:

    dist_x: float
    width: float
    height: float
    speed: float
    dino_y: float
    vel_y: float
    is_bird: float
    obstacle_gap: float
    next_obstacle_type: float

# ══════════════════════════════════════════════════════════════════════════════
# DINO
# ══════════════════════════════════════════════════════════════════════════════

class Dinosaur:

    def __init__(self, x_pos=DINO_X_POS):

        self.x_pos       = x_pos
        self.run_images  = RUNNING
        self.duck_images = DUCKING
        self.jump_image  = JUMPING

        self.reset()

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):

        self.image = self.run_images[0]

        self.rect = self.image.get_rect()

        self.rect.x = self.x_pos
        self.rect.y = DINO_Y_POS

        self.step_index = 0

        self.vel_y = 0.0

        self.is_running = True
        self.is_ducking = False
        self.is_jumping = False

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, action):

        if action == ACTION_JUMP and not self.is_jumping:

            self.is_running = False
            self.is_ducking = False
            self.is_jumping = True

            self.vel_y = DINO_JUMP_VEL

        elif action == ACTION_DUCK and not self.is_jumping:

            self.is_running = False
            self.is_ducking = True

        elif action == ACTION_RUN and not self.is_jumping:

            self.is_running = True
            self.is_ducking = False

        # animation state
        if self.is_running:
            self._run()

        elif self.is_ducking:
            self._duck()

        elif self.is_jumping:
            self._jump()

        if self.step_index >= 10:
            self.step_index = 0

    # ─────────────────────────────────────────────────────────────────────────

    def _run(self):

        self.image = self.run_images[self.step_index // 5]

        self.rect = self.image.get_rect(
            topleft=(self.x_pos, DINO_Y_POS)
        )

        self.step_index += 1

    # ─────────────────────────────────────────────────────────────────────────

    def _duck(self):

        self.image = self.duck_images[self.step_index // 5]

        self.rect = self.image.get_rect(
            topleft=(self.x_pos, DINO_Y_POS_DUCK)
        )

        self.step_index += 1

    # ─────────────────────────────────────────────────────────────────────────

    def _jump(self):

        self.image = self.jump_image

        self.rect.y += self.vel_y

        self.vel_y += GRAVITY

        if self.rect.y >= DINO_Y_POS:

            self.rect.y = DINO_Y_POS

            self.is_jumping = False
            self.is_running = True

            self.vel_y = 0.0

    # ─────────────────────────────────────────────────────────────────────────

    def draw(self, surface):

        surface.blit(
            self.image,
            self.rect
        )

        if DRAW_HITBOXES:

            pygame.draw.rect(
                surface,
                RED,
                self.rect,
                2
            )

# ══════════════════════════════════════════════════════════════════════════════
# CLOUD
# ══════════════════════════════════════════════════════════════════════════════

class Cloud:

    def __init__(self):

        self.reset()

    # ─────────────────────────────────────────────────────────────────────────

    def reset(self):

        self.x = SCREEN_W + random.randint(800, 1000)

        self.y = random.randint(50, 120)

        self.width = CLOUD.get_width()

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, speed):

        self.x -= speed

        if self.x < -self.width:

            self.x = SCREEN_W + random.randint(2500, 3000)

            self.y = random.randint(50, 120)

    # ─────────────────────────────────────────────────────────────────────────

    def draw(self, surface):

        surface.blit(
            CLOUD,
            (self.x, self.y)
        )

# ══════════════════════════════════════════════════════════════════════════════
# OBSTACLE
# ══════════════════════════════════════════════════════════════════════════════

class Obstacle:

    def __init__(self):

        self.is_bird = False

        r = random.randint(0, 2)

        # small cactus
        if r == 0:

            self.images = SMALL_CACTUS

            self.type = random.randint(0, 2)

            self.image = self.images[self.type]

            self.rect = self.image.get_rect()

            self.rect.x = SCREEN_W
            self.rect.y = 325

        # large cactus
        elif r == 1:

            self.images = LARGE_CACTUS

            self.type = random.randint(0, 2)

            self.image = self.images[self.type]

            self.rect = self.image.get_rect()

            self.rect.x = SCREEN_W
            self.rect.y = 300

        # bird
        else:

            self.images = BIRD

            self.type = 0

            self.is_bird = True

            self.image = self.images[0]

            self.rect = self.image.get_rect()

            self.rect.x = SCREEN_W
            self.rect.y = BIRD_Y

        self.bird_index = 0

        self.passed = False

    # ─────────────────────────────────────────────────────────────────────────

    def update(self, speed):

        self.rect.x -= speed

    # ─────────────────────────────────────────────────────────────────────────

    def draw(self, surface):

        if self.is_bird:

            if self.bird_index >= 10:
                self.bird_index = 0

            surface.blit(
                self.images[self.bird_index // 5],
                self.rect
            )

            self.bird_index += 1

        else:

            surface.blit(
                self.image,
                self.rect
            )

        if DRAW_HITBOXES:

            pygame.draw.rect(
                surface,
                BLUE,
                self.rect,
                2
            )

    # ─────────────────────────────────────────────────────────────────────────

    @property
    def off_screen(self):

        return self.rect.right < 0

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class DinoEnv:

    def __init__(
        self,
        render_mode=True
    ):

        self.render_mode = render_mode

        self.screen = pygame.display.set_mode(
            (SCREEN_W, SCREEN_H)
        )

        pygame.display.set_caption(TITLE)

        self.clock = pygame.time.Clock()

        self.reset()

    # ══════════════════════════════════════════════════════════════════════════

    def reset(self, n_dinos=1):

        self.done = False

        self.points = 0

        self.frame_count = 0

        self.game_speed = INITIAL_GAME_SPEED

        self.bg_x = 0

        self.dinos = [Dinosaur() for _ in range(n_dinos)]

        self.alive  = [True] * n_dinos

        self.cloud = Cloud()

        self.obstacles = []

        self.last_reward = 0.0

        self._spawn_obstacle()

        if n_dinos == 1:
            return self._get_obs()

        return [self._get_obs_for(d) for d in self.dinos]

    # ──────────────────────────────────────────────────────────────────────────

    @property
    def dino(self):
        return self.dinos[0]

    # ══════════════════════════════════════════════════════════════════════════

    def step(self, action):

        if self.done:
            raise RuntimeError("Environment finished. Call reset().")

        self.frame_count += 1

        reward = REWARD_ALIVE

        # dino
        self.dino.update(action)

        # obstacles
        self._update_obstacles()

        # cloud
        self.cloud.update(self.game_speed)

        # reward shaping
        reward += self._compute_reward(action)

        # collision
        collision = self._check_collision()

        if collision:

            reward += REWARD_DEATH

            self.done = True

        else:

            self.points += 1

            # speed increase
            if self.points % SPEED_INCREASE_EVERY == 0:

                if self.game_speed < MAX_GAME_SPEED:
                    self.game_speed += 1

        self.last_reward = reward

        obs = self._get_obs()

        info = self._build_info(collision)

        return obs, reward, self.done, info

    # ══════════════════════════════════════════════════════════════════════════

    def render(
        self,
        hud=None,
        hud_data=None,
        flip=True
    ):

        if not self.render_mode:
            return

        self.screen.fill(WHITE)

        self._draw_background()

        self.cloud.draw(self.screen)

        for obstacle in self.obstacles:
            obstacle.draw(self.screen)

        for dino in self.dinos:
            dino.draw(self.screen)

        if hud is not None:

            hud.draw(
                self.screen,
                mode=hud_data.get("mode", "dqn"),
                data=hud_data
            )

        if flip:
            pygame.display.flip()

    # ══════════════════════════════════════════════════════════════════════════

    def tick(self, fps=FPS):

        self.clock.tick(fps)

    # ══════════════════════════════════════════════════════════════════════════

    def close(self):
        pass  # pygame lifecycle is managed by the top-level caller

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNALS
    # ══════════════════════════════════════════════════════════════════════════

    def _spawn_obstacle(self):

        if len(self.obstacles) == 0:

            self.obstacles.append(
                Obstacle()
            )

    # ─────────────────────────────────────────────────────────────────────────

    def _update_obstacles(self):

        for obstacle in self.obstacles:

            obstacle.update(self.game_speed)

        # remove offscreen
        self.obstacles = [
            obs for obs in self.obstacles
            if not obs.off_screen
        ]

        # pre-spawn next obstacle when current one crosses the threshold
        if (
            len(self.obstacles) == 1
            and self.obstacles[0].rect.x < OBSTACLE2_SPAWN_THRESHOLD
            and random.random() < OBSTACLE2_SPAWN_PROB
        ):
            extra = random.randint(OBSTACLE2_MIN_EXTRA, OBSTACLE2_MAX_EXTRA)
            new_obs = Obstacle()
            new_obs.rect.x = SCREEN_W + extra
            self.obstacles.append(new_obs)

        if len(self.obstacles) == 0:
            self._spawn_obstacle()

    # ─────────────────────────────────────────────────────────────────────────

    def _check_collision(self):

        for obstacle in self.obstacles:

            if self.dino.rect.colliderect(obstacle.rect):

                return True

        return False

    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, action):

        reward = 0.0

        if not self.obstacles:
            return reward

        obstacle = self.obstacles[0]

        # obstacle passed — reward scales with speed (harder at high speed)
        if (
            not obstacle.passed
            and obstacle.rect.right < self.dino.rect.left
        ):
            obstacle.passed = True
            reward += REWARD_PASS_OBSTACLE * (self.game_speed / INITIAL_GAME_SPEED)

        # useless jump: penalize only when obstacle is farther than ~20 frames away
        # threshold is speed-relative so high-speed early jumps are not wrongly punished
        jump_too_early = (
            (obstacle.rect.x - self.dino.rect.left) > self.game_speed * 20
        )
        if action == ACTION_JUMP and (jump_too_early or obstacle.is_bird):
            reward += REWARD_USELESS_JUMP

        # useless duck
        if action == ACTION_DUCK and not obstacle.is_bird:
            reward += REWARD_USELESS_DUCK

        reward += self.game_speed * REWARD_SPEED_BONUS

        return reward

    # ─────────────────────────────────────────────────────────────────────────

    def _draw_background(self):

        image_width = BG.get_width()

        self.screen.blit(
            BG,
            (self.bg_x, GROUND_Y)
        )

        self.screen.blit(
            BG,
            (image_width + self.bg_x, GROUND_Y)
        )

        if self.bg_x <= -image_width:

            self.bg_x = 0

        self.bg_x -= self.game_speed

    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self):

        return self._get_obs_for(self.dino)

    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs_for(self, dino):

        if self.obstacles:

            obs = self.obstacles[0]

            dist_x = max(
                0.0,
                (obs.rect.x - dino.rect.x) / SCREEN_W
            )

            width = obs.rect.width / SCREEN_W

            height = (
                SCREEN_H - obs.rect.y
            ) / SCREEN_H

            is_bird = 1.0 if obs.is_bird else 0.0

        else:

            dist_x = 1.0
            width = 0.0
            height = 0.0
            is_bird = 0.0

        # second obstacle
        obstacle_gap = 0.0
        next_type = 0.0

        if len(self.obstacles) > 1:

            obs1 = self.obstacles[0]
            obs2 = self.obstacles[1]

            obstacle_gap = (
                obs2.rect.x - obs1.rect.x
            ) / SCREEN_W

            next_type = 1.0 if obs2.is_bird else 0.0

        vel_y = float(np.clip(dino.vel_y / 20.0, -1.0, 1.0))

        state = GameState(
            dist_x=dist_x,
            width=width,
            height=height,
            speed=self.game_speed / 100,
            dino_y=(SCREEN_H - dino.rect.y) / SCREEN_H,
            vel_y=vel_y,
            is_bird=is_bird,
            obstacle_gap=obstacle_gap,
            next_obstacle_type=next_type,
        )

        return np.array(
            list(asdict(state).values()),
            dtype=np.float32
        )

    # ─────────────────────────────────────────────────────────────────────────

    # ══════════════════════════════════════════════════════════════════════════
    # MULTI-DINO (NEAT)
    # ══════════════════════════════════════════════════════════════════════════

    def step_multi(self, actions):
        """One step for all parallel dinos. Used by NEAT training."""

        if self.done:
            raise RuntimeError("Environment finished. Call reset().")

        self.frame_count += 1

        # shared updates
        self._update_obstacles()
        self.cloud.update(self.game_speed)

        # obstacle-passed bonus fires once, given to every alive dino
        passed = self._check_pass_obstacle()

        rewards = [0.0] * len(self.dinos)
        dones   = [not a for a in self.alive]

        for i, dino in enumerate(self.dinos):

            if not self.alive[i]:
                continue

            dino.update(actions[i])

            r = REWARD_ALIVE

            if passed:
                r += REWARD_PASS_OBSTACLE

            r += self._reward_action(actions[i])

            r += self.game_speed * REWARD_SPEED_BONUS

            if self._check_collision_for(dino):
                r += REWARD_DEATH
                self.alive[i] = False

            rewards[i] = r
            dones[i]   = not self.alive[i]

        # score / speed update (once per frame while any dino alive)
        if any(self.alive):
            self.points += 1
            if self.points % SPEED_INCREASE_EVERY == 0:
                self.game_speed = min(
                    self.game_speed + 1,
                    MAX_GAME_SPEED
                )

        self.done = not any(self.alive)

        obs_list = [
            self._get_obs_for(self.dinos[i]) if self.alive[i] else None
            for i in range(len(self.dinos))
        ]

        info = {
            "score":       self.points,
            "speed":       self.game_speed,
            "alive_count": sum(self.alive),
            "fps":         self.clock.get_fps(),
            "sensor_data": self._sensor_data_first_alive(),
        }

        return obs_list, rewards, dones, info

    # ─────────────────────────────────────────────────────────────────────────

    def _check_pass_obstacle(self):
        """Returns True (once) when the front obstacle clears the dino column."""
        if not self.obstacles:
            return False
        obs = self.obstacles[0]
        if not obs.passed and obs.rect.right < DINO_X_POS:
            obs.passed = True
            return True
        return False

    # ─────────────────────────────────────────────────────────────────────────

    def _check_collision_for(self, dino):
        for obstacle in self.obstacles:
            if dino.rect.colliderect(obstacle.rect):
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────────

    def _reward_action(self, action):
        """Per-dino penalty for wasteful actions."""
        if not self.obstacles:
            return 0.0
        obs = self.obstacles[0]
        reward = 0.0
        if action == ACTION_JUMP and (obs.rect.x > 350 or obs.is_bird):
            reward += REWARD_USELESS_JUMP
        if action == ACTION_DUCK and not obs.is_bird:
            reward += REWARD_USELESS_DUCK
        return reward

    # ─────────────────────────────────────────────────────────────────────────

    def _sensor_data_first_alive(self):
        obstacle = self.obstacles[0] if self.obstacles else None
        if not obstacle:
            return None
        for i, dino in enumerate(self.dinos):
            if self.alive[i]:
                return {
                    "start": (dino.rect.right, dino.rect.centery),
                    "end":   (obstacle.rect.left, obstacle.rect.centery),
                }
        return None

    # ══════════════════════════════════════════════════════════════════════════

    def _build_info(self, collision):

        obstacle = self.obstacles[0] if self.obstacles else None

        sensor_data = None

        if obstacle:

            sensor_data = {
                "start": (
                    self.dino.rect.right,
                    self.dino.rect.centery,
                ),
                "end": (
                    obstacle.rect.left,
                    obstacle.rect.centery,
                ),
            }

        return {
            "score": self.points,
            "speed": self.game_speed,
            "collision": collision,
            "fps": self.clock.get_fps(),
            "sensor_data": sensor_data,
            "obstacle": obstacle,
        }