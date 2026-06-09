# core/constants.py

from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════

ROOT_DIR   = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT_DIR / "Assets"

# ══════════════════════════════════════════════════════════════════════════════
# SCREEN
# ══════════════════════════════════════════════════════════════════════════════

SCREEN_W = 1100
SCREEN_H = 600

FPS = 60

TITLE = "Dino AI"

# ══════════════════════════════════════════════════════════════════════════════
# COLORS
# ══════════════════════════════════════════════════════════════════════════════

WHITE  = (255, 255, 255)
BLACK  = (0,   0,   0)

GREEN  = (0,   210, 80)
RED    = (220, 50,  50)
BLUE   = (60,  130, 240)

YELLOW = (240, 190, 0)
CYAN   = (0,   200, 200)
PURPLE = (160, 80,  220)

GRAY       = (160, 160, 160)
LIGHT_GRAY = (210, 210, 210)
DARK_GRAY  = (70,  70,  70)

DARK  = (20, 20, 30)

ORANGE = (255, 140, 0)

# ══════════════════════════════════════════════════════════════════════════════
# GAME
# ══════════════════════════════════════════════════════════════════════════════

GROUND_Y = 380

INITIAL_GAME_SPEED = 16
MAX_GAME_SPEED     = 31

SPEED_INCREASE_EVERY = 100

# ══════════════════════════════════════════════════════════════════════════════
# DINO
# ══════════════════════════════════════════════════════════════════════════════

DINO_X_POS      = 80
DINO_Y_POS      = 310
DINO_Y_POS_DUCK = 340

DINO_JUMP_VEL = -18
GRAVITY       = 1.1

# ══════════════════════════════════════════════════════════════════════════════
# ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

ACTION_RUN  = 0
ACTION_JUMP = 1
ACTION_DUCK = 2

ACTION_NAMES = {
    ACTION_RUN:  "RUN",
    ACTION_JUMP: "JUMP",
    ACTION_DUCK: "DUCK",
}

N_ACTIONS = 3

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATIONS
# ══════════════════════════════════════════════════════════════════════════════

OBS_DIST_X      = 0
OBS_WIDTH       = 1
OBS_HEIGHT      = 2
OBS_SPEED       = 3
OBS_DINO_Y      = 4
OBS_VEL_Y       = 5
OBS_IS_BIRD     = 6
OBS_OBS_GAP     = 7
OBS_NEXT_TYPE   = 8

N_OBS = 9

OBS_LABELS = [
    "dist_x",
    "width",
    "height",
    "speed",
    "dino_y",
    "vel_y",
    "is_bird",
    "gap",
    "next_type",
]

# ══════════════════════════════════════════════════════════════════════════════
# REWARDS
# ══════════════════════════════════════════════════════════════════════════════

REWARD_ALIVE           = 0.1
REWARD_DEATH           = -10.0
REWARD_PASS_OBSTACLE   = 5.0
REWARD_SPEED_BONUS     = 0.01
REWARD_USELESS_JUMP    = -0.02
REWARD_USELESS_DUCK    = -0.01

# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════

HUD_ALPHA = 180

HUD_FONT_SMALL  = 14
HUD_FONT_MEDIUM = 18
HUD_FONT_LARGE  = 28

HUD_PANEL_BG = (10, 10, 20, HUD_ALPHA)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MAX_EPISODES = 20_000

SAVE_EVERY = 50

# ══════════════════════════════════════════════════════════════════════════════
# DQN
# ══════════════════════════════════════════════════════════════════════════════

DQN_TRAIN_EVERY  = 4
DQN_FRAME_SKIP   = 4

DQN_GAMMA = 0.99

DQN_LR = 1e-3

DQN_BATCH_SIZE  = 64
DQN_BUFFER_SIZE = 100_000

DQN_MIN_BUFFER = 1_000

DQN_TARGET_UPDATE = 500

DQN_EPS_START = 1.0
DQN_EPS_END   = 0.02
DQN_EPS_DECAY = 0.995

DQN_HIDDEN_SIZE = 128

# Soft update (applied every learn step)
DQN_TAU = 0.005

# Prioritized Experience Replay
DQN_PER_ALPHA         = 0.6    # prioritization exponent (0=uniform, 1=full)
DQN_PER_BETA_START    = 0.4    # IS correction start
DQN_PER_BETA_INCREMENT = 2e-7  # anneals beta toward 1.0 over ~3M learn steps

# ══════════════════════════════════════════════════════════════════════════════
# NEAT
# ══════════════════════════════════════════════════════════════════════════════

NEAT_MAX_GENERATIONS = 300

NEAT_CHECKPOINT_EVERY = 10

# ══════════════════════════════════════════════════════════════════════════════
# OBSTACLE SPAWNING
# ══════════════════════════════════════════════════════════════════════════════

BIRD_Y = 245

OBSTACLE2_SPAWN_THRESHOLD = SCREEN_W // 3   # trigger x for pre-spawning second obstacle
OBSTACLE2_SPAWN_PROB      = 0.5             # probability a second obstacle appears at all
OBSTACLE2_MIN_EXTRA       = 100             # minimum extra pixels beyond SCREEN_W
OBSTACLE2_MAX_EXTRA       = 400             # maximum extra pixels beyond SCREEN_W

# ══════════════════════════════════════════════════════════════════════════════
# DEBUG
# ══════════════════════════════════════════════════════════════════════════════

DRAW_HITBOXES     = False
DRAW_SENSOR_LINES = True
DRAW_DEBUG_TEXT   = False