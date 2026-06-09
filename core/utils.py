# core/utils.py

from collections import deque
import numpy as np
import pygame
import random

# ══════════════════════════════════════════════════════════════════════════════
# RANDOM
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """
    Define seed global.
    """

    random.seed(seed)
    np.random.seed(seed)

# ══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def normalize(value, min_value, max_value):
    """
    Normaliza valor para intervalo [0, 1].
    """

    if max_value - min_value == 0:
        return 0.0

    return (value - min_value) / (max_value - min_value)

# ══════════════════════════════════════════════════════════════════════════════
# MOVING AVERAGE
# ══════════════════════════════════════════════════════════════════════════════

def moving_average(data, window=10):
    """
    Média móvel simples.
    """

    if len(data) < window:
        return data

    weights = np.ones(window) / window

    return np.convolve(data, weights, mode="valid")

# ══════════════════════════════════════════════════════════════════════════════
# COLOR
# ══════════════════════════════════════════════════════════════════════════════

def random_color(min_value=80, max_value=255):
    """
    Gera cor RGB aleatória.
    """

    return (
        random.randint(min_value, max_value),
        random.randint(min_value, max_value),
        random.randint(min_value, max_value),
    )

# ══════════════════════════════════════════════════════════════════════════════
# DRAW
# ══════════════════════════════════════════════════════════════════════════════

def draw_text(
    surface,
    text,
    font,
    color,
    x,
    y,
    center=False
):
    """
    Desenha texto.
    """

    txt = font.render(str(text), True, color)

    rect = txt.get_rect()

    if center:
        rect.center = (x, y)
    else:
        rect.topleft = (x, y)

    surface.blit(txt, rect)

    return rect

# ══════════════════════════════════════════════════════════════════════════════
# PANELS
# ══════════════════════════════════════════════════════════════════════════════

def draw_panel(
    surface,
    x,
    y,
    w,
    h,
    color=(10, 10, 20, 180),
    border_radius=8
):
    """
    Painel semi-transparente.
    """

    panel = pygame.Surface((w, h), pygame.SRCALPHA)

    pygame.draw.rect(
        panel,
        color,
        (0, 0, w, h),
        border_radius=border_radius
    )

    surface.blit(panel, (x, y))

# ══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ══════════════════════════════════════════════════════════════════════════════

def draw_line_chart(
    surface,
    data,
    x,
    y,
    w,
    h,
    color,
    background=(10, 10, 20, 160),
    smooth=True
):
    """
    Desenha gráfico de linha.
    """

    if len(data) < 2:
        return

    draw_panel(surface, x, y, w, h, background)

    values = np.array(data, dtype=np.float32)

    if smooth and len(values) >= 10:
        values = moving_average(values, window=10)

    mn = float(values.min())
    mx = float(values.max())

    rng = mx - mn

    if rng == 0:
        rng = 1.0

    points = []

    for i, v in enumerate(values):

        px = x + 6 + int(i / max(len(values) - 1, 1) * (w - 12))

        py = y + h - 6 - int((v - mn) / rng * (h - 20))

        points.append((px, py))

    if len(points) >= 2:
        pygame.draw.lines(
            surface,
            color,
            False,
            points,
            2
        )

# ══════════════════════════════════════════════════════════════════════════════
# FPS
# ══════════════════════════════════════════════════════════════════════════════

class FPSCounter:

    def __init__(self):

        self.clock = pygame.time.Clock()

    def tick(self, fps):

        self.clock.tick(fps)

    @property
    def fps(self):

        return self.clock.get_fps()

# ══════════════════════════════════════════════════════════════════════════════
# HISTORY BUFFER
# ══════════════════════════════════════════════════════════════════════════════

class HistoryBuffer:

    def __init__(self, maxlen=100):

        self.data = deque(maxlen=maxlen)

    def append(self, value):

        self.data.append(value)

    def clear(self):

        self.data.clear()

    def values(self):

        return list(self.data)

    def __len__(self):

        return len(self.data)