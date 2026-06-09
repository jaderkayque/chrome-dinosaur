# core/assets.py
import os

import pygame

from core.constants import ASSETS_DIR

# ══════════════════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════════════════

# IMPORTANTE:
# pygame.init() deve ser chamado ANTES de importar este arquivo.

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_image(
    folder,
    filename,
    alpha=True,
    scale=None
):

    path = os.path.join(
        ASSETS_DIR,
        folder,
        filename
    )

    image = pygame.image.load(path)

    # evita erro quando display ainda não existe
    if pygame.display.get_surface():

        if alpha:
            image = image.convert_alpha()
        else:
            image = image.convert()

    if scale is not None:

        image = pygame.transform.scale(
            image,
            scale
        )

    return image

def load_images(folder, filenames, alpha=True, scale=None):
    """
    Carrega múltiplas imagens.

    Exemplo:
        load_images("Dino", ["Run1.png", "Run2.png"])
    """

    return [
        load_image(folder, filename, alpha=alpha, scale=scale)
        for filename in filenames
    ]


# ══════════════════════════════════════════════════════════════════════════════
# DINO
# ══════════════════════════════════════════════════════════════════════════════

RUNNING = load_images(
    "Dino",
    [
        "DinoRun1.png",
        "DinoRun2.png",
    ]
)

DUCKING = load_images(
    "Dino",
    [
        "DinoDuck1.png",
        "DinoDuck2.png",
    ]
)

JUMPING = load_image(
    "Dino",
    "DinoJump.png"
)

# ══════════════════════════════════════════════════════════════════════════════
# CACTUS
# ══════════════════════════════════════════════════════════════════════════════

SMALL_CACTUS = load_images(
    "Cactus",
    [
        "SmallCactus1.png",
        "SmallCactus2.png",
        "SmallCactus3.png",
    ]
)

LARGE_CACTUS = load_images(
    "Cactus",
    [
        "LargeCactus1.png",
        "LargeCactus2.png",
        "LargeCactus3.png",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# BIRD
# ══════════════════════════════════════════════════════════════════════════════

BIRD = load_images(
    "Bird",
    [
        "Bird1.png",
        "Bird2.png",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# BACKGROUND
# ══════════════════════════════════════════════════════════════════════════════

CLOUD = load_image(
    "Other",
    "Cloud.png"
)

BG = load_image(
    "Other",
    "Track.png"
)

# ══════════════════════════════════════════════════════════════════════════════
# FONT
# ══════════════════════════════════════════════════════════════════════════════

def get_font(size=16, bold=False):
    """
    Retorna fonte padrão do projeto.
    """

    return pygame.font.SysFont(
        "consolas",
        size,
        bold=bold
    )

# ══════════════════════════════════════════════════════════════════════════════
# CACHE INFO
# ══════════════════════════════════════════════════════════════════════════════

ALL_ASSETS = {
    "RUNNING": RUNNING,
    "DUCKING": DUCKING,
    "JUMPING": JUMPING,
    "SMALL_CACTUS": SMALL_CACTUS,
    "LARGE_CACTUS": LARGE_CACTUS,
    "BIRD": BIRD,
    "CLOUD": CLOUD,
    "BG": BG,
}