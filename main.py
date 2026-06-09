# main.py  —  Menu principal do Dino AI

import pygame
pygame.init()

import sys

from core.constants import *
from core.assets import get_font
from core.utils import draw_text, draw_panel

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

pygame.display.set_caption("Dino AI")
clock = pygame.time.Clock()

font_lg = get_font(HUD_FONT_LARGE, bold=True)
font_md = get_font(HUD_FONT_MEDIUM)
font_sm = get_font(HUD_FONT_SMALL)
fonts   = (font_lg, font_md, font_sm)

# ══════════════════════════════════════════════════════════════════════════════
# MENU HELPER
# ══════════════════════════════════════════════════════════════════════════════

def current_screen():
    """Sempre retorna a surface de display atual, mesmo após set_mode externo."""
    s = pygame.display.get_surface()
    if s is None:
        s = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    return s


def draw_menu(title, options, subtitle=None):
    """
    options: list of (key_label, description, color)
    """
    surface = current_screen()
    surface.fill(WHITE)

    cx = SCREEN_W // 2

    draw_text(surface, title, font_lg, DARK, cx, 110, center=True)

    if subtitle:
        draw_text(surface, subtitle, font_sm, GRAY, cx, 175, center=True)

    for i, (key_label, desc, color) in enumerate(options):
        y = 220 + i * 52
        draw_text(surface, f"[{key_label}]", font_md, DARK_GRAY, cx - 180, y)
        draw_text(surface, desc,             font_md, color,      cx - 110, y)

    pygame.display.flip()


def wait_key():
    """Retorna o pygame.key code da próxima tecla pressionada, ou None no QUIT."""
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                return event.key


# ══════════════════════════════════════════════════════════════════════════════
# MENUS
# ══════════════════════════════════════════════════════════════════════════════

def menu_principal():
    draw_menu(
        "DINO  AI",
        [
            ("1",   "Jogar vs IA",  BLUE),
            ("2",   "Replay DQN",   GREEN),
            ("3",   "Replay NEAT",  CYAN),
            ("4",   "Treinar",      ORANGE),
            ("ESC", "Sair",         GRAY),
        ],
    )
    key = wait_key()
    if key == pygame.K_1:     return "versus"
    if key == pygame.K_2:     return "replay_dqn"
    if key == pygame.K_3:     return "replay_neat"
    if key == pygame.K_4:     return "train"
    if key != pygame.K_ESCAPE: return "menu_principal"
    return None   # ESC ou QUIT


def menu_treinar():
    draw_menu(
        "TREINAR",
        [
            ("1",   "DQN",    BLUE),
            ("2",   "NEAT",   GREEN),
            ("ESC", "Voltar", GRAY),
        ],
    )
    key = wait_key()
    if key == pygame.K_1:   return "dqn"
    if key == pygame.K_2:   return "neat"
    return "back"

def menu_dificuldade():
    draw_menu(
        "Dificuldade",
        [
            ("1", "easy", BLUE),
            ("2", "expert", GREEN),
            ("ESC", "Voltar", GRAY),
        ],
    )
    key = wait_key()
    if key == pygame.K_1:   return "easy"
    if key == pygame.K_2:   return "expert"
    return None


def menu_modo_treino(label):
    draw_menu(
        f"TREINAR  {label}",
        [
            ("1",   "Novo treinamento",     BLUE),
            ("2",   "Continuar do último",  YELLOW),
            ("ESC", "Voltar",               GRAY),
        ],
        subtitle="[2] Retoma o último checkpoint salvo",
    )
    key = wait_key()
    if key == pygame.K_1:   return "new"
    if key == pygame.K_2:   return "resume"
    return "back"


# ══════════════════════════════════════════════════════════════════════════════
# VERSUS
# ══════════════════════════════════════════════════════════════════════════════

def run_versus():
    from core.versus import show_menu as versus_menu
    from core.versus import load_dqn, load_neat, play_round
    from core.game import DinoEnv

    pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Dino — Humano vs IA")

    screen = current_screen()
    choice = versus_menu(screen, fonts)

    if choice is None:
        return

    if choice == "DQN":
        ai_model = load_dqn()
    else:
        difficult = menu_dificuldade()
        if difficult is None:
            return
        ai_model = load_neat(difficult)

    #ai_model = load_dqn() if choice == "DQN" else load_neat()
    env = DinoEnv(render_mode=True)

    while True:
        result = play_round(ai_model, choice, env, fonts)
        if result in ("menu", "quit", None):
            break
        # "restart" → continua o loop


# ══════════════════════════════════════════════════════════════════════════════
# REPLAY
# ══════════════════════════════════════════════════════════════════════════════

def run_replay_dqn():
    pygame.display.set_caption("Dino — Replay DQN")
    from dqn.replay_dqn import replay
    replay()


def run_replay_neat():
    pygame.display.set_caption("Dino — Replay NEAT")
    from neat_ai.replay_neat import replay
    replay()


# ══════════════════════════════════════════════════════════════════════════════
# TREINAR
# ══════════════════════════════════════════════════════════════════════════════

def run_train_dqn(resume: bool):
    pygame.display.set_caption("Dino — Treino DQN")
    from dqn.train_dqn import train
    train(resume=resume)


def run_train_neat(resume: bool):
    pygame.display.set_caption("Dino — Treino NEAT")
    from neat_ai.train_neat import run
    run(resume=resume)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():

    pygame.display.set_mode((SCREEN_W, SCREEN_H))

    while True:

        pygame.display.set_caption("Dino AI")
        action = menu_principal()

        if action is None:
            break

        # ── versus ────────────────────────────────────────────────────────────
        elif action == "versus":
            run_versus()

        # ── replays ───────────────────────────────────────────────────────────
        elif action == "replay_dqn":
            run_replay_dqn()

        elif action == "replay_neat":
            run_replay_neat()

        # ── treinar ───────────────────────────────────────────────────────────
        elif action == "train":

            algo = menu_treinar()
            if algo == "back":
                continue

            mode = menu_modo_treino(algo.upper())
            if mode == "back":
                continue

            resume = (mode == "resume")

            if algo == "dqn":
                run_train_dqn(resume)
            else:
                run_train_neat(resume)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()