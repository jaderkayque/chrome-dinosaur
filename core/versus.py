# versus.py  —  Humano vs IA (DQN ou NEAT)
#
# Controles:
#   Seta UP / SPACE  = pular
#   Seta DOWN        = abaixar
#   R                = reiniciar
#   ESC              = sair

import pygame
pygame.init()

import os
import sys
import pickle
import numpy as np

from core.constants import *
from core.game import DinoEnv
from core.assets import get_font
from core.utils import draw_text, draw_panel

# ══════════════════════════════════════════════════════════════════════════════
# INDICES
# ══════════════════════════════════════════════════════════════════════════════

HUMAN_IDX = 0
AI_IDX    = 1

# ══════════════════════════════════════════════════════════════════════════════
# PATHS  (absolutos a partir da raiz do projeto)
# ══════════════════════════════════════════════════════════════════════════════

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DQN_MODEL_PATH   = os.path.join(_ROOT, "dqn/models", "dqn_best_official.pth")
NEAT_MODEL_PATH  = os.path.join(_ROOT, "neat_ai/models", "neat_best_official_expert.pkl")
NEAT_MODEL_PATH_EASY = os.path.join(_ROOT, "neat_ai/models", "neat_best_official_easy.pkl")
NEAT_CONFIG_PATH = os.path.join(_ROOT, "neat_ai", "config.txt")

# ══════════════════════════════════════════════════════════════════════════════
# LOADERS
# ══════════════════════════════════════════════════════════════════════════════

def load_dqn():

    from dqn.agent import DQNAgent

    if not os.path.exists(DQN_MODEL_PATH):
        raise FileNotFoundError(f"Modelo DQN nao encontrado: {DQN_MODEL_PATH}")

    agent = DQNAgent(N_OBS, N_ACTIONS)
    agent.load(DQN_MODEL_PATH)
    agent.epsilon = 0.0

    print(f"[DQN] Modelo carregado: {DQN_MODEL_PATH}")
    return agent


def load_neat(difficult='expert'):

    import neat

    if difficult == 'expert':
        path = NEAT_MODEL_PATH
    else:
        path = NEAT_MODEL_PATH_EASY

    if not os.path.exists(path):
        raise FileNotFoundError(f"Genoma NEAT nao encontrado: {path}")

    with open(path, "rb") as f:
        genome = pickle.load(f)

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        NEAT_CONFIG_PATH,
    )

    network = neat.nn.FeedForwardNetwork.create(genome, config)

    print(f"[NEAT] Genoma carregado: {path}")
    return network

# ══════════════════════════════════════════════════════════════════════════════
# ACTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_human_action(keys):

    if keys[pygame.K_UP] or keys[pygame.K_SPACE]:
        return ACTION_JUMP

    if keys[pygame.K_DOWN]:
        return ACTION_DUCK

    return ACTION_RUN


def get_ai_action(model, label, state):

    if label == "DQN":
        return model.select_action(state)

    outputs = model.activate(state)
    return int(np.argmax(outputs))

# ══════════════════════════════════════════════════════════════════════════════
# MENU
# ══════════════════════════════════════════════════════════════════════════════

def show_menu(screen, fonts):
    """Retorna 'DQN', 'NEAT' ou None (sair)."""

    font_lg, font_md, font_sm = fonts
    clock = pygame.time.Clock()
    cx = SCREEN_W // 2

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "DQN"
                if event.key == pygame.K_2:
                    return "NEAT"
                if event.key == pygame.K_ESCAPE:
                    return None

        screen.fill(WHITE)

        draw_text(screen, "HUMANO  VS  IA",   font_lg, DARK,      cx, 150, center=True)
        draw_text(screen, "Escolha o oponente:", font_md, DARK_GRAY, cx, 250, center=True)

        draw_text(screen, "[1]   DQN",  font_md, BLUE,  cx - 60, 310)
        draw_text(screen, "[2]   NEAT", font_md, GREEN, cx - 60, 360)

        draw_text(screen, "Controles:  Seta UP / SPACE = pular     Seta DOWN = abaixar",
                  font_sm, GRAY, cx, 460, center=True)
        draw_text(screen, "ESC = sair     R = Resetar", font_sm, GRAY, cx, 490, center=True)

        pygame.display.flip()
        clock.tick(60)

# ══════════════════════════════════════════════════════════════════════════════
# OVERLAYS
# ══════════════════════════════════════════════════════════════════════════════

def draw_dino_labels(surface, font_sm, env):
    """Contorno colorido + label acima de cada dino."""

    names  = ["VOCE", "IA"]
    colors = [GREEN,  BLUE]
    y_off  = [-34, -18]   # stacked: IA mais perto, VOCE acima

    for i, dino in enumerate(env.dinos):

        alive = env.alive[i]
        color = colors[i] if alive else RED
        label = names[i]  if alive else f"{names[i]} [X]"

        # contorno
        border = pygame.Rect(
            dino.rect.x - 2,
            dino.rect.y - 2,
            dino.rect.w + 4,
            dino.rect.h + 4,
        )
        pygame.draw.rect(surface, color, border, 2)

        # label
        surf = font_sm.render(label, True, color)
        surface.blit(
            surf,
            (dino.rect.centerx - surf.get_width() // 2,
             dino.rect.top + y_off[i]),
        )


def draw_versus_panel(surface, font_md, font_sm, env, death_scores, ai_label):
    """Painel lateral com status de cada jogador."""

    x, y, w, h = SCREEN_W - 215, 8, 205, 175
    draw_panel(surface, x, y, w, h)
    draw_text(surface, "VERSUS", font_md, WHITE, x + 10, y + 8)

    entries = [
        ("VOCE", HUMAN_IDX, GREEN),
        (ai_label, AI_IDX,  BLUE),
    ]

    for label, idx, color in entries:

        yy    = y + 45 + idx * 62
        alive = env.alive[idx]
        c     = color if alive else RED
        score = env.points if alive else death_scores[idx]
        status = "VIVO"  if alive else "MORTO"

        draw_text(surface, f"{label}  [{status}]", font_sm, c,     x + 10, yy)
        draw_text(surface, f"Score: {score}",       font_sm, WHITE, x + 10, yy + 18)

    draw_text(surface, f"Speed: {env.game_speed}", font_sm, GRAY,       x + 10, y + 148)
    draw_text(surface, "R=reiniciar  F=menu",       font_sm, DARK_GRAY, x + 10, y + 164)


def draw_game_over(surface, font_lg, font_md, font_sm, env, death_scores):
    """Overlay de fim de jogo com vencedor."""

    h_score = death_scores[HUMAN_IDX] if not env.alive[HUMAN_IDX] else env.points
    a_score = death_scores[AI_IDX]    if not env.alive[AI_IDX]    else env.points

    if h_score > a_score:
        msg, color = "VOCE GANHOU!", BLUE
    elif a_score > h_score:
        msg, color = "IA GANHOU!", RED
    else:
        msg, color = "EMPATE!", YELLOW

    overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    surface.blit(overlay, (0, 0))

    cx = SCREEN_W // 2

    draw_text(surface, msg,                       font_lg, color, cx, SCREEN_H // 2 - 70, center=True)
    draw_text(surface, f"Seu score:  {h_score}",  font_md, GREEN, cx, SCREEN_H // 2,      center=True)
    draw_text(surface, f"Score IA:   {a_score}",  font_md, BLUE,  cx, SCREEN_H // 2 + 40, center=True)
    draw_text(surface,
              "R = jogar novamente   F = menu   ESC = sair",
              font_sm, GRAY, cx, SCREEN_H // 2 + 110, center=True)

    pygame.display.flip()

# ══════════════════════════════════════════════════════════════════════════════
# COUNTDOWN
# ══════════════════════════════════════════════════════════════════════════════

def show_countdown(env, font_lg, font_sm):
    """Exibe 3 → 2 → 1 → GO! antes da corrida."""

    clock = pygame.time.Clock()

    steps = [
        ("3",   RED,    700),
        ("2",   ORANGE, 700),
        ("1",   YELLOW, 700),
        ("GO!", GREEN,  500),
    ]

    for text, color, duration_ms in steps:

        t0 = pygame.time.get_ticks()

        while pygame.time.get_ticks() - t0 < duration_ms:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    import sys; sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        import sys; sys.exit()

            # renderiza o jogo parado como fundo
            env.render(flip=False)

            # escurece levemente
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 110))
            env.screen.blit(overlay, (0, 0))

            # número grande centralizado
            draw_text(env.screen, text, font_lg, color,
                      SCREEN_W // 2, SCREEN_H // 2 - 30, center=True)

            draw_text(env.screen, "Prepara...", font_sm, LIGHT_GRAY,
                      SCREEN_W // 2, SCREEN_H // 2 + 40, center=True)

            pygame.display.flip()
            clock.tick(60)

# ══════════════════════════════════════════════════════════════════════════════
# ROUND
# ══════════════════════════════════════════════════════════════════════════════

def play_round(ai_model, ai_label, env, fonts):
    """
    Joga uma rodada.
    Retorna 'restart' ou 'quit'.
    """

    font_lg, font_md, font_sm = fonts

    env.reset(n_dinos=2)

    # human dino slightly ahead (30 px forward) and will render in green
    env.dinos[HUMAN_IDX].x_pos  = DINO_X_POS + 100
    env.dinos[HUMAN_IDX].rect.x = DINO_X_POS + 100

    states       = [env._get_obs_for(d) for d in env.dinos]
    death_scores = [0, 0]

    show_countdown(env, font_lg, font_sm)

    # ── main loop ─────────────────────────────────────────────────────────────

    while True:

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "quit"
                if event.key == pygame.K_r:
                    return "restart"
                if event.key == pygame.K_f:
                    return "menu"

        # actions
        keys  = pygame.key.get_pressed()
        h_act = get_human_action(keys) if env.alive[HUMAN_IDX] else ACTION_RUN
        a_act = get_ai_action(ai_model, ai_label, states[AI_IDX]) if env.alive[AI_IDX] else ACTION_RUN

        # step
        prev_alive = list(env.alive)
        obs_list, _, _, _ = env.step_multi([h_act, a_act])

        for i in range(2):
            if prev_alive[i] and not env.alive[i]:
                death_scores[i] = env.points
            if obs_list[i] is not None:
                states[i] = obs_list[i]

        # render
        env.render(flip=False)
        draw_dino_labels(env.screen, font_sm, env)
        draw_versus_panel(env.screen, font_md, font_sm, env, death_scores, ai_label)
        pygame.display.flip()

        env.tick(FPS)

        # game over
        if env.done:
            draw_game_over(env.screen, font_lg, font_md, font_sm, env, death_scores)

            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return "quit"
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            return "restart"
                        if event.key == pygame.K_f:
                            return "menu"
                        if event.key == pygame.K_ESCAPE:
                            return "quit"

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # janela inicial para o menu
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Dino — Humano vs IA")

    fonts = (
        get_font(HUD_FONT_LARGE, bold=True),
        get_font(HUD_FONT_MEDIUM),
        get_font(HUD_FONT_SMALL),
    )

    choice = show_menu(screen, fonts)

    if choice is None:
        pygame.quit()
        sys.exit()

    # carrega modelo escolhido
    if choice == "DQN":
        ai_model = load_dqn()
    else:
        ai_model = load_neat()

    env = DinoEnv(render_mode=True)

    # loop de rodadas
    while True:
        result = play_round(ai_model, choice, env, fonts)
        if result == "quit":
            break

    pygame.quit()
    sys.exit()