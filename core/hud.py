# core/hud.py

from collections import deque

import numpy as np
import pygame

from core.constants import *
from core.assets import get_font
from core.utils import (
    draw_text,
    draw_panel,
    draw_line_chart,
)

# ══════════════════════════════════════════════════════════════════════════════
# HUD
# ══════════════════════════════════════════════════════════════════════════════

class HUD:

    def __init__(self):

        self.font_sm = get_font(HUD_FONT_SMALL)
        self.font_md = get_font(HUD_FONT_MEDIUM)
        self.font_lg = get_font(HUD_FONT_LARGE, bold=True)

        self.score_history  = deque(maxlen=200)
        self.loss_history   = deque(maxlen=400)
        self.reward_history = deque(maxlen=200)

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY
    # ══════════════════════════════════════════════════════════════════════════

    def push_score(self, score):

        self.score_history.append(score)

    def push_loss(self, loss):

        if loss is not None:
            self.loss_history.append(loss)

    def push_reward(self, reward):

        self.reward_history.append(reward)

    def push_episode(
            self,
            score,
            reward
    ):

        self.score_history.append(score)

        self.reward_history.append(reward)


    # ══════════════════════════════════════════════════════════════════════════
    # MAIN DRAW
    # ══════════════════════════════════════════════════════════════════════════

    def draw(
        self,
        surface,
        mode,
        data
    ):
        """
        mode:
            'dqn'
            'neat'
        """

        self._draw_stats_panel(surface, mode, data)

        self._draw_inputs_panel(
            surface,
            data.get("state", None)
        )

        self._draw_score_chart(surface)

        self._draw_loss_chart(surface)

        if mode == "dqn":
            self._draw_q_values(
                surface,
                data.get("q_values", None)
            )

        if mode == "neat":
            self._draw_neural_network(
                surface,
                data.get("state", None),
                data.get("neural_outputs", None),
                data.get("genome", None),
            )

        if data.get("sensor_data", None):
            self._draw_sensor_line(
                surface,
                data["sensor_data"]
            )

    # ══════════════════════════════════════════════════════════════════════════
    # STATS PANEL
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_stats_panel(
        self,
        surface,
        mode,
        data
    ):

        draw_panel(surface, 8, 8, 300, 240)

        lines = []

        if mode == "dqn":

            lines = [
                (f"Mode:        DQN", WHITE),
                (f"Episode:     {data.get('episode', 0)}", WHITE),
                (f"Score:       {data.get('score', 0)}", YELLOW),
                (f"Best Score:  {data.get('best_score', 0)}", CYAN),
                (f"Action:      {data.get('action_name', '-')}", GREEN),
                (f"Epsilon:     {data.get('epsilon', 0):.4f}", PURPLE),
                (f"Loss:        {data.get('loss', 0):.5f}" if data.get('loss', None) is not None else "Loss:        -", GRAY),
                (f"Reward:      {data.get('reward', 0):.2f}", GREEN),
                (f"Steps:       {data.get('steps', 0)}", GRAY),
                (f"FPS:         {data.get('fps', 0):.1f}", ORANGE),
            ]

        elif mode == "neat":

            lines = [
                (f"Mode:        NEAT", WHITE),
                (f"Generation:  {data.get('generation', 0)}", WHITE),
                (f"Alive:       {data.get('alive', 0)}", GREEN),
                (f"Score:       {data.get('score', 0)}", YELLOW),
                (f"Best Score:  {data.get('best_score', 0)}", CYAN),
                (f"Fitness:     {data.get('fitness', 0):.2f}", PURPLE),
                (f"Species:     {data.get('species', 0)}", BLUE),
                (f"Speed:       {data.get('speed', 0):.1f}", GRAY),
                (f"Reward:      {data.get('reward', 0):.2f}", GREEN),
                (f"FPS:         {data.get('fps', 0):.1f}", ORANGE),
            ]

        for i, (txt, color) in enumerate(lines):

            draw_text(
                surface,
                txt,
                self.font_sm,
                color,
                16,
                16 + i * 22
            )

    # ══════════════════════════════════════════════════════════════════════════
    # INPUT PANEL
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_inputs_panel(
        self,
        surface,
        obs
    ):

        if obs is None:
            return

        draw_panel(surface, 316, 8, 300, 145)

        draw_text(
            surface,
            "Network Inputs",
            self.font_md,
            WHITE,
            324,
            16
        )

        for i, (label, value) in enumerate(zip(OBS_LABELS, obs)):

            col = i % 2
            row = i // 2

            x = 324 + col * 145
            y = 45 + row * 22

            draw_text(
                surface,
                f"{label}: {value:.3f}",
                self.font_sm,
                GRAY,
                x,
                y
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SCORE CHART
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_score_chart(
        self,
        surface
    ):

        draw_line_chart(
            surface,
            list(self.score_history),
            x=8,
            y=415,
            w=300,
            h=90,
            color=CYAN
        )

        draw_text(
            surface,
            "Score History",
            self.font_sm,
            WHITE,
            16,
            420
        )

    # ══════════════════════════════════════════════════════════════════════════
    # LOSS CHART
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_loss_chart(
        self,
        surface
    ):

        if len(self.loss_history) < 2:
            return

        draw_line_chart(
            surface,
            list(self.loss_history),
            x=8,
            y=515,
            w=300,
            h=75,
            color=YELLOW
        )

        draw_text(
            surface,
            "Loss",
            self.font_sm,
            WHITE,
            16,
            520
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Q VALUES
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_q_values(
            self,
            surface,
            q_values
    ):

        if q_values is None:
            return

        x = SCREEN_W - 220
        y = 8
        w = 210
        h = 120

        draw_panel(surface, x, y, w, h)

        draw_text(
            surface,
            "Q Values",
            self.font_md,
            WHITE,
            x + 10,
            y + 10
        )

        labels = [
            "RUN",
            "JUMP",
            "DUCK"
        ]

        colors = [
            GRAY,
            GREEN,
            BLUE,
        ]

        mn = float(np.min(q_values))
        mx = float(np.max(q_values))

        rng = mx - mn

        if rng <= 0:
            rng = 1.0

        for i, (label, qv, color) in enumerate(
                zip(labels, q_values, colors)
        ):
            yy = y + 40 + i * 24

            # fundo da barra
            pygame.draw.rect(
                surface,
                DARK_GRAY,
                (x + 75, yy, 110, 14)
            )

            # barra
            normalized = (qv - mn) / rng

            normalized = max(0.0, min(1.0, normalized))

            bar_w = int(normalized * 110)

            pygame.draw.rect(
                surface,
                color,
                (x + 75, yy, bar_w, 14)
            )

            # texto separado da barra
            draw_text(
                surface,
                f"{label}",
                self.font_sm,
                WHITE,
                x + 10,
                yy - 1
            )

            draw_text(
                surface,
                f"{qv:.2f}",
                self.font_sm,
                WHITE,
                x + 190,
                yy - 1
            )

    # ══════════════════════════════════════════════════════════════════════════
    # NETWORK VISUALIZATION
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_neural_network(
        self,
        surface,
        inputs,
        outputs,
        genome=None
    ):

        if inputs is None or outputs is None:
            return

        n_inputs  = len(inputs)
        n_outputs = len(outputs)

        out_labels = ["RUN", "JUMP", "DUCK"]

        # ── panel geometry ────────────────────────────────────────────────────

        x = SCREEN_W - 320
        y = 8
        w = 310

        # height adapts to the number of input nodes
        h = max(220, n_inputs * 22 + 50)

        draw_panel(surface, x, y, w, h)

        draw_text(
            surface,
            "Neural Network",
            self.font_md,
            WHITE,
            x + 10,
            y + 10
        )

        # ── column x-positions ────────────────────────────────────────────────

        x_in  = x + 55
        x_hid = x + 160
        x_out = x + 265

        content_h = h - 40     # vertical space for nodes

        # ── node positions ────────────────────────────────────────────────────

        node_pos = {}

        # inputs — keys are negative in neat-python: -1 … -n_inputs
        input_keys = list(range(-n_inputs, 0))   # [-8, …, -1]
        step = content_h / max(n_inputs, 1)
        for i, key in enumerate(input_keys):
            node_pos[key] = (x_in, int(y + 30 + i * step + step / 2))

        # outputs — keys 0 … n_outputs-1
        output_keys = list(range(n_outputs))
        step = content_h / max(n_outputs, 1)
        for i, key in enumerate(output_keys):
            node_pos[key] = (x_out, int(y + 30 + i * step + step / 2))

        if genome is not None:
            # hidden — any key in genome.nodes that is not an output key
            hidden_keys = sorted(
                k for k in genome.nodes if k not in output_keys
            )
            if hidden_keys:
                step = content_h / max(len(hidden_keys), 1)
                for i, key in enumerate(hidden_keys):
                    node_pos[key] = (x_hid, int(y + 30 + i * step + step / 2))

        # ── connections ───────────────────────────────────────────────────────

        if genome is not None:

            for (src, dst), conn in genome.connections.items():

                if not conn.enabled:
                    continue

                if src not in node_pos or dst not in node_pos:
                    continue

                w_val = float(conn.weight)
                alpha = min(255, int(abs(w_val) * 60 + 40))

                if w_val >= 0:
                    color = (0, alpha, 60)
                else:
                    color = (alpha, 0, 60)

                pygame.draw.line(
                    surface,
                    color,
                    node_pos[src],
                    node_pos[dst],
                    1
                )

        else:
            # static fallback: all inputs → all outputs
            for ip in node_pos.values():
                for op_key in output_keys:
                    pygame.draw.line(
                        surface,
                        DARK_GRAY,
                        ip,
                        node_pos[op_key],
                        1
                    )

        # ── input nodes ───────────────────────────────────────────────────────

        for i, key in enumerate(input_keys):

            pos       = node_pos[key]
            intensity = max(50, min(255, int(abs(float(inputs[i])) * 255)))

            pygame.draw.circle(surface, (0, intensity, 120), pos, 7)
            pygame.draw.circle(surface, WHITE, pos, 7, 1)

            draw_text(
                surface,
                OBS_LABELS[i],
                self.font_sm,
                GRAY,
                pos[0] - 40,
                pos[1] - 6
            )

        # ── hidden nodes ─────────────────────────────────────────────────────

        if genome is not None:

            for key in hidden_keys:

                pos = node_pos[key]

                pygame.draw.circle(surface, PURPLE, pos, 7)
                pygame.draw.circle(surface, WHITE,  pos, 7, 1)

                draw_text(
                    surface,
                    f"h{key}",
                    self.font_sm,
                    LIGHT_GRAY,
                    pos[0] + 10,
                    pos[1] - 6
                )

        # ── output nodes ─────────────────────────────────────────────────────

        best_out = int(np.argmax(outputs))

        for i, key in enumerate(output_keys):

            pos   = node_pos[key]
            color = GREEN if i == best_out else RED

            pygame.draw.circle(surface, color, pos, 9)
            pygame.draw.circle(surface, WHITE, pos, 9, 1)

            draw_text(
                surface,
                out_labels[i],
                self.font_sm,
                WHITE,
                pos[0] + 14,
                pos[1] - 6
            )

    # ══════════════════════════════════════════════════════════════════════════
    # SENSOR LINE
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_sensor_line(
        self,
        surface,
        sensor_data
    ):

        start = sensor_data["start"]
        end   = sensor_data["end"]

        pygame.draw.line(
            surface,
            ORANGE,
            start,
            end,
            2
        )