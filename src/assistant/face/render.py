"""Draws a computed FaceFrame with pygame. No animation logic lives here."""

from __future__ import annotations

import pygame

from assistant.face.logic import EyeFrame, FaceFrame

# Drawing constants — art design, like the layout fractions in logic.py.
CORNER_RADIUS = 0.45  # rounded-rect corner radius, × the smaller eye dimension
HIGHLIGHT_COLOR = (245, 248, 255)
HIGHLIGHT_SCALE = 0.30  # highlight radius, × pupil radius
HIGHLIGHT_OFFSET = 0.40  # highlight offset toward upper-left, × pupil radius
CURVE_COVER_WIDTH = 1.7  # happy-arc cover ellipse width, × eye width
CURVE_COVER_HEIGHT = 1.25  # happy-arc cover ellipse height, × eye height
CURVE_DEPTH = 0.6  # how deep the arc bites at curve=1, × eye height
MIN_PUPIL_PX = 2


def _rgb(color: tuple[float, float, float]) -> tuple[int, int, int]:
    r, g, b = (max(0, min(255, round(c))) for c in color)
    return (r, g, b)


def draw_face(screen: pygame.Surface, frame: FaceFrame) -> None:
    screen.fill(_rgb(frame.bg))
    eye_color = _rgb(frame.eye_color)
    iris_color = _rgb(frame.iris_color)
    bg = _rgb(frame.bg)
    for eye in frame.eyes:
        _draw_eye(screen, eye, eye_color, iris_color, bg)


def _draw_eye(
    screen: pygame.Surface,
    eye: EyeFrame,
    eye_color: tuple[int, int, int],
    iris_color: tuple[int, int, int],
    bg: tuple[int, int, int],
) -> None:
    rect = pygame.Rect(
        round(eye.center_x - eye.width / 2),
        round(eye.center_y - eye.height / 2),
        max(round(eye.width), 1),
        max(round(eye.height), 2),
    )
    radius = min(
        round(min(eye.width, eye.height) * CORNER_RADIUS),
        rect.width // 2,
        rect.height // 2,
    )
    pygame.draw.rect(screen, eye_color, rect, border_radius=radius)

    # Pupil, highlight, and the happy-arc cover are clipped to this eye so an
    # extreme gaze or a wide cover ellipse can never touch the neighbor.
    screen.set_clip(rect)
    pupil_r = round(eye.pupil_r)
    if pupil_r >= MIN_PUPIL_PX:
        px, py = round(eye.pupil_x), round(eye.pupil_y)
        pygame.draw.circle(screen, iris_color, (px, py), pupil_r)
        highlight_r = max(round(pupil_r * HIGHLIGHT_SCALE), 1)
        offset = round(pupil_r * HIGHLIGHT_OFFSET)
        pygame.draw.circle(screen, HIGHLIGHT_COLOR, (px - offset, py - offset), highlight_r)
    if eye.curve > 0.01:
        cover_w = eye.width * CURVE_COVER_WIDTH
        cover_h = eye.height * CURVE_COVER_HEIGHT
        bite = eye.curve * eye.height * CURVE_DEPTH
        cover = pygame.Rect(
            round(eye.center_x - cover_w / 2),
            round(rect.bottom - bite),
            round(cover_w),
            round(cover_h),
        )
        pygame.draw.ellipse(screen, bg, cover)
    screen.set_clip(None)
