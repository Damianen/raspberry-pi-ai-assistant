"""Animated procedural face — pure System 1.

Everything in this package reacts instantly and locally: no network, no LLM,
no disk on the render path. `logic` and `styles` are pure math (pytest
territory); `render` and `module` are the only files that import pygame, so
tests never need a display. Import `Face` from `assistant.face.module`.
"""
