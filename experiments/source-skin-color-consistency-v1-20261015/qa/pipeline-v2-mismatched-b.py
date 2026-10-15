"""Byte-level foreground solve for a fixed final background and fixed geometry.

Experimental contract only. Source/alpha/material gates are caller-owned.
"""
import numpy as np

from solver import solve


def correct_foreground(source, alpha, foreground, observed_background,
                       final_background, skin_core, allowed, transition,
                       radius=6):
    """Return the byte foreground used by the frozen native sampler.

    RGB inputs are float 0..1. The transition acts on color only; it never
    changes alpha or material ownership.
    """
    solved = solve(source, alpha, foreground, observed_background,
                   skin_core, allowed, radius=radius)
    corrected = foreground + (solved['foreground'] - foreground) * transition[..., None]
    foreground_byte = np.floor(np.clip(corrected, 0, 1) * 255 + .5).astype('float32')
    return {**solved, 'correctedForegroundByte': foreground_byte}
