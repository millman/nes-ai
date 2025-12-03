from __future__ import annotations

import numpy as np

NdArrayUint8 = np.ndarray[np.dtype[np.uint8]]

# From: nes/peripherals.py:323:
#   self.is_pressed[self.A] = int(state[self.A])
#   self.is_pressed[self.B] = int(state[self.B])
#   self.is_pressed[self.SELECT] = int(state[self.SELECT])
#   self.is_pressed[self.START] = int(state[self.START])
#   self.is_pressed[self.UP] = int(state[self.UP])
#   self.is_pressed[self.DOWN] = int(state[self.DOWN])
#   self.is_pressed[self.LEFT] = int(state[self.LEFT])
#   self.is_pressed[self.RIGHT] = int(state[self.RIGHT])
CONTROLLER_STATE_DESC = ["A", "B", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT"]


def _describe_controller_vector(is_pressed: NdArrayUint8) -> str:
    pressed = [
        desc
        for is_button_pressed, desc in zip(is_pressed, CONTROLLER_STATE_DESC)
        if is_button_pressed
    ]
    return str(pressed)


def _describe_controller_vector_compact(is_pressed: NdArrayUint8) -> str:
    pressed = [
        desc
        for is_button_pressed, desc in zip(is_pressed, CONTROLLER_STATE_DESC)
        if is_button_pressed
    ]
    if not pressed:
        return "NOOP"
    return "+".join(pressed)


def _to_controller_presses(buttons: list[str]) -> NdArrayUint8:
    is_pressed = np.zeros(8, dtype=np.uint8)
    for button in buttons:
        button_index = CONTROLLER_STATE_DESC.index(button.upper())
        is_pressed[button_index] = 1
    return is_pressed


CONTROLLER_NOOP = _to_controller_presses([])


# Action spaces adapted from:
# https://github.com/Kautenja/gym-super-mario-bros/blob/master/gym_super_mario_bros/actions.py
# venv/lib/python3.11/site-packages/gym_super_mario_bros/actions.py

# actions for the simple run right environment
RIGHT_ONLY = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
]


# actions for very simple movement
SIMPLE_MOVEMENT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
]


# actions for more complex movement
COMPLEX_MOVEMENT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
    ['left', 'a'],
    ['left', 'b'],
    ['left', 'a', 'b'],
    ['down'],
    ['up'],
]

COMPLEX_LEFT_RIGHT = [
    [],
    ['right'],
    ['right', 'a'],
    ['right', 'b'],
    ['right', 'a', 'b'],
    ['a'],
    ['left'],
    ['left', 'a'],
    ['left', 'b'],
    ['left', 'a', 'b'],
]


__all__ = [
    "NdArrayUint8",
    "CONTROLLER_STATE_DESC",
    "_describe_controller_vector",
    "_describe_controller_vector_compact",
    "_to_controller_presses",
    "CONTROLLER_NOOP",
    "RIGHT_ONLY",
    "SIMPLE_MOVEMENT",
    "COMPLEX_MOVEMENT",
    "COMPLEX_LEFT_RIGHT",
]
