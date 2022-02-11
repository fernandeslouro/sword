import numpy as np

from typing import NamedTuple
from enum import Enum


class Sensor(Enum):
    SENSOR_1 = 1
    SENSOR_2 = 2
    SENSOR_3 = 3
    SENSOR_4 = 4
    SENSOR_5 = 5


class SensorPosition(Enum):
    CHEST = 1
    RIGHT_THIGH = 2
    LEFT_THIGH = 3
    RIGHT_SHANK = 4
    LEFT_SHANK = 5


class SensorData(NamedTuple):
    vec: np.array
    acc: float
