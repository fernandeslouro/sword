from abc import ABC, abstractmethod
from typing import Dict

from supportlib.data_types import Sensor, SensorPosition, SensorData


class SensorPositionRequester(ABC):
    """
    Base class for the callbacks SensorPositionFinder uses to signal the
    identification of sensor positions.
    """

    @abstractmethod
    def on_sensor_position_found(self,
                                 sensor: Sensor,
                                 position: SensorPosition) -> None:
        """
        Callback called when a position of a given sensor is identified.
        :param sensor: the sensor whose position was identified
        :param position: the position of that sensor
        """
        pass

    @abstractmethod
    def on_finish(self) -> None:
        """
        Callback called when all sensor positions were identified.
        """
        pass


class SensorPositionFinder(ABC):
    """
    Base class for classes that receive sensor data streams and use them to
    identify the position of the sensors.
    """

    def __init__(self, position_requester: SensorPositionRequester):
        """
        Constructor.
        :param position_requester: the PositionRequester to be called when
        sensor positions are identified
        """
        self.position_requester = position_requester

    @abstractmethod
    def on_new_sensor_sample(self,
                             sensor_data: Dict[Sensor, SensorData]) -> None:
        """
        Callback called each time a new sample is received from the sensors
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """
        pass
