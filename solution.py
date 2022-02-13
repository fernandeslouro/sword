from supportlib.sensor_position_finding import (
    SensorPositionFinder,
    SensorPositionRequester,
)
from supportlib.data_types import Sensor, SensorPosition, SensorData
from supportlib.data_types import SensorPosition
from typing import Dict
from enum import Enum
import numpy as np


class MovementState(Enum):
    BEFORE_RIGHT_RAISE = 1
    DURING_RIGHT_RAISE = 2
    BEFORE_LEFT_RAISE = 3
    DURING_LEFT_RAISE = 4


class SolutionPositionFinder(SensorPositionFinder):

    N = 50
    past_data = np.array([])
    mov_state = MovementState(1)

    def __init__(self, position_requester: SensorPositionRequester):
        """
        Constructor.
        :param position_requester: the PositionRequester to be called when
        sensor positions are identified
        """
        self.position_requester = position_requester

    def on_new_sensor_sample(self, sensor_data: Dict[Sensor, SensorData]) -> None:
        """
        Callback called each time a new sample is received from the sensors
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """

        self.past_data, _ = self.position_acceleration_solution(
            sensor_data, self.past_data
        )

    def position_acceleration_solution(self, dict_data, data_array):
        """
        A more robust solution. When it detects a sensor with values of z close to 0
        and values of x close to 1, it assigns the sensor to the right/left thigh,
        depending on whether the movement has been detected before. At the same time,
        we expect the shank of the same leg to be rising with the thigh. However,
        there is the chance the shank stays in a mostly upright position, with no
        visible change on x, y or z. However, the shank sensor would have the highest
        acceleration value of all the other sensors, so it can be identified as well.
        In order to separate the second movement from the first, a flag is created,
        signaling the values of e.g. x and z are still "high" because of a movement
        already detected. When the values drop to an acceptable level again, we are
        ready to pick up the movements of the left leg.
        It keeps a rolling window of 100 counts
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        :param past_data: numpy array of dimensions 5 x 4 x N, holding past values for
        all the data.
        """
        # TODO: Update comment
        # TODO: Add comments explaining the bottom part
        data_array = self.update_past_data(dict_data, data_array)
        aggs = self.data_aggregations(data_array)

        # for n_sensor in range(1, 6):
        for sensor in self.unassigned_sensors():
            n_sensor = sensor.value

            if self.mov_state in [
                MovementState.BEFORE_RIGHT_RAISE,
                MovementState.BEFORE_LEFT_RAISE,
            ]:
                if aggs[n_sensor - 1, 2] > -0.25 and aggs[n_sensor - 1, 0] > 0.7:
                    if self.mov_state == MovementState.BEFORE_RIGHT_RAISE:
                        thigh_rising = SensorPosition.RIGHT_THIGH
                        shank_rising = SensorPosition.RIGHT_SHANK
                        self.mov_state = MovementState.DURING_RIGHT_RAISE
                    elif self.mov_state == MovementState.BEFORE_LEFT_RAISE:
                        thigh_rising = SensorPosition.LEFT_THIGH
                        shank_rising = SensorPosition.LEFT_SHANK
                        self.mov_state = MovementState.DURING_LEFT_RAISE
                    self.position_requester.on_sensor_position_found(
                        Sensor(n_sensor), thigh_rising
                    )
                    accelerations = aggs[:, -1]
                    max_acc_sensor = np.argsort(accelerations)[-1] + 1
                    if max_acc_sensor == n_sensor:
                        max_acc_sensor = np.argsort(accelerations)[-2] + 1
                    self.position_requester.on_sensor_position_found(
                        Sensor(max_acc_sensor), shank_rising
                    )
            else:
                if (
                    self.mov_state == MovementState.DURING_RIGHT_RAISE
                    and aggs[n_sensor - 1, 2] < -0.7
                    and aggs[n_sensor - 1, 0] < 0.3
                ):
                    self.mov_state = MovementState.BEFORE_LEFT_RAISE

        if self.mov_state == MovementState.DURING_LEFT_RAISE:
            remaining_sensor = self.unassigned_sensors()[0]
            self.position_requester.on_sensor_position_found(
                remaining_sensor, SensorPosition.CHEST
            )
            self.position_requester.on_finish()

        return data_array, aggs

    def update_past_data(self, dict_data, data_array):
        """
        Converts the dict provided by the generator into a numpy array of 5 x 4 x 1,
        to be appended to the past_data numpy array
        :param dict_data: dictionary corresponding each sensor to the orientation
        vector and the acceleration value
        :param arr: the array to apend to, which contains the past data
        """
        to_append = np.zeros((5, 4))

        for sensor, values in dict_data.items():
            single_sensor = np.append(values.vec, values.acc)
            np.reshape(single_sensor, (1, 4))
            to_append[sensor.value - 1] = single_sensor

        if data_array.size == 0:
            data_array = to_append
        elif np.shape(data_array) == (5, 4):
            data_array = np.stack((data_array, to_append), axis=2)
        elif np.shape(data_array)[2] < self.N:
            data_array = np.append(data_array, np.atleast_3d(to_append), axis=2)
        elif np.shape(data_array)[2] == self.N:
            data_array = np.append(data_array, np.atleast_3d(to_append), axis=2)
            data_array = data_array[:, :, 1:]
        return data_array

    def data_aggregations(self, data_array, chosen_agg="mean"):
        """
        Some data aggregations are necessary to implement our algorith:
        - Average x, y, z, acc values
        - Rows are each of the sensors
        - Columns are the values of x, y, z, a
        """
        if len(np.shape(data_array)) == 3:
            aggregation = np.zeros((5, 4))
            if chosen_agg == "mean":
                for i in range(5):
                    aggregation[i, :] = [
                        np.mean(data_array[i, 0, :]),
                        np.mean(data_array[i, 1, :]),
                        np.mean(data_array[i, 2, :]),
                        np.mean(data_array[i, 3, :]),
                    ]
            if chosen_agg == "std":
                for i in range(5):
                    aggregation[i, :] = [
                        np.std(data_array[i, 0, :]),
                        np.std(data_array[i, 1, :]),
                        np.std(data_array[i, 2, :]),
                        np.std(data_array[i, 3, :]),
                    ]
        else:
            aggregation = np.zeros((5, 4))
        return aggregation

    def unassigned_sensors(self):
        """
        Ouputs a list containing the sensors not yet assigned to any body part
        :param sensor_positions: dict corresponding each of the Sensor variables to its
        SensorPosition
        :return: list of usassigned Sensor variables
        """
        all_sensors = set(list(Sensor))
        assigned_sensors = set(self.position_requester.sensor_positions.keys())
        return list(all_sensors - assigned_sensors)
