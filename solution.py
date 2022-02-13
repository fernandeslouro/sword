from supportlib.sensor_position_finding import (
    SensorPositionFinder,
    SensorPositionRequester,
)
from supportlib.data_types import Sensor, SensorPosition, SensorData
from supportlib.data_types import SensorPosition
from typing import Dict
from enum import Enum
import numpy as np
import itertools

N_TRACKERS = 5


class MovementState(Enum):
    BEFORE_RIGHT_RAISE = 1
    DURING_RIGHT_RAISE = 2
    BEFORE_LEFT_RAISE = 3
    DURING_LEFT_RAISE = 4


class SolutionPositionFinder(SensorPositionFinder):

    past_data = np.array([])
    past_angles = np.array([])
    mov_state = MovementState(1)

    def __init__(self, position_requester: SensorPositionRequester, rolling_agg=50):
        """
        Constructor.
        :param position_requester: the PositionRequester to be called when
        sensor positions are identified
        : rolling_agg: number of past sensor samples to consider when calculating
        "rolling" (i.e applied to the last N samples) aggregations, such as the mean
        of the x, y, z and acceleration values
        """
        self.position_requester = position_requester
        self.rolling_agg = rolling_agg

    def on_new_sensor_sample(self, sensor_data: Dict[Sensor, SensorData]) -> None:
        """
        Callback called each time a new sample is received from the sensors. When
        it detects a sensor with values of z close to 0 and values of x close to 1,
        it assigns the sensor to the right/left thigh, depending on whether the
        movement has been detected before (i.e. the MovementState). At the same time,
        we expect the shank of the same leg to be rising with the thigh. However,
        there is the chance the shank stays in a mostly upright position, with no
        visible change on x, y or z. However, the shank sensor would have the highest
        acceleration value of all the other sensors, so it can be identified as well.
        In order to separate the second movement from the first, a flag is created,
        signaling the values of e.g. x and z are still "high" because of a movement
        already detected. When the values drop to an acceptable level again, we are
        ready to pick up the movements of the left leg.
        All values are calculated based on a rolling mean, in order to increase
        robustness and consistency of results.
        The past values are stored in a 3D numpy array of dimensions 5 (number of
        sensors) x 4 (x, y, z, a) x 50 (rolling_agg). The aggregations are stored
        in a numpy array of dimensions 5 x 4.
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """
        # TODO: Consider adding a kalman filter to the acceleration

        # Get data from latest sample into rolling window and calculating aggregations
        self.past_data, self.past_angles = self.rolling_windows(
            sensor_data, self.past_data, self.past_angles
        )
        aggs = data_aggregations(self.past_data)

        for sensor in self.unassigned_sensors():
            n_sensor = sensor.value

            if self.mov_state in [
                MovementState.BEFORE_RIGHT_RAISE,
                MovementState.BEFORE_LEFT_RAISE,
            ]:
                # If we are in a state before a leg raise, and we detect values of z close
                # to zero (thigh raising close to paralled) and values of x close to 1
                # (thigh close to pointing straight forward), we can consider the leg is
                # being raised. The sensor where these values are verified will be the thigh,
                # and the sensor with the highest average acceleration in the rolling window
                # will be the shank of the same leg. These values are an heuristic, but seem
                # to work well.
                if aggs[n_sensor - 1, 2] > -0.2 and aggs[n_sensor - 1, 0] > 0.7:
                    if self.mov_state == MovementState.BEFORE_RIGHT_RAISE:
                        # If no legs have been raised yet, the raised thigh and shank are the
                        # ones of the right leg. We also swithc to the next movement state.
                        thigh_rising = SensorPosition.RIGHT_THIGH
                        shank_rising = SensorPosition.RIGHT_SHANK
                        self.mov_state = MovementState.DURING_RIGHT_RAISE
                    elif self.mov_state == MovementState.BEFORE_LEFT_RAISE:
                        # If we are expecting the raise of the left leg and a leg is raised, we
                        # assume it's the left leg :^)
                        thigh_rising = SensorPosition.LEFT_THIGH
                        shank_rising = SensorPosition.LEFT_SHANK
                        self.mov_state = MovementState.DURING_LEFT_RAISE

                    self.position_requester.on_sensor_position_found(
                        Sensor(n_sensor), thigh_rising
                    )
                    # We then get the average acceleration of the 5 sensors from the aggregations
                    # array, in order to find the sensor with the highest acceleration out of the
                    # others.
                    accelerations = aggs[:, -1]
                    max_acc_sensor = np.argsort(accelerations)[-1] + 1
                    if max_acc_sensor == n_sensor:
                        max_acc_sensor = np.argsort(accelerations)[-2] + 1
                    if (
                        Sensor(max_acc_sensor)
                        in self.position_requester.sensor_positions.keys()
                    ):
                        raise ValueError(
                            "The segment thought to be the shank has already been assigned. This\
                             is based on an assumption related to the shank being the highest\
                             acceleration segment, which may not hold up."
                        )
                    self.position_requester.on_sensor_position_found(
                        Sensor(max_acc_sensor), shank_rising
                    )
            else:
                if (
                    self.mov_state == MovementState.DURING_RIGHT_RAISE
                    and aggs[n_sensor - 1, 2] < -0.7
                    and aggs[n_sensor - 1, 0] < 0.3
                ):
                    # If after raising the right leg, the right thigh sensor returns
                    # to a position closed to vertical (on x and z), we consider that
                    # the left leg will soon be raised. This will cause the next thigh
                    # lift to be associated to the left leg. These values are heuristics,
                    # but seem to work well.
                    self.mov_state = MovementState.BEFORE_LEFT_RAISE

        if self.mov_state == MovementState.DURING_LEFT_RAISE:
            remaining_sensor = self.unassigned_sensors()[0]
            self.position_requester.on_sensor_position_found(
                remaining_sensor, SensorPosition.CHEST
            )
            self.position_requester.on_finish()

    def rolling_windows(
        self,
        dict_data: Dict[Sensor, SensorData],
        data_array: np.array,
        angles_array: np.array,
    ) -> (np.array, np.array):
        """
        Creates an array containing a rolling window of past samples' data. We
        always take action based on rolling window aggregations as opposed to
        individual values of samples, to add robstness. This
        array has dimensions 5 (number of sensors) x 4 (x, y, z, a) x 50
        (rolling_agg). We start by getting the 5 x 4 array of the current sample.
        We then append it to the array containing the past data, which is empty
        initially.
        :param dict_data: dictionary corresponding each sensor to the orientation
        vector and the acceleration value
        :param arr: the array to apend to, which contains the past data
        """
        # TODO: add comments related to angles
        data_to_append = np.zeros((N_TRACKERS, 4))

        # The data is extracted from the dict provided by the generator from
        # iterate_sensor_data. It starts by exracting the data from the dict into a
        # 5 x 4 numpy array, filling the array one sensor at a time.
        for sensor, values in dict_data.items():
            single_sensor = np.append(values.vec, values.acc)
            np.reshape(single_sensor, (1, 4))
            data_to_append[sensor.value - 1] = single_sensor

        pairs = list(itertools.combinations(range(N_TRACKERS), 2))
        # Pairs are generated in a sorted manner. So we can always know where a
        # specific pair is in the numpy array.
        angles_to_append = np.zeros((len(pairs)))
        for i, p in enumerate(pairs):
            angles_to_append[i] = angle_between(
                np.array(
                    (
                        data_to_append[p[0], 0],
                        data_to_append[p[0], 1],
                        data_to_append[p[0], 2],
                    )
                ),
                np.array(
                    (
                        data_to_append[p[1], 0],
                        data_to_append[p[1], 1],
                        data_to_append[p[1], 2],
                    )
                ),
            )

        # Since we are starting from an empty array, the commands to joing a 5 x 4
        # array with the previous data changes, according to the shape of the data.
        # Before data from 50 samples is present, we simply append data to the array
        # but when the third dimension gets to 50, we must start removing the older
        # data.
        if data_array.size == 0:
            data_array = data_to_append
            angles_array = angles_to_append
        elif np.shape(data_array) == (N_TRACKERS, 4):
            data_array = np.stack((data_array, data_to_append), axis=2)
            angles_array = np.stack((angles_array, angles_to_append))
        elif np.shape(data_array)[2] < self.rolling_agg:
            data_array = np.append(data_array, np.atleast_3d(data_to_append), axis=2)
            angles_array = np.vstack((angles_array, angles_to_append))
        elif np.shape(data_array)[2] == self.rolling_agg:
            data_array = np.append(data_array, np.atleast_3d(data_to_append), axis=2)
            data_array = data_array[:, :, 1:]
            angles_array = np.vstack((angles_array, angles_to_append))
            angles_array = angles_array[1:, :]

        return data_array, angles_array

    def unassigned_sensors(self) -> list(Sensor):
        """
        Ouputs a list containing the sensors not yet assigned to any body part
        :return: list of Sensor variables corresponding to sensors not yer assigned to any
        body segment.
        """
        all_sensors = set(list(Sensor))
        assigned_sensors = set(self.position_requester.sensor_positions.keys())
        return list(all_sensors - assigned_sensors)


def data_aggregations(data_array: np.array, chosen_agg="mean") -> np.array:
    """
    Calculates aggregations from the numpy array containing the data of the rolling
    window of past samples. Several aggregations are implemented, such as the mean
    and standard deviation. If there are less than two data samples worth of data in
    data_array, the function outputs an array of zeros.
    :param data_array: 5 x 4 x N numpy array containing the data from the rolling
    window of past samples.
    :param chosen_agg: string with the chosen aggregation to perform on the rolling
    window data.
    :return: a 5 x 4 matrix with the containing the results of the chosed aggregation
    performed on the rolling window data.
    """
    if len(np.shape(data_array)) == 3:
        aggregation = np.zeros((N_TRACKERS, 4))
        if chosen_agg == "mean":
            for i in range(N_TRACKERS):
                aggregation[i, :] = [
                    np.mean(data_array[i, 0, :]),
                    np.mean(data_array[i, 1, :]),
                    np.mean(data_array[i, 2, :]),
                    np.mean(data_array[i, 3, :]),
                ]
        elif chosen_agg == "std":
            for i in range(N_TRACKERS):
                aggregation[i, :] = [
                    np.std(data_array[i, 0, :]),
                    np.std(data_array[i, 1, :]),
                    np.std(data_array[i, 2, :]),
                    np.std(data_array[i, 3, :]),
                ]
        else:
            raise ValueError("Only supported aggregations are 'mean' and 'std'")
    else:
        aggregation = np.zeros((N_TRACKERS, 4))
    return aggregation


def angle_between(v1, v2):
    """
    Returns the unit vector of the vector.
    Returns the angle in radians between vectors 'v1' and 'v2':
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
