from supportlib.data_types import Sensor, SensorPosition, SensorData
from supportlib.data_types import SensorPosition
from supportlib.sensor_position_finding import (
    SensorPositionFinder,
    SensorPositionRequester,
)
from typing import Dict
from enum import Enum
import numpy as np
import itertools
import math

N_TRACKERS = 5


class MovementState(Enum):
    BEFORE_RIGHT_RAISE = 1
    DURING_RIGHT_RAISE = 2
    BEFORE_LEFT_RAISE = 3
    DURING_LEFT_RAISE = 4


class SolutionPositionFinder(SensorPositionFinder):

    # Arrays where rolling windows will be stored, and setting initial movement state.
    # We are storing rolling windows of past values of x, y, z, a, for each sensor,
    # as well as past values of anglues between all pairs of sensors/vectors (a
    # rolling window with values of 10 angles in total.
    past_data = np.array([])
    past_angles = np.array([])
    mov_state = MovementState(1)

    def __init__(self, position_requester: SensorPositionRequester, rolling_agg=50):
        """
        Constructor.
        :param position_requester: the PositionRequester to be called when
        sensor positions are identified.
        : rolling_agg: number of past sensor samples to consider when calculating
        "rolling" (i.e applied to the last N samples) aggregations, such as the mean
        of the x, y, z and acceleration values, and the angles between vectors.
        """
        self.position_requester = position_requester
        self.rolling_agg = rolling_agg

    def on_new_sensor_sample(self, sensor_data: Dict[Sensor, SensorData]) -> None:
        """
        Callback called each time a new sample is received from the sensors. When
        it detects a patters matching a thigh raise in one of the sensors,
        it assigns the sensor to the right/left thigh, depending on whether the
        movement has been detected before (i.e. the MovementState). At the same time,
        we expect the shank of the same leg to be rising with the thigh. However,
        there is the chance the shank stays in a mostly upright position, with no
        visible change on x, y or z. However, the shank sensor would have the highest
        acceleration value of all the other sensors, so it can be identified as well.
        In order to separate the second movement from the first, a the movement state
        enum is used. When the thigh is no longer raised, we areready to pick up the
        movements of the left leg.
        All values are calculated based on a rolling mean, in order to increase
        robustness and consistency of results.
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """

        # Get data from latest sample into rolling window and calculating aggregations
        self.past_data, self.past_angles = self.rolling_windows(
            sensor_data, self.past_data, self.past_angles
        )
        data_aggs = data_aggregations(self.past_data)
        angle_aggs = angle_aggregations(self.past_angles)

        for sensor in self.unassigned_sensors():
            n_sensor = sensor.value

            if self.mov_state in [
                MovementState.BEFORE_RIGHT_RAISE,
                MovementState.BEFORE_LEFT_RAISE,
            ]:
                # If we are in a state before a leg raise, and we detect a thigh raise, The
                # sensor where these values are verified will be assigned to the thigh,
                # and the sensor with the highest average acceleration in the rolling window
                # will be the shank of the same leg.
                if self.is_thigh_raised(data_aggs, angle_aggs, n_sensor):
                    if self.mov_state == MovementState.BEFORE_RIGHT_RAISE:
                        # If no legs have been raised yet, the raised thigh and shank are the
                        # ones of the right leg. We also switch to the next movement state.
                        thigh_rising = SensorPosition.RIGHT_THIGH
                        shank_rising = SensorPosition.RIGHT_SHANK
                        self.mov_state = MovementState.DURING_RIGHT_RAISE
                    elif self.mov_state == MovementState.BEFORE_LEFT_RAISE:
                        # If we are expecting the raise of the left leg and a leg is raised, we
                        # assume it's the left leg
                        thigh_rising = SensorPosition.LEFT_THIGH
                        shank_rising = SensorPosition.LEFT_SHANK
                        self.mov_state = MovementState.DURING_LEFT_RAISE

                    self.position_requester.on_sensor_position_found(
                        Sensor(n_sensor), thigh_rising
                    )

                    # After assigning a sensor to the thing, we look through the remaining
                    # sensors, and assign the one with the highest average acceleration to
                    # the shank of the same leg
                    raised_shank_sensor = self.find_raised_shank(data_aggs, n_sensor)

                    self.position_requester.on_sensor_position_found(
                        Sensor(raised_shank_sensor), shank_rising
                    )
            # If a leg is raised, we check for when it drops, in order to chage to a state
            # where we can detect the left leg
            elif (
                self.mov_state == MovementState.DURING_RIGHT_RAISE
                and not self.is_thigh_raised(data_aggs, angle_aggs, n_sensor)
            ):
                # If after raising the right leg, the right thigh sensor returns
                # to a position closed to vertical, we consider that
                # the left leg will soon be raised. This will cause the next thigh
                # lift to be associated to the left leg. These values are heuristics,
                # but seem to work well.
                self.mov_state = MovementState.BEFORE_LEFT_RAISE

        # The remaining sensor is assigned to the chest. This sensor has not shown patters
        # consistent with thighs raisisng, and its acceleration is lower
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
        Creates two arrays containing  rolling window of past samples' vecto/acc and
        of angles between all sensors.
        The past values of vectors and acceleration are stored in a 3D numpy array
        of dimensions 5 (number of sensors) x 4 (x, y, z, a) x 50 (rolling_agg) and the
        past angle values are stored in a numpy array with dimensions 50 (rolling_agg)
        x 10 (number of pairs of sensors, when we have 5 sensors). The aggregations are
        stored in a numpy array of dimensions 5 x 4 for the vectors/acc, and of size 10
        for past angles.
        :param dict_data: dictionary corresponding each sensor to the orientation
        vector and the acceleration value
        :param data_array: numpy array containing the rolling window of all the sensor data,
        initially an empty array
        :param angles_array: numpy array containinf the rolling window data of past angles
        between all sensors, initially an empty array
        :return: data_array, with data for the last sample appended to it
        :return: angles_array, with data for the last sample appended to it
        """

        data_to_append = np.zeros((N_TRACKERS, 4))

        # The data is extracted from the dict provided by the generator from
        # iterate_sensor_data. It starts by exracting the data from the dict into a
        # 5 x 4 numpy array, filling the array one sensor at a time.
        for sensor, values in dict_data.items():
            single_sensor = np.append(values.vec, values.acc)
            np.reshape(single_sensor, (1, 4))
            data_to_append[sensor.value - 1] = single_sensor

        # Generating the angles between all pairs of angles. angles_to_append is
        # an array of size 10, corresponding to all angles between sensors of the latest
        # sample.
        # Pairs are generated in a sorted manner. So we can always know where a
        # specific pair is in the numpy array.
        pairs = list(itertools.combinations(range(N_TRACKERS), 2))
        angles_to_append = np.zeros((len(pairs)))
        for i, p in enumerate(pairs):
            angles_to_append[i] = vectors_angle(
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

        # Since we are adding dimensions to our numpy arrays (they start out empyu),
        #  the commands for merging arrays with the previous data change, according
        # to the shape of the data.
        # Before data from 50 samples is present, we simply append data to the array
        # but when a dimension gets to 50, we must start removing the older
        # data. This is similar for both angles and sensor data.
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
        :return: list of Sensor variables corresponding to sensors not yet assigned to any
        body segment.
        """
        all_sensors = set(list(Sensor))
        assigned_sensors = set(self.position_requester.sensor_positions.keys())
        return list(all_sensors - assigned_sensors)

    def is_thigh_raised(
        self,
        data_aggregations: np.array,
        angle_aggregations: np.array,
        sensor_number: int,
        method="angles",
    ) -> bool:
        """
        Based on rolling window data, detects wheter a sensor corresponds to a thigh
        beimg raised.
        :param data_aggregations: mean values of a rolling window of past sensor data
        :param angle_aggregations: mean values of a rolling window of past data of all
        angles between sensors
        :param sensor_number: sensor to evaluate (we check if a specific sensor is a
        thigh being raised)
        :param method: method to use to check for rising thigh. Two methods were
        implemented, 'angles' based on angles between all sensors, and 'vector_values'
        which looks at values for specific axis. There is also the option to choose
        "both", so that both conditions implemented for a rising thigh have to be met.
        """

        # If we are in a state before a leg raise, and we detect values of z close
        # to zero (thigh raising close to paralled) and values of x close to 1
        # (thigh close to pointing straight forward), we can consider the leg is
        thigh_raised = True
        if method in ["angles", "both"]:
            # The "angles" method checks for whether all the angles where a sensor is
            # included are higher than 70 degrees. In a leg raise, the vector of the
            # thigh is close to perpendicular with all other vectors, which are
            # relatively close to being parallel between themselves

            pairs = list(itertools.combinations(range(1, N_TRACKERS + 1), 2))
            relevant_angles_indices = [
                i for i, p in enumerate(pairs) if sensor_number in p
            ]

            for angle_index in relevant_angles_indices:
                if angle_aggregations[angle_index] < 70:
                    thigh_raised &= False
        elif method in ["vector_values", "both"]:
            # The "vector_values" method checks for a value of z higher than -0.2,
            # simmultaneous with a value of x higher than 0.6. If both these conditions
            # are met, we consider the thigh is raised. These values are heuristics, but
            # seem to work well.
            if (
                data_aggregations[sensor_number - 1, 2] < -0.2
                or data_aggregations[sensor_number - 1, 0] < 0.6
            ):
                thigh_raised = False
        else:
            raise ValueError(
                "Possible values for 'method' include 'angles', 'vector_values' and \
                    'both'"
            )

        return thigh_raised

    def find_raised_shank(self, data_aggregations: np.array, n_sensor: int) -> int:
        """
        After a thigh raise is detected, this function outputs the sensor of the shank
        being raised. The assumption is that the shank will have the highest acceleration
        mean in the rolling window of all the other sensors.
        :param data_aggregation: numpy array containing the mean of sensors data (rolling)
        :param n_sensor: sensor number of the detected rising thigh. This will be excluded
        when looking at accelerations.
        :return: the sensor number to be assigned to the shank
        """

        # We get the average acceleration of the 5 sensors from the aggregations
        # array, in order to find the sensor with the highest acceleration out of the
        # others.
        accelerations = data_aggregations[:, -1]
        max_acc_sensor = np.argsort(accelerations)[-1] + 1
        if max_acc_sensor == n_sensor:
            max_acc_sensor = np.argsort(accelerations)[-2] + 1
        if Sensor(max_acc_sensor) in self.position_requester.sensor_positions.keys():
            raise ValueError(
                "The segment thought to be the shank has already been assigned. This\
                    is based on an assumption related to the shank being the highest\
                    acceleration segment, which may not hold up."
            )
        return max_acc_sensor


def data_aggregations(data_array: np.array, chosen_agg="mean") -> np.array:
    """
    Calculates aggregations from the numpy array containing the data of the rolling
    window of past samples. Several aggregations are implemented, such as the mean
    and standard deviation, but at the moment only the mean is used. If there are
    less than two data samples worth of data in data_array, the function outputs an
    array of zeros.
    :param data_array: 5 x 4 x N numpy array containing the data from the rolling
    window of past samples.
    :param chosen_agg: string with the chosen aggregation to perform on the rolling
    window data.
    :return: a 5 x 4 matrix with the containing the results of the chosed aggregation
    performed on the rolling window data.
    """
    aggregation = np.zeros((N_TRACKERS, 4))

    if len(np.shape(data_array)) == 3:
        if chosen_agg == "mean":
            for i in range(N_TRACKERS):
                aggregation[i, :] = [np.mean(data_array[i, val, :]) for val in range(4)]
        elif chosen_agg == "std":
            for i in range(N_TRACKERS):
                aggregation[i, :] = [np.mean(data_array[i, val, :]) for val in range(4)]
        else:
            raise ValueError("Only supported aggregations are 'mean' and 'std'")
    return aggregation


def angle_aggregations(angles_array: np.array, chosen_agg="mean") -> np.array:
    """
    Calculates aggregations from the numpy array containing the data of the rolling
    window of past amgles. Several aggregations are implemented, such as the mean
    and standard deviation, but at the moment only the mean is used. If there are
    less than two data samples worth of data in data_array, the function outputs an
    array of zeros.
    :param angles_array: N x 10 numpy array containing the data from the rolling
    window of past samples.
    :param chosen_agg: string with the chosen aggregation to perform on the rolling
    window data.
    :return: a size 10 array with the containing the results of the chosed aggregation
    performed on the rolling window data.
    """
    aggregation = np.zeros((10))

    if len(np.shape(angles_array)) == 2:
        if chosen_agg == "mean":
            aggregation = [np.mean(angles_array[:, val]) for val in range(10)]
        elif chosen_agg == "std":
            aggregation = [np.mean(angles_array[:, val]) for val in range(10)]
        else:
            raise ValueError("Only supported aggregations are 'mean' and 'std'")
    return aggregation


def vectors_angle(v1, v2, deg=True):
    """
    Calculates the angle between two 3D vectors. It starts by calculating the unit
    vector of each vector.
    :param v1: the first vector.
    :param v2: the second vector.
    :param deg: whether the result should be in degrees. If false, output is is
    radians.
    :return; the angle between vectors 'v1' and 'v2':
    """
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    if deg:
        angle *= 180 / math.pi
    return angle
