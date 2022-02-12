from supportlib.sensor_position_finding import SensorPositionFinder, SensorPositionRequester
from supportlib.data_types import Sensor, SensorPosition, SensorData
from typing import Dict
from supportlib.data_types import SensorPosition
import numpy as np


CORRECT_POSITIONS = {
    Sensor.SENSOR_1: SensorPosition.RIGHT_SHANK,
    Sensor.SENSOR_2: SensorPosition.CHEST,
    Sensor.SENSOR_3: SensorPosition.LEFT_THIGH,
    Sensor.SENSOR_4: SensorPosition.LEFT_SHANK,
    Sensor.SENSOR_5: SensorPosition.RIGHT_THIGH
}

class SolutionPositionFinder(SensorPositionFinder):

    current_sample = 0

    def __init__(self, position_requester: SensorPositionRequester):
        """
        Constructor.
        :param position_requester: the PositionRequester to be called when
        sensor positions are identified
        """
        self.position_requester = position_requester

    def on_new_sensor_sample(self,
                             sensor_data: Dict[Sensor, SensorData]) -> None:
        """
        Callback called each time a new sample is received from the sensors
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """

        # TODO: implement a better algorithm
        self.initial_naive_solution(sensor_data)




    def initial_naive_solution(self,
                             sensor_data: Dict[Sensor, SensorData], verbose=False):
        """
        Naive function to assign sensors to body parts (Sensor to SensorPosition)
        It is heavily informed by the actual data we are receiving, and is not robust
        whatsoever. It assigns the first sensor to have a positive value of z as the
        right thigh, and if a sensor presents a positive value of z after the sample
        number 300, it is considered as being in the left thigh. The same logic is 
        applied for shanks, but with a value of x of -0.5. The last sensor to be 
        assigned to any body part is considered to be in the Chest.
        This function takes into account the number of the sample, which is increased
        in each run.
        :param sensor_data: a dict containing sensor data as values and the
        corresponding sensors as keys
        """

        self.current_sample += 1

        for i in range(1, 6):
            if not Sensor(i) in self.position_requester.sensor_positions.keys():
                if (sensor_data[Sensor(i)].vec[2] > 0) :
                    # this sensor is on a thigh
                    if verbose: print(f"\n\n{self.current_sample} - Detecting Right thigh by being having the first z value to reach 0")
                    self.position_requester.on_sensor_position_found(Sensor(i), SensorPosition.RIGHT_THIGH)
                    if self.current_sample > 300:
                        if verbose: print(f"\n\n{self.current_sample} - Detecting Left Thigh by having z rech zero after sample 1500")
                        self.position_requester.on_sensor_position_found(Sensor(i), SensorPosition.LEFT_THIGH)
            
            if not Sensor(i) in self.position_requester.sensor_positions.keys():
                if sensor_data[Sensor(i)].vec[0] < -0.5:
                    # this sensor is on a shank
                    if verbose: print(f"\n\n{self.current_sample} - Detecting Right Shank by being having the first x value to reach -0.5")
                    self.position_requester.on_sensor_position_found(Sensor(i), SensorPosition.RIGHT_SHANK)
                    if self.current_sample > 300:
                        if verbose: print(f"\n\n{self.current_sample} - Detecting Left Shank by having x rech -0.5 after sample 1500")
                        self.position_requester.on_sensor_position_found(Sensor(i), SensorPosition.LEFT_SHANK)
        
        if len(self.position_requester.sensor_positions.keys()) == 4:
            if verbose: print(f"\n\n{self.current_sample} - Detecting chest by being the last one remaining")
            remaining_sensor = self.unassigned_sensors()[0]
            self.position_requester.on_sensor_position_found(remaining_sensor, SensorPosition.CHEST)
            self.position_requester.on_finish()
    
    def unassigned_sensors(self):
        """
        Ouputs a list containing the sensors not yet assigned to any body part
        :param sensor_positions: dict corresponding each of the Sensor variables to its
        SensorPosition
        :return: list of usassigned Sensor variables
        """
        # TODO: find a better way to get a list of all sensors
        all_sensors = [Sensor.SENSOR_1, Sensor.SENSOR_2, Sensor.SENSOR_3, Sensor.SENSOR_4, Sensor.SENSOR_5]
        return list(set(CORRECT_POSITIONS.keys()) - set(self.position_requester.sensor_positions.keys()))

# %%


from supportlib.file_reading import iterate_sensor_data
import numpy as np

N = 50

def position_acceleration_solution(dict_data, data_array):
    """
    A more robust solution. When it detects a sensor with values of z close to 0
    and values of x close to 1, it assigns the sensor to the right/left thigh, 
    depending on whether the movement has been detected before. At the same time,
    we expect the shank of the same leg to be rising with the thigh. However, 
    there is the chance the shank stays in a mostly upright position, with no
    visible change on x, y or z. However, the shank sensor would have the highest 
    acceleration value of all the other sensors, so it can be identified as well.
    It keeps a rolling window of 100 counts
    :param sensor_data: a dict containing sensor data as values and the
    corresponding sensors as keys
    :param past_data: numpy array of dimensions 5 x N x 4, holding past values for
    all the data.
    """

    # move here what I'm putting inside the loop
    pass
    

def update_past_data(dict_data, data_array):
    """
    Converts the dict provided by the generator into a numpy array of 5 x 4 x 1,
    to be appended to the past_data numpy array
    :param dict_data: dictionary corresponding each sensor to the orientation
    vector and the acceleration value
    :param arr: the array to apend to, which contains the past data
    """
    to_append = np.zeros((5,4))

    for sensor, values in dict_data.items():
        single_sensor = np.append(values.vec, values.acc)
        np.reshape(single_sensor, (1,4))
        to_append[sensor.value-1] = single_sensor

    if data_array.size==0:
        data_array = to_append
    elif np.shape(data_array) == (5,4):
        data_array = np.stack((data_array, to_append), axis=2)
    elif np.shape(data_array)[2] < N:
        data_array = np.append(data_array, np.atleast_3d(to_append), axis=2)
    elif np.shape(data_array)[2] == N:
        data_array = np.append(data_array, np.atleast_3d(to_append), axis=2)
        data_array = data_array[:,:,1:]
    return data_array
    

def data_aggregations(data_array):
    """
    Some data aggregations are necessary to implement our algorith:
     - Average x, y, z, acc values
     - Rows are each of the sensors
     - Columns are the values of x, y, z, a
    """
    if len(np.shape(data_array)) == 3:
        averages = np.zeros((5,4))
        for i in range(5):
            averages[i,:] = [np.mean(data_array[0,0,:]),
                        np.mean(data_array[0,1,:]),
                        np.mean(data_array[0,2,:]),
                        np.mean(data_array[0,3,:])]
    else:
        averages = np.zeros((5,4))
    return averages

past_data = np.array([])
for data_dict in iterate_sensor_data("resources/sensor_data.csv",5):
    past_data = update_past_data(data_dict, past_data)
    aggs = data_aggregations(past_data)
