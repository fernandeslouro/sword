# %%
from supportlib.sensor_position_finding import SensorPositionFinder, SensorPositionRequester
from supportlib.data_types import Sensor, SensorPosition, SensorData
from typing import Dict

CORRECT_POSITIONS = {
    Sensor.SENSOR_1: SensorPosition.RIGHT_SHANK,
    Sensor.SENSOR_2: SensorPosition.CHEST,
    Sensor.SENSOR_3: SensorPosition.LEFT_THIGH,
    Sensor.SENSOR_4: SensorPosition.LEFT_SHANK,
    Sensor.SENSOR_5: SensorPosition.RIGHT_THIGH
}




# %%

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

        self.initial_naive_solution(sensor_data)




    def initial_naive_solution(self,
                             sensor_data: Dict[Sensor, SensorData], verbose=False):
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
        return list(set(CORRECT_POSITIONS.keys()) - set(self.position_requester.sensor_positions.keys()))





#%%
