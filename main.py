from supportlib.sensor_position_finding import SensorPositionRequester
from supportlib.data_types import Sensor, SensorPosition
from supportlib.file_reading import iterate_sensor_data
from solution import SolutionPositionFinder

MOVEMENT_DATA_FILE_PATH = "resources/sensor_data.csv"
N_TRACKERS = 5

CORRECT_POSITIONS = {
    Sensor.SENSOR_1: SensorPosition.RIGHT_SHANK,
    Sensor.SENSOR_2: SensorPosition.CHEST,
    Sensor.SENSOR_3: SensorPosition.LEFT_THIGH,
    Sensor.SENSOR_4: SensorPosition.LEFT_SHANK,
    Sensor.SENSOR_5: SensorPosition.RIGHT_THIGH,
}


class SensorPositionFinderTester(SensorPositionRequester):
    def __init__(self):
        self.position_finder = SolutionPositionFinder(self)
        self.sensor_positions = dict()
        self.finished = False

    def run(self):
        for data_dict in iterate_sensor_data(MOVEMENT_DATA_FILE_PATH, N_TRACKERS):
            self.position_finder.on_new_sensor_sample(data_dict)

            if self.finished:
                break

        if not self.finished:
            print("EOF reached before on_finish was called")

    def on_sensor_position_found(
        self, sensor: Sensor, position: SensorPosition
    ) -> None:
        print(f"{sensor}'s position identified as {position}")
        self.sensor_positions[sensor] = position

    def on_finish(self) -> None:
        self.finished = True

        if CORRECT_POSITIONS == self.sensor_positions:
            print("All sensor positions correctly identified!")
        else:
            print("Sensor positions are not correct!")

            print("Expected:")
            for key in CORRECT_POSITIONS.keys():
                print(f"{key}:\t{CORRECT_POSITIONS[key]}")

            print("Actual:")
            for key in CORRECT_POSITIONS.keys():
                print(f"{key}:\t{self.sensor_positions.get(key)}")


if __name__ == "__main__":
    SensorPositionFinderTester().run()
