from csv import DictReader

import numpy as np

from .data_types import Sensor, SensorData


class _FileTags:
    SAMPLE_INDEX = "sample_index"

    SENSOR = "sensor"

    VEC_X = "vec_x"
    VEC_Y = "vec_y"
    VEC_Z = "vec_z"

    ACC = "acc"
    
    HEADER = [SAMPLE_INDEX, SENSOR, VEC_X, VEC_Y, VEC_Z, ACC]


def iterate_sensor_data(file_path, n_trackers):
    with open(file_path, "r") as file:
        reader = DictReader(file)

        if _FileTags.HEADER != reader.fieldnames:
            raise ValueError("Sensor data file has wrong format. "
                             f"Expected header: {_FileTags.HEADER}")

        data_dict = dict()
        for row in reader:
            sensor = Sensor(int(row[_FileTags.SENSOR]))

            vec = np.array([
                float(row[_FileTags.VEC_X]),
                float(row[_FileTags.VEC_Y]),
                float(row[_FileTags.VEC_Z])
            ])

            acc = float(row[_FileTags.ACC])

            data_dict[sensor] = SensorData(vec, acc)

            if len(data_dict) >= n_trackers:
                yield data_dict
                data_dict = dict()
