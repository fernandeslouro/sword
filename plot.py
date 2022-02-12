# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
plt.style.use('fivethirtyeight')
# %%
mov = pd.read_csv("resources/sensor_data.csv")

def unit_vector(vector):
    """ 
    Returns the unit vector of the vector.  
    """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ 
    Returns the angle in radians between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))




# %%

def diff(x):
    return x.iloc[-1] - x.iloc[0]

ALPHA_STD=0.1
ALPHA_VALUES = 0.6

for s in range(1,6):
    plt.figure(figsize=(10,10))
    plt.title(f"Sensor {s}")
    sensor_data = mov[mov.sensor==s]
    plt.plot(sensor_data.vec_x, color="red", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_x), color="red", alpha=ALPHA_STD)
    #plt.plot(sensor_data.vec_x.rolling(20).apply(diff), color="red")

    plt.plot(sensor_data.vec_y, color="green", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_y), color="green", alpha=ALPHA_STD)
    #plt.plot(sensor_data.vec_y.rolling(20).apply(diff), color="green")

    plt.plot(sensor_data.vec_z, color="blue", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_z), color="blue", alpha=ALPHA_STD)
    #plt.plot(sensor_data.vec_z.rolling(20).apply(diff), color="blue")


    plt.plot(sensor_data.acc, color="yellow", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.acc), color="yellow", alpha=ALPHA_STD)
# %%
statistics.stdev(sensor_data.vec_x)

#%%

mov
# %%
