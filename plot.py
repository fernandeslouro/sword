# %%
import pandas as pd
import matplotlib.pyplot as plt
import statistics
# %%
mov = pd.read_csv("resources/sensor_data.csv")
plt.style.use('fivethirtyeight')
# %%
ALPHA_STD=0.2
for s in range(1,6):
    plt.figure(figsize=(10,10))
    plt.title(f"Sensor {s}")
    sensor_data = mov[mov.sensor==s]
    plt.plot(sensor_data.vec_x, color="red")
    plt.axhline(y=statistics.stdev(sensor_data.vec_x), color="red", alpha=ALPHA_STD)
    plt.plot(sensor_data.vec_y, color="green")
    plt.axhline(y=statistics.stdev(sensor_data.vec_y), color="green", alpha=ALPHA_STD)
    plt.plot(sensor_data.vec_z, color="blue")
    plt.axhline(y=statistics.stdev(sensor_data.vec_z), color="blue", alpha=ALPHA_STD)
    plt.plot(sensor_data.acc, color="yellow")
    plt.axhline(y=statistics.stdev(sensor_data.acc), color="yellow", alpha=ALPHA_STD)
# %%
statistics.stdev(sensor_data.vec_x)