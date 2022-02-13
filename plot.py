# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import statistics
import itertools

plt.style.use("fivethirtyeight")
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


dict_dfs = {}
for i in range(1, 6):
    dict_dfs[i] = mov[mov.sensor == i].add_suffix(f"_{i}")
    dict_dfs[i] = dict_dfs[i].drop([f"sensor_{i}"], axis=1)
    dict_dfs[i] = dict_dfs[i].rename(columns={f"sample_index_{i}": "sample_index"})

data_frames = [dict_dfs[i] for i in range(1, 6)]

df_merged = reduce(
    lambda left, right: pd.merge(left, right, on=["sample_index"], how="outer"),
    data_frames,
)

# %%
pairs = list(itertools.combinations(range(1, 6), 2))
for p in pairs:
    df_merged[f"angle_{p[0]}_{p[1]}"] = df_merged.apply(
        lambda row: angle_between(
            [
                row[f"vec_x_{p[0]}"],
                row[f"vec_y_{p[0]}"],
                row[f"vec_z_{p[0]}"],
            ],
            [
                row[f"vec_x_{p[1]}"],
                row[f"vec_y_{p[1]}"],
                row[f"vec_z_{p[1]}"],
            ],
        ),
        axis=1,
    )


for p in pairs:
    plt.figure(figsize=(10, 10))
    plt.plot(df_merged[f"angle_{p[0]}_{p[1]}"] * 57.2958)
    plt.title(f"Sensor {p[0]} - Sensor {p[1]}")
    plt.savefig(f"plots/s{p[0]}_s{p[1]}.png")
# %%


def diff(x):
    return x.iloc[-1] - x.iloc[0]


ALPHA_STD = 0.1
ALPHA_VALUES = 0.6

for s in range(1, 6):
    plt.figure(figsize=(10, 10))
    plt.title(f"Sensor {s}")
    sensor_data = mov[mov.sensor == s]
    plt.plot(sensor_data.vec_x, color="red", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_x), color="red", alpha=ALPHA_STD)
    # plt.plot(sensor_data.vec_x.rolling(50).apply(diff), color="red")
    plt.plot(sensor_data.vec_x.rolling(50).mean(), color="red")

    plt.plot(sensor_data.vec_y, color="green", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_y), color="green", alpha=ALPHA_STD)
    plt.plot(sensor_data.vec_y.rolling(50).mean(), color="green")

    plt.plot(sensor_data.vec_z, color="blue", alpha=ALPHA_VALUES)
    plt.axhline(y=statistics.stdev(sensor_data.vec_z), color="blue", alpha=ALPHA_STD)
    plt.plot(sensor_data.vec_z.rolling(50).mean(), color="blue")

    plt.plot(sensor_data.acc, color="pink", alpha=ALPHA_VALUES)
    plt.plot(sensor_data.acc.rolling(50).mean(), color="pink")
    plt.axhline(y=statistics.stdev(sensor_data.acc), color="pink", alpha=ALPHA_STD)
# %%
statistics.stdev(sensor_data.vec_x)

#%%

mov
# %%
