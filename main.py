"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.
"""
from pandas import read_csv, DataFrame, to_datetime, concat
from numpy import vectorize, atan2, pi, arange
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

def load_trials_data(file_path: str) -> DataFrame:
    """
    Load the lobster movement trials dataset from a CSV file.
    """
    df: DataFrame = read_csv(file_path)
    date_col = "date"
    df[date_col] = to_datetime(df[date_col])
    return df.set_index("trial")

def clock_string_to_seconds(clock_str: str) -> int:
    """
    Convert a clock string in the format 'HH:MM:SS' to total seconds.
    """
    hours, minutes, seconds = map(int, clock_str.split(':'))
    return hours * 3600 + minutes * 60 + seconds

def load_interval_data(file_path: str) -> DataFrame:
    """
    Load the lobster movement intervals dataset from a CSV file.
    """
    df: DataFrame = read_csv(file_path)
    start = "start"
    stop = "stop"
    convert = vectorize(clock_string_to_seconds)
    df[start] = convert(df[start])
    df[stop] = convert(df[stop])
    df["duration"] = df[stop] - df[start]
    return df.set_index("trial")

def load_events_data(file_path: str) -> DataFrame:
    """
    Load the lobster movement events dataset from a CSV file.
    """
    df: DataFrame = read_csv(file_path)
    time = "time"
    field = "field"
    entangled = "entangled"
    convert = vectorize(clock_string_to_seconds)
    df[time] = convert(df[time])
    df[field] = df[field].replace("F", "").astype(bool)
    df[entangled] = df[entangled].fillna(False).astype(bool)
    return df.set_index("trial")

def join_intervals_and_events(intervals: DataFrame, events: DataFrame) -> DataFrame:
    """
    Join intervals and events dataframes on the trial index.
    Where the trial index matches, and an event falls within the interval, 
    the time of the event in elapsed seconds is added 
    to the intervals dataframe.
    """
    joined = intervals.join(events, lsuffix="_intervals", rsuffix="_events")
    event_in_interval = joined["time"].between(joined["start"], joined["stop"])
    filtered = joined.loc[event_in_interval]
    filtered["valid"] = filtered["stop"] - filtered["time"]
    return filtered

def load_single_trial(file_path: str) -> DataFrame:
    """
    Load a single lobster movement trial dataset from a CSV file.
    """
    df: DataFrame = read_csv(file_path)
    pos = "angular_position"
    time = "elapsed_time"
    convert = vectorize(clock_string_to_seconds)
    df[time] = convert(df[time])
    df.set_index(time, inplace=True)
    df.sort_index(inplace=True)

    x = (df["x_head"] + df["x_tail"]) / 2
    y = (df["y_head"] + df["y_tail"]) / 2
    df[pos] = atan2(x, y) * 180 / pi
    df["velocity"] = (x.diff()**2 + y.diff()**2)**0.5

    dx = df["x_head"] - df["x_tail"]
    dy = df["y_head"] - df["y_tail"]
    df["heading"] = atan2(dx, dy) * 180 / pi
    # d1 = "heading_first_derivative"
    # df[d1] = df["heading"].diff()
    # df[d1] = df[d1].where(df[d1] > -180, df[d1] + 360)
    # df[d1] = df[d1].where(df[d1] < 180, df[d1] - 360)
    return df

def join_elapsed_time_since_event(trial_data: DataFrame, trial: int, events: DataFrame) -> DataFrame:
    """
    Calculate the time since the last event in each interval
    """
    mask = events.index == trial
    selected_trial = events[mask]
    rows = trial_data.shape[0]
    trial_data["coil_on"] = False
    trial_data["since_event"] = trial_data.index.to_series()
    for row in selected_trial.itertuples():
        loc = trial_data.index.get_loc(row.time)
        trial_data["coil_on"][loc:rows] = row.field
        trial_data["since_event"][loc:rows] = arange(rows - loc)

def load_control_data(file_path: str) -> DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.
    """
    df: DataFrame = read_csv(file_path)
    time = "elapsed_time"
    convert = vectorize(clock_string_to_seconds)
    df[time] = convert(df[time])
    df.set_index("trial", inplace=True)
    return df

def plot_control_position(df: DataFrame) -> None:
    N = 12
    position = df["sector"].value_counts(normalize=True)
    radians = position.index / 360 * 2 * np.pi
    width = 2 * np.pi / N
    ax = plt.subplot(projection='polar')
    ax.bar(radians, position.values, width=width, bottom=0.0, color="gray")
    plt.savefig("figures/control_position_polar_plot.png")

def plot_control_heading(df: DataFrame) -> None:
    N = 12
    heading = df["heading"].value_counts(normalize=True)
    radians = heading.index / 360 * 2 * np.pi
    width = 2 * np.pi / N
    ax = plt.subplot(projection='polar')
    ax.bar(radians, heading.values, width=width, bottom=0.0, color="gray")
    plt.savefig("figures/control_heading_polar_plot.png")

def calc_histogram(df, bins: int, ax: Axes, color: str, alpha: float, label: str) -> None:
    area, edges = np.histogram(df+180, bins=bins, range=(0, 360), density=True)
    radians = (edges[:-1] / 360 + (360 / bins / 2)) * 2 * np.pi
    width = 2 * np.pi / bins * 0.5
    ax.bar(radians, area, width=width, bottom=0.0, color="none", edgecolor=color, alpha=alpha, label=label)

def plot_trial_position(
    df: DataFrame,
    bins: int = 12,
    alpha: float = 0.75
) -> None:
    """
    Plot the angular position distribution of a lobster trial. Split data
    by whether the coil was on or off.
    """
    series = df["angular_position"]
    mask = df["compass"] > 0
    rows = df.shape[0]
    on = series.loc[mask]
    off = series.loc[~mask]
    ax = plt.subplot(projection='polar')
    calc_histogram(on, bins, ax, "red", alpha, "coil on")
    calc_histogram(off, bins, ax, "blue", alpha, "coil off")
    ax.grid(False)
    ax.set_title(f"Angular position distribution (N={rows})")
    ax.set_yticklabels([])
    plt.legend(loc="best")
    plt.savefig("figures/trial_position_polar_plot.png")
    plt.close()

def plot_trial_heading(
    df: DataFrame,
    bins: int = 12,
    alpha: float = 0.75
) -> None:
    """
    Plot the angular position distribution of a lobster trial. Split data
    by whether the coil was on or off.
    """
    series = df["heading"]
    mask = df["compass"] > 0
    rows = df.shape[0]
    on = series.loc[mask]
    off = series.loc[~mask]
    ax = plt.subplot(projection='polar')
    calc_histogram(on, bins, ax, "red", alpha, "coil on")
    calc_histogram(off, bins, ax, "blue", alpha, "coil off")
    ax.grid(False)
    ax.set_title(f"Heading distribution (N={rows})")
    ax.set_yticklabels([])
    plt.legend(loc="best")
    plt.savefig("figures/trial_heading_polar_plot.png")
    plt.close()

if __name__ == "__main__":
    # trials = load_trials_data("data/trials.csv")
    # print(trials.dtypes)
    # print(trials.head())

    # Data are already annotated with time since polarity switch,
    # Can't join events and intervals because some time offset information
    # was lost.
    # intervals = load_interval_data("data/intervals.csv")
    # timing = intervals["duration"].groupby(["trial"]).sum()
    # print(timing)
    # print(intervals.dtypes)
    dfs = []
    for trial_id in [16, 17, 18, 19, 20, 21, 22, 23, 24]:
        trial = load_single_trial(f"data/trials/{trial_id}.csv")
        trial["trial"] = trial_id
        print(trial.dtypes)
        print(trial.head())
        dfs.append(trial)
    all_trials = concat(dfs)
    plot_trial_position(all_trials, bins=12)
    plot_trial_heading(all_trials, bins=12)

    # control = load_control_data("data/control.csv")
    # print(control.dtypes)
    # print(control.head())
    # plot_control_heading(control)
    # plot_control_position(control)

