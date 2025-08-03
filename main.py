"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.
"""
from pandas import read_csv, DataFrame, to_datetime
from numpy import vectorize, atan2, pi, arange

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
    df["elapsed"] = df[stop] - df[start]
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
    d1 = "heading_first_derivative"
    df[d1] = df["heading"].diff()
    df[d1] = df[d1].where(df[d1] > -180, df[d1] + 360)
    df[d1] = df[d1].where(df[d1] < 180, df[d1] - 360)

    d2 = "heading_second_derivative"
    df[d2] = df[d1].diff()
    df[d2] = df[d2].where(df[d2] > -180, df[d2] + 360)
    df[d2] = df[d2].where(df[d2] < 180, df[d2] - 360)

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


if __name__ == "__main__":
    # trials = load_trials_data("data/trials.csv")
    # print(trials.dtypes)
    # print(trials.head())

    # intervals = load_interval_data("data/intervals.csv")
    # print(intervals.dtypes)
    # print(intervals.head())

    events = load_events_data("data/events.csv")
    # print(events.dtypes)
    # print(events.head())

    # joined = join_intervals_and_events(intervals, events)
    # print(joined.dtypes)
    # print(joined.head())

    example_trial = load_single_trial("data/trial-16.csv")
    join_elapsed_time_since_event(example_trial, 16, events)
    print(example_trial.dtypes)
    print(example_trial.head())

