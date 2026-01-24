"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.
"""

from pandas import read_csv
import pandas as pd
from numpy import vectorize, atan2, pi, arange, histogram, append
from matplotlib.pyplot import subplots
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes
from click import option, group, argument
from enum import Enum
from pathlib import Path
from polars import col, Struct, Float64, struct, Int64
import polars as pl
from roifile import roiread, ImagejRoi
from typing import List

FIGURES = Path(__file__).parent / "figures"

class Commands(Enum):
    """
    Consistent command names for the CLI.
    """
    PLOT = "plot"
    DESCRIBE = "describe"
    LOBSTER = "lobster"

class SourceData(Enum):
    """
    Docstring for Metadata
    """
    CONTROL = "data/control.csv"
    TRIALS_DIR = "data/trials/"
    TRIALS = "data/trials.csv"
    INTERVALS = "data/intervals.csv"
    EVENTS = "data/events.csv"
    LOBSTERS = "data/lobsters.csv"
    GEOMAGNETIC_FIELD = "data/magnetic-field.csv"
    CONTROL_EVENTS = "data/control-events.csv"

class Palette(Enum):
    """
    Color palette for plotting.
    """
    ON = "red"
    OFF = "blue"
    CONTROL = "black"

@group()
def cli():
    """
    Commandline interface for better UX when working with lobster
    movement dataset.
    """


@group(name=Commands.PLOT.value)
def cli_plot():
    """
    Commands for plotting experiment data.
    """


cli.add_command(cli_plot)


@group(name=Commands.DESCRIBE.value)
def cli_describe():
    """
    Commands for describing experiment data.
    """


cli.add_command(cli_describe)


@group(name=Commands.LOBSTER.value)
def cli_describe_lobster():
    """
    Commands for describing lobster movement data.
    """


cli_describe.add_command(cli_describe_lobster)


def load_trials_data_polars(file_path: str) -> pl.DataFrame:
    """
    Load the lobster movement trial polar dataset from a CSV file.
    """
    df: pl.DataFrame = pl.read_csv(file_path)
    date_col = "date"
    df = df.with_columns(col(date_col).str.to_datetime())
    return df


def clock_string_to_seconds(clock_str: str) -> int:
    """
    Convert a clock string in the format 'HH:MM:SS' to total seconds.
    """
    hours, minutes, seconds = map(int, clock_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


def load_interval_data(file_path: str) -> pd.DataFrame:
    """
    Load the lobster movement intervals dataset from a CSV file.
    """
    df = read_csv(file_path)
    start = "start"
    stop = "stop"
    convert = vectorize(clock_string_to_seconds)
    df[start] = convert(df[start])
    df[stop] = convert(df[stop])
    df["duration"] = df[stop] - df[start]
    return df.set_index("trial")




def join_intervals_and_events(intervals: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
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



def load_single_trial_polars(
    file_path: str, time_column: str = "elapsed_time"
) -> pl.DataFrame:
    """
    Load a single lobster movement trial dataset from a CSV file.
    """
    def calc_heading(row: dict) -> dict | None:
        """
        Calculate the heading and position from head and tail coordinates.
        """
        if any(x is None for x in row.values()):
            return None
        dx = row["x_head"] - row["x_tail"]
        dy = row["y_head"] - row["y_tail"]
        x = (row["x_head"] + row["x_tail"]) / 2
        y = (row["y_head"] + row["y_tail"]) / 2
        return {"position": atan2(x, y), "heading": atan2(dx, dy)}
    alias = "computed"
    return (
        pl.read_csv(file_path)
        .with_columns(
            col(time_column)
            .map_elements(clock_string_to_seconds, return_dtype=Int64)
            .alias(time_column),
            struct(["x_head", "x_tail", "y_head", "y_tail"])
            .map_elements(
                calc_heading,
                return_dtype=Struct({"position": Float64, "heading": Float64}),
            )
            .alias(alias),
        )
        .unnest(alias)
    )

def load_control_data(file_path: str, time_column: str = "elapsed_time") -> pl.DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.

    Convert time to elapsed seconds, and position and heading to radians.
    """
    return (
        pl.read_csv(file_path)
        .with_columns(
            col(time_column)
            .map_elements(clock_string_to_seconds, return_dtype=Int64)
            .alias(time_column),
            (col("sector") / 360 * 2 * pi).alias("position"),
            (col("heading") / 360 * 2 * pi).alias("heading")
        )
    )


def join_elapsed_time_since_event(
    trial_data: pd.DataFrame, trial: int, events: pd.DataFrame
) -> pd.DataFrame:
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



@cli_describe_lobster.command("trials")
def cli_describe_lobster_trials():
    """
    Describe lobster position data.
    """
    trials = load_trials_data_polars("data/trials.csv")
    print(trials.describe())

def load_events_data(file_path: str) -> pl.DataFrame:
    """
    Load the lobster movement events dataset from a CSV file.
    """
    df = pl.read_csv(file_path)
    time = "time"
    field = "field"
    entangled = "entangled"
    return df.with_columns(
        col(time).map_elements(clock_string_to_seconds, return_dtype=Int64).alias(time),
        (col(field) == "T").alias(field),
        (col(entangled).fill_null("F") == "T").alias(entangled)
    )


@cli_describe_lobster.command("roi")
def cli_describe_lobster_roi():
    """
    Describe lobster position data.
    """
    trial = 22
    data = roiread(SourceData.TRIALS_DIR.value + f"0{trial}_ROI.zip")
    if isinstance(data, list):
        print((data[0].coordinates() - 200.0) / 200.0)
    
    events = load_events_data(SourceData.EVENTS.value).filter(col("trial") == trial)
    print(events)


@cli_plot.command("lobster")
@argument("parameter", default="heading")
@option("--bins", default=12, help="Number of bins for the histogram.")
def cli_plot_lobster(parameter: str, bins: int = 12):
    """
    Plot the position or heading distribution of lobster trials. 
    This doesn't use derivatives, so we load and concatenate all the flat files. 
    The records are then partitioned by whether the coil was on or off.
    """
    control = load_control_data(SourceData.CONTROL.value)
    series = control[parameter]
    area, edges = histogram(series, bins=bins, range=(0, 2*pi), density=True)
    area = append(area, area[0])  # close the circle
    control_label = "Control (N=" + str(series.count()) + ")"
    trials = load_single_trial_polars(SourceData.TRIALS_DIR.value + "*.csv")
    fig, ax = subplots(subplot_kw={"projection": "polar"})
    ax.plot(edges, area, color=Palette.CONTROL.value, label=control_label)
    state = col("compass")
    for mask, color, label in [
        (state <= 0, Palette.OFF.value, "Off"),
        (state > 0, Palette.ON.value, "On"),
    ]:
        series = trials.filter(mask)[parameter]
        area, edges = histogram(series, bins=bins, range=(-pi, pi), density=True)
        area = append(area, area[0])
        ax.plot(
            edges,
            area,
            color=color,
            label=label + f" (N={series.count()})",
        )
    ax.grid(False)
    ax.set_title(f"Lobster {parameter} distribution by coil state")
    ax.set_yticklabels([])
    fig.legend(loc="lower right", frameon=False)
    fig.savefig(FIGURES / f"{Commands.LOBSTER.value}_{parameter}.png")
    print("OK")


if __name__ == "__main__":
    cli()

    # Data are already annotated with time since polarity switch,
    # Can't join events and intervals because some time offset information
    # was lost.
    # intervals = load_interval_data("data/intervals.csv")
    # timing = intervals["duration"].groupby(["trial"]).sum()
    # print(timing)
    # print(intervals.dtypes)
