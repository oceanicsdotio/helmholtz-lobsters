"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.

- `plot lobster heading`: OK
- `plot lobster position`: OK
- `describe lobster trials`: WIP
- `describe lobster roi`: WIP

"""

from enum import Enum
from pathlib import Path
from pandas import read_csv
import pandas as pd
from numpy import vectorize, atan2, pi, arange, histogram, append, array
from matplotlib.pyplot import subplots
from click import option, group, argument, Choice
from polars import col, Struct, Float64, struct, Int64, DataFrame, Config, all
import polars as pl
from zipfile import ZipFile
from roifile import roiread, ImagejRoi

FIGURES = Path(__file__).parent / "figures"
Config.set_tbl_rows(-1)


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


class Dim(Enum):
    """
    Dimensions used in lobster movement dataset.
    """

    X_HEAD = "x_head"  # normalized position of the head
    Y_HEAD = "y_head"
    X_TAIL = "x_tail"  # normalized position of the tail
    Y_TAIL = "y_tail"
    DURATION = "duration"
    TRIAL_TYPE = "trial_type"
    LOBSTER_ID = "lobster_id"
    COMPASS = "compass"
    TRIAL = "trial"
    FIELD = "field"
    ENTANGLED = "entangled"  # whether lobster is entangled
    NOTES = "notes"  # additional notes or comments
    ACCLIMATION = "acclimation"  # minutes in tank before trial
    ELAPSED_TIME = "elapsed_time"
    SINCE_EVENT = "since_event"
    DATA_PERIOD = "data_period"
    DATE = "date"
    EVENT = "event"
    TIME = "time"
    SECTOR = "sector"
    HEADING = "heading" # computed / observed
    POSITION = "position" # computed


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


def join_intervals_and_events(
    intervals: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
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

def calc_heading(row: dict) -> dict | None:
    """
    Calculate the heading and position from head and tail coordinates.
    """
    if any(x is None for x in row.values()):
        return None
    dx = row[Dim.X_HEAD.value] - row[Dim.X_TAIL.value]
    dy = row[Dim.Y_HEAD.value] - row[Dim.Y_TAIL.value]
    x = (row[Dim.X_HEAD.value] + row[Dim.X_TAIL.value]) / 2
    y = (row[Dim.Y_HEAD.value] + row[Dim.Y_TAIL.value]) / 2
    return {Dim.POSITION.value: atan2(x, y), Dim.HEADING.value: atan2(dx, dy)}


def load_single_trial_polars(
    file_path: str, time_column: str = "elapsed_time"
) -> DataFrame:
    """
    Load either a single lobster movement trial dataset from a CSV file, or
    use a pattern to load and concatenate many files.
    """
    alias = "computed"  # temporary column during unnest
    return (
        pl.read_csv(file_path)
        .with_columns(
            col(time_column)
            .map_elements(clock_string_to_seconds, return_dtype=Int64)
            .alias(time_column),
            struct([Dim.X_HEAD.value, Dim.X_TAIL.value, Dim.Y_HEAD.value, Dim.Y_TAIL.value])
            .map_elements(
                calc_heading,
                return_dtype=Struct({Dim.POSITION.value: Float64, Dim.HEADING.value: Float64}),
            )
            .alias(alias),
        )
        .unnest(alias)
    )


def load_control_data(file_path: str, time_column: str = "elapsed_time") -> DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.

    Convert time to elapsed seconds, and position and heading to radians.
    """
    return pl.read_csv(file_path).with_columns(
        col(time_column)
        .map_elements(clock_string_to_seconds, return_dtype=Int64)
        .alias(time_column),
        (col("sector") / 360 * 2 * pi).alias("position"),
        (col("heading") / 360 * 2 * pi).alias("heading"),
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
    Get most recent molt date for the lobster in each trial.

    Then subtract that from the trial date to get time since molt.
    """
    lobsters = (
        pl.read_csv(SourceData.LOBSTERS.value)
        .with_columns(col(Dim.DATE.value).str.to_datetime().dt.date().alias(Dim.DATE.value))
        .filter(col(Dim.EVENT.value) == "molt")
    )
    date_right = Dim.DATE.value + "_right"  # temporary column during join
    trials = (
        pl.read_csv(SourceData.TRIALS.value)
        .with_columns(col(Dim.DATE.value).str.to_datetime().dt.date().alias(Dim.DATE.value))
        .join(lobsters, on=Dim.LOBSTER_ID.value, how="left")
        .with_columns(
            (col(Dim.DATE.value) - col(date_right)).alias("days_since_molt"),
        )
        .drop([date_right])
        .group_by((Dim.TRIAL.value, Dim.LOBSTER_ID.value))
        .agg(
            all().exclude(("days_since_molt", Dim.DATE.value)).unique().item(),
            col("days_since_molt").max(),
        )
        .sort(Dim.TRIAL.value)
    )
    print(trials)


def load_events_data(file_path: str) -> DataFrame:
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
        (col(entangled).fill_null("F") == "T").alias(entangled),
    )


@cli_describe_lobster.command("roi")
def cli_describe_lobster_roi():
    """
    Describe lobster position data.
    """
    trial = 24
    filename = SourceData.TRIALS_DIR.value + f"0{trial}_ROI.zip"
    rois = roiread(filename)
    with ZipFile(filename, "r") as files:
        data = zip(files.namelist(), rois)

    def parse_row(item: tuple[str, ImagejRoi]) -> list[float]:
        file, roi = item
        frame = int(file.split("-")[0])
        coords = (roi.coordinates() - 200.0) / 200.0
        return [frame, *coords.flatten()]
    
    frames = array([parse_row(row) for row in data]).T
    df = DataFrame(frames.T, schema=["image", "x_head", "y_head", "x_tail", "y_tail"]).with_columns(
        col("image").cast(Int64),
    )
    print(df)
    # events = load_events_data(SourceData.EVENTS.value).filter(col("trial") == trial)
    # print(events)

class LobsterParameters(Enum):
    """
    Parameters for lobster plotting.
    """

    POSITION = "position"
    HEADING = "heading"

@cli_plot.command(Commands.LOBSTER.value)
@argument("parameter", type=Choice(LobsterParameters, case_sensitive=False))
@option("--bins", default=12, help="Number of bins for the histogram.")
def cli_plot_lobster(parameter: LobsterParameters, bins: int = 12):
    """
    Plot the position or heading distribution of lobster trials.
    This doesn't use derivatives, so we load and concatenate all the flat files.
    The records are then partitioned by whether the coil was on or off.
    """
    control = load_control_data(SourceData.CONTROL.value)
    series = control[parameter.value]
    area, edges = histogram(series, bins=bins, range=(0, 2 * pi), density=True)
    area = append(area, area[0])  # close the circle
    control_label = "Control (N=" + str(series.count()) + ")"
    trials = load_single_trial_polars(SourceData.TRIALS_DIR.value + "*.csv")
    fig, ax = subplots(subplot_kw={"projection": "polar"})
    ax.plot(edges, area, color=Palette.CONTROL.value, label=control_label)
    state = col(Dim.COMPASS.value)
    for mask, color, label in [
        (state <= 0, Palette.OFF.value, "Off"),
        (state > 0, Palette.ON.value, "On"),
    ]:
        series = trials.filter(mask)[parameter.value]
        area, edges = histogram(series, bins=bins, range=(-pi, pi), density=True)
        area = append(area, area[0])
        ax.plot(
            edges,
            area,
            color=color,
            label=f"{label} (N={series.count()})",
        )
    ax.grid(False)
    ax.set_title(f"Lobster {parameter.value} distribution by coil state")
    ax.set_yticklabels([])
    fig.legend(loc="lower right", frameon=False)
    fig.savefig(FIGURES / f"{Commands.LOBSTER.value}_{parameter.value}.png")
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
