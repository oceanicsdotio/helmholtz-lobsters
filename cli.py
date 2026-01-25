"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.

- `plot lobster histogram heading`: OK
- `plot lobster histogram position`: OK
- `plot lobster trajectory <TRIAL>`: OK
- `describe lobster trials`: WIP
- `describe lobster roi`: WIP

"""

from enum import Enum
from pathlib import Path
from itertools import pairwise
from zipfile import ZipFile
from numpy import atan2, pi, histogram, append, vstack, meshgrid
import numpy as np
from matplotlib.pyplot import subplots
from click import option, group, argument, Choice
from polars import (
    col,
    Struct,
    Float64,
    struct,
    Int64,
    DataFrame,
    Config
)
import polars as pl
from roifile import roiread, ImagejRoi
from sklearn.neighbors import KernelDensity

FIGURES = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"

class Commands(Enum):
    """
    Consistent command names for the CLI.
    """

    PLOT = "plot"
    DESCRIBE = "describe"
    LOBSTER = "lobster"
    HISTOGRAM = "histogram"


class PlotLobsterOptions(Enum):
    """
    Parameters for lobster plotting.
    """

    POSITION = "position"
    HEADING = "heading"


class SourceData(Enum):
    """
    Docstring for Metadata
    """

    CONTROL = DATA / "control.csv"
    TRIALS_DIR = DATA / "trials"
    TRIALS = DATA / "trials.csv"
    INTERVALS = DATA / "intervals.csv"
    EVENTS = DATA / "events.csv"
    LOBSTERS = DATA / "lobsters.csv"
    GEOMAGNETIC_FIELD = DATA / "magnetic-field.csv"
    CONTROL_EVENTS = DATA / "control-events.csv"


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
    HEADING = "heading"  # computed / observed
    POSITION = "position"  # computed


class Derived(Enum):
    """
    Derived dimensions used in lobster movement dataset.
    """

    THETA = "position"
    RADIUS = "radius"
    HEADING = "heading"


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


@group(name=Commands.LOBSTER.value)
def cli_plot_lobster():
    """
    Commands for describing lobster movement data.
    """


@group(name=Commands.DESCRIBE.value)
def cli_describe():
    """
    Commands for describing experiment data.
    """


@group(name=Commands.LOBSTER.value)
def cli_describe_lobster():
    """
    Commands for describing lobster movement data.
    """





def clock_string_to_seconds(clock_str: str) -> int:
    """
    Convert a clock string in the format 'HH:MM:SS' to total seconds.
    """
    hours, minutes, seconds = map(int, clock_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


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
    return {
        Dim.POSITION.value: atan2(x, y),
        Dim.HEADING.value: atan2(dx, dy),
        "radius": (x**2 + y**2) ** 0.5,
    }


def parse_row(item: tuple[str, ImagejRoi], res: float) -> list[float]:
    file, roi = item
    frame = int(file.split("-")[0])
    coords = roi.coordinates().flatten()  # type: ignore
    coords = (coords - (res / 2)) / (res / 2)
    coords[1] *= -1  # invert y axis from image coordinates to cartesian
    coords[3] *= -1
    return [frame, *coords]


def load_roi_time_series(file_path: Path, res: float) -> DataFrame:
    """
    Load lobster movement trial dataset with ROI data from a CSV file.
    """

    with ZipFile(file_path, "r") as files:
        data = zip(files.namelist(), roiread(file_path))  # type: ignore
        frames = [parse_row(item, res) for item in data]

    alias = "computed"  # temporary column during unnest
    names = [Dim.X_HEAD.value, Dim.Y_HEAD.value, Dim.X_TAIL.value, Dim.Y_TAIL.value]
    computed_struct = Struct(
        {
            Dim.POSITION.value: Float64,
            Dim.HEADING.value: Float64,
            "radius": Float64,
        }
    )
    computed = (
        struct(names)
        .map_elements(calc_heading, return_dtype=computed_struct)
        .alias(alias)
    )
    time_col = col(Dim.ELAPSED_TIME.value).cast(Int64)
    return (
        DataFrame(frames, schema=[Dim.ELAPSED_TIME.value, *names], orient="row")
        .with_columns(time_col, time_col.diff().alias("time_diff"), computed)
        .unnest(alias)
    )


def load_trial_time_series(file_path: str) -> DataFrame:
    """
    Load either a single lobster movement trial dataset from a CSV file, or
    use a pattern to load and concatenate many files.
    """
    time = (
        col(Dim.ELAPSED_TIME.value)
        .map_elements(clock_string_to_seconds, return_dtype=Int64)
        .alias(Dim.ELAPSED_TIME.value)
    )
    alias = "computed"  # temporary column during unnest
    names = [Dim.X_HEAD.value, Dim.X_TAIL.value, Dim.Y_HEAD.value, Dim.Y_TAIL.value]
    computed = (
        struct(names)
        .map_elements(
            calc_heading,
            return_dtype=Struct(
                {
                    Dim.POSITION.value: Float64,
                    Dim.HEADING.value: Float64,
                    "radius": Float64,
                }
            ),
        )
        .alias(alias)
    )
    return pl.read_csv(file_path).with_columns(time, computed).unnest(alias)


def load_control_data(file_path: Path, time_column: str = "elapsed_time") -> DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.

    Convert time to elapsed seconds, and position and heading to radians.
    """
    return pl.read_csv(file_path).with_columns(
        col(time_column)
        .map_elements(clock_string_to_seconds, return_dtype=Int64)
        .alias(time_column),
        (col(Dim.SECTOR.value) / 360 * 2 * pi).alias(Dim.POSITION.value),
        (col(Dim.HEADING.value) / 360 * 2 * pi).alias(Dim.HEADING.value),
    )

@cli_describe_lobster.command("trials")
def cli_describe_lobster_trials():
    """
    Get most recent molt date for the lobster in each trial.

    Then subtract that from the trial date to get time since molt.
    """
    lobsters = (
        pl.read_csv(SourceData.LOBSTERS.value)
        .with_columns(
            col(Dim.DATE.value).str.to_datetime().dt.date().alias(Dim.DATE.value)
        )
        .filter(col(Dim.EVENT.value) == "molt")
    )
    date_right = Dim.DATE.value + "_right"  # temporary column during join
    trials = (
        pl.read_csv(SourceData.TRIALS.value)
        .with_columns(
            col(Dim.DATE.value).str.to_datetime().dt.date().alias(Dim.DATE.value)
        )
        .join(lobsters, on=Dim.LOBSTER_ID.value, how="left")
        .with_columns(
            (col(Dim.DATE.value) - col(date_right)).alias("days_since_molt"),
        )
        .drop([date_right])
        .group_by((Dim.TRIAL.value, Dim.LOBSTER_ID.value))
        .agg(
            pl.all().exclude(("days_since_molt", Dim.DATE.value)).unique().item(),
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
    df = load_roi_time_series(SourceData.TRIALS_DIR.value / "024_ROI.zip", 400.0)
    print(df.head())
    # events = load_events_data(SourceData.EVENTS.value).filter(col("trial") == trial)
    # print(events)


res_lookup = {
    16: 456,
    17: 450,
    18: 450,
    19: 400,
    20: 400,
    21: 400,
    22: 400,
    23: 300,
    24: 400,
}


@cli_plot_lobster.command("trajectory")
@argument("trial", type=int)
def cli_plot_lobster_trajectory(trial: int):
    """
    Plot the trajectory of lobster trials.
    """

    df = load_roi_time_series(
        SourceData.TRIALS_DIR.value / f"{trial:03d}_ROI.zip", res_lookup[trial]
    )
    indices = df.select((col("time_diff") > 1).arg_true().alias("indices"))
    X = vstack([df[Dim.X_HEAD.value], df[Dim.Y_HEAD.value]]).T
    kde = KernelDensity(bandwidth=0.05, kernel="gaussian").fit(X)
    xx, yy = meshgrid(np.linspace(-1, 1, 120), np.linspace(-1, 1, 50))
    xy_grid = vstack([xx.ravel(), yy.ravel()]).T

    # Predict the log density
    log_dens = kde.score_samples(xy_grid)
    dens = np.exp(log_dens)
    dens = dens.reshape(xx.shape)
    theta_grid = np.arctan2(xx, yy)
    r_grid = np.sqrt(xx**2 + yy**2)

    fig, ax = subplots(subplot_kw={"projection": "polar"})
    c = ax.pcolormesh(theta_grid, r_grid, dens, cmap="cool", shading="gouraud")
    fig.colorbar(c, ax=ax, label="Density")

    for current, next_item in pairwise([0, *indices["indices"], len(df) + 1]):
        subset = df.slice(offset=current, length=next_item - current)
        x = subset[Dim.POSITION.value]
        y = subset["radius"]
        ax.plot(
            x,
            y,
            color="black",
            alpha=0.6,
            linestyle="-",
            linewidth=0.5,
        )

    ax.grid(False)
    ax.set_title(f"Trajectory for trial {trial}")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    fig.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"lobster_trajectory_{trial}.png")
    print("OK")


@cli_plot_lobster.command(Commands.HISTOGRAM.value)
@argument("parameter", type=Choice(PlotLobsterOptions, case_sensitive=False))
@option("--bins", default=12, help="Number of bins.")
def cli_plot_lobster_histogram(parameter: PlotLobsterOptions, bins: int):
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
    trials_csv = load_trial_time_series(str(SourceData.TRIALS_DIR.value) + "/*.csv")
    trials = DataFrame()
    for trial_num in range(16, 25):
        trial_data = load_roi_time_series(
            SourceData.TRIALS_DIR.value / f"{trial_num:03d}_ROI.zip",
            res_lookup[trial_num],
        )
        trials = pl.concat([trials, trial_data])
    fig, ax = subplots(subplot_kw={"projection": "polar"})
    ax.plot(edges, area, color=Palette.CONTROL.value, label=control_label)
    state = col(Dim.COMPASS.value)
    for mask, color, label in [
        (state <= 0, Palette.OFF.value, "Off"),
        (state > 0, Palette.ON.value, "On"),
    ]:
        series = trials_csv.filter(mask)[parameter.value]
        area, edges = histogram(series, bins=bins, range=(-pi, pi), density=True)
        area = append(area, area[0])
        ax.plot(
            edges,
            area,
            color=color,
            label=f"{label} (N={series.count()})",
        )
    area, edges = histogram(
        trials[parameter.value], bins=bins, range=(-pi, pi), density=True
    )
    area = append(area, area[0])
    ax.plot(edges, area, color="black", linestyle="--", label="All trials")
    ax.grid(False)
    ax.set_title(f"Lobster {parameter.value} distribution by coil state")
    ax.set_yticklabels([])
    fig.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{Commands.LOBSTER.value}-histogram-{parameter.value}.png")
    print("OK")


if __name__ == "__main__":
    cli.add_command(cli_describe)
    cli.add_command(cli_plot)
    cli_describe.add_command(cli_describe_lobster)
    cli_plot.add_command(cli_plot_lobster)
    cli()
