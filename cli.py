"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.

- `plot lobster histogram <PARAMETER>`: OK
- `plot lobster trajectory <TRIAL>`: OK

- `describe lobster trials`: WIP
- `describe lobster roi`: WIP

"""

from enum import Enum
from pathlib import Path
from itertools import pairwise
from zipfile import ZipFile
from numpy import atan2, pi, histogram, meshgrid, linspace, exp, array
from numpy.linalg import norm
from matplotlib.pyplot import subplots
from click import option, group, argument, Choice
from polars import col, Struct, Float64, struct, Int64, DataFrame, read_csv, Series
import polars as pl
from roifile import roiread, ImagejRoi
from sklearn.neighbors import KernelDensity

FIGURES = Path(__file__).parent / "figures"
DATA = Path(__file__).parent / "data"


class Cmd(Enum):
    """
    Consistent command names for the CLI.
    """

    PLOT = "plot"
    DESCRIBE = "describe"
    LOBSTER = "lobster"
    HISTOGRAM = "histogram"
    TRAJECTORY = "trajectory"
    TRIALS = "trials"


@group(name=Cmd.LOBSTER.value)
def cli_plot_lobster():
    """
    Cmd for describing lobster movement data.
    """


@group(name=Cmd.LOBSTER.value)
def cli_describe_lobster():
    """
    Cmd for describing lobster movement data.
    """


class PlotLobsterOptions(Enum):
    """
    Parameters for lobster plotting. Used for Click type checking
    with a Choice.
    """

    POSITION = "position"
    HEADING = "heading"


class SourceData(Enum):
    """
    Data and metadata file paths
    """

    CONTROL = DATA / "control.csv"
    TRIALS_DIR = DATA / "trials"
    TRIALS = DATA / "trials.csv"
    INTERVALS = DATA / "intervals.csv"
    EVENTS = DATA / "events.csv"
    LOBSTERS = DATA / "lobsters.csv"


class Dim(Enum):
    """
    Dimensions used in lobster movement dataset.
    """

    DATE = "date"
    TRIAL = "trial"
    LOBSTER_ID = "lobster_id"
    TRIAL_TYPE = "trial_type"
    ELAPSED_TIME = "elapsed_time"
    FIELD = "field"  # switching field on/off
    ENTANGLED = "entangled"  # whether lobster is entangled during switching
    START = "start"  # intervals
    STOP = "stop"  # intervals
    EVENT = "event"  # lobsters, molt / death
    HEADING = "heading"  # control
    POSITION = "position"  # control

    X_HEAD = "x_head"  # normalized position of the head
    Y_HEAD = "y_head"
    X_TAIL = "x_tail"  # normalized position of the tail
    Y_TAIL = "y_tail"
    COMPASS = "compass"  # coil state
    SINCE_EVENT = "since_event"  # time since coil state switch


class Derived(Enum):
    """
    Derived dimensions used in lobster movement dataset.
    """

    THETA = "position"
    RADIUS = "radius"
    HEADING = "heading"
    X = "x"
    Y = "y"
    TIME_DIFF = "time_diff"


class Palette(Enum):
    """
    Color palette for plotting.
    """

    ON = "red"
    OFF = "blue"
    CONTROL = "black"


# Image sizes for each trial number
image_size = {
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


def clock_string_to_seconds(clock_str: str) -> int:
    """
    Convert a clock string in the format 'HH:MM:SS' to total seconds.
    """
    hours, minutes, seconds = map(int, clock_str.split(":"))
    return hours * 3600 + minutes * 60 + seconds


computed_struct = Struct(
    {
        Derived.THETA.value: Float64,
        Derived.HEADING.value: Float64,
        Derived.X.value: Float64,
        Derived.Y.value: Float64,
        Derived.RADIUS.value: Float64,
    }
)


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
        Derived.THETA.value: atan2(y, x),
        Derived.HEADING.value: atan2(dy, dx),
        Derived.X.value: x,
        Derived.Y.value: y,
        Derived.RADIUS.value: (x**2 + y**2) ** 0.5,
    }


def parse_row(item: tuple[str, ImagejRoi], res: float) -> list[float]:
    """
    Parse ImageJ ROI data into a flat list of frame and normalized coordinates.
    """
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


def load_trial_time_series(file_path: Path) -> DataFrame:
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
    return read_csv(file_path).with_columns(time, computed).unnest(alias)


def load_control_data(file_path: Path, time_column: str = "elapsed_time") -> DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.

    Convert time to elapsed seconds, and position and heading to radians.
    """
    return read_csv(file_path).with_columns(
        col(time_column)
        .map_elements(clock_string_to_seconds, return_dtype=Int64)
        .alias(time_column),
        (col(Dim.POSITION.value) / 360 * 2 * pi).alias(Dim.POSITION.value),
        (col(Dim.HEADING.value) / 360 * 2 * pi).alias(Dim.HEADING.value),
    )


@cli_describe_lobster.command("trials")
def cli_describe_lobster_trials():
    """
    Get most recent molt date for the lobster in each trial.

    Then subtract that from the trial date to get time since molt.
    """
    lobsters = (
        read_csv(SourceData.LOBSTERS.value)
        .with_columns(
            col(Dim.DATE.value).str.to_datetime().dt.date().alias(Dim.DATE.value)
        )
        .filter(col(Dim.EVENT.value) == "molt")
    )
    date_right = Dim.DATE.value + "_right"  # temporary column during join
    trials = (
        read_csv(SourceData.TRIALS.value)
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
    df = read_csv(file_path)
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


# pylint: disable=too-many-locals
@cli_plot_lobster.command(Cmd.TRAJECTORY.value)
@argument("trial", type=int)
@option("--bandwidth", default=0.1, help="Bandwidth for kernel density estimate.")
@option("--colormap", default="Blues_r", help="Colormap for density plot.")
@option("--image-format", default="png", help="Image format for output file.")
def cli_plot_lobster_trajectory(
    trial: int,
    bandwidth: float,
    colormap: str,
    image_format: str,
):
    """
    Plot the trajectory of lobster trials. The visualization shows the
    superposition of the trajectory of a single trial over the output of
    a kernel density estimate of all positions in that trial.
    """
    path = SourceData.TRIALS_DIR.value / f"{trial:03d}_ROI.zip"
    df = load_roi_time_series(path, image_size[trial])
    fig, ax = subplots(subplot_kw={"projection": "polar"})

    # Project the predicted density to a polar grid.
    i, j = (50, 120)
    train = df.select([Derived.X.value, Derived.Y.value])
    grid = meshgrid(linspace(-1, 1, i), linspace(-1, 1, j))
    predict = (
        KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        .fit(train.to_numpy())
        .score_samples(array(grid).reshape(2, i * j).T)
        .reshape((j, i))
    )
    gradient = ax.pcolormesh(
        atan2(*reversed(grid)),
        norm(array(grid), axis=0),
        exp(predict),
        cmap=colormap,
        shading="gouraud"
    )
    fig.colorbar(gradient, ax=ax, label="Density")

    # Break position data on gaps in the time series, and plot each segment
    # over the density gradient.
    mask = (col(Derived.TIME_DIFF.value) > 1).arg_true()
    breakpoints = df.select(mask).get_column(Derived.TIME_DIFF.value)
    edges = [0, *breakpoints, len(df) + 1]
    select = (Derived.THETA.value, Derived.RADIUS.value)
    for start, end in pairwise(edges):
        ax.plot(
            *df.slice(offset=start, length=end - start).select(select),
            color=Palette.CONTROL.value,
            alpha=0.5,
            linestyle=":",
            linewidth=1,
        )

    # Extract coil toggle events from CSV (for now), and plot the
    # location of the animal at these times. Future case needs to
    # take the event timestamps for each trial and join with the
    # ROI trajectories to get positions from raw sources.
    # OK for now to drop rows with missing data...
    trials = load_trial_time_series(SourceData.TRIALS_DIR.value / f"{trial}.csv").with_columns(
        (col(Dim.COMPASS.value).diff()).alias("compass_diff")
    )
    dims = [Derived.THETA.value, Derived.RADIUS.value]
    filter_on = col("compass_diff")
    on_events = trials.filter(filter_on > 1).select(dims)
    ax.scatter(
        *on_events,
        edgecolors=Palette.CONTROL.value,
        facecolors="white",
        label=f"Coil On (N={on_events.height})",
        marker=".",
        s=100,
        zorder=3,
    )
    off_events = trials.filter(filter_on < -1).select(dims)
    ax.scatter(
        *off_events,
        edgecolors=Palette.CONTROL.value,
        facecolors=Palette.CONTROL.value,
        label=f"Coil Off (N={off_events.height})",
        marker=".",
        s=100,
        zorder=3,
    )

    # Style and save the figure.
    ax.grid(False)
    ax.set_title(f"Position (Trial {trial})")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    fig.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    filename = [Cmd.LOBSTER.value, Cmd.TRAJECTORY.value, f"{trial}.{image_format}"]
    fig.savefig(FIGURES / "-".join(filename))
    print("OK")


@cli_plot_lobster.command(Cmd.HISTOGRAM.value)
@argument("parameter", type=Choice(PlotLobsterOptions, case_sensitive=False))
@option("--bins", default=12, help="Number of bins.")
def cli_plot_lobster_histogram(parameter: PlotLobsterOptions, bins: int):
    """
    Plot the position or heading distribution of lobster trials.
    This doesn't use derivatives, so we load and concatenate all the flat files.
    The records are then partitioned by whether the coil was on or off.
    """
    fig, ax = subplots(subplot_kw={"projection": "polar"})

    def polar_hist(
        series: Series,
        label: str,
        bounds: tuple[float, float] = (-pi, pi),
        color: Palette = Palette.CONTROL,
        linestyle: str = "-",
    ):
        """Convenience function to plot polar histogram"""
        hist = histogram(series, bins=bins, range=bounds, density=True)
        count = series.is_not_nan().sum()
        ax.fill(
            *zip(*zip(*reversed(hist))),
            color=color.value,
            linestyle=linestyle,
            label=f"{label.capitalize()} (N={count})",
            fill=False,
        )

    # Load and combine all trial data from ROI files, then plot
    # the overall distribution.
    def trial(index: int):
        """Convenience function to load a single ROI file."""
        path = SourceData.TRIALS_DIR.value / f"{index:03d}_ROI.zip"
        df = load_roi_time_series(path, image_size[index])
        return df[parameter.value]

    polar_hist(
        pl.concat(map(trial, range(16, 25))),
        linestyle="--",
        label="all",
    )

    # Load data from control trials and plot the distribution as a
    # polygon on the polar plot. The zipping and unpacking is to
    # close the polygon and get coordinates in the right order.
    polar_hist(
        load_control_data(SourceData.CONTROL.value).get_column(parameter.value),
        bounds=(0, 2 * pi),
        label="control",
    )

    # Load and combine experiment data from ROI files, then partition the
    # observations by coil state to plot their distributions.
    trials_csv = load_trial_time_series(SourceData.TRIALS_DIR.value / "*.csv")
    state = col(Dim.COMPASS.value)
    polar_hist(
        trials_csv.filter(state <= 0).get_column(parameter.value),
        color=Palette.OFF,
        label="off",
    )
    polar_hist(
        trials_csv.filter(state > 0).get_column(parameter.value),
        color=Palette.ON,
        label="on",
    )

    # Style and save the figure.
    ax.grid(False)
    ax.set_title(f"Lobster {parameter.value} distribution by coil state")
    ax.set_yticklabels([])
    fig.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    filename = [Cmd.LOBSTER.value, Cmd.HISTOGRAM.value, parameter.value + ".png"]
    fig.savefig(FIGURES / "-".join(filename))
    print("OK")


if __name__ == "__main__":

    @group()
    def cli():
        """
        Commandline interface for better UX when working with lobster
        movement dataset.
        """

    @group(name=Cmd.PLOT.value)
    def cli_plot():
        """
        Cmd for plotting experiment data.
        """

    @group(name=Cmd.DESCRIBE.value)
    def cli_describe():
        """
        Cmd for describing experiment data.
        """

    cli.add_command(cli_describe)
    cli.add_command(cli_plot)
    cli_describe.add_command(cli_describe_lobster)
    cli_plot.add_command(cli_plot_lobster)
    cli()
