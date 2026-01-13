"""
Load, transform, and do basic analysis on a lobster
movement trials dataset.
"""

from pandas import read_csv, DataFrame
from numpy import vectorize, atan2, pi, arange, histogram, append
from matplotlib.pyplot import subplots
from click import option, group, argument
import polars as pl


@group()
def cli():
    """
    Commandline interface for better UX when working with lobster
    movement dataset.
    """


@group(name="plot")
def cli_plot():
    """
    Commands for plotting experiment data.
    """


cli.add_command(cli_plot)


@group(name="describe")
def cli_describe():
    """
    Commands for describing experiment data.
    """


cli.add_command(cli_describe)


@group(name="lobster")
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
    df = df.with_columns(pl.col(date_col).str.to_datetime())
    return df


def clock_string_to_seconds(clock_str: str) -> int:
    """
    Convert a clock string in the format 'HH:MM:SS' to total seconds.
    """
    hours, minutes, seconds = map(int, clock_str.split(":"))
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
    struct = "computed"
    return (
        pl.read_csv(file_path)
        .with_columns(
            pl.col(time_column)
            .map_elements(clock_string_to_seconds, return_dtype=pl.Int64)
            .alias(time_column),
            pl.struct(["x_head", "x_tail", "y_head", "y_tail"])
            .map_elements(
                calc_heading,
                return_dtype=pl.Struct({"position": pl.Float64, "heading": pl.Float64}),
            )
            .alias(struct),
        )
        .unnest(struct)
    )

def load_control_data(file_path: str, time_column: str = "elapsed_time") -> pl.DataFrame:
    """
    Load the control data for lobster movement trials from a CSV file.

    Convert time to elapsed seconds, and position and heading to radians.
    """
    return (
        pl.read_csv(file_path)
        .with_columns(
            pl.col(time_column)
            .map_elements(clock_string_to_seconds, return_dtype=pl.Int64)
            .alias(time_column),
            (pl.col("sector") / 360 * 2 * pi).alias("position"),
            (pl.col("heading") / 360 * 2 * pi).alias("heading")
        )
    )


def join_elapsed_time_since_event(
    trial_data: DataFrame, trial: int, events: DataFrame
) -> DataFrame:
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


@cli_plot.command("lobster")
@argument("column", default="heading")
@option("--bins", default=12, help="Number of bins for the histogram.")
def cli_plot_lobster(column: str, bins: int = 12):
    """
    Plot the position distribution of a lobster trial. This doesn't use derivatives,
    so we are free to load and concatenate all the trials in one go. The records are then
    partitioned by whether the coil was on or off.
    """
    fig, ax = subplots(subplot_kw={"projection": "polar"})
    df = load_control_data("data/control.csv")
    series = df[column]
    rows = series.count()
    area, edges = histogram(series, bins=bins, range=(0, 2*pi), density=True)
    area = append(area, area[0])
    ax.plot(edges, area, color="black", label="control")
    df = load_single_trial_polars("data/trials/*.csv")
    for mask, color, label in [
        (pl.col("compass") > 0, "red", "coil on"),
        (pl.col("compass") <= 0, "blue", "coil off"),
    ]:
        series = df.filter(mask)[column]
        rows += series.count()
        area, edges = histogram(series, bins=bins, range=(-pi, pi), density=True)
        area = append(area, area[0])
        ax.plot(
            edges,
            area,
            color=color,
            label=label
        )
    ax.grid(False)
    ax.set_title(f"{column.capitalize()} distribution (N={rows})")
    ax.set_yticklabels([])
    fig.legend()
    fig.savefig(f"figures/trial_{column}_polar_plot.png")
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
