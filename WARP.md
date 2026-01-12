# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This repository contains analysis code and data from a scientific study demonstrating that American lobsters (*Homarus americanus*) reorient in response to magnetic field reversals. The research uses Helmholtz coils to reverse local magnetic fields and tracks lobster movements in controlled arena trials.

## Development Environment

### Package Management
This project uses **pixi** for Python dependency management:
- Environment configuration: `pixi.toml`
- Dependencies: Python 3.13+, pandas, matplotlib
- Platform: osx-arm64

### Setup and Activation
```bash
# Install dependencies (if needed)
pixi install

# Run Python scripts in the pixi environment
pixi run python main.py
```

## Project Structure

### Code Organization
- `main.py`: Single monolithic analysis script containing all data loading, transformation, and plotting functions
- `data/`: Experimental data in CSV format
  - `trials.csv`: Metadata for each trial (dates, lobster IDs, trial types)
  - `events.csv`: Magnetic field reversal events with timestamps
  - `intervals.csv`: Time intervals for analysis
  - `control.csv`: Control trial data
  - `trials/`: Individual trial data files (16.csv through 24.csv) with position coordinates and headings
- `figures/`: Output directory for generated polar plots

### Data Processing Pipeline
The analysis follows this workflow:
1. Load trial metadata and event data
2. Parse time strings ("HH:MM:SS") into elapsed seconds
3. Join events with trial intervals to identify valid data windows
4. Calculate angular positions and headings from x/y coordinates using `atan2`
5. Split data by coil state (on/off) for comparative analysis
6. Generate polar histograms of lobster positions and headings

### Key Data Columns
Trial data files contain:
- `elapsed_time`: Time in HH:MM:SS format
- `x_head`, `y_head`, `x_tail`, `y_tail`: Lobster coordinates
- `compass`: Magnetic field strength/direction indicator
- `since_event`: Seconds elapsed since last field reversal

## Running the Analysis

### Execute Main Script
```bash
pixi run python main.py
```

This processes trials 16-24 and generates:
- `figures/trial_position_polar_plot.png`: Angular position distribution
- `figures/trial_heading_polar_plot.png`: Heading distribution

Both plots split data by coil state (red=on, blue=off) to visualize reorientation effects.

### Development Notes
- Much of the `if __name__ == "__main__"` block is commented out, showing iterative analysis development
- The script is designed for exploratory analysis rather than production use
- No automated tests exist in this repository
- Type hints are used consistently throughout functions

## Scientific Context

The research demonstrates magnetic field sensitivity in American lobsters, analogous to known navigation abilities in spiny lobsters (*Panulirus argus*). Experiments use:
- Helmholtz coils to create controlled magnetic field reversals
- Blindfolded and tethered juvenile lobsters to isolate magnetic sensing
- Video tracking to record reorientation events
- Statistical analysis of heading changes correlated with field reversals

Trial types:
- Type 1: Baseline trials (untethered, no field manipulation)
- Type 2A/2B: Initial EMF trials with single reversal
- Type 3: Regular reversals every 2 minutes
- Type 4: Opportunistic reversals every 15-60 seconds
