# SeaDrive-MLM

SeaDrive-MLM is a ROS/Gazebo workspace for USV COLREG encounter simulation and Monte Carlo evaluation. The public entrypoint is `boat_monte_carlo.bash`, with two maintained modes:

- `Geo`: geometry and local-planner baseline.
- `Local`: local OpenAI-compatible VLM/LLM service for COLREG decisions.

Build products, logs, Monte Carlo outputs, credentials, generated datasets, and training artifacts are intentionally excluded from the repository.

## Requirements

- Ubuntu 20.04 with ROS Noetic.
- Gazebo and `gazebo_ros`.
- `catkin_tools`.
- Python 3 packages used by the runtime nodes, including `numpy`, `Pillow`, `opencv-python`, and ROS Python packages.
- For `Local` mode, a local OpenAI-compatible chat/completions endpoint that can accept the configured VLM model.

Install common ROS dependencies with your normal ROS package manager setup, for example:

```bash
sudo apt update
sudo apt install python3-catkin-tools ros-noetic-desktop-full ros-noetic-gazebo-ros-pkgs
```

## Build

From the repository root:

```bash
catkin build -j2
source devel/setup.bash
```

The runner also checks for `devel/setup.bash` and will ask you to build first if it is missing.

## Run Monte Carlo

Show usage:

```bash
bash boat_monte_carlo.bash --help
```

Run one geometric baseline trial:

```bash
MC_GUI=false bash boat_monte_carlo.bash HeadOn Geo 1
```

Run one local VLM/LLM trial:

```bash
MC_GUI=false LOCAL_API_BASE=http://host:8000/v1 MODEL=qwen3-vl-2b-instruct-local \
  bash boat_monte_carlo.bash NarrowMulti Local 1
```

Supported scenarios are `HeadOn`, `Crossing`, `Overtaking`, `MultiShip`, and `NarrowMulti`.

## Common Environment Variables

- `MC_COUNT`: number of trials when the count argument is omitted.
- `MC_TIMEOUT_S`: timeout per trial in seconds.
- `MC_RESULT_DIR`: output directory override.
- `MC_GUI`: set `false` for headless Gazebo runs.
- `MC_SEED`: fixed base seed; trial N uses `MC_SEED + N - 1`.
- `MC_PLOT_TRAJECTORIES`: set `false` to skip trajectory plots.
- `LOCAL_API_BASE` or `API_BASE`: OpenAI-compatible local API base URL for `Local` mode.
- `MODEL` or `LOCAL_MODEL`: local model name for `Local` mode.

## Outputs

By default each run writes to:

```text
tmp/monte_carlo/<timestamp>_<scenario>_<mode>/
```

Important files include `results.csv`, `summary.txt`, per-trial trajectory CSVs, optional VLA event logs, and optional trajectory plots.

## Notes

- Do not commit generated `build/`, `devel/`, `logs/`, or `tmp/` directories.
- Do not commit API keys or local service credentials. Configure them through environment variables.
- Historical experiment results and model-training outputs are not part of this repository; regenerate them locally when needed.
