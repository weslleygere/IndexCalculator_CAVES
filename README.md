# Soundscape Acoustic Index Calculator

This project provides a modular batch-processing pipeline for extracting acoustic indices from audio recordings for soundscape analysis. It recursively discovers WAV files, loads and preprocesses each recording, splits it into fixed-duration segments, computes temporal and spectral alpha indices with [`scikit-maad`](https://scikit-maad.github.io/), and writes the results to CSV files.

The pipeline is designed for large audio collections: files are processed in parallel, results are streamed to disk, failures are recorded separately, and a checkpoint allows interrupted runs to resume without reprocessing completed files.

## Table of Contents

1. [Features](#features)
2. [Project Structure](#project-structure)
3. [Processing Workflow](#processing-workflow)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Output and Logs](#output-and-logs)
8. [Resuming an Interrupted Run](#resuming-an-interrupted-run)

## Features

- Recursive, case-insensitive discovery of `.wav` files
- Configurable audio channel selection and detrending
- Fixed-duration segmentation with tolerance for incomplete final segments
- Clipping detection and optional resampling and normalization
- Temporal and spectral acoustic indices computed with `scikit-maad`
- Multiprocessing with a configurable number of workers
- Incremental CSV writing to limit memory usage
- Separate error reporting for loading, preprocessing, and index-calculation failures
- Per-file checkpoints for safely resuming an interrupted execution
- Timestamped output directories and execution logs

## Project Structure

```text
.
|-- main.py                         # Application entry point
|-- config_params.json              # Audio preprocessing and index parameters
|-- requirements.txt                # Pinned Python dependencies
|-- .env.example                    # Example runtime environment variables
|-- src/
|   |-- pipeline.py                 # Pipeline orchestration and multiprocessing
|   |-- config/
|   |   |-- settings.py             # Environment loading and validation
|   |   `-- logging.py              # Console/file logging and run metadata
|   `-- core/
|       |-- audio_loader.py          # WAV loading through scikit-maad
|       |-- pre_processor.py         # Segmentation, clipping, resampling, normalization
|       |-- idx_processor.py         # Temporal and spectral index calculation
|       |-- params_loader.py         # JSON parameter loading and validation
|       `-- utils.py                 # Metadata, scanning, CSV, and checkpoint helpers
`-- README.md
```

The `output/` directory is created automatically when the pipeline runs.

## Processing Workflow

For every directory containing WAV files, the pipeline performs the following steps:

1. Scans the configured input directory recursively.
2. Excludes files already recorded in the selected run's checkpoint.
3. Loads each WAV file using the configured channel and detrending options.
4. Splits the waveform into fixed-duration segments.
5. Rejects clipped segments and incomplete final segments outside the configured tolerance.
6. Optionally downsamples and normalizes valid segments.
7. Calculates all temporal and spectral alpha indices exposed by `scikit-maad`.
8. Writes successful segment results and errors to separate CSV files.
9. Records the source file in the checkpoint after its results have been flushed.

Temporal index columns are prefixed with `t_`; spectral index columns are prefixed with `s_`.

## Installation

### Local Setup

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate it.

   On Windows PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   On Linux or macOS:

   ```bash
   source .venv/bin/activate
   ```

4. Install the pinned dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Create the local environment file:

   On Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

   On Linux or macOS:

   ```bash
   cp .env.example .env
   ```

## Configuration

Configuration is split between `.env`, which controls paths and execution settings, and `config_params.json`, which controls audio preprocessing and index calculation.

### Environment Variables

Edit `.env` before running the pipeline:

```dotenv
DATA_PATH=/path/to/wav/files
CONFIG_PARAMS_PATH=config_params.json
OUTPUT_DIR=output
RESUME_OUTPUT_DIR=
MAX_WORKERS=0
LOG_LEVEL=INFO
```

| Variable | Description | Default used by the code |
|---|---|---|
| `DATA_PATH` | Directory containing the WAV collection. Subdirectories are scanned recursively. | `data/raw` |
| `CONFIG_PARAMS_PATH` | Path to the JSON processing configuration. | `data/parameters/config_params.json` |
| `OUTPUT_DIR` | Base directory for new timestamped runs. | `output` |
| `RESUME_OUTPUT_DIR` | Existing run directory to resume. Leave empty for a new run. | Empty |
| `MAX_WORKERS` | Worker processes. `0` selects one fewer than the detected CPU count, with a minimum of one. | `0` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL`. | `INFO` |

The repository's `.env.example` already points `CONFIG_PARAMS_PATH` to the included root-level `config_params.json`. Replace its example `DATA_PATH` with the location of your own audio collection.

### Audio and Index Parameters

The included `config_params.json` has this structure:

```json
{
  "preprocess": {
    "channel": "right",
    "detrend": true,
    "segment_duration": 60,
    "clipping_threshold": 0.98,
    "target_sample_rate": null,
    "normalize_audio": false,
    "segment_tolerance_percent": 1.0
  },
  "indices": {
    "compatibility": "seewave"
  },
  "spectrogram": {
    "flims": [0, 20000]
  }
}
```

| Parameter | Description |
|---|---|
| `preprocess.channel` | Audio channel passed to the `scikit-maad` WAV loader, such as `left` or `right`. |
| `preprocess.detrend` | Whether to remove the waveform's linear trend while loading. |
| `preprocess.segment_duration` | Target segment length in seconds; must be an integer greater than zero. |
| `preprocess.clipping_threshold` | Maximum accepted absolute amplitude, from `0` to `1`. A segment meeting or exceeding it is rejected. |
| `preprocess.target_sample_rate` | Optional output sample rate. Use `null` to preserve the source rate. Resampling is only applied when this value is lower than the source rate. |
| `preprocess.normalize_audio` | Whether valid segments are normalized to a maximum amplitude of `1.0`. |
| `preprocess.segment_tolerance_percent` | Maximum percentage of missing duration accepted for a final partial segment. Accepted tails are zero-padded; others are reported as errors. |
| `indices.compatibility` | Compatibility mode passed to the temporal index calculation, for example `seewave`. |
| `spectrogram.flims` | Two integer frequency limits, in hertz, satisfying `0 <= low < high`. |

Choose an upper frequency limit that is valid for the recordings' sample rate (no higher than the Nyquist frequency, which is half the sample rate).

## Running the Pipeline

After installing dependencies and configuring `.env`, run:

```bash
python main.py
```

Progress and messages are displayed in the terminal. The run's `experiment.log` stores run metadata plus warnings and errors.

## Output and Logs

Each new execution creates a timestamped directory:

```text
output/
`-- YYYYMMDD_HHMMSS/
    |-- acoustic_indices.csv
    |-- acoustic_errors.csv              # Created only when an error is recorded
    |-- processed_files_checkpoint.txt
    `-- experiment.log
```

### `acoustic_indices.csv`

Contains one row per successfully processed audio segment. Its leading columns are:

| Column | Description |
|---|---|
| `directory_name` | Name of the source file's immediate parent directory. |
| `file_name` | Generated segment name. |
| `segment_id` | Zero-based segment number. |
| `sample_rate` | Segment sample rate after optional downsampling. |
| `processing_time` | Total processing time for the source file, in seconds. |
| `t_*` | Temporal acoustic-index values. |
| `s_*` | Spectral acoustic-index values. |

### `acoustic_errors.csv`

Contains failed files or segments with the processing stage, directory, file name, segment number, message, and exception type. It is created only if at least one failure occurs.

### `processed_files_checkpoint.txt`

Stores completed source-file paths and their success/failure counts. The checkpoint is updated only after the associated CSV records are flushed to disk.

### `experiment.log`

Records warnings and errors, platform and path metadata, and snapshots of `.env` and `config_params.json`. Because `.env` is copied into the log, avoid placing secrets in it.

## Resuming an Interrupted Run

To continue an existing run, set `RESUME_OUTPUT_DIR` in `.env` to that run's directory:

```dotenv
RESUME_OUTPUT_DIR=output/20260909_143000
```

Then run the same command again:

```bash
python main.py
```

Files already listed in `processed_files_checkpoint.txt` are skipped, existing acoustic-index data is appended to, and new errors are appended to the error CSV. Keep the same input directory and processing parameters when resuming so that all rows in the run remain comparable.
