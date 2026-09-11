# trace_v0.2

## Project Structure

## What Each Area Does

### Root files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Lists the Python dependencies. |
| `pytest.ini` | Configures pytest. |
| `README.md` | Project structure and file documentation. |

### `backend/`

| File | Purpose |
| --- | --- |
| `config.py` | Defines project paths, FastF1 cache settings, and race events. |
| `cleaner.py` | Cleans and validates extracted lap data. |
| `extract_laps.py` | Extracts lap timing data from FastF1 sessions. |
| `extract_telemetry.py` | Extracts car telemetry data. |
| `extract_other.py` | Extracts weather and race-control data. |
| `run_ingestion.py` | Loads configured sessions and writes processed Parquet files. |
| `feature_engineering.py` | Creates derived features for modelling. |
| `derive_acceleration.py` | Derives acceleration-related data from telemetry. |
| `track_intervals.py` | Builds track interval data. |
| `import_external_telemetry.py` | Imports telemetry from external data sources. |
| `merge_external_drs.py` | Merges external DRS data into processed data. |
| `inspect_track_status.py` | Inspects track-status records. |
| `verify.py` | Performs data and pipeline verification checks. |
| `test_load.py` | Checks that data-loading code works. |

### `backend/models/`

| File | Purpose |
| --- | --- |
| `race_state.py` | Defines enums and Pydantic models for cars, tyres, track conditions, and race state. |
| `result.py` | Defines lap, driver, and complete simulation result models. |
| `weather.py` | Intended weather model module; currently empty. |
| `car.py`, `race.py`, `strategy.py`, `track.py`, `tyre.py` | Model files currently present but empty. |

### `backend/physics/`

| File or group | Purpose |
| --- | --- |
| `lap_time_predictor.py` | Predicts lap time from circuit, tyre, fuel mass, and wetness. |
| `acceleration.py`, `acceleration_model.py` | Calculates and models vehicle acceleration. |
| `vehicle.py`, `mass.py`, `fuel.py`, `parameters.py` | Represents vehicle physics, mass, fuel, and constants. |
| `track.py`, `track_state.py`, `wet_track.py` | Represents circuit and track-condition effects. |
| `tyre.py`, `fuel_effect.py` | Models tyre degradation and fuel-related lap-time effects. |
| `classify_regimes.py` | Classifies driving or acceleration regimes. |
| `estimate_resistance.py`, `fit_resistance.py`, `fit_resistance_physics.py`, `fit_resistance_by_session.py`, `prepare_resistance.py` | Estimates and fits vehicle resistance models. |
| `prepare_physics_data.py`, `apply_physics.py`, `add_longitudinal_force.py` | Prepares or applies physics calculations; `apply_physics.py` is currently empty. |
| `evaluate_acceleration_model.py`, `backtest_predictor.py` | Evaluates and backtests model predictions. |
| `check_*.py` | Diagnostic scripts for columns, resistance, RPM, smoothing, gradients, DRS, vehicle data, and track shape. |

### `backend/simulation/` and `backend/strategy/`

| File | Purpose |
| --- | --- |
| `simulation/simulator.py` | Advances a deterministic race lap by lap, including fuel use and pit stops. |
| `strategy/strategy.py` | Represents pit-stop strategies and evaluates them through the simulator. |

### Other directories

| Directory | Purpose |
| --- | --- |
| `backend/monte_carlo/` | Reserved for Monte Carlo simulation code; package initializer is empty. |
| `backend/weather/` | Reserved for weather-specific code; package initializer is empty. |
| `backend/data/` | Stores raw and external backend datasets. |
| `data/` | Project-level data directory. |
| `docs/` | Project documentation; `interfaces.md` is currently empty. |
| `tests/` | Automated tests for physics, models, simulation, strategy, tyres, and vehicles. |

## Currently Empty Files

These files contain no code or text at present:

```text
backend/__init__.py
backend/data/README.md
backend/models/car.py
backend/models/race.py
backend/models/strategy.py
backend/models/track.py
backend/models/tyre.py
backend/models/weather.py
backend/monte_carlo/__init__.py
backend/physics/__init__.py
backend/physics/apply_physics.py
backend/weather/__init__.py
data/README.md
docs/interfaces.md
tests/__init__.py
```

```text
trace_v0.2/
├── backend/
│   ├── data/
│   │   ├── raw/
│   │   │   ├── 2024/
│   │   │   │   ├── 2024-03-02_Bahrain_Grand_Prix/
│   │   │   │   │   ├── 2024-02-29_Practice_1/
│   │   │   │   │   ├── 2024-02-29_Practice_2/
│   │   │   │   │   ├── 2024-03-01_Practice_3/
│   │   │   │   │   └── 2024-03-02_Race/
│   │   │   │   └── 2024-06-09_Canadian_Grand_Prix/
│   │   │   │       └── ...
│   │   │   └── external/
│   │   │       ├── Bahrain Grand Prix/
│   │   │       └── Canadian Grand Prix/
│   │   └── README.md
│   ├── models/
│   │   ├── car.py
│   │   ├── race.py
│   │   ├── race_state.py
│   │   ├── result.py
│   │   ├── strategy.py
│   │   ├── track.py
│   │   ├── tyre.py
│   │   └── weather.py
│   ├── monte_carlo/
│   ├── physics/
│   │   ├── acceleration.py
│   │   ├── acceleration_model.py
│   │   ├── apply_physics.py
│   │   ├── fuel.py
│   │   ├── lap_time_predictor.py
│   │   ├── mass.py
│   │   ├── parameters.py
│   │   ├── track.py
│   │   ├── track_state.py
│   │   ├── tyre.py
│   │   ├── vehicle.py
│   │   └── ...
│   ├── simulation/
│   │   └── simulator.py
│   ├── strategy/
│   │   └── strategy.py
│   ├── weather/
│   ├── cleaner.py
│   ├── config.py
│   ├── derive_acceleration.py
│   ├── extract_laps.py
│   ├── extract_other.py
│   ├── extract_telemetry.py
│   ├── feature_engineering.py
│   ├── import_external_telemetry.py
│   ├── inspect_track_status.py
│   ├── merge_external_drs.py
│   ├── run_ingestion.py
│   ├── test_load.py
│   ├── track_intervals.py
│   └── verify.py
├── data/
│   └── README.md
├── docs/
│   └── interfaces.md
├── tests/
│   ├── test_acceleration_model.py
│   ├── test_lap_time_predictor.py
│   ├── test_models.py
│   ├── test_simulation.py
│   ├── test_strategy.py
│   ├── test_track_grip.py
│   ├── test_track_state.py
│   ├── test_tyre.py
│   └── test_vehicle.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```
