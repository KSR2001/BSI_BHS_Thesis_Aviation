# Digital Twin and Machine-Learning-based ICS/OT Cybersecurity and Anomaly Detection in Airport Baggage Handling System

This repository contains the Part 1 of the code and generated research artifacts for a master's thesis prototype on **Digital Twin and Machine-Learning-based ICS/OT Cybersecurity and Anomaly Detection in Airport Baggage Handling System**.

Part 2 (Unity 3D Digital Twin Project Repository) of this Master's thesis can be found here : 


## Demo Video

Click the image below to watch the digital twin demonstration.

[![Watch the demo video](docs/demo_thumbnail.jpg)](https://drive.google.com/file/d/1iV6qbu4FeMz8gd3BeDgkF0C8kQvjsYWd/view?usp=sharing)


The project combines:

- a SimPy-based discrete-event simulation of an outbound airport BHS,
- synthetic event logs and fixed-interval telemetry generation,
- process-aware spatio-temporal feature engineering,
- an LSTM Autoencoder for reconstruction-based anomaly detection,
- a Temporal CNN for supervised multi-class attack classification,
- online proxy scripts for connecting trained models to a Unity digital twin visualization.

This is a research prototype. It uses synthetic simulation data, not real airport operational data.

## Project Overview

Airport baggage handling systems are cyber-physical systems that combine conveyors, sensors, diverters, routing logic, manual encoding, early-bag storage, and supervisory monitoring. Cyber-physical attacks or abnormal operating conditions can appear as process-level effects such as queue buildup, reduced throughput, abnormal sensor activity, diverter routing mismatches, or stopped conveyor segments.

This project investigates whether a simulation-driven digital twin can generate useful telemetry for machine-learning-based anomaly detection and attack classification. The simulation produces both normal and attack scenarios. These logs are transformed into fixed-length 60-second windows with 78 engineered features.

Two deep learning approaches are included:

1. **LSTM Autoencoder**: trained mainly on normal behavior and used for binary anomaly detection through reconstruction error.
2. **Temporal CNN**: trained as a supervised classifier over five classes: `normal`, `dos`, `spoof`, `fdi`, and `stopped_conv`.

The online part of the project streams SimPy events to a proxy, rebuilds rolling feature windows in real time, applies the trained model, and forwards process events, predictions, and alerts to Unity.

## Repository Structure

```text
`-- Simpy_models/
    |-- bhs_sim_behavioral.py
    |-- 02_build_features_lstm.py
    |-- 03_train_lstm_ae.py
    |-- 03_train_temporal_cnn_cls.py
    |-- 04_evaluate_lstm_ae.py
    |-- 04_evaluate_temporal_cnn_cls.py
    |-- 05_online_lstm_ae_proxy.py
    |-- 05_online_temporal_cnn_proxy.py
    |-- 06_make_quantile_threshold.py
    |-- inspect_npz.py
    |-- inspect_scaler.py
    |-- plot_lstm_ae_training_curve.py
    |-- plot_temporal_cnn_training_curve.py
    |-- viz_raw_csvs.py
    |-- data/
    |   |-- raw/
    |   |   |-- manifest.csv
    |   |   |-- normal_run*.csv
    |   |   |-- dos_run*.csv
    |   |   |-- spoof_run*.csv
    |   |   |-- fdi_run*.csv
    |   |   `-- stopped_conv_run*.csv
    |   `-- lstm_windows.npz
    |-- models_lstm/
    |   |-- lstm_ae.pt
    |   |-- scaler.npz
    |   |-- config.json
    |   |-- thresholds.json
    |   |-- thresholds_quantile_q0995.json
    |   `-- thresholds_quantile_q0999.json
    |-- models_tcnn_named/
    |   |-- temporal_cnn.pt
    |   |-- scaler.npz
    |   `-- config.json
    |-- plots_lstm/
    |-- plots_lstm_q0995/
    |-- plots_lstm_q0999/
    |-- plots_tcnn_named_perm/
    |-- plots_lstm_training_diagnostics*/
    |-- plots_tcnn_training_diagnostics_clean/
    `-- plots_raw/
|-- requirements.txt
|-- README.md
|-- .gitignore
```



## Main Scripts

| Script | Purpose |
| --- | --- |
| `bhs_sim_behavioral.py` | SimPy BHS simulation, scenario generation, CSV logging, and live TCP streaming |
| `02_build_features_lstm.py` | Converts raw logs into 60-second, 78-feature sliding windows |
| `03_train_lstm_ae.py` | Trains the LSTM Autoencoder anomaly detector |
| `04_evaluate_lstm_ae.py` | Evaluates LSTM-AE anomaly detection and creates plots/CSV summaries |
| `03_train_temporal_cnn_cls.py` | Trains the supervised Temporal CNN classifier |
| `04_evaluate_temporal_cnn_cls.py` | Evaluates Temporal CNN classification and creates thesis plots/metrics |
| `05_online_lstm_ae_proxy.py` | Online LSTM-AE proxy between SimPy and Unity |
| `05_online_temporal_cnn_proxy.py` | Online Temporal CNN proxy between SimPy and Unity |
| `06_make_quantile_threshold.py` | Generates optional quantile thresholds for LSTM-AE deployment-style calibration |
| `viz_raw_csvs.py` | Creates plots from raw simulation CSV files |

## Installation

### Option A: Conda Environment

```bash
conda create -n bhs_thesis_env python=3.10 -y
conda activate bhs_thesis_env
pip install -r requirements.txt
```

### Option B: Python Virtual Environment

```bash
python -m venv .venv
```

On Windows:

```cmd
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Then install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dependencies

The project uses the packages listed in `requirements.txt`:

```text
simpy
numpy
pandas
matplotlib
scikit-learn
joblib
torch
tensorflow
```

## Working Directory

Most commands should be run from inside the `Simpy_models` folder:

```cmd
cd Simpy_models
```

Example path:

```cmd
(bhs_thesis_env) F:\BSI_BHS_Thesis_Aviation\Simpy_models>
```

## Real-Time Unity Digital Twin Workflow

The real-time setup uses three runtime components:

1. **SimPy simulator** streams process events and telemetry to TCP port `8765`.
2. **Online proxy** receives the SimPy stream, rebuilds rolling 60-second feature windows, applies the trained model, and forwards events/predictions/alerts.
3. **Unity digital twin** connects to the proxy output port and visualizes baggage movement, component states, predictions, and alerts.

Recommended startup order:

1. Start the SimPy simulator in Terminal 1.
2. Start either the LSTM-AE proxy or the Temporal CNN proxy in Terminal 2.
3. Configure Unity to connect to the correct proxy port.
4. Press Play in Unity.

Port summary:

| Component | Host | Port | Purpose |
| --- | --- | ---: | --- |
| SimPy stream | `127.0.0.1` | `8765` | Upstream simulator stream |
| LSTM-AE proxy output | `127.0.0.1` | `9001` | Unity connection for anomaly alerts |
| Temporal CNN proxy output | `127.0.0.1` | `9002` | Unity connection for class predictions and alerts |

The multi-line proxy commands below use Windows CMD line continuation with `^`. In PowerShell, either run the command on one line or replace `^` with PowerShell's backtick continuation character.



## Example Real-Time Demonstration Sequences

### LSTM-AE Anomaly Detection Demo

Terminal 1:

```cmd
python bhs_sim_behavioral.py --mode attack --attacks fdi --fdi-flip-prob 0.7 --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

Terminal 2:

```cmd
python 05_online_lstm_ae_proxy.py ^
  --modeldir models_lstm ^
  --upstream-host 127.0.0.1 --upstream-port 8765 ^
  --listen-host 127.0.0.1 --listen-port 9001 ^
  --telemetry-dt 1.0 --window 60 ^
  --warmup-sec 120 ^
  --calib-sec 0 ^
  --online-thr-mult 1.05 ^
  --require-consecutive 3 ^
  --cooldown-sec 3 ^
  --log1p-counts ^
  --reset-on-new-run
```

Unity:

```text
Host: 127.0.0.1
Port: 9001
```

### Temporal CNN Attack Classification Demo

Terminal 1:

```cmd
python bhs_sim_behavioral.py --mode attack --attacks dos --dos-speed-factor 0.2 --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

Terminal 2:

```cmd
python 05_online_temporal_cnn_proxy.py ^
  --modeldir models_tcnn_named ^
  --upstream-host 127.0.0.1 --upstream-port 8765 ^
  --listen-host 127.0.0.1 --listen-port 9002 ^
  --telemetry-dt 1.0 --window 60 --stride 5 ^
  --warmup-sec 120 ^
  --thr-conf 0.75 ^
  --require-consecutive 3 ^
  --cooldown-sec 2 ^
  --log1p-counts ^
  --reset-on-new-run ^
  --heuristic-stopped --verbose --send-predictions ^
  --stop-lookback 15 --stop-zero-frac 0.8 --stop-queue-min 1.0 --stop-exits-max 0.2
```

Unity:

```text
Host: 127.0.0.1
Port: 9002
```

## Notes on Unity Integration

Unity is the visualization endpoint of the digital twin loop. The Unity scene should connect to the proxy, not directly to SimPy.

Use:

- port `9001` for LSTM-AE anomaly detection demonstrations,
- port `9002` for Temporal CNN classification demonstrations.

The Unity side should receive process events, prediction messages, and alert messages. Process events are used to animate bags and BHS components. Prediction and alert messages are used to visualize detected anomalies or predicted attack classes.

If Unity does not receive messages:

1. Check that SimPy is running and streaming to `127.0.0.1:8765`.
2. Check that the proxy has connected to SimPy.
3. Check that Unity is connected to the proxy output port, not the SimPy port.
4. Check that no other process is already using ports `8765`, `9001`, or `9002`.
5. Restart the components in this order: SimPy, proxy, Unity.



## SimPy Streaming Commands

Run one of these commands in Terminal 1 from the `Simpy_models` folder.

### Normal Run

```cmd
python bhs_sim_behavioral.py --mode normal --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

### DoS Attack

```cmd
python bhs_sim_behavioral.py --mode attack --attacks dos --dos-speed-factor 0.2 --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

### Spoofing Attack

```cmd
python bhs_sim_behavioral.py --mode attack --attacks spoof --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

### False Data Injection Attack

```cmd
python bhs_sim_behavioral.py --mode attack --attacks fdi --fdi-flip-prob 0.7 --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

### Stopped Conveyor Attack

```cmd
python bhs_sim_behavioral.py --mode attack --attacks stopped_conv --attack-start 200 --attack-duration 100 --arrival-rate 0.2 --runtime 600 --telemetry-dt 1.0 --realtime 1.0 --stream 127.0.0.1:8765 --stream-progress
```

## LSTM-AE Online Proxy for Unity

The LSTM-AE proxy sends output to Unity on port `9001`.

Unity connection:

```text
Host: 127.0.0.1
Port: 9001
```

### LSTM-AE Proxy for Normal Calibration Run

Use this mode when the SimPy run is normal and the proxy should calibrate a high-quantile reconstruction-error threshold.

```cmd
python 05_online_lstm_ae_proxy.py ^
  --modeldir models_lstm ^
  --upstream-host 127.0.0.1 --upstream-port 8765 ^
  --listen-host 127.0.0.1 --listen-port 9001 ^
  --telemetry-dt 1.0 --window 60 ^
  --warmup-sec 120 ^
  --calib-sec 300 ^
  --calib-quantile 0.999 ^
  --calib-min-n 60 ^
  --online-thr-mult 1.05 ^
  --require-consecutive 3 ^
  --cooldown-sec 3 ^
  --log1p-counts ^
  --reset-on-new-run
```

### LSTM-AE Proxy for Attack Run

Use this mode for attack demonstrations. Calibration is disabled with `--calib-sec 0` so attack scores do not contaminate the normal baseline.

```cmd
python 05_online_lstm_ae_proxy.py ^
  --modeldir models_lstm ^
  --upstream-host 127.0.0.1 --upstream-port 8765 ^
  --listen-host 127.0.0.1 --listen-port 9001 ^
  --telemetry-dt 1.0 --window 60 ^
  --warmup-sec 120 ^
  --calib-sec 0 ^
  --online-thr-mult 1.05 ^
  --require-consecutive 3 ^
  --cooldown-sec 3 ^
  --log1p-counts ^
  --reset-on-new-run
```

## Temporal CNN Online Proxy for Unity

The Temporal CNN proxy sends output to Unity on port `9002`.

Unity connection:

```text
Host: 127.0.0.1
Port: 9002
```

Run this command in Terminal 2 after starting SimPy:

```cmd
python 05_online_temporal_cnn_proxy.py ^
  --modeldir models_tcnn_named ^
  --upstream-host 127.0.0.1 --upstream-port 8765 ^
  --listen-host 127.0.0.1 --listen-port 9002 ^
  --telemetry-dt 1.0 --window 60 --stride 5 ^
  --warmup-sec 120 ^
  --thr-conf 0.75 ^
  --require-consecutive 3 ^
  --cooldown-sec 2 ^
  --log1p-counts ^
  --reset-on-new-run ^
  --heuristic-stopped --verbose --send-predictions ^
  --stop-lookback 15 --stop-zero-frac 0.8 --stop-queue-min 1.0 --stop-exits-max 0.2
```


## Dataset Generation Workflow

### 1. Generate Raw Simulation Runs

This command generates 12 runs for each scenario and writes raw CSV logs plus a manifest file.

```cmd
python bhs_sim_behavioral.py --generate --runs-per-case 12 --arrival-rate 0.20 --runtime 900 --telemetry-dt 1.0 --seed 42 --randomize-attack-window --out-dir data/raw --manifest data/raw/manifest.csv
```

Generated scenarios:

- `normal`
- `fdi`
- `spoof`
- `dos`
- `stopped_conv`

Main output:

```text
data/raw/manifest.csv
data/raw/normal_run01.csv ... normal_run12.csv
data/raw/fdi_run01.csv ... fdi_run12.csv
data/raw/spoof_run01.csv ... spoof_run12.csv
data/raw/dos_run01.csv ... dos_run12.csv
data/raw/stopped_conv_run01.csv ... stopped_conv_run12.csv
```

### 2. Build Feature Windows

```cmd
python 02_build_features_lstm.py --manifest data/raw/manifest.csv --out data/lstm_windows.npz --window 60 --stride 5 --test-size 0.30 --val-size 0.20 --random-state 42 --label-mode attack_window --log1p-counts
```

This creates:

```text
data/lstm_windows.npz
```

The generated dataset uses:

- window length: 60 seconds,
- stride: 5 seconds,
- number of features: 78,
- label mode: `attack_window`,
- count feature transform: `log1p`.

Dataset split from the current run:

| Split | Shape | Label Counts |
| --- | --- | --- |
| Train | `(6760, 60, 78)` | normal: 4635, dos: 528, spoof: 548, fdi: 497, stopped_conv: 552 |
| Validation | `(845, 60, 78)` | normal: 541, dos: 87, spoof: 77, fdi: 78, stopped_conv: 62 |
| Test | `(2535, 60, 78)` | normal: 1727, dos: 266, spoof: 173, fdi: 183, stopped_conv: 186 |

## LSTM Autoencoder Workflow

The LSTM Autoencoder is used for binary anomaly detection. It learns normal behavior and flags windows with high reconstruction error.

### Train LSTM-AE

```cmd
python 03_train_lstm_ae.py --data data/lstm_windows.npz --outdir models_lstm --epochs 60 --batch 128 --lr 1e-3 --thr-mode f1
```

Main outputs:

```text
models_lstm/lstm_ae.pt
models_lstm/scaler.npz
models_lstm/config.json
models_lstm/thresholds.json
```

Training summary from the current run:

```text
[thr] mode=f1 best_val: thr=0.831796 f1=0.790 precision=0.730 recall=0.862
[save] model+scaler+thresholds+config -> models_lstm/
```

### Evaluate LSTM-AE

```cmd
python 04_evaluate_lstm_ae.py --data data/lstm_windows.npz --modeldir models_lstm --plotdir plots_lstm
```

Current test results:

```text
[binary CM]
 [[1329  398]
 [ 162  646]]
[binary] precision=0.619 recall=0.800 f1=0.698 thr=0.831796
[prevalence] attack_rate=0.319 baseline_PR=0.319
[curves] PR-AUC=0.598 ROC-AUC=0.801
```

Main outputs:

```text
plots_lstm/metrics_summary.csv
plots_lstm/confusion_matrix_binary.png
plots_lstm/pr_curve.png
plots_lstm/roc_curve.png
plots_lstm/score_boxplot.png
plots_lstm/per_scenario_flag_rate.csv
plots_lstm/threshold_sweep.csv
```

### Optional Quantile Thresholds

For deployment-style normal-only threshold calibration:

```cmd
python 06_make_quantile_threshold.py --data data/lstm_windows.npz --modeldir models_lstm --q 0.995
```

```cmd
python 06_make_quantile_threshold.py --data data/lstm_windows.npz --modeldir models_lstm --q 0.999
```

The repository currently contains:

```text
models_lstm/thresholds_quantile_q0995.json
models_lstm/thresholds_quantile_q0999.json
```

## Temporal CNN Workflow

The Temporal CNN is used for supervised multi-class classification over:

```text
normal, dos, spoof, fdi, stopped_conv
```

### Train Temporal CNN

```cmd
python 03_train_temporal_cnn_cls.py --data data/lstm_windows.npz --outdir models_tcnn_named --epochs 60 --batch 256 --lr 1e-3 --weighting balanced --seed 42
```

Training summary from the current run:

```text
[epoch 01] train_loss=0.517603 val_acc=0.7361 val_macro_f1=0.5695
[epoch 02] train_loss=0.186388 val_acc=0.8533 val_macro_f1=0.6785
[epoch 03] train_loss=0.118725 val_acc=0.8663 val_macro_f1=0.6977
[epoch 04] train_loss=0.082245 val_acc=0.8462 val_macro_f1=0.6572
[epoch 05] train_loss=0.061393 val_acc=0.8722 val_macro_f1=0.7019
[epoch 06] train_loss=0.043269 val_acc=0.8663 val_macro_f1=0.6812
[epoch 07] train_loss=0.037643 val_acc=0.8604 val_macro_f1=0.6729
[epoch 08] train_loss=0.031073 val_acc=0.8556 val_macro_f1=0.6613
[epoch 09] train_loss=0.026854 val_acc=0.8521 val_macro_f1=0.6656
[epoch 10] train_loss=0.023713 val_acc=0.8497 val_macro_f1=0.6587
[epoch 11] train_loss=0.019890 val_acc=0.8675 val_macro_f1=0.6786
[epoch 12] train_loss=0.020027 val_acc=0.8710 val_macro_f1=0.6852
[epoch 13] train_loss=0.019140 val_acc=0.8710 val_macro_f1=0.6861
[early-stop] patience=8
[save] model -> models_tcnn_named\temporal_cnn.pt
[save] scaler+config -> models_tcnn_named/
[labels] {0: 'normal', 1: 'dos', 2: 'spoof', 3: 'fdi', 4: 'stopped_conv'}
```

Main outputs:

```text
models_tcnn_named/temporal_cnn.pt
models_tcnn_named/scaler.npz
models_tcnn_named/config.json
```

### Evaluate Temporal CNN

```cmd
python 04_evaluate_temporal_cnn_cls.py --data data/lstm_windows.npz --modeldir models_tcnn_named --plotdir plots_tcnn_named_perm --importance permutation --imp-subsample 1500 --perm-repeats 3
```

Current test results:

```text
[test] accuracy= 0.8781 macro_f1= 0.8084 weighted_f1= 0.8835
[speed] ms/window= 0.121 windows/sec= 8296.53
[delay] min_detection_delay_seconds= 60.0
[save] thesis plots + csv + json -> plots_tcnn_named_perm
```

Main outputs:

```text
plots_tcnn_named_perm/metrics_tcnn_full.json
plots_tcnn_named_perm/classification_report.csv
plots_tcnn_named_perm/confusion_matrix_raw.png
plots_tcnn_named_perm/confusion_matrix_normalized.png
plots_tcnn_named_perm/pr_curves_ovr.png
plots_tcnn_named_perm/pr_auc_ovr.csv
plots_tcnn_named_perm/per_class_metrics.csv
plots_tcnn_named_perm/per_class_bars_precision_recall_f1.png
plots_tcnn_named_perm/feature_importance_permutation.csv
plots_tcnn_named_perm/feature_importance_permutation_top25.png
plots_tcnn_named_perm/misclassified_samples.csv
plots_tcnn_named_perm/misclassification_pairs.csv
```


## Feature Schema

The trained models use 78 features. The feature groups include:

- queue length per conveyor/component,
- busy state per conveyor/component,
- exit counts per conveyor,
- mean transit time per conveyor,
- diverter decision counts,
- diverter mismatch counts,
- main sensor hit count,
- total throughput,
- total queue,
- total busy components.

The full feature schema is saved in the model config files and plot outputs:

```text
models_lstm/config.json
models_tcnn_named/config.json
plots_lstm/feature_schema.csv
plots_lstm_q0995/feature_schema.csv
plots_lstm_q0999/feature_schema.csv
```

## Research Status

This repository demonstrates a controlled experimental pipeline for simulation-driven BHS cybersecurity research:

1. Generated synthetic BHS behavior.
2. Converted process logs into spatio-temporal windows.
3. Trained and evaluated anomaly detection and attack classification models.
4. Connected live simulation output to Unity through online inference proxies.

The current implementation should be interpreted as a thesis research prototype, not a validated deployment for real airport operations.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Author

Kuldeep Singh  
M.Sc. Geoinformatics and Spatial Data Science  
University of Muenster, Germany
