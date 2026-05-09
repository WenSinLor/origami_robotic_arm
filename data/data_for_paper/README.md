# Data for Paper

This folder contains the curated analysis outputs used to frame the paper story for the origami robotic arm. The files here are primarily publication-ready figures and exported model artifacts generated from the scripts in `src/examples/`.

The data are organized around four experimental questions:

1. Can marker/bending-sensor dynamics classify payload direction and magnitude?
2. Does the classifier generalize across payload mass, arm stiffness, sensor modality, and workspace radius?
3. Can the arm trajectory be predicted autoregressively from previous marker states?
4. Can the same sensing/readout approach estimate center-of-mass offset in the COM demo?

## Folder Hierarchy

```text
data_for_paper/
├── com_demo_100g_outer/
├── com_demo_100g_regression/
├── com_demo_60g/
│   ├── dynamic_summary_feature_vector/
│   └── raw_time_series/
├── com_demo_60g_near/
│   ├── cross_distance_generalization/
│   │   ├── dynamic_summary_feature_vector/
│   │   └── raw_time_series/
│   └── self_test/
│       ├── dynamic_summary_feature_vector/
│       └── raw_time_series/
├── mix_state_20g/
├── soft_state_20g/
├── soft_state_20g_bending_sensor/
├── soft_state_40g/
├── soft_state_40g_bending_sensor/
├── soft_state_100g/
│   └── trajectory_prediction/
│       ├── cross_trajectory_generalization/
│       │   ├── train_sample_0/
│       │   └── train_sample_0-1-2-3/
│       └── self_test/
│           └── coor_0/
├── soft_state_100g_bending_sensor/
├── soft_state_100g_near/
│   ├── cross_radius_generalization/
│   │   ├── coor_0-four_leak/
│   │   ├── coor_0-one_leak/
│   │   ├── coor_0-three_leak/
│   │   ├── coor_0-two_leak/
│   │   └── no_leak/
│   └── self_test/
├── soft_state_100g_near_bending_sensor/
│   ├── cross_radius_generalization/
│   │   ├── coor_0-one_leak/
│   │   ├── coor_0-two_leak/
│   │   └── no_leak/
│   └── self_test/
├── stiff_state_20g/
└── stiff_state_100g/
```

## Naming Conventions

- `soft_state_*`: soft arm state experiments.
- `stiff_state_*`: stiff arm state experiments.
- `mix_state_*`: mixed-state experiments.
- `20g`, `40g`, `100g`: payload mass condition.
- `_near`: near-radius or shifted-radius workspace condition.
- `_bending_sensor`: bending-sensor validation condition.
- `coor_0`, `coor_1`, `coor_2`, `coor_3`: four commanded displacement / payload-direction classes.
- `loo_*`: leave-one-out cross-validation figures.
- `perclass_*`: fixed train/test split figures for each class.
- `raw_time_series`: frame-level raw time-series readout.
- `dynamic_summary_feature_vector`: trial-level dynamic-summary feature readout.

## Repeated Figure Types

Most classification folders contain the same six figure types:

- `loo_confusion_matrix.pdf`: leave-one-out confusion matrix. Use this to show within-condition classification robustness when each sample is held out once.
- `loo_per_sample.pdf`: leave-one-out accuracy by held-out sample. Use this to show whether failures are sample-specific.
- `perclass_confusion_matrix.pdf`: confusion matrix for the configured independent train/test split.
- `perclass_staircase.pdf`: time-resolved predicted readout trajectories, including the trial-level mean decision. This is useful for showing how predictions stabilize over time.
- `perclass_polar.pdf`: predicted class/readout geometry in target space.
- `perclass_pca.pdf`: PCA projection of feature space, useful for showing class separability.

## Payload Classification Blocks

### `soft_state_20g/`, `soft_state_40g/`, `soft_state_100g/`

Main soft-arm payload classification conditions at three masses. These are the core baseline results for showing that the deformation dynamics encode payload direction.

Each folder contains both leave-one-out and per-class train/test figures.

### `stiff_state_20g/`, `stiff_state_100g/`

Stiff-arm comparison conditions. Use these to contrast the sensing/readout behavior between soft and stiff structural states.

### `mix_state_20g/`

Mixed-state 20 g condition. Use this as evidence for performance when state conditions are not purely soft or stiff.

### `soft_state_20g_bending_sensor/`, `soft_state_40g_bending_sensor/`, `soft_state_100g_bending_sensor/`

Bending-sensor validation for the corresponding payload masses. These folders mirror the marker-based payload classification outputs, but for the bending-sensor data stream.

Use these to support the claim that the classification story is not limited to image-marker tracking.

## Near-Radius and Cross-Radius Generalization

### `soft_state_100g_near/self_test/`

Self-test classification for the near-radius 100 g condition. This validates performance when both training and testing are drawn from the near-radius condition.

### `soft_state_100g_near/cross_radius_generalization/`

Cross-radius generalization experiments for the marker-based 100 g near condition.

Subfolders:

- `no_leak/`: no near-radius samples included in training for the tested generalization setting.
- `coor_0-one_leak/`, `coor_0-two_leak/`, `coor_0-three_leak/`, `coor_0-four_leak/`: progressively include more `coor_0` near-radius examples in the training set. Use these to show how a small amount of target-radius calibration changes generalization.

Each subfolder contains the per-class figure set: confusion matrix, staircase, polar plot, and PCA.

### `soft_state_100g_near_bending_sensor/self_test/`

Self-test classification for the near-radius bending-sensor condition.

### `soft_state_100g_near_bending_sensor/cross_radius_generalization/`

Bending-sensor version of the cross-radius generalization experiment.

Subfolders:

- `no_leak/`
- `coor_0-one_leak/`
- `coor_0-two_leak/`

Use these as the bending-sensor counterpart to the marker-based cross-radius results.

## Trajectory Prediction

### `soft_state_100g/trajectory_prediction/self_test/`

Autoregressive trajectory prediction on the same configured trajectory condition.

Files:

- `traj_base_sample_0.pdf`: 2D marker trajectory prediction. Solid colored curves are autoregressive predictions; gray/reference curves show the measured trajectory.
- `phase_base_sample_0.pdf`: predicted vs true marker displacement over time, separated by marker and coordinate.

### `soft_state_100g/trajectory_prediction/cross_trajectory_generalization/`

Autoregressive trajectory prediction under cross-trajectory train/test settings.

Subfolders:

- `train_sample_0/`: model trained using sample 0, then tested on other samples.
- `train_sample_0-1-2-3/`: model trained using samples 0-3, then tested on held-out samples.

File meanings:

- `traj_base_sample_*.pdf`: predicted and true 2D marker paths.
- `phase_base_sample_*.pdf`: predicted and true displacement traces over time.
- `base_sample_*_fft.pdf`: frequency-domain diagnostic for selected test samples.

Use this block to show whether learned motion dynamics roll out beyond the seed frames and how generalization improves with more training trajectories.

## COM Demo Blocks

### `com_demo_60g/dynamic_summary_feature_vector/`

COM readout for the 60 g demo using trial-level dynamic-summary features.

Files:

- `dynamic_summary_confusion_matrix.pdf`: classification of positive and negative COM offset.
- `dynamic_summary_readout_points.pdf`: predicted readout points relative to target offset locations.

### `com_demo_60g/raw_time_series/`

COM readout for the 60 g demo using frame-level raw time-series logic, matching the payload classifier style.

Files:

- `raw_confusion_matrix.pdf`: confusion matrix from the raw time-series readout.
- `raw_readout_points.pdf`: predicted readout points from the raw time-series readout.

Use this pair with the dynamic-summary folder to compare frame-level raw readout vs compact trial-level summary features.

### `com_demo_60g_near/self_test/`

COM readout for the 60 g near condition when training/testing are drawn from the near condition.

Subfolders:

- `dynamic_summary_feature_vector/`
- `raw_time_series/`

These mirror the 60 g COM outputs and can be used to show near-condition self-test performance.

### `com_demo_60g_near/cross_distance_generalization/`

COM readout under cross-distance generalization.

Subfolders:

- `dynamic_summary_feature_vector/`: dynamic-summary confusion matrix and readout points.
- `raw_time_series/`: raw time-series confusion matrix and readout points.

Additional file:

- `dynamic_summary_ridge_weights.mat`: exported Ridge readout weights and normalization data for the dynamic-summary model.

Use this block to support the COM generalization story across distance/radius conditions.

### `com_demo_100g_outer/`

Actual bar-outer COM demo condition. This block treats the outer-bar case as a `+4` offset target and is intended to support the real-time demo interpretation.

Training is from the 60 g bar-base dynamic-summary readout (`bar_base/coor_0 -> +3`, `bar_base/coor_2 -> -3`). Testing is the actual outer-bar positive-side demo (`bar_outer/coor_0 -> +4`). There is intentionally no `-4` target in this condition.

Files:

- `bar_outer_confusion_matrix.pdf`: confusion matrix for decoding the actual `+4` outer-bar trials against the allowed targets.
- `bar_outer_readout_points.pdf`: final trial-level predicted offset/readout points relative to the trained `-3`, trained `+3`, and actual-demo `+4` target locations.
- `bar_outer_streaming_readout.pdf`: streaming predicted offset over elapsed time. The model is trained once, then each test trial is evaluated using progressively more frames to mimic a live demo. The plot marks the `+4` target, the `+/-0.5` target band, and the time when the mean predicted offset first enters that band.

Use this block as the clearest demo-facing result: it shows whether the trained readout moves toward the unseen outer-bar `+4` target and how quickly the streaming estimate reaches the target band.

### `com_demo_100g_regression/`

Scalar COM-offset regression experiment.

File:

- `scalar_regression_scatter.png`: continuous predicted offset vs true offset. This figure shows whether the scalar regressor can interpolate across offset magnitudes (train with -3, -1, +1, +2 ; test with -2, +2).

Note: this plot reflects the configured test set in `train_com_demo_regression.py`; if the test set only contains near-condition samples, only the corresponding true offsets appear on the x-axis.

## Recommended Paper Usage

- Use the payload classification folders to establish the main sensing/classification result.
- Use the bending-sensor folders to show that the method transfers beyond visual marker tracking.
- Use near-radius and cross-radius folders to discuss generalization and calibration.
- Use trajectory prediction to support the broader claim that the arm dynamics are learnable, not only classifiable.
- Use COM demo folders to show a second task: mapping dynamics to physically meaningful center-of-mass offset.

## Script Provenance

Primary scripts used to generate these results:

- `src/examples/train_payload_classifier.py`: per-class payload classification plots.
- `src/examples/train_payload_loo_cv.py`: leave-one-out payload classification plots.
- `src/examples/trajectory_predictor.py`: autoregressive trajectory prediction plots.
- `src/examples/trajectory_predictor_failure_analysis.py`: trajectory prediction diagnostics.
- `src/examples/train_com_demo.py`: COM classification/readout plots.
- `src/examples/train_com_demo_regression.py`: scalar COM-offset regression plot.
