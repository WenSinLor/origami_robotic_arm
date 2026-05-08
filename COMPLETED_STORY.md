# Completed Story: Contact-Parametric Ringdown Sensing in an Origami Arm

This update completes the original PDF story with the data now present in `data/`, especially the bending-sensor validations, the sensor-count ablation, and the COM demo.

## Core Thesis

A bistable origami arm can be used as a contact-parametric physical reservoir: a fixed impulse produces a ringdown trajectory whose shape is systematically changed by hidden payload mass, angular payload location, grip radius, and payload COM. The information is recoverable with simple linear readouts from either camera-tracked markers or embedded bending sensors. The strongest final claim is not full invariance across all payloads, radii, and configurations; the defensible claim is structured decodability plus useful transfer when the readout is calibrated to the relevant mechanical regime.

## Updated Story Status

| Story | Updated status | What the data now supports |
| --- | --- | --- |
| Story 1: coupling mechanism | Complete | Loaded trials depart from the empty-box baseline after amplitude normalization. The window-sensitivity summary shows monotone shape-deviation means at the robust 0.8 s normalization window: empty `0.077`, 20 g `0.980`, 40 g `1.319`, 100 g `11.036`. |
| Story 2: configuration dependence | Complete, with refined wording | Decoding survives configuration changes, but quality is not configuration-invariant. In recomputed all-common LOO-CV, optical decoding remains above chance for soft, stiff, and mixed states. At 20 g, mixed is strongest (`89.6%`), soft is `78.8%`, and stiff is `75.0%`. At 100 g, both soft and stiff are strong (`91.7%` and `93.8%` in this recomputation), while earlier continuous-error plots should still be used to discuss finer soft/stiff differences. |
| Story 3a: angular position | Complete | Angular payload position is decodable from ringdown dynamics. Optical marker LOO-CV gives soft 20 g `78.8%`, soft 40 g `91.7%`, and soft 100 g `91.7%`; bending sensors strengthen this trend at higher mass: 20 g `67.5%`, 40 g `85.0%`, 100 g `97.5%`. |
| Story 3b: radial generalization | Complete, but phrase as partial generalization | Optical marker readouts transfer above chance between outer and nearer radii: outer-to-near `70.8%`, near-to-outer `72.9%`, chance `25%`. Bending sensors are excellent within-radius at 100 g (`97.5%` outer and `97.5%` near) but weak under zero-shot radius transfer (`46.3%` outer-to-near; `22.5%` outer-to-nearest). The right claim is partial radial generalization, not radius invariance. |
| Sensor-count ablation | Complete | The full 19-marker optical readout reaches `91.7%` LOO-CV in the ablation bundle. Random marker removal degrades average accuracy quickly: 18 markers `63.7%`, 17 markers `53.4%`, 13 markers `39.8%`. Some subsets still reach high accuracy, so information is partly redundant but strongly sensor-placement dependent. |
| Story 4: COM localization demo | Complete | The 60 g bar-payload bending-sensor demo works with a dynamic-summary ridge readout. Training on outer bar grips (`+/-3`) and testing on nearer bar grips (`+/-2`) gives `92.5%` exact offset classification, `97.5%` sign accuracy, and `0.15` cell MAE. This completes the functional demo, with the caveat that point-payload calibration does not transfer directly to bar payloads. |

## Completed Narrative

The work should now be framed as a progression from mechanism to sensing function.

First, the empty-box-referenced analysis establishes that the payload is not merely adding noise or changing response amplitude. After amplitude normalization, the ringdown shape changes with loading. This is the experimental counterpart of the theory: payload mass and placement perturb the effective inertia and gravity/stiffness landscape, and those perturbations are visible in the measured transient response.

Second, angular payload position is recoverable from that transient response. The optical marker datasets show reliable four-class angular decoding across payload masses, with the expected weakening at lower mass. The bending-sensor datasets now add an important validation: the same reservoir state can be read electrically, not only by camera. In the soft 100 g condition, bending sensors reach `97.5%` LOO-CV at the outer radius and `97.5%` at the nearer radius, which is the cleanest evidence that embedded sensing can replace external tracking for the main angular task.

Third, the configuration story should be presented as robustness with dependence, not invariance. The arm remains informative in soft, stiff, and mixed configurations, but the readout quality changes. The most defensible 20 g statement is that mixed is competitive or strongest in the recomputed LOO result, while stiff is not catastrophic but remains configuration-sensitive. For 100 g, both soft and stiff optical conditions are strong in discrete LOO-CV; use the earlier continuous MSE figures to discuss whether soft has lower continuous readout error in that specific split.

Fourth, radius transfer is real but limited. Optical marker data show above-chance outer-near transfer, so the angular representation is not tied completely to one radius. The bending sensors sharpen the interpretation: within-radius bending-sensor decoding is excellent, but zero-shot cross-radius bending-sensor transfer is much weaker. This means the body creates structured state spaces at each radius, but a readout trained at one radius should not be claimed to be radius-invariant without calibration or radius-aware supervision.

Fifth, the sensor-count ablation is no longer pending. It shows why the marker array matters: the full marker set is strong, random removals degrade the average quickly, but selected subsets can remain good. The paper should use this as a practical sensing-design result: the reservoir signal is distributed, but not uniformly distributed.

Finally, Story 4 is now the application endpoint. The bar COM demo shows that the same ringdown principle can move beyond abstract four-way classification into signed COM/offset localization for an occluded payload. The successful 60 g bar protocol is not a universal payload-transfer result; a point-payload-trained readout transferred poorly to bar data. The correct claim is stronger and cleaner: with bar-specific calibration, bending-sensor ringdown features localize the bar COM with high exact offset accuracy and near-perfect left/right sign accuracy.

## Recommended Final Claim Wording

Use this wording in the paper:

> These results demonstrate a contact-parametric physical reservoir in which hidden payload state is encoded into the free ringdown dynamics of a bistable origami arm. The encoding is observable through both camera markers and embedded bending sensors, supports payload mass and angular-position inference, survives changes in bistable configuration with condition-dependent quality, partially transfers across grip radius, and enables a calibrated COM-localization demo for an occluded bar payload.

Avoid these stronger claims:

- Do not claim radius invariance; say partial radial generalization.
- Do not claim all configurations are equivalent; say decodability is retained with configuration-dependent quality.
- Do not claim point-payload calibration directly generalizes to bar payloads; the bar demo needs bar-specific calibration.
- Do not claim bending sensors solve every condition; stiff 100 g bending-sensor LOO-CV is weak (`40.0%`) and should be treated as a sensor-placement/readout limitation or a separate failure mode.

## Figure/Data Mapping

Use these existing outputs as the clean figure sequence:

- Mechanism / Story 1: `data/soft_state_noload/story1_shape_curves.png`, `story1_raw_vs_shape.png`, `story1_window_sensitivity.csv`.
- Optical angular decoding: `data/soft_state_20g`, `data/soft_state_40g`, `data/soft_state_100g`, and `data/soft_state_100g_near` confusion/PCA/staircase figures.
- Bending-sensor validation: `data/soft_state_20g_bending_sensor`, `data/soft_state_40g_bending_sensor`, `data/soft_state_100g_bending_sensor`, `data/soft_state_100g_near_bending_sensor`.
- Modal/radius support: `data/modal_validation_improved__soft_state_100g__vs__soft_state_100g_near/aggregate_summary.json`.
- Sensor-count ablation: `data/ablation_study/loo_ablation/ablation_curve_loo.png`, `ablation_lcs_guided_loo.png`.
- COM demo: `data/com_demo_60g_near/dynamic_summary_confusion_matrix.png`, `dynamic_summary_readout_points.png`, `bar_transfer_confusion.png`, `bar_transfer_predictions.png`.
- Continuous offset bridge: `data/com_bridge_demo_output_2d/offset_confusion_test.png`, `pred_xy_test.png`.

## Numbers Used In This Update

The headline metrics above were recomputed from the HDF5 files using the repo's existing modeling convention: baseline-subtracted trajectories, 0-3 s window, z-scored features, linear or ridge readout, and held-out trial evaluation. The PDF's older split-specific figures can still be used visually, but the updated story should prefer these recomputed LOO/held-out numbers when giving exact claims.
