# Experiment Protocol

## Objective

Demonstrate a measurable improvement while preserving a fair comparison with the supplied five-class baseline, then build a separate 21-class portfolio extension.

## Track A: direct baseline improvement

The following values are fixed:

- Same five classes and 500 images
- Same persisted train/validation/test manifest
- Same 350/75/75 split
- Same test set, evaluated once after model selection
- Seed 42 for the initial parity run
- Accuracy and macro F1 as mandatory comparison metrics

The accepted reference is 74.67% test accuracy and 0.733 macro F1 from the 102,277-parameter custom CNN. First reproduce it within reasonable stochastic variation. Then compare transfer learning and fine-tuning on the identical manifest.

Because the historical manifest contains visually related scenes across split boundaries, report a second five-class sensitivity experiment using a group-aware manifest. The historical result answers “did the model improve on the supplied assignment?”; the group-aware result answers “does the improvement survive a more credible leakage control?” Never merge or substitute these two claims.

## Track B: 21-class extension

Use all 2,100 images and create a separately named, stratified, group-aware manifest. Before splitting, detect exact hashes and perceptual near-duplicates; manually verified related samples must stay in the same group. Report accuracy, macro F1, balanced accuracy, top-3 accuracy, per-class metrics, confusion matrix, calibration, and error examples.

Track B metrics must never be described as a direct improvement over Track A because the prediction task is more difficult.

## Model-selection rules

1. Training data fits model parameters.
2. Validation data selects checkpoints and hyperparameters.
3. Test data remains untouched until the experiment is frozen.
4. Every reported result names its manifest, seed, model version, and selected epoch.
5. Failed and unfavorable runs remain in the experiment log.
6. No expected or desired accuracy may be presented as an achieved result.

## Required submission evidence

- Clean Colab notebook runnable from a fresh runtime
- Dataset source and acquisition metadata
- Versioned experiment configuration
- Split manifest or deterministic manifest-generation cell
- Baseline and improved model table
- Training curves and per-class evaluation
- Misclassification analysis
- Limitations and responsible-use statement
- Git commit identifier for the submitted notebook
