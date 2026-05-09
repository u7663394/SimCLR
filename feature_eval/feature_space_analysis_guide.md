# Feature Space Visualization Guide

This analysis compares two encoders on the same CIFAR-10 test images:

- Baseline SimCLR encoder trained with the projection head.
- Randomly initialized ResNet encoder as an untrained control.

The script produces qualitative t-SNE plots for the encoder feature space.

## Run

Run from the repository root after activating the project environment:

```bash
python feature_eval/visualize_feature_space.py \
  --baseline-result feature_eval/results/projection_head_yes.json \
  --data ./datasets_local \
  --sample-size 1000 \
  --device auto
```

## Outputs

All outputs are written to `feature_eval/report/feature_space_analysis/`.

- `tsne_comparison.png`: side-by-side t-SNE feature-space comparison.
- `tsne_baseline_with_projection_head.png`: baseline-only t-SNE plot.
- `tsne_random_initialization.png`: random-encoder-only t-SNE plot.
- `tsne_coordinates.csv`: t-SNE coordinates if the scatter plot needs to be customized.
- `feature_space_analysis_report.md`: generated report draft with figures and interpretation.

## Suggested Report Framing

Use the t-SNE plot as a qualitative visualization of global class structure:
clearer same-class clusters suggest a more semantically organized feature space.

The random initialization is important because it controls for structure caused
only by the ResNet architecture or image statistics. If the trained SimCLR
encoder produces clearer clusters than the random encoder, the difference can be
attributed to contrastive pretraining.

Remember that t-SNE is qualitative: use it to discuss visible cluster separation
and class mixing, not as a precise quantitative metric.
