# Symbol CNN

PyTorch code for training a CNN classifier on the symbol image folders in `datasets`.

Expected dataset structure:

```text
datasets/
  trainset/
    circles/
    squares/
    stars/
    triangles/
  validset/
    circles/
    squares/
    stars/
    triangles/
  testset/
    circles/
    squares/
    stars/
    triangles/
```

The code also accepts `validateset` if that is the validation folder name.

## Install

From the `nn_training` folder:

```bash
python3 -m pip install -r cnn/requirements.txt
```

## Train

```bash
python3 -m cnn.train --epochs 25 --batch-size 32
```

Outputs:

- `cnn/checkpoints/symbols_cnn_best.pt`
- `cnn/checkpoints/symbols_cnn_latest.pt`
- `cnn/runs/symbols_cnn/history.csv`
- `cnn/runs/symbols_cnn/test_metrics.json`

Useful options:

```bash
python3 -m cnn.train --epochs 40 --batch-size 64 --image-size 128 --device auto
python3 -m cnn.train --device cpu
python3 -m cnn.train --device mps
```

## Evaluate

```bash
python3 -m cnn.evaluate --checkpoint cnn/checkpoints/symbols_cnn_best.pt --split testset
python3 -m cnn.evaluate --checkpoint cnn/checkpoints/symbols_cnn_best.pt --split validset
```

## Plot Results

```bash
python3 -m cnn.plot_results
```

Default output:

- `cnn/runs/symbols_cnn/results_summary.png`

## Predict

```bash
python3 -m cnn.predict --checkpoint cnn/checkpoints/symbols_cnn_best.pt --image path/to/image.jpg
python3 -m cnn.predict --checkpoint cnn/checkpoints/symbols_cnn_best.pt --image-dir path/to/images
```
