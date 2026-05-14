# Symbol MobileNetV2

PyTorch code for training a MobileNetV2 classifier on the symbol image folders in `datasets`.

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

The model labels are singular: `circle`, `square`, `star`, and `triangle`. The dataset folder names can stay plural.

## Install

From the `nn_training` folder:

```bash
python3 -m pip install -r mobilenetv2/requirements.txt
```

## Train

```bash
python3 -m mobilenetv2.train --epochs 25 --batch-size 32
```

Outputs:

- `mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt`
- `mobilenetv2/checkpoints/symbols_mobilenetv2_latest.pt`
- `mobilenetv2/runs/symbols_mobilenetv2/history.csv`
- `mobilenetv2/runs/symbols_mobilenetv2/test_metrics.json`

Useful options:

```bash
python3 -m mobilenetv2.train --epochs 40 --batch-size 32 --image-size 224 --device auto
python3 -m mobilenetv2.train --pretrained
python3 -m mobilenetv2.train --pretrained --freeze-features
python3 -m mobilenetv2.train --device cpu
python3 -m mobilenetv2.train --device mps
```

`--pretrained` uses ImageNet MobileNetV2 weights from torchvision. If those weights are not already cached, torchvision may need network access to download them. Without `--pretrained`, the MobileNetV2 architecture trains from random initialization.

## Evaluate

```bash
python3 -m mobilenetv2.evaluate --checkpoint mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt --split testset
python3 -m mobilenetv2.evaluate --checkpoint mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt --split validset
```

## Plot Results

```bash
python3 -m mobilenetv2.plot_results
```

Default output:

- `mobilenetv2/runs/symbols_mobilenetv2/results_summary.png`

## Predict

```bash
python3 -m mobilenetv2.predict --checkpoint mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt --image path/to/image.jpg
python3 -m mobilenetv2.predict --checkpoint mobilenetv2/checkpoints/symbols_mobilenetv2_best.pt --image-dir path/to/images
```

