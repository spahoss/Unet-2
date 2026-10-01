# Embryo U-Net Segmentation

Train a U-Net model on embryo image/mask pairs and use the saved checkpoint to segment new images. The model is provided by `segmentation-models-pytorch` and uses a ResNet-18 encoder.

## Tested environment

- Ubuntu or Ubuntu on WSL2
- Python 3.13
- PyTorch 2.7.1
- CUDA 12.6
- NVIDIA GeForce GTX 1080

CPU execution is also supported.

## Files

- `train.py`: validates the dataset, creates grouped train/validation/test splits, trains the model, evaluates the held-out test split, and saves results.
- `predict.py`: loads a saved `best_model.pt` checkpoint and segments a new image.
- `data_utils.py`: validates dataset files, creates grouped splits, resizes images with letterboxing, and prepares training tensors.
- `dataset.json`: dataset manifest containing sample IDs, relative image/mask paths, hashes, source names, group names, and mask label values.
- `requirements.txt`: pinned PyTorch/CUDA dependencies.

## NVIDIA driver and WSL2

When using WSL2, install the current NVIDIA driver in Windows. Do not install a Linux NVIDIA display driver inside WSL2.

Confirm that Ubuntu can access the GPU:

```bash
nvidia-smi
```

## Install `uv`

```bash
sudo apt update
sudo apt install pipx
pipx install uv
pipx ensurepath
source ~/.bashrc
```

## Create the environment

From the repository directory:

```bash
uv python install 3.13
uv venv --python 3.13 .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 uv pip install \
  -r requirements.txt \
  --index-strategy unsafe-best-match
```

The initial download is large and may take several minutes.

Verify PyTorch and CUDA:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

Expected GPU output is similar to:

```text
PyTorch: 2.7.1+cu126
CUDA: 12.6
GPU available: True
GPU: NVIDIA GeForce GTX 1080
```

## Dataset manifest

Paths in `dataset.json` are relative to the directory supplied with `--data-root`. Each sample must provide an image and a single-channel mask with identical dimensions. The files must match the SHA-256 hashes stored in the manifest.

Samples from the same embryo group remain in the same split. Each source must contain at least five independent groups so that train, validation, and test splits can all be created.

## Validate the dataset

Always validate the dataset before training:

```bash
python train.py \
  --data-root "/absolute/path/to/embryos1,2,3,4" \
  --manifest dataset.json \
  --size 512 \
  --check
```

This verifies file paths, hashes, image/mask dimensions, label values, non-empty masks, grouped splits, and resized masks. It does not train a model.

## Train with the GPU

```bash
python train.py \
  --data-root "/absolute/path/to/embryos1,2,3,4" \
  --manifest dataset.json \
  --epochs 30 \
  --batch-size 2 \
  --size 512 \
  --lr 0.0001 \
  --weights imagenet \
  --device cuda \
  --patience 10 \
  --run-name embryo_unet
```

The first run with `--weights imagenet` may download pretrained ResNet-18 encoder weights. Use `--weights none` when pretrained weights are not wanted or internet access is unavailable.

If GPU memory is insufficient, lower `--batch-size` first, then lower `--size`. The size must be at least 64 and divisible by 32.

To let the script select CUDA when available and otherwise use the CPU:

```bash
python train.py --data-root "/absolute/path/to/data" --device auto
```

## Training output

Each run is written to `runs/<run-name>/` and contains:

- `best_model.pt`: checkpoint selected by validation Dice score.
- `config.json`: training configuration and package versions.
- `history.csv`: training loss and validation metrics for every epoch.
- `split.json`: samples assigned to train, validation, and test splits.
- `test_metrics.json`: final metrics on the held-out test split.
- `test_predictions/`: masks and visual comparisons for held-out test images.

The test split is evaluated once after the best checkpoint has been selected using the validation split.

## Predict a new image

```bash
python predict.py \
  --checkpoint runs/embryo_unet/best_model.pt \
  --image "/absolute/path/to/image.jpg" \
  --device cuda
```

By default, predictions are saved in a new timestamped directory under `predictions/`.

Choose an explicit output directory and threshold:

```bash
python predict.py \
  --checkpoint runs/embryo_unet/best_model.pt \
  --image "/absolute/path/to/image.jpg" \
  --output predictions/example_01 \
  --threshold 0.5 \
  --border-ratio 0.0 \
  --device cuda
```

The output directory must not already exist.

To compare the prediction with a reference mask and print Dice/IoU:

```bash
python predict.py \
  --checkpoint runs/embryo_unet/best_model.pt \
  --image "/absolute/path/to/image.jpg" \
  --mask "/absolute/path/to/reference_mask.png" \
  --device cuda
```

The current prediction code expects reference-mask background pixels to be `0` and foreground pixels to be `1`. The reference mask must have the same dimensions as the input image.

Prediction output contains:

- `prediction_raw.png`: thresholded mask before border removal.
- `prediction.png`: final binary mask.
- `probability.png`: foreground probability map.
- `overlay.png`: prediction overlaid on the input image.
- `comparison.png`: input, optional reference mask, prediction, and overlay panels.

## Monitor GPU usage

During training, run this in another terminal:

```bash
watch -n 1 nvidia-smi
```

## Resume work later

Activate the environment whenever a new terminal is opened:

```bash
cd ~/Unet-2
source .venv/bin/activate
```

Deactivate it with:

```bash
deactivate
```

The `.venv`, `runs`, and generated prediction directories should not be committed to Git.
