# Accelerating `training.py` on macOS (Apple Silicon) and Windows (Intel)

This README explains why your CPU usage looks low on macOS when using **MPS**, and how to actually make training faster.  
It also covers Windows optimizations.

---

## Why low CPU ≠ slow training on macOS

- When `device=mps`, most computation runs on the **Apple GPU**.  
  CPU usage will look low in Activity Monitor, but GPU usage (check the **GPU tab**) is where the work happens.
- Data loading and augmentations still run on CPU. If they are slow, GPU will idle.
- By default, your script enables **AMP only for CUDA**. On MPS, you need to switch to `torch.amp.autocast` for Apple GPUs.

---

## Code Changes for macOS MPS

In `training.py`, replace device and AMP setup with:

```python
# Device selection
use_cuda = torch.cuda.is_available()
use_mps  = torch.backends.mps.is_available()
device   = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
print("Using device:", device)

# AMP setup
amp_device_type = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
use_amp = amp_device_type in ("cuda", "mps")
scaler = torch.amp.GradScaler(device_type=amp_device_type, enabled=use_amp)
```

And in your training loop:

```python
with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
    logits = model(x)
    loss   = criterion(logits, y)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Remove old `torch.cuda.amp.*` calls.

---

## DataLoader Tuning

Update your loaders:

```python
train_loader = DataLoader(
    train_ds, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=(device.type != "cpu"),
    persistent_workers=(args.workers > 0), prefetch_factor=2
)
test_loader  = DataLoader(
    test_ds, batch_size=max(2*args.batch_size, 128), shuffle=False,
    num_workers=args.workers, pin_memory=(device.type != "cpu"),
    persistent_workers=(args.workers > 0), prefetch_factor=2
)
```

Tips:
- On Apple Silicon, `--workers 2` or `--workers 4` often works best.
- Use bigger batches (`--batch-size 128` or higher) until memory is full.
- Set `pin_memory=False` for MPS.

---

## More Speedups

- **Channels-last format** for CNNs:
  ```python
  model = model.to(memory_format=torch.channels_last)
  x = x.to(device, non_blocking=False).to(memory_format=torch.channels_last)
  ```
- **Autocast** enables FP16/BF16 on MPS, increasing throughput.
- **Profiling data vs compute**:
  ```python
  t0 = time.perf_counter()
  ...
  t1 = time.perf_counter()
  print("Data:", t1-t0, "Compute:", t2-t1)
  ```

---

## macOS MPS Env Vars

- Allow more GPU memory:
  ```bash
  export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
  ```

---

## Windows (Intel CPU)

### CPU-only (MKL/oneDNN)

By default, PyTorch uses MKL/oneDNN. Set threads:

```bat
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
python -c "import torch; print(torch.get_num_threads())"
python training.py --workers 8 --batch-size 64
```

### Intel Extension for PyTorch (IPEX)

```bat
pip install intel-extension-for-pytorch
```

In code:
```python
import intel_extension_for_pytorch as ipex
model, optimizer = ipex.optimize(model, optimizer=optimizer, dtype=torch.float32)
```

### GPU Options

- **NVIDIA GPU**: install CUDA build of PyTorch.
- **Intel GPU**: try `torch-directml` or OpenVINO.

---

## Example Runs

macOS MPS:
```bash
python training.py --epochs 30 --batch-size 128 --workers 4
```

Windows CPU-only:
```bat
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
python training.py --epochs 30 --batch-size 64 --workers 8
```

---

## FAQ

- **Why is CPU usage low?**  
  Because the GPU (MPS) does the work. Monitor GPU tab.

- **Why is training still slow?**  
  Increase batch size, adjust workers, enable AMP for MPS, and use channels-last.

- **Why do many workers crash on Windows?**  
  Use ≤ number of cores. Try `persistent_workers=True`.

---
