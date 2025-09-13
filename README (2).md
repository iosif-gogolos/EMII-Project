# EMII V2 — Brain Tumor MRI Scanner (Control & XAI)

> **Educational research prototype — not a medical device.**  
> This repo contains two lightweight desktop apps used in a within‑subjects A/B usability study:
> - **Control app**: prediction only (no explanations).  
> - **XAI app (EMII)**: prediction + explanations (Grad‑CAM, SHAP, LIME).

Both apps target **brain‑tumor classification** on 2D MRI images (4 classes: *glioma, meningioma, pituitary, no tumor*) using a small CNN (demo) or a transfer‑learned **ResNet‑18** model you train locally.

---

## Product goals

1) **Set clear user expectations (G1/G2).** Make explicit what the system can and cannot do, and how well it tends to perform.  
2) **Aid plausibility checks (G11).** Provide concise, legible overlays so clinicians can challenge/accept predictions.  
3) **Be responsive and unobtrusive (G3/G4).** Autoplay once, fast slider navigation, no modal dialogs during the core task.  
4) **Support feedback loops (G15/G16).** Lightweight thumbs up/down and session logs to learn from user behavior and explain consequences.  
5) **Provide global controls (G17).** Toggle XAI methods and display settings; load/replace model weights easily.  
6) **Notify about changes (G18).** On first run and after updates, show a short “What’s new / Limitations” banner.

(“G#” refers to **Amershi et al., 2019**: *Guidelines for Human‑AI Interaction*.)

---

## Components

- **V2_EMII_brain_tumor_MRI_Scanner_Control.py** — control desktop app (no XAI)  
- **V2_EMII_brain_tumor_MRI_Scanner_XAI.py** — XAI desktop app (Grad‑CAM, SHAP, LIME)  
- **requirements_control.txt** and **requirements_xai.txt** — per‑app dependencies  

---

## Dataset

We use the public **Brain Tumor MRI Dataset** (4 classes) hosted on Kaggle (CC0). It aggregates Figshare, Br35H, and other sources; image sizes vary.  
**Local layout used in this project** (as provided by the user):

```
/Users/iosifgogolos/.../brain-tumor-mri-dataset/
  Training/{glioma,meningioma,notumor,pituitary}
  Testing/{glioma,meningioma,notumor,pituitary}
```

> **Important**: This dataset contains *2D images*, not 3D NIfTI volumes. The V2 desktop apps load 2D images for classification. (A separate BraTS/NIfTI flow is kept only for earlier experiments.)

---

## Ethics, privacy, scope

- **No PHI**: Only use public or fully anonymized data.  
- **Local‑only**: Inference runs on your machine. No uploads.  
- **Research‑only**: These tools are **not for clinical use**; they are study prototypes to observe perception, trust, and usability effects of XAI.

---

## Install & run

Create a virtual environment and install dependencies. Use CPU or CUDA wheels as appropriate.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements_control.txt
python V2_EMII_brain_tumor_MRI_Scanner_Control.py

# For the XAI app
pip install -r requirements_xai.txt
python V2_EMII_brain_tumor_MRI_Scanner_XAI.py
```

**Model weights**  
Train the ResNet‑18 as described below; place the output file at: `./models/brain_tumor_resnet18.pth`.  
Both apps will attempt to load it and fall back to the small demo CNN if missing.

---

## Notes on XAI

- **Grad‑CAM (torchcam)**: fast, intuitive **“where”** view at last conv block.  
- **SHAP (KernelExplainer)**: model‑agnostic local importance; we keep samples modest for responsiveness.  
- **LIME (lime_image)**: superpixel perturbation; shows coarse reliance.  
- **Explanations are aids for plausibility, not ground truth. Use critically.**

---

## Training report — working steps (local dataset)

Below is a **compact, reproducible plan** to train and export `brain_tumor_resnet18.pth` using your local folders:

### Train & test folders

```
/Users/iosifgogolos/.../brain-tumor-mri-dataset/Training/{glioma,meningioma,notumor,pituitary}
/Users/iosifgogolos/.../brain-tumor-mri-dataset/Testing/{glioma,meningioma,notumor,pituitary}
```

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or CUDA wheel
pip install scikit-learn matplotlib tqdm
```

### 2) Data transforms

- **Train**: resize 256 → random crop 224; light augmentation; normalize (ImageNet).  
- **Test** : resize 256 → center crop 224; normalize.

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_root = "/Users/iosifgogolos/.../brain-tumor-mri-dataset/Training"
test_root  = "/Users/iosifgogolos/.../brain-tumor-mri-dataset/Testing"

train_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

test_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(train_root, transform=train_tf)
test_ds  = datasets.ImageFolder(test_root,  transform=test_tf)

from torch.utils.data import DataLoader
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

class_names = train_ds.classes  # ['glioma','meningioma','notumor','pituitary']
```

### 3) Model & loss

```python
import torch, torch.nn as nn, torch.optim as optim
from torchvision import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
in_f = model.fc.in_features
model.fc = nn.Linear(in_f, 4)
model = model.to(device)

# Optional: compute class weights to counter imbalance
import numpy as np
targets = np.array([y for _, y in train_ds.samples])
class_counts = np.bincount(targets, minlength=4)
weights = 1.0 / (class_counts + 1e-6)
weights = weights / weights.sum() * 4
class_weights = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)  # or None
optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

### 4) Train loop with early stopping

```python
from tqdm import tqdm
import os

best_acc = 0.0
patience, bad_epochs = 5, 0
save_path = "./models/brain_tumor_resnet18.pth"
os.makedirs("./models", exist_ok=True)

def eval_epoch(loader):
    model.eval()
    correct, total, loss_sum = 0,0,0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            probs = torch.softmax(logits, dim=1)
            pred = probs.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += float(loss.item()) * y.size(0)
    return loss_sum/total, correct/total

for epoch in range(30):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        pbar.set_postfix(loss=float(loss.item()))
    scheduler.step()

    val_loss, val_acc = eval_epoch(test_loader)
    print(f"Val loss {val_loss:.4f} | Val acc {val_acc*100:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), save_path)
        bad_epochs = 0
        print(f"[SAVE] {save_path}")
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break
```

### 5) Evaluation & confusion matrix

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model.load_state_dict(torch.load(save_path, map_location=device))
model.eval()
all_y, all_p = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        all_p.append(pred)
        all_y.append(y.numpy())
all_p = np.concatenate(all_p); all_y = np.concatenate(all_y)

print(classification_report(all_y, all_p, target_names=class_names))
print(confusion_matrix(all_y, all_p))
```

### 6) (Optional) Temperature scaling (logit calibration)

```python
# quick scalar T learned on validation to improve probability calibration
T = torch.nn.Parameter(torch.ones(1, device=device))
optT = optim.LBFGS([T], lr=0.1, max_iter=50)

# collect validation logits & labels
val_logits, val_y = [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        val_logits.append(model(x).cpu())
        val_y.append(y)
val_logits = torch.cat(val_logits, dim=0).to(device)
val_y     = torch.cat(val_y, dim=0).to(device)

def nll():
    optT.zero_grad()
    scaled = val_logits / T
    loss = nn.CrossEntropyLoss()(scaled, val_y)
    loss.backward()
    return loss

optT.step(nll)
print("Learned temperature:", float(T.item()))
# At inference time: probs = softmax(logits / T)
```

**Export** the uncalibrated weights as `./models/brain_tumor_resnet18.pth` (the apps load that). If you want calibration inside the app, divide logits by `T` before softmax.

### 7) Place weights for the apps

```
./models/brain_tumor_resnet18.pth
```

Run the Control/XAI apps. They’ll show per‑image prediction, confidence, uncertainty (entropy), and overlays (XAI).

---

## Practical notes for your usability study

- **Consistent display**: Apply CLAHE/normalization only for display; do not leak into model inputs during evaluation.  
- **Explainability choice**: Default to **Grad‑CAM** in sessions for speed; offer **SHAP/LIME** for deeper inspection.  
- **Quality framing**: Keep the “how well” banner visible (sets expectations — *Amershi G2*).  
- **Feedback**: Collect thumbs up/down during sessions; include file name, method, and probs in logs.

---

## Citations (APA)

- Amershi, S., Weld, D. S., Vorvoreanu, M., Fourney, A., Nushi, B., Collisson, P., Suh, J., Iqbal, S., Bennett, P. N., Inkpen, K., Teevan, J., Kikin‑Gil, R., & Horvitz, E. (2019). **Guidelines for Human‑AI Interaction**. *CHI ’19 Proceedings*. https://doi.org/10.1145/3290605.3300233  
- Nickparvar, M. (n.d.). **Brain Tumor MRI Dataset** [Dataset]. Kaggle. (CC0 Public Domain). *Retrieved from* https://www.kaggle.com/ (search: *Brain Tumor MRI Dataset*).  
- GusLovesMath. (n.d.). **CNN Brain Tumor Classification | 99% Accuracy** [Notebook]. Kaggle. *Retrieved from* https://www.kaggle.com/

> If your thesis requires dates/URLs precisely, replace “n.d.” and add the exact Kaggle URLs and access dates from your download history.
