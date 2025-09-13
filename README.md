# EMII — Brain Tumor MRI Scanner (Control & XAI)

**Purpose.** Two lightweight, local desktop apps to support a within-subjects usability study on explainability for brain tumor MRI classification using the Kaggle *Brain Tumor MRI Dataset*.  
- **Control app:** prediction only (ResNet18; 4 classes).  
- **XAI app:** prediction + **Grad-CAM**, **SHAP (KernelExplainer)**, and **LIME** overlays.

> Research prototype. Not for clinical use.

---

## Product goals (aligned with Human–AI Guidelines)
- **G1/G2 Make clear what/how well:** banner explains capabilities; metrics and confidence shown per image.
- **G11 Why it did that:** Grad-CAM / SHAP / LIME overlays give “where/what” cues for plausibility checks.
- **G15 Feedback:** in-app thumbs up/down logs (“feedback/” folder) for iterative improvement.
- **G17 Global controls:** CLAHE toggle (display only), overlay opacity slider.
- **G16 Consequences:** report & UI reflect method choice; overlay updates immediately.

---

## Dataset (4-class)
- Classes: `glioma`, `meningioma`, `notumor`, `pituitary`
- Folder structure used for training:
```
Training/
  glioma/ *.jpg|*.png
  meningioma/
  notumor/
  pituitary/
Testing/
  glioma/
  meningioma/
  notumor/
  pituitary/
```

### APA citation for thesis
Nickparvar, M. (2021). *Brain Tumor MRI Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

### Optional inspiration (modeling)
GusLovesMath. (2023). *CNN Brain Tumor Classification | 99% Accuracy* [Kaggle notebook]. https://www.kaggle.com/code/guslovesmath/cnn-brain-tumor-classification-99-accuracy

### Human–AI guidelines
Amershi, S., Weld, D. S., Vorvoreanu, M., Fourney, A., Nushi, B., Collisson, P., … Horvitz, E. (2019). Guidelines for human-AI interaction. *CHI 2019*. https://doi.org/10.1145/3290605.3300233

---

## Install

```bash
# (Recommended) Python 3.10–3.12 virtual env
pip install -r requirements-xai.txt   # includes everything
```

## Run

### To run training script  (training.py) on mac m1 with gui
```bash
❯ python training.py --train-root "./brain-tumor-mri-dataset/Training" \
                   --test-root "./brain-tumor-mri-dataset/Testing" \
                   --epochs 30 --batch-size 128 --lr 2e-4 \
                   --use-class-weights --gui
```

```bash
# Control (prediction-only)
python EMII_brain_tumor_MRI_Scanner_Control.py

# XAI (Grad-CAM, SHAP, LIME)
python EMII_brain_tumor_MRI_Scanner_XAI.py
```

Trained weights are placed at: `./models/brain_tumor_resnet18.pth`.

---

## Model
- Backbone: **ResNet18**, final fc → 4 classes.
- Input: RGB 224×224 (ImageNet mean/std normalization).
- Reports export to `./reports/` (PDF with canvases).

---

## Notes on XAI
- **Grad-CAM (torchcam):** fast, intuitive “where” view at last conv block.
- **SHAP (KernelExplainer):** model-agnostic local importance; we keep samples modest for responsiveness.
- **LIME (lime_image):** superpixel perturbation; shows coarse reliance.

> Explanations are aids for *plausibility*, not ground truth. Use critically.

---
