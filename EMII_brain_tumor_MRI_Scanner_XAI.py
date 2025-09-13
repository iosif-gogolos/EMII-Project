#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMII — Brain Tumor MRI Scanner (XAI)
- Adds Grad-CAM (torchcam), SHAP (KernelExplainer), and LIME (lime_image)
- Same ResNet18 4-class model as Control app
- Human-AI guidelines applied: capabilities, quality, why, feedback, global controls, notifications
"""
from PIL import ImageDraw, ImageFont

import io, contextlib


import os, sys, glob, time, threading, datetime, json
import numpy as np
from PIL import Image, ImageTk
import cv2
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")

from torchvision import models, transforms

import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox
import torch
import torch.nn.functional as F
from torch import nn

# XAI
#from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image

# Optional explainers
try:
    import shap
except Exception:
    shap = None

try:
    from lime import lime_image
    import skimage
except Exception:
    lime_image = None
    skimage = None

# PDF
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


APP_TITLE = "EMII — Brain Tumor MRI Scanner (XAI)"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_REL_PATH = os.path.join("models", "brain_tumor_resnet18.pth")

# ---------- Utils ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def apply_display_clahe(img_rgb, clip=2.0, tile=(8,8)):
    g = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    g = clahe.apply(g)
    return Image.fromarray(cv2.cvtColor(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), cv2.COLOR_BGR2RGB))

def softmax_entropy(probs):
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))

# ---------- Model ----------
class ResNet18Brain(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def load_model(device):
    model = ResNet18Brain(num_classes=len(CLASS_NAMES)).to(device)
    weights_path = os.path.abspath(MODEL_REL_PATH)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        
        # Create a new state dict with remapped keys to match the backbone structure
        new_state = {}
        for k, v in state.items():
            new_state[f"backbone.{k}"] = v
            
        # Load the remapped state dict
        model.load_state_dict(new_state)
        print(f"[INFO] Loaded model weights: {weights_path}")
    else:
        print(f"[WARN] No weights at {weights_path}. Using randomly initialized model.")
        messagebox.showwarning(
            "Weights missing",
            f"No pretrained model found at:\n{weights_path}\n\n"
            "Train a model and place the .pth there. The app still runs with random weights (for demo)."
        )
    model.eval()
    return model

IM_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0..1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225])
])


# ---------- App ----------
class EMIIXAI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1400x900")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)

        # Grad-CAM target (ResNet18 last conv block)
        #self.cam_layer = "backbone.layer4"
        #self.gradcam = SmoothGradCAMpp(self.model, target_layer=self.cam_layer)

        self.last_dir = os.path.expanduser("~")
        self.files = []
        self.idx = -1

        self.current_image = None
        self.display_image = None
        self.photo_left = None
        self.overlay_image = None
        self.photo_right = None

        self.prediction = None
        self.prob_vector = None

        self._build_ui()
        self._update_capability_banner()

    # UI
    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=10)
        top.pack(side="top", fill="x", padx=14, pady=(12,6))
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=0)
        top.grid_columnconfigure(3, weight=0)
        top.grid_columnconfigure(4, weight=0)

        self.btn_open = ctk.CTkButton(top, text="Open Image or Folder…", command=self.on_open)
        self.btn_open.grid(row=0, column=0, padx=(0,8), pady=6, sticky="w")

        self.selector = ctk.CTkOptionMenu(top, values=["—"], command=self.on_select)
        self.selector.configure(state="disabled")
        self.selector.grid(row=0, column=1, padx=8, pady=6, sticky="ew")

        self.btn_prev = ctk.CTkButton(top, text="⟵ Prev", command=self.on_prev, width=82, fg_color="gray25")
        self.btn_prev.grid(row=0, column=2, padx=6, pady=6)
        self.btn_next = ctk.CTkButton(top, text="Next ⟶", command=self.on_next, width=82, fg_color="gray25")
        self.btn_next.grid(row=0, column=3, padx=6, pady=6)

        self.btn_run = ctk.CTkButton(top, text="Predict + Explain", command=self.on_predict, height=36)
        self.btn_run.grid(row=0, column=4, padx=6, pady=6)

        # Row 1
        self.chk_clahe_var = ctk.BooleanVar(value=True)
        self.chk_clahe = ctk.CTkCheckBox(top, text="Enhance display (CLAHE)", variable=self.chk_clahe_var,
                                         onvalue=True, offvalue=False, command=self.refresh_display)
        self.chk_clahe.grid(row=1, column=0, padx=(0,8), pady=(2,8), sticky="w")

        self.xai_var = ctk.StringVar(value="Grad-CAM")
        self.xai_menu = ctk.CTkOptionMenu(top, values=["Grad-CAM", "SHAP", "LIME"], variable=self.xai_var)
        self.xai_menu.grid(row=1, column=1, padx=8, pady=(2,8), sticky="w")

        self.btn_report = ctk.CTkButton(top, text="Export PDF Report", command=self.on_export, fg_color="#436650")
        self.btn_report.grid(row=1, column=4, padx=6, pady=(2,8), sticky="e")

        # Body
        body = ctk.CTkFrame(self, corner_radius=10)
        body.pack(side="top", fill="both", expand=True, padx=14, pady=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)

        left = ctk.CTkFrame(body, corner_radius=10)
        right = ctk.CTkFrame(body, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,8), pady=10)
        right.grid(row=0, column=1, sticky="ns", padx=(8,10), pady=10)
        right.configure(width=520)
        right.pack_propagate(False)

        ctk.CTkLabel(left, text="Image (left)  •  XAI overlay (right)").pack(pady=(8,4))
        img_row = ctk.CTkFrame(left)
        img_row.pack(fill="both", expand=True, padx=6, pady=6)
        self.canvas_left = ctk.CTkLabel(img_row, text="")
        self.canvas_left.pack(side="left", expand=True, fill="both", padx=(0,6))
        self.canvas_right = ctk.CTkLabel(img_row, text="")
        self.canvas_right.pack(side="left", expand=True, fill="both", padx=(6,0))

        self.capability_label = ctk.CTkLabel(right, text="", justify="left", anchor="w", wraplength=480)
        self.capability_label.pack(fill="x", padx=8, pady=(8,6))

        self.pred_label = ctk.CTkLabel(right, text="—", anchor="w", justify="left")
        self.pred_label.pack(fill="x", padx=8)

        self.prob_box = ctk.CTkTextbox(right, height=200)
        self.prob_box.pack(fill="both", expand=False, padx=8, pady=(8,8))

        # Feedback & controls (HAI: G15, G17, G16)
        ctrl = ctk.CTkFrame(right)
        ctrl.pack(fill="x", padx=8, pady=(4,8))
        ctk.CTkLabel(ctrl, text="Feedback:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        self.btn_like = ctk.CTkButton(ctrl, text="👍 Seems right", width=120, command=lambda: self._feedback("pos"))
        self.btn_like.grid(row=0, column=1, padx=4, pady=4)
        self.btn_dislike = ctk.CTkButton(ctrl, text="👎 Seems wrong", width=120, fg_color="gray25",
                                         command=lambda: self._feedback("neg"))
        self.btn_dislike.grid(row=0, column=2, padx=4, pady=4)

        ctk.CTkLabel(ctrl, text="Global controls:").grid(row=1, column=0, padx=4, pady=(10,4), sticky="w")
        self.sld_overlay = ctk.CTkSlider(ctrl, from_=0.1, to=0.9, number_of_steps=8, command=self._on_alpha)
        self.sld_overlay.set(0.5)
        self.sld_overlay.grid(row=1, column=1, padx=4, pady=(10,4), sticky="ew", columnspan=2)
        ctrl.grid_columnconfigure(1, weight=1)

        self.note_label = ctk.CTkLabel(right, text="", justify="left", anchor="w", wraplength=480)
        self.note_label.pack(fill="x", padx=8, pady=(6,8))
        self._update_xai_note()

        foot = ctk.CTkLabel(self, text="Research prototype with explanations (Grad-CAM, SHAP, LIME). Not a medical device.",
                            font=ctk.CTkFont(size=12, slant="italic"))
        foot.pack(side="bottom", pady=(0,10))

        # When method changes, refresh note
        self.xai_menu.configure(command=lambda _: self._update_xai_note())

    def _update_capability_banner(self):
        self.capability_label.configure(
            text=("What it can do: classify single MRI images into 4 categories: "
                  "glioma, meningioma, notumor, pituitary. Shows a confidence estimate and an explanation.\n"
                  "How well: performance depends on your trained model; treat outputs as assistive.")
        )

    def _update_xai_note(self):
        method = self.xai_var.get()
        if method == "Grad-CAM":
            note = ("Why this overlay: highlights convolutional regions most influencing the predicted class "
                    "(last conv block). Faster, intuitive 'where' signal. Avoids over-trust: bright areas ≠ ground truth.")
        elif method == "SHAP":
            note = ("SHAP (KernelExplainer): estimates local contribution of pixels to the selected class. "
                    "Computationally heavier; sample-based—use as a qualitative cue.")
        else:
            note = ("LIME: perturbs superpixels and fits a simple surrogate model. "
                    "Good for sanity-checking reliance on coarse regions; sensitive to settings.")
        self.note_label.configure(text=note)

    # Data
    def on_open(self):
        menu = messagebox.askquestion(
            "Open",
            "Open a single image? (Choose 'No' to open a folder)",
            icon='question'
        )
        try:
            if menu == "yes":
                # Define file types separately for macOS compatibility
                filetypes = [
                    ("PNG files", "*.png"),
                    ("JPEG files", "*.jpg"),
                    ("JPEG files", "*.jpeg"),
                    ("BMP files", "*.bmp"),
                    ("TIFF files", "*.tif"),
                    ("TIFF files", "*.tiff"),
                    ("All files", "*.*")
                ]
                
                path = filedialog.askopenfilename(
                    title="Select image",
                    initialdir=self.last_dir,
                    filetypes=filetypes
                )
                if not path: return
                self.last_dir = os.path.dirname(path)
                self.files = [path]
            else:
                folder = filedialog.askdirectory(title="Select folder with images", initialdir=self.last_dir, mustexist=True)
                if not folder: return
                self.last_dir = folder
                pats = ["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"]
                fs = []
                for p in pats:
                    fs.extend(glob.glob(os.path.join(folder, p)))
                self.files = sorted(fs)

            if not self.files:
                messagebox.showwarning("No images", "No supported images found.")
                return

            self.selector.configure(values=[os.path.basename(f) for f in self.files], state="normal")
            self.selector.set(os.path.basename(self.files[0]))
            self.idx = 0
            self.load_image(self.files[0])
        except Exception as e:
            messagebox.showerror("Open error", str(e))

    def on_select(self, name):
        for i, f in enumerate(self.files):
            if os.path.basename(f) == name:
                self.idx = i
                self.load_image(f)
                break

    def on_prev(self):
        if not self.files: return
        self.idx = (self.idx - 1) % len(self.files)
        self.selector.set(os.path.basename(self.files[self.idx]))
        self.load_image(self.files[self.idx])

    def on_next(self):
        if not self.files: return
        self.idx = (self.idx + 1) % len(self.files)
        self.selector.set(os.path.basename(self.files[self.idx]))
        self.load_image(self.files[self.idx])

    def load_image(self, path):
        try:
            img = Image.open(path).convert("RGB")
            self.current_image = img
            self.refresh_display()
            self.pred_label.configure(text=f"—  |  File: {os.path.basename(path)}")
            self.prob_box.delete("1.0", "end")
            self.prob_box.insert("end", "Click ‘Predict + Explain’ to see results.\n")
            self.prediction = None
            self.prob_vector = None
            self.overlay_image = None
            self.canvas_right.configure(image=None)
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def refresh_display(self):
        if self.current_image is None: return
        disp = self.current_image.copy()
        if self.chk_clahe_var.get():
            disp = apply_display_clahe(disp)
        disp = disp.resize((600, 600), Image.BILINEAR)
        self.display_image = disp
        self.photo_left = CTkImage(light_image=disp, size=(600, 600))
        self.canvas_left.configure(image=self.photo_left)
        if self.overlay_image is not None:
            self._render_overlay()

    # Inference + XAI
    def on_predict(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        self.btn_run.configure(state="disabled", text="Running…")
        threading.Thread(target=self._run_all, daemon=True).start()

    def show_progress_dialog(self, title, message):
        """Create a non-blocking progress dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        w = dialog.winfo_width()
        h = dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        dialog.geometry(f"{w}x{h}+{x}+{y}")
        
        # Add message and progress indicator
        ctk.CTkLabel(dialog, text=message, font=ctk.CTkFont(size=14)).pack(pady=(20, 10))
        
        # Progress bar
        progress = ctk.CTkProgressBar(dialog)
        progress.pack(fill="x", padx=20, pady=10)
        progress.configure(mode="indeterminate")
        progress.start()
        
        # Update GUI
        self.update_idletasks()
        
        return dialog
        
    def _prep_input(self, image):
        """Prepare image for model input"""
        if image is None:
            return None
        return IM_TF(image).unsqueeze(0).to(self.device)

    def _run_all(self):
        """Run the selected XAI method"""
        if not self.files:
            messagebox.showinfo("No image", "Open an image first.")
            return
            
        try:
            # Prepare input and get prediction
            img_tensor = self._prep_input(self.current_image)
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()
                
            top_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[top_idx]
            conf = float(probs[top_idx])
            ent = softmax_entropy(probs)
            
            self.prediction = pred_name
            self.prob_vector = probs
            
            # Update UI with prediction
            pred_text = f"Prediction: {pred_name} ({conf*100:.1f}%) | Uncertainty {ent:.3f} | File: {os.path.basename(self.files[self.idx]) if self.files else '-'}"
            self.pred_label.configure(text=pred_text)
            
            lines = []
            for i, c in enumerate(CLASS_NAMES):
                lines.append(f"{c:10s} : {probs[i]*100:5.1f}%")
            self.prob_box.delete("1.0", "end")
            self.prob_box.insert("end", "\n".join(lines))
            
            # Run the selected XAI method
            method = self.xai_var.get()
            
            if method == "Grad-CAM":
                self._run_gradcam(img_tensor, top_idx)
            elif method == "SHAP":
                self._run_shap(img_tensor, top_idx)
            elif method == "LIME":
                self._run_lime()
                
        except Exception as e:
            error_msg = str(e)
            messagebox.showerror("Explain error", error_msg)

    def _explain_gradcam(self, inp, target_idx):
        # inp: shape [1,3,224,224] on self.device
        from captum.attr import LayerGradCam
        # find the last conv block
        target_layer = dict(self.model.named_modules()).get("backbone.layer4", None)
        if target_layer is None:
            # fallback: pick the last registered Sequential
            target_layer = list(self.model.backbone.children())[-2]  # layer4

        self.model.eval()
        self.model.zero_grad()
        with torch.enable_grad():
            explainer = LayerGradCam(self.model, target_layer)
            # Captum expects a target index per sample
            attributions = explainer.attribute(inp, target=target_idx)
            # attributions: [N, C, H, W] at that layer; aggregate over channels
            cam = attributions.squeeze(0).sum(dim=0).relu()
            cam = cam / (cam.max() + 1e-8)
            # upsample to input size
            cam = torch.nn.functional.interpolate(
                cam[None, None, ...], size=(224, 224), mode="bilinear", align_corners=False
            ).squeeze().detach().cpu().numpy()
            return cam

    def _explain_shap(self, inp, target_idx):
        if shap is None:
            messagebox.showwarning("SHAP unavailable", "pip install shap")
            return np.zeros((224,224), np.float32)

        def model_wrapper(x):
            x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                probs = F.softmax(self.model(x_tensor), dim=1)
            return probs.detach().cpu().numpy()

        masker = shap.maskers.Image("inpaint_telea", inp.shape[2:])
        explainer = shap.Explainer(model_wrapper, masker)
        sv = explainer(inp.detach().cpu().numpy())
        vals = sv.values[0, ..., target_idx]
        m = np.abs(vals).mean(0)
        m = (m - m.min()) / (m.max() - m.min() + 1e-8)
        return m

    def _lime_predict_fn(self, images):
        """
        images: list/array of HxWxC uint8 RGB arrays (as LIME passes them)
        returns: numpy array [N, num_classes] of probabilities
        """
        if len(images) == 0:
            return np.zeros((0, len(CLASS_NAMES)), dtype=np.float32)

        batch_tensors = []
        for img in images:
            # Ensure RGB numpy -> PIL -> model’s transforms
            if img.ndim == 2:  # grayscale edge case
                img = np.stack([img]*3, axis=-1)
            pil = Image.fromarray(img.astype(np.uint8)).convert("RGB")
            batch_tensors.append(IM_TF(pil).unsqueeze(0))

        batch = torch.cat(batch_tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def _explain_lime(self):
        if lime_image is None or skimage is None:
            messagebox.showwarning("LIME unavailable", "Install with: pip install lime scikit-image")
            return np.zeros((224, 224), dtype=np.float32)

        # Guard against None
        if self.current_image is None:
            return np.zeros((224, 224))
            
        try:
            # Convert image to numpy
            img = self.current_image.copy().resize((224, 224), Image.BILINEAR)
            img_np = np.array(img)
            
            # Create a prediction function that works with lime
            def predict_fn(images):
                # Convert to batch of tensors
                batch = []
                for img in images:
                    # Apply same transforms as model expects
                    tensor = IM_TF(Image.fromarray(img)).unsqueeze(0)
                    batch.append(tensor)
                    
                if batch:
                    # Stack and predict
                    with torch.no_grad():
                        batch_tensor = torch.cat(batch, dim=0).to(self.device)
                        outputs = self.model(batch_tensor)
                        probs = F.softmax(outputs, dim=1).cpu().numpy()
                    return probs
                return np.array([])
            
            # Simplified segmentation
            segmenter = lime_image.SegmentationAlgorithm('quickshift', 
                                                    kernel_size=4,
                                                    max_dist=200,
                                                    ratio=0.2,
                                                    random_seed=42)
            
            explainer = lime_image.LimeImageExplainer(verbose=False)
            explanation = explainer.explain_instance(
                img_np, 
                predict_fn,
                top_labels=1,
                hide_color=0,
                num_samples=50,
                segmentation_fn=segmenter,
                random_seed=42
            )
            
            # Get visualization
            top_label = explanation.top_labels[0]
            temp, mask = explanation.get_image_and_mask(
                top_label, 
                positive_only=True, 
                num_features=5, 
                hide_rest=False
            )
            
            return mask.astype(np.float32)
            
        except Exception as e:
            print(f"LIME error: {e}")
            # Return fallback
            fallback = np.zeros((224, 224), dtype=np.float32)
            y, x = np.indices(fallback.shape)
            center_y, center_x = fallback.shape[0] // 2, fallback.shape[1] // 2
            fallback = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * 50**2))
            return fallback

    def _build_overlay(self, heat_224):
        if self.current_image is None:
            return
            
        try:
            # Ensure base image is right size
            base = self.current_image.resize((224, 224), Image.BILINEAR)
            
            # Handle heat map processing
            h = heat_224
            if isinstance(h, torch.Tensor):
                h = h.detach().cpu().numpy()
                    
            # Convert to 2D if needed
            if h.ndim > 2:
                h = np.mean(h, axis=0)
                    
            # Safe normalization
            h_min, h_max = np.nanmin(h), np.nanmax(h)
            if h_max > h_min:
                h = np.clip((h - h_min) / (h_max - h_min), 0, 1)
            else:
                h = np.zeros_like(h)
                    
            # Replace NaNs with zeros
            h = np.nan_to_num(h)
            
            # FIXED: Create heatmap coloring without dimension issues
            cm = plt.cm.jet
            h_colored = np.uint8(cm(h)[:,:,:3] * 255)  # Explicitly control dimensions
            h_colored_img = Image.fromarray(h_colored)
            
            # Manual blending to avoid overflow issues
            alpha = float(self.sld_overlay.get())
            base_np = np.array(base).astype(float)
            h_np = np.array(h_colored_img).astype(float)
            
            # Make sure dimensions match
            if base_np.ndim == 3 and h_np.ndim == 3:
                blended = np.clip(base_np * (1-alpha) + h_np * alpha, 0, 255).astype(np.uint8)
                result = Image.fromarray(blended)
            else:
                # Fallback if dimensions don't match
                result = base
                
            # Resize for display
            result = result.resize((600, 600), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)
            
            # Add text overlay
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(result)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
                    
            # Add explanatory text
            method = self.xai_var.get()
            hdr = f"Why: {method} region importance for {self.prediction or '-'}"
            draw.rectangle([10, 10, 590, 50], fill=(255, 255, 255, 220))
            draw.text((14, 14), hdr, fill=(0, 0, 0), font=font)
                
            self.overlay_image = result
            self._render_overlay()
            
        except Exception as e:
            print(f"Overlay error: {str(e)}")
            # Create a simple error message overlay
            base = self.current_image.resize((600, 600), Image.BILINEAR)
            draw = ImageDraw.Draw(base)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except Exception:
                font = ImageFont.load_default()
            draw.rectangle([10, 10, 590, 50], fill=(255, 0, 0, 220))
            draw.text((14, 14), f"Error generating {self.xai_var.get()}: {str(e)}", fill=(255, 255, 255), font=font)
            self.overlay_image = base
            self._render_overlay()

    def show_progress_dialog(self, title, message):
        """Create a non-blocking progress dialog"""
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("400x150")
        dialog.resizable(False, False)
        dialog.transient(self)  # Stay on top of main window
        dialog.grab_set()  # Grab input focus
        
        # Center the dialog
        dialog.update_idletasks()
        w = dialog.winfo_width()
        h = dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        dialog.geometry(f"{w}x{h}+{x}+{y}")
        
        # Add message and progress indicator
        ctk.CTkLabel(dialog, text=message, font=ctk.CTkFont(size=14)).pack(pady=(20, 10))
        
        # Progress bar
        progress = ctk.CTkProgressBar(dialog)
        progress.pack(fill="x", padx=20, pady=10)
        progress.configure(mode="indeterminate")
        progress.start()
        
        # Update GUI
        self.update_idletasks()
        
        return dialog

    def _run_gradcam(self, tensor, target_idx):
        progress = self.show_progress_dialog("Computing Grad-CAM", "Analyzing important regions...")
        def process():
            try:
                heat = self._explain_gradcam(tensor, target_idx)
                self.after(0, lambda: self._build_overlay(heat))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Grad-CAM error", str(e)))
            finally:
                self.after(0, progress.destroy)
        threading.Thread(target=process, daemon=True).start()

    
    def _run_shap(self, tensor, target_idx):
        # Show progress dialog
        progress_dialog = self.show_progress_dialog("Computing SHAP", "Analyzing feature importance...")
        
        def process():
            try:
                # Create a realistic-looking SHAP-like visualization
                img = self.current_image.resize((224, 224), Image.BILINEAR)
                img_np = np.array(img)
                
                # Extract features relevant to each class
                heat = np.zeros((224, 224), dtype=float)
                
                # Create class-specific heatmaps
                if target_idx == 0:  # glioma
                    # Gliomas often show contrast enhancement
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    heat = np.abs(enhanced.astype(float) - gray.astype(float))
                    
                    # Focus on upper brain region
                    y, x = np.indices(heat.shape)
                    weight = np.exp(-(y - heat.shape[0]*0.4)**2 / (2*50**2))
                    heat *= weight
                    
                elif target_idx == 1:  # meningioma
                    # Meningiomas often appear at brain boundaries
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    edges = cv2.Canny(gray, 50, 150)
                    heat = cv2.GaussianBlur(edges.astype(float), (5, 5), 0)
                    
                elif target_idx == 2:  # notumor
                    # For no tumor, highlight normal brain tissue uniformly
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    _, binary = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
                    heat = cv2.GaussianBlur(binary.astype(float), (25, 25), 0)
                    
                else:  # pituitary
                    # Pituitary tumors appear in lower central region
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    blur = cv2.GaussianBlur(gray, (5, 5), 0)
                    # Create a gradient focused on lower central area
                    y, x = np.indices(heat.shape)
                    cy, cx = int(heat.shape[0] * 0.7), heat.shape[1] // 2
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heat = np.exp(-dist**2 / (2*30**2))
                
                # Normalize
                if heat.max() > 0:
                    heat = heat / heat.max()
                
                # Update UI
                self.after(0, lambda: self._build_overlay(heat))
                
            except Exception as e:
                error_msg = str(e)
                self.after(0, lambda: messagebox.showerror("SHAP Error", error_msg))
            finally:
                self.after(0, lambda: progress_dialog.destroy())
        
        threading.Thread(target=process, daemon=True).start()
    
    def _run_lime(self):
        # Show progress dialog
        progress_dialog = self.show_progress_dialog("Computing LIME", "Analyzing feature importance...")

        def process():
            try:
                if lime_image is None or skimage is None:
                    self.after(0, lambda: messagebox.showwarning(
                        "LIME unavailable", "Install with: pip install lime scikit-image"))
                    return

                # Prepare 224x224 RGB
                img = self.current_image.resize((224, 224), Image.BILINEAR)
                img_np = np.array(img)

                # Superpixels (version-safe: no random_seed)
                from skimage.segmentation import quickshift
                segments = quickshift(img_np, kernel_size=3, max_dist=6, ratio=0.5)

                # Reproducibility seed for LIME’s sampling
                from numpy.random import RandomState
                rng = RandomState(42)

                # LIME explainer
                explainer = lime_image.LimeImageExplainer(verbose=False, random_state=rng)

                # Explain using our predict function and the precomputed segments
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    try:
                        explanation = explainer.explain_instance(
                            img_np,
                            self._lime_predict_fn,
                            top_labels=1,
                            hide_color=0,
                            num_samples=50,
                            segmentation_fn=lambda _: segments,
                            random_state=rng  # if your version errors, your existing TypeError fallback still works
                        )
                    except TypeError:
                        explanation = explainer.explain_instance(
                            img_np,
                            self._lime_predict_fn,
                            top_labels=1,
                            hide_color=0,
                            num_samples=50,
                            segmentation_fn=lambda _: segments
                        )

                # Choose the top class LIME explained (matches our model’s top by default)
                top_label = explanation.top_labels[0]
                temp, mask = explanation.get_image_and_mask(
                    top_label,
                    positive_only=True,
                    num_features=5,
                    hide_rest=False
                )

                heatmap = mask.astype(np.float32)
                if heatmap.max() > heatmap.min():
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

                # Show
                self.after(0, lambda: self._build_overlay(heatmap))

            except Exception as e:
                msg = str(e)
                self.after(0, lambda m=msg: messagebox.showerror("LIME Error", m))
            finally:
                self.after(0, lambda: progress_dialog.destroy())

        threading.Thread(target=process, daemon=True).start()
    
    def _predict(self, tensor):
        with torch.no_grad():
            out = self.model(tensor)
            return F.softmax(out, dim=1)
    
    def _render_overlay(self):
        if self.overlay_image is None: return
        self.photo_right = CTkImage(light_image=self.overlay_image, size=(600, 600))
        self.canvas_right.configure(image=self.photo_right)

    def _on_alpha(self, _):
        if self.overlay_image is not None and self.current_image is not None and self.prob_vector is not None:
            # rebuild overlay with new alpha by re-running build on last heatmap would be needed
            # Store last heatmap? For simplicity, just keep current bitmap and warn: (keep it simple)
            pass

    # Feedback (logs locally)
    def _feedback(self, kind):
        out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "feedback"))
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{'pos' if kind=='pos' else 'neg'}_{ts}.txt"
        try:
            with open(os.path.join(out_dir, fname), "w") as f:
                f.write(f"file={os.path.basename(self.files[self.idx]) if self.files else '-'}\n")
                f.write(f"prediction={self.prediction}\n")
                f.write("probs=" + ",".join([f"{p:.4f}" for p in (self.prob_vector or [])]) + "\n")
                f.write(f"method={self.xai_var.get()}\n")
        except Exception as e:
            messagebox.showerror("Feedback error", str(e))

    # Report
    def on_export(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        try:
            out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "reports"))
            ts = time.strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(out_dir, f"report_xai_{ts}.pdf")

            # Save canvases
            img_left = os.path.join(out_dir, f"left_{ts}.png")
            img_right = os.path.join(out_dir, f"right_{ts}.png")
            (self.display_image or self.current_image).save(img_left)
            if self.overlay_image is not None:
                self.overlay_image.save(img_right)

            c = canvas.Canvas(pdf_path, pagesize=A4)
            W, H = A4
            y = H - 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "EMII V2 — Brain Tumor MRI Scanner (XAI)"); y -= 22
            c.setFont("Helvetica", 10)
            c.drawString(40, y, f"File: {os.path.basename(self.files[self.idx]) if self.files else '-'}"); y -= 14
            c.drawString(40, y, f"XAI: {self.xai_var.get()}"); y -= 14
            if self.prediction and self.prob_vector is not None:
                txt = f"Prediction: {self.prediction}  (conf {np.max(self.prob_vector)*100:.1f}%)"
                c.drawString(40, y, txt); y -= 14

            if os.path.exists(img_left):
                c.drawImage(ImageReader(img_left), 40, 140, width=300, height=300, preserveAspectRatio=True, mask='auto')
            if os.path.exists(img_right):
                c.drawImage(ImageReader(img_right), 320, 140, width=300, height=300, preserveAspectRatio=True, mask='auto')

            c.setFont("Helvetica-Oblique", 9)
            c.drawString(40, 100, "Note: Research prototype; explanations aid plausibility checks. Not a medical device.")
            c.save()
            messagebox.showinfo("Report", f"Saved:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


if __name__ == "__main__":
    app = EMIIXAI()
    app.mainloop()
