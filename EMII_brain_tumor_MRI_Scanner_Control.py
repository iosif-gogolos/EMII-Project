#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMII V2 — Brain Tumor MRI Scanner (Control / No-XAI)
- Prediction-only desktop app for the 4-class Kaggle "Brain Tumor MRI Dataset"
- Classes: glioma, meningioma, notumor, pituitary
- Loads a fine-tuned ResNet18 .pth from ./models/brain_tumor_resnet18.pth
- Local, offline, research prototype (NOT for clinical use)
"""

import os, sys, glob, time, threading
import numpy as np
from PIL import Image, ImageTk
import cv2
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms

import customtkinter as ctk
from customtkinter import CTkImage
from tkinter import filedialog, messagebox

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader


APP_TITLE = "EMII — Brain Tumor MRI Scanner (Control)"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_REL_PATH = os.path.join("models", "brain_tumor_resnet18.pth")


# ---------- Utils ----------
def ensure_dir(d):
    os.makedirs(d, exist_ok=True)
    return d

def apply_display_clahe(img_rgb, clip=2.0, tile=(8,8)):
    """Display-only CLAHE (keeps model input untouched)."""
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
            "You can train a model (see README) and place the .pth there.\n"
            "The app will still run with random weights (for demo only)."
        )
    model.eval()
    return model

# ImageNet normalization for ResNet18
IM_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),  # [0..1], CHW
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225])
])


# ---------- App ----------
class EMIIControl(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x820")
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)

        self.last_dir = os.path.expanduser("~")
        self.files = []
        self.idx = -1

        self.current_image = None
        self.display_image = None
        self.photo_left = None

        self.prediction = None
        self.prob_vector = None

        self._build_ui()

    # UI
    def _build_ui(self):
        top = ctk.CTkFrame(self, corner_radius=10)
        top.pack(side="top", fill="x", padx=14, pady=(12, 6))
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)
        top.grid_columnconfigure(2, weight=0)
        top.grid_columnconfigure(3, weight=0)

        self.btn_open = ctk.CTkButton(top, text="Open Image or Folder…", command=self.on_open)
        self.btn_open.grid(row=0, column=0, padx=(0,8), pady=6, sticky="w")

        self.selector = ctk.CTkOptionMenu(top, values=["—"], command=self.on_select)
        self.selector.configure(state="disabled")
        self.selector.grid(row=0, column=1, padx=8, pady=6, sticky="ew")

        self.btn_prev = ctk.CTkButton(top, text="⟵ Prev", command=self.on_prev, width=80, fg_color="gray25")
        self.btn_prev.grid(row=0, column=2, padx=6, pady=6)
        self.btn_next = ctk.CTkButton(top, text="Next ⟶", command=self.on_next, width=80, fg_color="gray25")
        self.btn_next.grid(row=0, column=3, padx=6, pady=6)

        # Row 1
        self.btn_run = ctk.CTkButton(top, text="Run Prediction", command=self.on_predict, height=36)
        self.btn_run.grid(row=1, column=0, padx=(0,8), pady=4, sticky="w")

        self.chk_clahe_var = ctk.BooleanVar(value=True)
        self.chk_clahe = ctk.CTkCheckBox(top, text="Enhance display (CLAHE)", variable=self.chk_clahe_var,
                                         onvalue=True, offvalue=False, command=self.refresh_display)
        self.chk_clahe.grid(row=1, column=1, padx=8, pady=4, sticky="w")

        self.btn_report = ctk.CTkButton(top, text="Export PDF Report", command=self.on_export, fg_color="#436650")
        self.btn_report.grid(row=1, column=3, padx=6, pady=4, sticky="e")

        # Panels
        body = ctk.CTkFrame(self, corner_radius=10)
        body.pack(side="top", fill="both", expand=True, padx=14, pady=10)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=0)

        left = ctk.CTkFrame(body, corner_radius=10)
        right = ctk.CTkFrame(body, corner_radius=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(10,8), pady=10)
        right.grid(row=0, column=1, sticky="ns", padx=(8,10), pady=10)
        right.configure(width=420)
        right.pack_propagate(False)

        ctk.CTkLabel(left, text="Image").pack(pady=(8,4))
        self.canvas_left = ctk.CTkLabel(left, text="")
        self.canvas_left.pack(fill="both", expand=True, padx=8, pady=8)

        ctk.CTkLabel(right, text="Prediction").pack(pady=(8,4))
        self.pred_label = ctk.CTkLabel(right, text="—", anchor="w", justify="left")
        self.pred_label.pack(fill="x", padx=8)

        self.prob_box = ctk.CTkTextbox(right, height=240)
        self.prob_box.pack(fill="both", expand=True, padx=8, pady=(8,8))

        foot = ctk.CTkLabel(self, text="Research prototype. Not for clinical use. Local files only.",
                            font=ctk.CTkFont(size=12, slant="italic"))
        foot.pack(side="bottom", pady=(0, 10))

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
            self.prob_box.insert("end", "Run prediction to see results.\n")
            self.prediction = None
            self.prob_vector = None
        except Exception as e:
            messagebox.showerror("Load error", str(e))

    def refresh_display(self):
        if self.current_image is None: return
        disp = self.current_image.copy()
        if self.chk_clahe_var.get():
            disp = apply_display_clahe(disp)
        disp = disp.resize((700, 700), Image.BILINEAR)
        self.display_image = disp
        self.photo_left = CTkImage(light_image=disp, size=(700, 700))
        self.canvas_left.configure(image=self.photo_left)

    # Prediction
    def on_predict(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return

        self.btn_run.configure(state="disabled", text="Running…")
        threading.Thread(target=self._run_prediction, daemon=True).start()

    def _run_prediction(self):
        try:
            # Model input
            tens = IM_TF(self.current_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tens)
                probs = F.softmax(logits, dim=1).cpu().numpy().squeeze()

            top_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[top_idx]
            conf = float(probs[top_idx])
            ent = softmax_entropy(probs)

            self.prediction = pred_name
            self.prob_vector = probs

            # Update UI
            txt = f"Prediction: {pred_name}  (confidence {conf*100:.1f}%)  |  Uncertainty (entropy) {ent:.3f}"
            self.after(0, lambda: self.pred_label.configure(
                text=f"{txt}  |  File: {os.path.basename(self.files[self.idx]) if self.files else '-'}"
            ))
            lines = []
            for i, c in enumerate(CLASS_NAMES):
                lines.append(f"{c:10s} : {probs[i]*100:5.1f}%")
            self.after(0, lambda: [self.prob_box.delete("1.0", "end"),
                                   self.prob_box.insert("end", "\n".join(lines))])
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Prediction error", str(e)))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="Run Prediction"))

    # Report
    def on_export(self):
        if self.current_image is None:
            messagebox.showinfo("No image", "Open an image first.")
            return
        try:
            out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "reports"))
            ts = time.strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(out_dir, f"report_control_{ts}.pdf")

            # Save display image for embedding
            img_path = os.path.join(out_dir, f"display_{ts}.png")
            (self.display_image or self.current_image).save(img_path)

            c = canvas.Canvas(pdf_path, pagesize=A4)
            W, H = A4
            y = H - 40
            c.setFont("Helvetica-Bold", 14)
            c.drawString(40, y, "EMII V2 — Brain Tumor MRI Scanner (Control)"); y -= 22
            c.setFont("Helvetica", 10)
            c.drawString(40, y, f"File: {os.path.basename(self.files[self.idx]) if self.files else '-'}"); y -= 14
            pred_txt = self.pred_label.cget("text")
            c.drawString(40, y, pred_txt[:110]); y -= 14
            if len(pred_txt) > 110:
                c.drawString(40, y, pred_txt[110:220]); y -= 14

            if os.path.exists(img_path):
                c.drawImage(ImageReader(img_path), 40, 140, width=300, height=300, preserveAspectRatio=True, mask='auto')

            c.setFont("Helvetica-Oblique", 9)
            c.drawString(40, 100, "Note: Research prototype for educational use only. Not a medical device.")
            c.save()
            messagebox.showinfo("Report", f"Saved:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Export error", str(e))


if __name__ == "__main__":
    app = EMIIControl()
    app.mainloop()
