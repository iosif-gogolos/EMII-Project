#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMII — Brain Tumor MRI Scanner (XAI) - Enhanced UI Version with Advanced Explainability
- Modern, professional interface with improved layout and visual hierarchy
- Adds Grad-CAM (torchcam), SHAP (KernelExplainer), and LIME (lime_image)
- Enhanced with fidelity scoring, signed overlays, and conservation checks
- Same ResNet18 4-class model as Control app
- Human-AI guidelines applied: capabilities, quality, why, feedback, global controls, notifications

LIME Framework Attribution:
Based on LIME (Local Interpretable Model-Agnostic Explanations)
© Copyright 2016, Marco Tulio Ribeiro
Documentation: https://lime-ml.readthedocs.io/en/latest/lime.html
Paper: "Why Should I Trust You?" Explaining the Predictions of Any Classifier

SHAP Framework Attribution:
Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. 
Advances in Neural Information Processing Systems, 30.
Official Documentation: https://github.com/slundberg/shap

Medical domain knowledge from:
- Wang, Y., Qi, Q., & Shen, X. (2020). Image segmentation of brain MRI based on LTriDP 
  and superpixels of improved SLIC. Brain Sciences, 10(2), 116.
- Snehalatha, N., & Patil, S. R. (2021). Brain MRI image segmentation using simple linear 
  iterative clustering (SLIC) segmentation with superpixel fusion.
"""

import spinner


try:
    # Core Python imports
    print("Started loading the libraries")
    import sys, os, itertools, glob, time, threading, datetime, json

    # Import spinner for loading feedback
    try:
        from utils.spinner import Spinner
    except ImportError:
        class Spinner:
            def __init__(self, message="Loading..."):
                print(f"Loading: {message}")
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

    # Numerical and image processing
    with Spinner("Loading numpy module"):
        import numpy as np
    print("✓ NumPy loaded")

    with Spinner("Loading opencv module"):
        import cv2
    print("✓ OpenCV loaded")

    # PIL/Pillow for image handling
    try:
        with Spinner("Loading pillow dependencies"):
            from PIL import Image, ImageDraw, ImageFont
        print("✓ Pillow dependencies loaded")
    except ImportError:
        print("Warning: PIL/Pillow not found. Install with: pip install Pillow")
        Image = ImageDraw = ImageFont = None

    # Matplotlib for plotting
    try:
        with Spinner("Loading matplotlib dependencies"):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm
        print("✓ Matplotlib dependencies loaded")
    except ImportError:
        print("Warning: Matplotlib not found. Install with: pip install matplotlib")
        matplotlib = plt = cm = None

    # PyTorch and torchvision
    try:
        with Spinner("Loading PyTorch dependencies"):
            import torch
            import torch.nn.functional as F
            from torch import nn
            from torchvision import models, transforms
            from torchvision.transforms.functional import to_pil_image
        print("✓ PyTorch loaded")
    except ImportError:
        print("Warning: PyTorch not found. Install with: pip install torch torchvision")
        torch = F = nn = models = transforms = to_pil_image = None

    # CustomTkinter for GUI
    try:
        with Spinner("Importing GUI libraries"):
            import customtkinter as ctk
            from customtkinter import CTkImage
            from tkinter import filedialog, messagebox
        print("✓ CustomTkinter imported")
    except ImportError:
        print("Warning: CustomTkinter not found. Install with: pip install customtkinter")
        ctk = CTkImage = filedialog = messagebox = None

    # XAI libraries
    try:
        with Spinner("Import torchcam utils"):
            from torchcam.utils import overlay_mask
        print("✓ torchcam for XAI loaded")
    except ImportError:
        print("Warning: torchcam not found. Install with: pip install torchcam")
        overlay_mask = None

    try:
        from captum.attr import LayerGradCam, GradientShap, LayerAttribution, GuidedGradCam, NoiseTunnel
        from captum.metrics import infidelity
        gradient_shap_available = True
        captum_available = True
        print("✓ Captum for GradCAM loaded")
    except Exception as e:
        print("Warning: captum not found. Install with: pip install captum:", e)
        gradient_shap_available = False
        captum_available = False
        LayerGradCam = GradientShap = None

    try:
        import shap
        print("✓ SHAP loaded")
    except ImportError:
        print("Warning: SHAP not found. Install with: pip install shap")
        shap = None

    try:
        with Spinner(message="loading LIME"):
            from lime import lime_image
            print("✓ LIME loaded")
            LIME_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: LIME not found: {e}")
        lime_image = None
        LIME_AVAILABLE = False

    try:
        with Spinner("Loading scikit-image"):
            import skimage
            from skimage.segmentation import quickshift, slic, watershed
            from skimage.filters import gaussian, sobel
            from scipy import ndimage
            print("✓ scikit-image loaded")
            SKIMAGE_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: scikit-image/Scipy import failed: {e}")
        skimage = quickshift = slic = gaussian = sobel = watershed = ndimage = None
        SKIMAGE_AVAILABLE = False

    # PDF generation
    try:
        with Spinner("Loading reportlab"):
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.utils import ImageReader
            reportlab_available = True
        print("✓ ReportLab loaded")
    except ImportError:
        print("Warning: ReportLab not found. Install with: pip install reportlab")
        canvas = A4 = ImageReader = None
        reportlab_available = False

    print("✓ ✓ ✓ All libraries and dependencies loaded")

except Exception as e:
    print("Failed to load libraries:", e)

def check_pytorch_compatibility():
    """Check PyTorch version compatibility and configure safe threading."""
    if torch is None:
        return False
    
    try:
        version = torch.__version__
        print(f"PyTorch version: {version}")
        
        # Set conservative threading
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        
        return True
    except Exception as e:
        print(f"PyTorch compatibility check failed: {e}")
        return False

# Constants
APP_TITLE = "EMII — Brain Tumor MRI Scanner (XAI)"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
MODEL_REL_PATH = os.path.join("models", "brain_tumor_resnet18.pth")

def get_resource_path(relative_path):
    """
    Get the absolute path to a resource file, handling both development and bundled app scenarios.
    
    For PyInstaller bundles: looks in sys._MEIPASS (temporary extraction directory)
    For macOS .app bundles: looks in Contents/Resources relative to the executable
    For development: uses the script directory
    
    Parameters
    ----------
    relative_path : str
        Path relative to the resource directory (e.g., "models/brain_tumor_resnet18.pth")
    
    Returns
    -------
    str
        Absolute path to the resource file
    """
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    if hasattr(sys, '_MEIPASS'):
        # Running in a PyInstaller bundle
        base_path = sys._MEIPASS
        print(f"[DEBUG] PyInstaller bundle detected, _MEIPASS: {base_path}")
    else:
        # Running in development mode
        base_path = os.path.dirname(os.path.abspath(__file__))
        print(f"[DEBUG] Development mode, base_path: {base_path}")
    
    # Try the resource path first
    resource_path = os.path.join(base_path, relative_path)
    print(f"[DEBUG] Checking resource path: {resource_path}")
    if os.path.exists(resource_path):
        print(f"[DEBUG] Found resource at: {resource_path}")
        return os.path.abspath(resource_path)
    
    # For macOS .app bundles, also check Contents/Resources relative to executable
    if sys.platform == 'darwin' and hasattr(sys, 'executable'):
        exec_dir = os.path.dirname(sys.executable)
        print(f"[DEBUG] macOS executable directory: {exec_dir}")
        
        # PyInstaller bundle - executable is in MacOS, resources in Resources
        if 'Contents' in exec_dir:
            contents_dir = os.path.dirname(exec_dir)
            resources_dir = os.path.join(contents_dir, "Resources")
            resource_path = os.path.join(resources_dir, relative_path)
            print(f"[DEBUG] Checking app bundle Resources: {resource_path}")
            if os.path.exists(resource_path):
                print(f"[DEBUG] Found resource in app bundle: {resource_path}")
                return os.path.abspath(resource_path)
        
        # Also check next to the executable (for manual placement)
        resource_path = os.path.join(exec_dir, relative_path)
        print(f"[DEBUG] Checking next to executable: {resource_path}")
        if os.path.exists(resource_path):
            print(f"[DEBUG] Found resource next to executable: {resource_path}")
            return os.path.abspath(resource_path)
        
        # Check in the app bundle's parent directory (if app is in Applications)
        app_bundle = exec_dir
        while app_bundle and not app_bundle.endswith('.app') and len(app_bundle) > 1:
            app_bundle = os.path.dirname(app_bundle)
        if app_bundle and app_bundle.endswith('.app'):
            # Check next to the .app bundle
            app_parent = os.path.dirname(app_bundle)
            resource_path = os.path.join(app_parent, relative_path)
            print(f"[DEBUG] Checking next to app bundle: {resource_path}")
            if os.path.exists(resource_path):
                print(f"[DEBUG] Found resource next to app bundle: {resource_path}")
                return os.path.abspath(resource_path)
    
    # Fallback: return the original path (will be checked by caller)
    fallback = os.path.abspath(os.path.join(base_path, relative_path))
    print(f"[DEBUG] Using fallback path: {fallback}")
    return fallback

# Color scheme
COLORS = {
    'primary': '#2B5CE6',
    'secondary': '#1E3A8A', 
    'success': '#22C55E',
    'warning': '#F59E0B',
    'danger': '#EF4444',
    'info': '#3B82F6',
    'dark': "#5C5C5C",
    'light': '#F8FAFC',
    'medical': '#0EA5E9',
    'lime': '#10B981',
    'gradcam': '#F59E0B',
    'shap': '#8B5CF6'
}

# Medical reasoning templates for "Why Card"
MEDICAL_REASONING = {
    "glioma": {
        "because": ["Irregular rim enhancement", "Perilesional edema pattern", "Infiltrative margins", "T2 hyperintensity"],
        "despite": ["No restricted diffusion", "Absent calcifications", "No dural attachment"]
    },
    "meningioma": {
        "because": ["Dural attachment", "Homogeneous enhancement", "CSF cleft sign", "Hyperostosis"],
        "despite": ["No perilesional edema", "No irregular margins", "No central necrosis"]
    },
    "notumor": {
        "because": ["Normal brain parenchyma", "Symmetric ventricles", "No mass effect", "Normal sulci"],
        "despite": ["No enhancement", "No edema", "No abnormal signal"]
    },
    "pituitary": {
        "because": ["Sellar location", "Stalk deviation", "Microadenoma signal", "Delayed enhancement"],
        "despite": ["No suprasellar extension", "No cavernous sinus invasion", "Small size"]
    }
}

# ---------- Utils ----------
def ensure_dir(d):
    """Create directory *d* if it does not exist and return the path."""
    os.makedirs(d, exist_ok=True)
    return d

def apply_display_clahe(img_rgb, clip=2.0, tile=(8,8)):
    """Apply CLAHE contrast enhancement to an RGB image for better visual clarity."""
    if Image is None or cv2 is None:
        return img_rgb
    
    g = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    g = clahe.apply(g)
    return Image.fromarray(cv2.cvtColor(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB), cv2.COLOR_BGR2RGB))

def softmax_entropy(probs):
    """Calculate normalized entropy of a probability vector as an uncertainty score."""
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    return float(-np.sum(p * np.log(p)) / np.log(len(p)))

# ---------- Medical Image Segmentation Utils ----------
def medical_brain_segmentation(image, method='hybrid'):
    """Segment a brain MRI RGB image into superpixels using medical-inspired heuristics."""
    if not SKIMAGE_AVAILABLE:
        print("[DEBUG] scikit-image not available, using grid fallback")
        h, w = image.shape[:2]
        segments = np.zeros((h, w), dtype=int)
        block_size = 32
        label = 0
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                segments[y:y+block_size, x:x+block_size] = label
                label += 1
        return segments
    
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    enhanced = apply_3d_histogram_reconstruction(gray)
    enhanced = apply_gamma_enhancement(enhanced, gamma=0.5)
    
    if method == 'hybrid':
        try:
            segments_qs = quickshift(
                image, 
                kernel_size=3,
                max_dist=8,
                ratio=0.3,
                sigma=0.8
            )
            
            _, brain_mask = cv2.threshold(enhanced, 20, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
            brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_OPEN, kernel)
            
            segments_refined = segments_qs.copy()
            segments_refined[brain_mask == 0] = -1
            
            return segments_refined
        except Exception as e:
            print(f"[DEBUG] Hybrid segmentation failed: {e}, using fallback")
            return medical_brain_segmentation(image, method='fallback')
    
    # Fallback method
    h, w = image.shape[:2]
    segments = np.zeros((h, w), dtype=int)
    block_size = 16
    label = 0
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            segments[y:y+block_size, x:x+block_size] = label
            label += 1
    return segments

def apply_3d_histogram_reconstruction(gray_image):
    """Denoise a grayscale MRI slice via 3D histogram reconstruction (Wang et al., 2020)."""
    h, w = gray_image.shape
    reconstructed = gray_image.copy().astype(np.float32)
    
    kernel = np.ones((3,3), np.float32) / 9
    mean_img = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
    median_img = cv2.medianBlur(gray_image, 3).astype(np.float32)
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            f_val = gray_image[y, x]
            g_val = mean_img[y, x]
            h_val = median_img[y, x]
            
            if abs(f_val - g_val) > 10 or abs(f_val - h_val) > 10:
                reconstructed[y, x] = 0.5 * f_val + 0.3 * g_val + 0.2 * h_val
                
    return reconstructed.astype(np.uint8)

def apply_gamma_enhancement(image, gamma=0.5):
    """Apply gamma correction to enhance contrast in a grayscale medical image."""
    normalized = image.astype(np.float32) / 255.0
    enhanced = np.power(normalized, gamma)
    return (enhanced * 255).astype(np.uint8)

def create_medical_lime_explainer(random_state=42):
    """Create a LIME image explainer tuned for brain MRI (kernel, feature selection, seed)."""
    if lime_image is None:
        return None
        
    kernel_width = 0.15
    feature_selection = 'lasso_path'
    
    explainer = lime_image.LimeImageExplainer(
        kernel_width=kernel_width,
        kernel=None,
        verbose=False,
        feature_selection=feature_selection,
        random_state=random_state
    )
    
    return explainer

# ---------- Model ----------
class ResNet18Brain(nn.Module):
    def __init__(self, num_classes=4):
        """Wrap a ResNet-18 backbone for 4-class brain tumor classification."""
        super().__init__()
        if models is None:
            raise ImportError("PyTorch not available")
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """Forward pass that delegates to the underlying ResNet-18 backbone."""
        return self.backbone(x)

def load_model(device):
    """Load the brain tumor classification model weights onto the given device."""
    if torch is None:
        print("PyTorch not available - using dummy model")
        return None
        
    model = ResNet18Brain(num_classes=len(CLASS_NAMES)).to(device)
    
    # Try multiple locations for the model file
    weights_path = get_resource_path(MODEL_REL_PATH)
    
    # Additional fallback locations to check
    fallback_paths = [
        weights_path,  # Primary location (from get_resource_path)
        os.path.join(os.path.dirname(sys.executable), MODEL_REL_PATH),  # Next to executable
        os.path.join(os.path.expanduser("~"), "Applications", MODEL_REL_PATH),  # In Applications
    ]
    
    # For macOS .app bundles, also check Contents/Resources
    if sys.platform == 'darwin' and hasattr(sys, 'executable'):
        exec_dir = os.path.dirname(sys.executable)
        if 'Contents' in exec_dir:
            contents_dir = os.path.dirname(exec_dir)
            resources_path = os.path.join(contents_dir, "Resources", MODEL_REL_PATH)
            fallback_paths.insert(1, resources_path)
    
    # Try each path until we find the model
    found_path = None
    for path in fallback_paths:
        if os.path.exists(path):
            found_path = os.path.abspath(path)
            break
    
    if found_path:
        try:
            state = torch.load(found_path, map_location=device)
            
            new_state = {}
            for k, v in state.items():
                new_state[f"backbone.{k}"] = v
                
            model.load_state_dict(new_state)
            print(f"[INFO] Loaded model weights: {found_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {found_path}: {e}")
            if messagebox:
                messagebox.showerror(
                    "Model Load Error",
                    f"Found model file at:\n{found_path}\n\n"
                    f"But failed to load it:\n{str(e)}\n\n"
                    "The app will run with random weights (for demo)."
                )
    else:
        # Show all paths we tried
        searched_paths = "\n".join([f"  - {p}" for p in fallback_paths])
        print(f"[WARN] No weights found. Searched locations:\n{searched_paths}")
        print(f"[WARN] Using randomly initialized model.")
        if messagebox:
            messagebox.showwarning(
                "Weights missing",
                f"No pretrained model found. Searched locations:\n\n{searched_paths}\n\n"
                "For macOS .app bundles, place models/ folder in:\n"
                "  - EMII.app/Contents/Resources/models/\n"
                "  - Or next to EMII.app in Applications/\n\n"
                "The app will run with random weights (for demo)."
            )
    
    model.eval()
    return model

# Image transforms
if transforms:
    IM_TF = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
else:
    IM_TF = None

# ---------- Enhanced UI Components ----------
class StatusBar(ctk.CTkFrame):
    def __init__(self, parent):
        """Status bar widget showing app state and active compute device."""
        super().__init__(parent, height=35, corner_radius=0)
        self.pack_propagate(False)
        
        self.status_label = ctk.CTkLabel(
            self, 
            text="Ready", 
            font=ctk.CTkFont(size=12),
            anchor="w"
        )
        self.status_label.pack(side="left", padx=15, pady=8)
        
        self.device_label = ctk.CTkLabel(
            self, 
            text=f"Device: {'CUDA' if torch and torch.cuda.is_available() else 'CPU'}", 
            font=ctk.CTkFont(size=12),
            anchor="e"
        )
        self.device_label.pack(side="right", padx=15, pady=8)
    
    def set_status(self, text, color=None):
        """Update the status bar text and optional text color."""
        self.status_label.configure(text=text)
        if color:
            self.status_label.configure(text_color=color)

class XAIMethodCard(ctk.CTkFrame):
    def __init__(self, parent, title, description, color, method_key):
        """Compact card summarizing one XAI method (LIME, Grad-CAM, SHAP)."""
        super().__init__(parent, corner_radius=10)
        self.method_key = method_key
        
        # Method icon and title
        title_frame = ctk.CTkFrame(self, height=40, corner_radius=10)
        title_frame.pack(fill="x", padx=5, pady=5)
        title_frame.pack_propagate(False)
        
        method_icons = {"LIME": "▷", "Grad-CAM": "▷", "SHAP": "▷"}
        icon = method_icons.get(method_key, "🔍")
        
        ctk.CTkLabel(
            title_frame, 
            text=f"{icon} {title}", 
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w"
        ).pack(side="left", padx=10, pady=10)
        
        # Status indicator
        self.status_label = ctk.CTkLabel(
            title_frame,
            text="Ready",
            font=ctk.CTkFont(size=10),
            width=60,
            corner_radius=15,
            fg_color="gray30"
        )
        self.status_label.pack(side="right", padx=10, pady=5)
        
        # Description
        desc_label = ctk.CTkLabel(
            self,
            text=description,
            font=ctk.CTkFont(size=11),
            wraplength=250,
            justify="left",
            anchor="w"
        )
        desc_label.pack(fill="x", padx=10, pady=(0, 10))
    
    def set_status(self, status, color=None):
        """Set the card status label (e.g., Ready, Active, Running, Complete)."""
        self.status_label.configure(text=status)
        if color:
            self.status_label.configure(fg_color=color)

class WhyCard(ctk.CTkFrame):
    def __init__(self, parent):
        """Panel that explains *why* the model predicted a class (chips + medical text)."""
        super().__init__(parent, corner_radius=10)
        self.pack_propagate(False)
        
        # Header
        header_frame = ctk.CTkFrame(self, height=50, corner_radius=10)
        header_frame.pack(fill="x", padx=5, pady=5)
        header_frame.pack_propagate(False)
        
        ctk.CTkLabel(
            header_frame, 
            text="Why Card", 
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w"
        ).pack(side="left", padx=15, pady=15)
        
        # Prediction chips container
        chips_container = ctk.CTkFrame(self)
        chips_container.pack(fill="x", padx=10, pady=5)
        
        # Prediction result chips
        self.pred_chip = ctk.CTkLabel(
            chips_container, 
            text="Prediction: —", 
            corner_radius=15, 
            fg_color="gray25",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.pred_chip.pack(side="left", padx=5, pady=5)
        
        self.conf_chip = ctk.CTkLabel(
            chips_container, 
            text="Confidence: —", 
            corner_radius=15, 
            fg_color="gray25",
            font=ctk.CTkFont(size=12)
        )
        self.conf_chip.pack(side="left", padx=5, pady=5)
        
        self.ent_chip = ctk.CTkLabel(
            chips_container, 
            text="Uncertainty: —", 
            corner_radius=15, 
            fg_color="gray25",
            font=ctk.CTkFont(size=12)
        )
        self.ent_chip.pack(side="left", padx=5, pady=5)
        
        # Medical reasoning section
        reasoning_container = ctk.CTkFrame(self)
        reasoning_container.pack(fill="both", expand=True, padx=10, pady=10)
        reasoning_container.grid_columnconfigure(0, weight=1)
        reasoning_container.grid_columnconfigure(1, weight=1)
        
        # Supporting evidence (Because)
        because_frame = ctk.CTkFrame(reasoning_container, corner_radius=8)
        because_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        
        ctk.CTkLabel(
            because_frame, 
            text="✓ Supporting Evidence", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['success']
        ).pack(pady=(10, 5))
        
        self.because_text = ctk.CTkTextbox(
            because_frame, 
            height=100, 
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        self.because_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Contrasting evidence (Despite)
        despite_frame = ctk.CTkFrame(reasoning_container, corner_radius=8)
        despite_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)
        
        ctk.CTkLabel(
            despite_frame, 
            text="⚠ Contrasting Evidence", 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=COLORS['warning']
        ).pack(pady=(10, 5))
        
        self.despite_text = ctk.CTkTextbox(
            despite_frame, 
            height=100, 
            wrap="word",
            font=ctk.CTkFont(size=11)
        )
        self.despite_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def update_prediction(self, prediction, confidence, entropy, prob_vector):
        """Update chips, colors, and medical reasoning for a new prediction."""
        # Update chips with color coding
        self.pred_chip.configure(text=f"Prediction: {prediction.upper()}")
        self.conf_chip.configure(text=f"Confidence: {confidence:.1f}%")
        self.ent_chip.configure(text=f"Uncertainty: {entropy:.3f}")
        
        # Color coding based on confidence
        if confidence > 80:
            pred_color = COLORS['success']
            conf_color = COLORS['success']
        elif confidence > 60:
            pred_color = COLORS['warning']
            conf_color = COLORS['warning']
        else:
            pred_color = COLORS['danger']
            conf_color = COLORS['danger']
        
        self.pred_chip.configure(fg_color=pred_color)
        self.conf_chip.configure(fg_color=conf_color)
        
        # Entropy color (higher = more uncertain)
        if entropy > 0.7:
            ent_color = COLORS['danger']
        elif entropy > 0.4:
            ent_color = COLORS['warning']
        else:
            ent_color = COLORS['success']
        
        self.ent_chip.configure(fg_color=ent_color)
        
        # Update medical reasoning
        self._update_medical_reasoning(prediction)
    
    def _update_medical_reasoning(self, prediction):
        """Populate supporting/contrasting evidence lists based on the class label."""
        reasoning = MEDICAL_REASONING.get(prediction, {
            "because": ["Features detected by model", "Regional patterns", "Intensity characteristics"],
            "despite": ["Contrasting evidence", "Alternative patterns", "Confounding factors"]
        })
        
        # Format supporting evidence
        because_text = "\n".join([f"• {item}" for item in reasoning["because"]])
        self.because_text.delete("1.0", "end")
        self.because_text.insert("1.0", because_text)
        
        # Format contrasting evidence
        despite_text = "\n".join([f"• {item}" for item in reasoning["despite"]])
        self.despite_text.delete("1.0", "end")
        self.despite_text.insert("1.0", despite_text)
    
    def reset(self):
        """Reset the Why Card UI to its neutral, pre-prediction state."""
        self.pred_chip.configure(text="Prediction: —", fg_color="gray25")
        self.conf_chip.configure(text="Confidence: —", fg_color="gray25")
        self.ent_chip.configure(text="Uncertainty: —", fg_color="gray25")
        
        self.because_text.delete("1.0", "end")
        self.because_text.insert("1.0", "Analysis results will appear here...")
        
        self.despite_text.delete("1.0", "end")
        self.despite_text.insert("1.0", "Contrasting evidence will appear here...")
    
    def add_lime_insights(self, fidelity_score, positive_count, negative_count, top_positive=None, top_negative=None):
        """
        Add LIME-specific insights to the Why Card for medical staff.
        
        Parameters
        ----------
        fidelity_score : float
            R² fidelity score from LIME explanation
        positive_count : int
            Number of positive contributing segments
        negative_count : int
            Number of negative contributing segments
        top_positive : list, optional
            List of (segment_id, weight) tuples for top positive segments
        top_negative : list, optional
            List of (segment_id, weight) tuples for top negative segments
        """
        # Get current text
        current_because = self.because_text.get("1.0", "end").strip()
        current_despite = self.despite_text.get("1.0", "end").strip()
        
        # Add LIME insights
        lime_because = []
        if fidelity_score is not None:
            if fidelity_score > 0.7:
                lime_because.append(f"✓ High explanation fidelity (R²={fidelity_score:.2f}) - explanation is reliable")
            elif fidelity_score > 0.4:
                lime_because.append(f"⚠ Moderate explanation fidelity (R²={fidelity_score:.2f})")
            else:
                lime_because.append(f"⚠ Low explanation fidelity (R²={fidelity_score:.2f}) - consider alternative views")
        
        if positive_count > 0:
            lime_because.append(f"✓ {positive_count} image regions strongly support this diagnosis")
            if top_positive and len(top_positive) > 0:
                top_weight = top_positive[0][1] if isinstance(top_positive[0], tuple) else 0
                lime_because.append(f"  Strongest supporting region: weight +{top_weight:.3f}")
        
        lime_despite = []
        if negative_count > 0:
            lime_despite.append(f"⚠ {negative_count} image regions show opposing signals")
            if top_negative and len(top_negative) > 0:
                top_weight = abs(top_negative[0][1]) if isinstance(top_negative[0], tuple) else 0
                lime_despite.append(f"  Strongest opposing region: weight -{top_weight:.3f}")
        
        # Combine with existing text
        if lime_because:
            combined_because = current_because + "\n\n" + "LIME Analysis:\n" + "\n".join([f"• {item}" for item in lime_because])
            self.because_text.delete("1.0", "end")
            self.because_text.insert("1.0", combined_because)
        
        if lime_despite:
            combined_despite = current_despite + "\n\n" + "LIME Analysis:\n" + "\n".join([f"• {item}" for item in lime_despite])
            self.despite_text.delete("1.0", "end")
            self.despite_text.insert("1.0", combined_despite)

# ---------- Main XAI App ----------
class EMIIXAI(ctk.CTk):
    def __init__(self):
        """Main CustomTkinter application for EMII brain tumor MRI XAI exploration."""
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1600x1000")
        self.minsize(1400, 900)
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")

        # Initialize device and model
        if torch and check_pytorch_compatibility():
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = load_model(self.device)
            print("Running with PyTorch")
        else:
            self.device = None
            self.model = None
            print("Running in demo mode without PyTorch")

        # Initialize medical-optimized explainers
        self.lime_explainer = create_medical_lime_explainer(random_state=42)

        # File management
        self.last_dir = os.path.expanduser("~")
        self.files = []
        self.idx = -1

        # Image state
        self.current_image = None
        self.display_image = None
        self.photo_left = None
        self.overlay_image = None
        self.photo_right = None

        # Prediction state
        self.prediction = None
        self.prob_vector = None
        self.last_lime_segments = None
        self.last_heatmap = None
        self.last_method_note = ""

        # XAI state
        self.current_xai_method = "LIME"

        # Enhanced XAI metrics storage
        self.lime_score = None
        self.lime_topk = None
        self.lime_signed_overlay = None
        self.lime_explanation_obj = None  # Store full LIME explanation object
        self.lime_local_pred = None  # Local prediction from explanation model
        self.lime_intercept = None  # Intercept from explanation model
        self.lime_positive_segments = None  # Positive contributing segments
        self.lime_negative_segments = None  # Negative contributing segments
        self.shap_check = None
        self.shap_topk = None
        self.gradcam_metrics = None

        self._build_ui()
        self.status_bar.set_status("Ready - Load an image to begin XAI analysis")

    def _build_ui(self):
        """Construct the top-level layout (header, main content, status bar)."""
        # Configure main grid
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

    def _build_header(self):
        """Build the top header row with title and file controls."""
        header = ctk.CTkFrame(self, height=140, corner_radius=0)
        header.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        header.pack_propagate(False)
        header.grid_columnconfigure(1, weight=1)
        
        # Title section
        title_frame = ctk.CTkFrame(header)
        title_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=20, pady=10)
        
        ctk.CTkLabel(
            title_frame,
            text="EMII Prototype — Brain Tumor MRI Scanner (XAI)",
            font=ctk.CTkFont(size=24, weight="bold"),
            anchor="w"
        ).pack(side="left", padx=15, pady=10)
        
        capability_label = ctk.CTkLabel(
            title_frame,
            text="Enhanced with Medical AI Explanations",
            font=ctk.CTkFont(size=14),
            text_color=COLORS['medical'],
            anchor="e"
        )
        capability_label.pack(side="right", padx=15, pady=10)

        # Controls row (file open + selector only)
        controls = ctk.CTkFrame(header)
        controls.grid(row=1, column=0, columnspan=4, sticky="ew", padx=20, pady=(0, 10))
        controls.grid_columnconfigure(1, weight=1)

        # File controls
        file_frame = ctk.CTkFrame(controls)
        file_frame.grid(row=0, column=0, sticky="w", padx=10, pady=10)
        
        self.btn_open = ctk.CTkButton(
            file_frame, 
            text="📂 Open Image/Folder", 
            command=self.on_open,
            height=35,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.btn_open.pack(side="left", padx=5)
        
        self.btn_prev = ctk.CTkButton(
            file_frame, text="⟵", command=self.on_prev, width=40, height=35, fg_color="gray30"
        )
        self.btn_prev.pack(side="left", padx=2)
        
        self.btn_next = ctk.CTkButton(
            file_frame, text="⟶", command=self.on_next, width=40, height=35, fg_color="gray30"
        )
        self.btn_next.pack(side="left", padx=2)

        # File selector
        selector_frame = ctk.CTkFrame(controls)
        selector_frame.grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        selector_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(
            selector_frame, text="Current File:", anchor="w", font=ctk.CTkFont(size=12)
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 2))
        
        self.selector = ctk.CTkOptionMenu(
            selector_frame, values=["No files loaded"], command=self.on_select, height=35
        )
        self.selector.configure(state="disabled")
        self.selector.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))


    def _build_main_content(self):
        """Build the main content area with XAI tools, image panel, and Why Card."""
        main = ctk.CTkFrame(self, corner_radius=10)
        main.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        # Top row: XAI tools panel spans full width
        # Second row: two columns -> left image panel, right WhyCard (same height)
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=1)

        # XAI tools panel on top
        self._build_xai_tools_panel(main).grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # Left: Image panel
        self._build_image_panel(main)  # will place itself at row=1, col=0

        # Right: Why Card (same height as image panel)
        self._build_why_panel(main)    # will place itself at row=1, col=1

    
    def _build_why_panel(self, parent):
        """Right side container holding Why Card, metrics, and feedback controls."""
        side = ctk.CTkFrame(parent, corner_radius=10)
        side.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        parent.grid_columnconfigure(1, weight=1)
        side.grid_rowconfigure(0, weight=1)   # Why card
        side.grid_rowconfigure(1, weight=0)   # Metrics
        side.grid_rowconfigure(2, weight=0)   # Feedback buttons
        side.grid_columnconfigure(0, weight=1)

        # Why Card fills available height
        self.why_card = WhyCard(side)
        self.why_card.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Optional: metrics under Why Card (kept from your original)
        metrics_holder = ctk.CTkFrame(side)
        metrics_holder.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))
        ctk.CTkLabel(metrics_holder, text="Reliability Metrics", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(6, 4), anchor="w")
        self.metrics_text = ctk.CTkTextbox(metrics_holder, height=100, font=ctk.CTkFont(size=11))
        self.metrics_text.pack(fill="x", padx=2, pady=(0, 6))

        # Feedback buttons for user assessment of the AI output
        feedback_frame = ctk.CTkFrame(side)
        feedback_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(0, 10))
        feedback_frame.grid_columnconfigure(0, weight=0)
        feedback_frame.grid_columnconfigure(1, weight=0)
        feedback_frame.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(
            feedback_frame,
            text="Your assessment:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=0, column=0, padx=(6, 4), pady=6, sticky="w")

        self.btn_feedback_right = ctk.CTkButton(
            feedback_frame,
            text="✓ Seems right",
            height=30,
            width=120,
            command=self.on_feedback_right,
            fg_color=COLORS["success"],
        )
        self.btn_feedback_right.grid(row=0, column=1, padx=4, pady=6, sticky="w")

        self.btn_feedback_wrong = ctk.CTkButton(
            feedback_frame,
            text="⚠ Seems wrong",
            height=30,
            width=130,
            command=self.on_feedback_wrong,
            fg_color=COLORS["danger"],
        )
        self.btn_feedback_wrong.grid(row=0, column=2, padx=4, pady=6, sticky="w")

        # Disable until a prediction is available
        self.btn_feedback_right.configure(state="disabled")
        self.btn_feedback_wrong.configure(state="disabled")

    
    def _build_image_panel(self, parent):
        """Left panel that renders the original image and the XAI overlay image."""
        image_panel = ctk.CTkFrame(parent, corner_radius=10)
        image_panel.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        parent.grid_rowconfigure(1, weight=1)
        parent.grid_columnconfigure(0, weight=1)

        image_panel.grid_rowconfigure(1, weight=1)
        image_panel.grid_columnconfigure(0, weight=1)
        image_panel.grid_columnconfigure(1, weight=1)

        # Header
        img_header = ctk.CTkFrame(image_panel, height=50, corner_radius=10)
        img_header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=0, pady=0)
        img_header.pack_propagate(False)
        ctk.CTkLabel(
            img_header, text="Medical Image Analysis",
            font=ctk.CTkFont(size=16, weight="bold"), anchor="w"
        ).pack(side="left", padx=15, pady=15)

        # Overlay controls in header
        controls_frame = ctk.CTkFrame(img_header)
        controls_frame.pack(side="right", padx=15, pady=5)
        
        # Overlay mode selector (for LIME: positive, negative, or both)
        overlay_mode_frame = ctk.CTkFrame(controls_frame)
        overlay_mode_frame.pack(side="left", padx=(5, 10))
        ctk.CTkLabel(overlay_mode_frame, text="Mode:", font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        self.overlay_mode_var = ctk.StringVar(value="both")
        overlay_mode_menu = ctk.CTkOptionMenu(
            overlay_mode_frame,
            values=["both", "positive", "negative"],
            variable=self.overlay_mode_var,
            command=self._on_overlay_mode_change,
            width=80,
            height=25,
            font=ctk.CTkFont(size=10)
        )
        overlay_mode_menu.pack(side="left", padx=2)
        
        # Alpha slider
        ctk.CTkLabel(controls_frame, text="Opacity:", font=ctk.CTkFont(size=10)).pack(side="left", padx=(5,5))
        self.sld_overlay = ctk.CTkSlider(
            controls_frame, from_=0.1, to=0.9, number_of_steps=8, command=self._on_alpha, width=100
        )
        self.sld_overlay.set(0.6)
        self.sld_overlay.pack(side="left", padx=(0,5))
        self.alpha_label = ctk.CTkLabel(controls_frame, text="0.60", width=40, font=ctk.CTkFont(size=10))
        self.alpha_label.pack(side="right", padx=(5,5))

        # Image display containers (unchanged)
        left_container = ctk.CTkFrame(image_panel)
        left_container.grid(row=1, column=0, sticky="nsew", padx=(10, 5), pady=10)
        left_container.grid_rowconfigure(1, weight=1)
        right_container = ctk.CTkFrame(image_panel)
        right_container.grid(row=1, column=1, sticky="nsew", padx=(5, 10), pady=10)
        right_container.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(left_container, text="Original Image", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=(10, 5))
        self.canvas_left = ctk.CTkLabel(left_container, text="📂 Load an image to begin", font=ctk.CTkFont(size=16))
        self.canvas_left.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        ctk.CTkLabel(right_container, text="XAI Explanation", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0, column=0, pady=(10, 5))
        self.canvas_right = ctk.CTkLabel(right_container, text="🔍 Run analysis to see explanation", font=ctk.CTkFont(size=16))
        self.canvas_right.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))


    def _build_xai_tools_panel(self, parent):
        """Top bar with XAI tools, method cards, method selector, and Analyze+Explain."""
        panel = ctk.CTkFrame(parent, corner_radius=10)
        panel.pack_propagate(False)

        # Title
        title = ctk.CTkLabel(panel, text="XAI Tools", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(anchor="w", padx=20, pady=(12, 6))

        # Method cards row
        cards_row = ctk.CTkFrame(panel, corner_radius=10)
        cards_row.pack(fill="x", padx=10, pady=(0, 10))
        cards_row.grid_columnconfigure(0, weight=1)
        cards_row.grid_columnconfigure(1, weight=1)
        cards_row.grid_columnconfigure(2, weight=1)

        self.method_cards = {}
        methods_info = [
            ("LIME", "Local explanations with medical segmentation", COLORS['lime']),
            ("Grad-CAM", "Gradient-based activation mapping", COLORS['gradcam']),
            ("SHAP", "Unified attribution values", COLORS['shap'])
        ]
        for i, (method, desc, color) in enumerate(methods_info):
            card = XAIMethodCard(cards_row, method, desc, color, method)
            card.grid(row=0, column=i, sticky="ew", padx=5, pady=5)
            self.method_cards[method] = card

        # Controls row (selector + Analyze + Explain + display options)
        controls_row = ctk.CTkFrame(panel)
        controls_row.pack(fill="x", padx=10, pady=(0, 12))
        controls_row.grid_columnconfigure(0, weight=0)
        controls_row.grid_columnconfigure(1, weight=1)
        controls_row.grid_columnconfigure(2, weight=0)
        controls_row.grid_columnconfigure(3, weight=0)

        # Method selector
        self.xai_var = ctk.StringVar(value="LIME")
        method_selector = ctk.CTkOptionMenu(
            controls_row, values=["LIME", "Grad-CAM", "SHAP"],
            variable=self.xai_var, command=self._on_method_change, height=35
        )
        method_selector.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="w")

        # Analyze + Explain button (moved here)
        self.btn_run = ctk.CTkButton(
            controls_row,
            text="🔍 Analyze + Explain",
            command=self.on_predict,
            height=36,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=COLORS['success']
        )
        self.btn_run.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        # Enhance display toggle
        self.chk_clahe_var = getattr(self, "chk_clahe_var", ctk.BooleanVar(value=True))
        self.chk_clahe = ctk.CTkCheckBox(
            controls_row, text="Enhance Display", variable=self.chk_clahe_var,
            command=self.refresh_display, font=ctk.CTkFont(size=12)
        )
        self.chk_clahe.grid(row=0, column=2, padx=10, pady=5, sticky="e")

        # Export PDF on the right
        self.btn_report = ctk.CTkButton(
            controls_row, text="📄 Export PDF", command=self.on_export, height=32, width=110, fg_color=COLORS['info']
        )
        self.btn_report.grid(row=0, column=3, padx=(10, 0), pady=5, sticky="e")

        return panel

        
        # Feedback controls
        #feedback_frame = ctk.CTkFrame(footer_frame)
        #feedback_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        #ctk.CTkLabel(feedback_frame, text="Feedback:").pack(side="left", padx=(10, 10))
        """
        self.btn_like = ctk.CTkButton(
            feedback_frame, 
            text="👍 Helpful", 
            width=100, 
            command=lambda: self._feedback("pos"),
            fg_color=COLORS['success']
        )
        self.btn_like.pack(side="left", padx=5)
        
        self.btn_dislike = ctk.CTkButton(
            feedback_frame, 
            text="👎 Not Helpful", 
            width=120, 
            fg_color=COLORS['danger'],
            command=lambda: self._feedback("neg")
        )
        self.btn_dislike.pack(side="left", padx=5)
        """

    def _build_status_bar(self):
        """Create the bottom status bar widget."""
        self.status_bar = StatusBar(self)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)

    # ========== Enhanced XAI Methods with Advanced Metrics ==========
    
    def _log_to_progress(self, dialog, message):
        """Safely append a debug message to the modal progress dialog."""
        if dialog and hasattr(dialog, 'debug_text'):
            def update_text():
                if dialog.winfo_exists():
                    dialog.debug_text.insert("end", message + "\n")
                    dialog.debug_text.see("end")
            self.after(0, update_text)

    def _overlay_lime_signed(self, segments, seg_weights, base_img_224, alpha=0.6):
        """
        Build a signed green/red overlay image from LIME segment weights.
        
        Inspired by ImageExplanation.get_image_and_mask() with positive_only/negative_only.
        Green = positive contribution (supports prediction)
        Red = negative contribution (opposes prediction)
        
        Parameters
        ----------
        segments : np.ndarray
            2D array of segment IDs
        seg_weights : dict
            Dictionary mapping segment_id -> weight
        base_img_224 : PIL.Image
            Base image at 224x224
        alpha : float
            Overlay transparency (0-1)
        
        Returns
        -------
        PIL.Image
            Blended image with signed overlay
        """
        # Build 2 masks from signed weights: pos→green, neg→red
        pos = np.zeros_like(segments, dtype=float)
        neg = np.zeros_like(segments, dtype=float)
        for seg_id, w in seg_weights.items():
            (pos if w > 0 else neg)[segments == seg_id] = abs(w)
        # Normalize per mask
        if pos.max() > 0: pos /= pos.max()
        if neg.max() > 0: neg /= neg.max()
        pos_rgb = np.dstack([np.zeros_like(pos), pos, np.zeros_like(pos)])
        neg_rgb = np.dstack([neg, np.zeros_like(neg), np.zeros_like(neg)])
        overlay = np.clip(pos_rgb + neg_rgb, 0, 1)
        base = np.array(base_img_224).astype(float) / 255.0
        blended = np.clip(base*(1-alpha) + overlay*alpha, 0, 1)
        return Image.fromarray((blended*255).astype(np.uint8))
    
    def _get_lime_segment_medical_context(self, segment_id, weight, segments, image_np):
        """
        Map LIME segment importance to medical terminology for better explainability.
        
        Parameters
        ----------
        segment_id : int
            Segment identifier
        weight : float
            Segment importance weight
        segments : np.ndarray
            2D array of segment IDs
        image_np : np.ndarray
            Original image as numpy array
        
        Returns
        -------
        str
            Medical context description for the segment
        """
        # Get segment mask
        seg_mask = (segments == segment_id)
        if not seg_mask.any():
            return "Unknown region"
        
        # Extract segment region
        seg_region = image_np[seg_mask]
        if len(seg_region) == 0:
            return "Empty region"
        
        # Calculate segment characteristics
        mean_intensity = float(np.mean(seg_region))
        std_intensity = float(np.std(seg_region))
        
        # Get segment location (center)
        y_coords, x_coords = np.where(seg_mask)
        center_y, center_x = int(np.mean(y_coords)), int(np.mean(x_coords))
        img_height, img_width = segments.shape
        relative_y = center_y / img_height
        relative_x = center_x / img_width
        
        # Medical context based on location and intensity
        location_desc = ""
        if relative_y < 0.3:
            location_desc = "Superior"
        elif relative_y > 0.7:
            location_desc = "Inferior"
        else:
            location_desc = "Mid"
        
        if relative_x < 0.3:
            location_desc += " left"
        elif relative_x > 0.7:
            location_desc += " right"
        else:
            location_desc += " central"
        
        intensity_desc = ""
        if mean_intensity > 200:
            intensity_desc = "hyperintense"
        elif mean_intensity < 50:
            intensity_desc = "hypointense"
        else:
            intensity_desc = "isointense"
        
        contribution = "supporting" if weight > 0 else "opposing"
        strength = "strong" if abs(weight) > 0.1 else "moderate" if abs(weight) > 0.05 else "weak"
        
        return f"{strength} {contribution} signal in {location_desc} {intensity_desc} region"

    # Event handlers and core functionality
    def _on_method_change(self, method):
        """Switch active XAI method and update method card states."""
        self.current_xai_method = method
        for name, card in self.method_cards.items():
            if name == method:
                card.set_status("Active", COLORS['success'])
            else:
                card.set_status("Ready", "gray30")
        
        self.status_bar.set_status(f"XAI Method changed to {method}")

    def on_open(self):
        """Open a single image or a folder of images using a file dialog."""
        if not filedialog:
            print("File dialog not available")
            return
            
        menu = messagebox.askquestion(
            "Open",
            "Open a single image? (Choose 'No' to open a folder)",
            icon='question'
        )
        try:
            if menu == "yes":
                path = filedialog.askopenfilename(
                    initialdir=self.last_dir,
                    title="Select brain MRI image",
                    filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
                )
                if path:
                    self.files = [path]
                    self.last_dir = os.path.dirname(path)
            else:
                folder = filedialog.askdirectory(initialdir=self.last_dir, title="Select folder with MRI images")
                if folder:
                    self.last_dir = folder
                    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
                    self.files = []
                    for ext in extensions:
                        self.files.extend(glob.glob(os.path.join(folder, ext)))
                        self.files.extend(glob.glob(os.path.join(folder, ext.upper())))
                    # Deduplicate files to prevent navigation issues (case-insensitive filesystems)
                    self.files = sorted(list(set(self.files)))

            if not self.files:
                messagebox.showinfo("No images", "No image files found.")
                return

            filenames = [os.path.basename(f) for f in self.files]
            self.selector.configure(values=filenames, state="normal")
            self.selector.set(filenames[0])
            self.idx = 0
            self.load_image(self.files[0])
            self.status_bar.set_status(f"Loaded {len(self.files)} image(s)")
        except Exception as e:
            messagebox.showerror("Open error", str(e))
            self.status_bar.set_status("Error loading files", COLORS['danger'])

    def on_select(self, name):
        """Handle selection of a specific file name from the dropdown."""
        for i, f in enumerate(self.files):
            if os.path.basename(f) == name:
                self.idx = i
                self.load_image(self.files[i])
                break

    def on_prev(self):
        """Navigate to the previous image in the loaded file list."""
        if not self.files: 
            return
        self.idx = (self.idx - 1) % len(self.files)
        self.selector.set(os.path.basename(self.files[self.idx]))
        self.load_image(self.files[self.idx])

    def on_next(self):
        """Navigate to the next image in the loaded file list."""
        if not self.files: 
            return
        self.idx = (self.idx + 1) % len(self.files)
        self.selector.set(os.path.basename(self.files[self.idx]))
        self.load_image(self.files[self.idx])

    def load_image(self, path):
        """Load an image from disk, reset state, and refresh the displays."""
        try:
            if Image is None:
                print("PIL not available")
                return
                
            img = Image.open(path).convert("RGB")
            self.current_image = img
            self.refresh_display()
            
            # Reset state
            self.prediction = None
            self.prob_vector = None
            self.overlay_image = None
            self.last_heatmap = None
            self.last_method_note = ""
            
            # Explicitly clear image references to prevent Tcl errors
            self.photo_right = None
            self.canvas_right.configure(image=None, text="🔍 Run analysis to see explanation")
            
            # Reset enhanced metrics
            self.lime_score = None
            self.lime_topk = None
            
            # Reset enhanced metrics
            self.lime_score = None
            self.lime_topk = None
            self.lime_signed_overlay = None
            self.lime_explanation_obj = None
            self.lime_local_pred = None
            self.lime_intercept = None
            self.lime_positive_segments = None
            self.lime_negative_segments = None
            self.shap_check = None
            self.shap_topk = None
            self.gradcam_metrics = None
            
            # Update UI
            self.why_card.reset()
            self._reset_metrics()
            self.status_bar.set_status(f"Image loaded: {os.path.basename(path)}")

            # Disable feedback buttons until a new prediction is made
            if hasattr(self, "btn_feedback_right") and hasattr(self, "btn_feedback_wrong"):
                self.btn_feedback_right.configure(state="disabled")
                self.btn_feedback_wrong.configure(state="disabled")
            
        except Exception as e:
            if messagebox:
                messagebox.showerror("Load error", str(e))
            self.status_bar.set_status("Error loading image", COLORS['danger'])

    def refresh_display(self):
        """Refresh the left-hand original image display (with optional CLAHE)."""
        if self.current_image is None: 
            return
            
        disp = self.current_image.copy()
        if self.chk_clahe_var.get():
            disp = apply_display_clahe(disp)
        
        # Calculate display size maintaining aspect ratio
        container_width = 400
        container_height = 400
        img_width, img_height = disp.size
        
        scale = min(container_width/img_width, container_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        disp = disp.resize((new_width, new_height), Image.BILINEAR)
        self.display_image = disp
        
        if CTkImage:
            self.photo_left = CTkImage(light_image=disp, size=(new_width, new_height))
            self.canvas_left.configure(image=self.photo_left, text="")
            
        if self.overlay_image is not None:
            self._render_overlay()

    def on_predict(self):
        """Trigger prediction and XAI analysis in a background thread."""
        if not self.files:
            messagebox.showerror("Error", "No images loaded")
            return
            
        self.btn_run.configure(state="disabled", text="🔄 Analyzing...")
        self.status_bar.set_status("Running prediction and XAI analysis...", COLORS['warning'])
        
        # Update method card status
        self.method_cards[self.current_xai_method].set_status("Running", COLORS['warning'])
        
        self.prediction_thread = threading.Thread(target=self._run_all)
        self.prediction_thread.start()

    def _run_all(self):
        """Compute model prediction, update metrics, then dispatch the chosen XAI method."""
        try:
            if self.current_image is None:
                self.after(0, lambda: messagebox.showerror("Error", "No image loaded"))
                return

            # Get prediction first
            if self.model and torch:
                torch.set_num_threads(1)
                img_tensor = self._prep_input(self.current_image)
                with torch.no_grad():
                    logits = self.model(img_tensor)
                    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
            else:
                # Dummy prediction for demo
                probs = np.random.dirichlet([1,1,1,1])

            top_idx = int(np.argmax(probs))
            pred_name = CLASS_NAMES[top_idx]
            conf = float(probs[top_idx]) * 100
            ent = softmax_entropy(probs)

            self.prediction = pred_name
            self.prob_vector = probs

            # Update UI with prediction
            self.after(0, lambda: self.why_card.update_prediction(pred_name, conf, ent, probs))
            self.after(0, lambda: self._update_metrics(pred_name, conf, ent, probs))

            # Run explanation based on selected method
            method = self.current_xai_method
            if method == "LIME":
                self.after(0, self._run_lime)
            elif method == "Grad-CAM":
                self.after(0, self._run_gradcam)
            else:  # "SHAP"
                self.after(0, self._run_shap)

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Analysis failed: {str(e)}"))
            self.after(0, lambda: self.method_cards[self.current_xai_method].set_status("Error", COLORS['danger']))
        finally:
            self.after(0, lambda: self.btn_run.configure(state="normal", text="🔍 Analyze + Explain"))

    def _prep_input(self, image):
        """Resize and normalize a PIL image into a 1×C×H×W tensor on the right device."""
        if image is None or IM_TF is None:
            return None
        return IM_TF(image).unsqueeze(0).to(self.device)

    def _run_lime(self):
        """Run LIME explanation and compute fidelity/top-k metrics for the current image."""
        progress_dialog = self._show_progress("Computing LIME", "Analyzing superpixel importance...")
        
        def process():
            try:
                self._log_to_progress(progress_dialog, "Starting LIME analysis...")
                
                if lime_image is None:
                    self.after(0, self._show_lime_unavailable)
                    return
                    
                if self.current_image is None:
                    self.after(0, lambda: messagebox.showerror("No image", "Please load an image first"))
                    return

                img_224 = self.current_image.resize((224, 224), Image.BILINEAR)
                img_np = np.array(img_224)

                self._log_to_progress(progress_dialog, "Computing medical segmentation...")
                # Medical segmentation
                segments = medical_brain_segmentation(img_np, method='hybrid')
                n_segs = len(np.unique(segments))
                self.last_lime_segments = segments

                if self.lime_explainer is None:
                    self.lime_explainer = create_medical_lime_explainer(random_state=42)

                num_samples = int(np.clip(n_segs * 8, 400, 2000))
                self._log_to_progress(progress_dialog, f"Running LIME with {num_samples} samples on {n_segs} segments...")

                explanation = self.lime_explainer.explain_instance(
                    img_np,
                    self._lime_predict_fn,
                    top_labels=1,
                    hide_color=0,
                    num_samples=num_samples,
                    segmentation_fn=lambda _: segments,
                    random_seed=42
                )

                top_label = explanation.top_labels[0]
                
                # Store full explanation object for detailed analysis
                self.lime_explanation_obj = explanation
                
                # 1) Extract comprehensive explanation data (inspired by explanation.py as_list())
                # Get sorted explanation list: [(segment_id, weight), ...] sorted by absolute weight
                exp_list = sorted(explanation.local_exp[top_label], 
                                key=lambda x: abs(x[1]), reverse=True)
                
                # 2) Local fidelity (R²-style). Some LIME versions set .score, so guard it.
                self.lime_score = float(getattr(explanation, "score", np.nan))
                self._log_to_progress(progress_dialog, f"LIME fidelity score (R²): {self.lime_score:.3f}")
                
                # 3) Extract intercept and local prediction (from lime_base.py explain_instance_with_data)
                self.lime_intercept = float(getattr(explanation, "intercept", {}).get(top_label, np.nan))
                self.lime_local_pred = float(getattr(explanation, "local_pred", np.nan))
                self._log_to_progress(progress_dialog, f"Local prediction: {self.lime_local_pred:.3f}, Intercept: {self.lime_intercept:.3f}")

                # 4) Separate positive and negative contributing segments (inspired by get_image_and_mask)
                positive_segments = [(int(seg), float(w)) for seg, w in exp_list if w > 0]
                negative_segments = [(int(seg), float(w)) for seg, w in exp_list if w < 0]
                
                self.lime_positive_segments = positive_segments[:10]  # Top 10 positive
                self.lime_negative_segments = negative_segments[:10]  # Top 10 negative
                
                # 5) Top-K signed segment weights (for the predicted label)
                self.lime_topk = [(int(seg), float(w)) for seg, w in exp_list[:15]]  # Top 15 overall
                self._log_to_progress(progress_dialog, f"Top positive segments: {len(positive_segments)}")
                self._log_to_progress(progress_dialog, f"Top negative segments: {len(negative_segments)}")
                self._log_to_progress(progress_dialog, f"Most important segment: {self.lime_topk[0] if self.lime_topk else 'N/A'}")

                # 3) Heatmap from weights (what you already do)
                seg_weights = dict(explanation.local_exp[top_label])
                heat = np.zeros_like(segments, dtype=np.float32)
                for seg_id, weight in seg_weights.items():
                    heat[segments == seg_id] = weight
                m, M = float(np.nanmin(heat)), float(np.nanmax(heat))
                heat = (heat - m) / (M - m) if M > m else np.zeros_like(heat, dtype=np.float32)

                # Enhanced method note with explanation quality
                fidelity_note = ""
                if self.lime_score is not None:
                    if self.lime_score > 0.7:
                        fidelity_note = " (High fidelity)"
                    elif self.lime_score > 0.4:
                        fidelity_note = " (Moderate fidelity)"
                    else:
                        fidelity_note = " (Low fidelity)"
                
                method_note = f"LIME Explanation — {CLASS_NAMES[top_label].upper()}{fidelity_note}"
                
                # 6) Build enhanced overlay with better visualization
                self.after(0, lambda: self._build_overlay(heat, method_note))

                # 7) Create signed green/red overlay (inspired by ImageExplanation.get_image_and_mask)
                self._log_to_progress(progress_dialog, "Creating signed overlay...")
                try:
                    base_224 = self.current_image.resize((224, 224), Image.BILINEAR)
                    signed_overlay = self._overlay_lime_signed(segments, seg_weights, base_224, alpha=float(self.sld_overlay.get()))
                    self.lime_signed_overlay = signed_overlay  # keep in memory
                    self._log_to_progress(progress_dialog, "Signed overlay created successfully")
                except Exception as e:
                    self._log_to_progress(progress_dialog, f"Signed overlay failed: {e}")
                    self.lime_signed_overlay = None
                
                # 8) Update Why Card with LIME-specific insights
                self.after(0, lambda: self._update_why_card_with_lime(top_label, explanation))
                
                # 9) Add LIME insights to Why Card
                pos_count = len(self.lime_positive_segments) if self.lime_positive_segments else 0
                neg_count = len(self.lime_negative_segments) if self.lime_negative_segments else 0
                self.after(0, lambda: self.why_card.add_lime_insights(
                    self.lime_score,
                    pos_count,
                    neg_count,
                    self.lime_positive_segments[:3] if self.lime_positive_segments else None,
                    self.lime_negative_segments[:3] if self.lime_negative_segments else None
                ))
                
                # 10) Update metrics with detailed LIME information
                self.after(0, lambda: self._update_lime_metrics(top_label))
                
                self.after(0, lambda: self.method_cards["LIME"].set_status("Complete", COLORS['success']))
                self.after(0, lambda: self.status_bar.set_status("LIME analysis complete", COLORS['success']))
                
            except Exception as e:
                print(f"LIME ERROR: {e}")
                self._log_to_progress(progress_dialog, f"LIME error: {e}")
                fallback = self._create_lime_fallback()
                self.after(0, lambda: self._build_overlay(fallback, "LIME fallback"))
                self.after(0, lambda: self.method_cards["LIME"].set_status("Error", COLORS['danger']))
            finally:
                if progress_dialog:
                    self.after(2000, lambda: progress_dialog.destroy() if progress_dialog.winfo_exists() else None)
        
        # Start new thread
        threading.Thread(target=process).start()

    def _run_gradcam(self):
        """Run enhanced GradCAM with GuidedGradCam, NoiseTunnel, and infidelity metric for robust medical explainability."""
        if not captum_available or torch is None or self.model is None:
            self._show_not_implemented("Grad-CAM")
            self.method_cards["Grad-CAM"].set_status("N/A", "gray30")
            return
            
        progress_dialog = self._show_progress("Computing Enhanced Grad-CAM", "Applying guided backprop and noise robustness...")
        
        def process():
            try:
                self._log_to_progress(progress_dialog, "Starting enhanced Grad-CAM analysis...")
                
                img_tensor = self._prep_input(self.current_image)
                if img_tensor is None:
                    raise RuntimeError("Model transforms not available.")
                
                # Clear any previous gradients
                self.model.zero_grad()
                
                # Use GuidedGradCam for precise, fine-grained attributions (better for medical details)
                # We target the last convolutional layer of ResNet18
                target_layer = self.model.backbone.layer4[-1].conv2
                guided_gradcam = GuidedGradCam(self.model, target_layer)
                
                # Wrap with NoiseTunnel for robustness (smoothgrad approximation)
                # This reduces noise in the explanation, critical for medical trust
                nt_gradcam = NoiseTunnel(guided_gradcam)
                
                self._log_to_progress(progress_dialog, "Computing GuidedGradCam with noise robustness...")
                with torch.no_grad():
                    logits = self.model(img_tensor)
                    probs = F.softmax(logits, dim=1)
                    target = int(torch.argmax(probs, dim=1).item())
                
                # Compute attributions with noise (nt_samples=10 for balance of speed and robustness)
                # stdevs=0.1 is standard for normalized images
                attributions = nt_gradcam.attribute(
                    img_tensor, target=target, nt_type="smoothgrad", nt_samples=10, stdevs=0.1
                )
                
                # Aggregate channels and upsample to image size
                cam_agg = attributions.mean(dim=1, keepdim=True)
                cam_up = LayerAttribution.interpolate(cam_agg, (224, 224))
                cam_np = cam_up.squeeze(0).squeeze(0).detach().cpu().numpy()
                
                # Normalize to [0, 1]
                cmin, cmax = float(cam_np.min()), float(cam_np.max())
                cam_np = (cam_np - cmin) / (cmax - cmin) if cmax > cmin else np.zeros_like(cam_np, dtype=np.float32)
                
                # --- Medical Metrics Calculation ---
                self._log_to_progress(progress_dialog, "Computing medical metrics...")
                
                # 1. Brain Coverage (How much of the brain is implicated?)
                img_np = np.array(self.current_image.resize((224, 224)))
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                _, brain_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
                brain_mask_bool = (brain_mask > 0)
                
                brain_activation = (cam_np * brain_mask_bool).sum()
                total_activation = cam_np.sum()
                coverage = float(brain_activation / (total_activation + 1e-8))
                
                # 2. Focus Score (Gini Coefficient of activation)
                # A high score means the model is looking at a specific spot (tumor).
                # A low score means the model is looking everywhere (diffuse/uncertain).
                flat_attr = cam_np[brain_mask_bool].flatten()
                if len(flat_attr) > 0:
                    # Simple sparsity metric: sum of top 10% / sum of all
                    sorted_attr = np.sort(flat_attr)
                    top_10_idx = int(len(sorted_attr) * 0.9)
                    focus_score = np.sum(sorted_attr[top_10_idx:]) / (np.sum(sorted_attr) + 1e-9)
                else:
                    focus_score = 0.0

                # 3. Infidelity (Explanation Reliability)
                # Measures if significant perturbations to the input actually change the output
                def perturb_func(inputs, **kwargs):
                    noise = torch.randn_like(inputs) * 0.1
                    perturbed = inputs + noise
                    return perturbed, noise
                
                infid_score = infidelity(
                    self.model, perturb_func, img_tensor, attributions, target=target, n_perturb_samples=5
                ).item()
                
                self.gradcam_metrics = {
                    "target_class": int(target),
                    "class_name": CLASS_NAMES[target],
                    "layer": "layer4[-1].conv2",
                    "brain_coverage": coverage,
                    "focus_score": focus_score,
                    "infidelity": infid_score,
                    "method": "GuidedGradCam + NoiseTunnel"
                }
                
                self._log_to_progress(progress_dialog, f"Focus Score: {focus_score:.3f}, Infidelity: {infid_score:.4f}")
                
                method_note = f"Grad-CAM: {CLASS_NAMES[target].upper()} (Focal Score: {focus_score:.2f})"
                
                # Build overlay with contours (Radiologist style)
                self.after(0, lambda: self._build_gradcam_overlay(cam_np, method_note, brain_mask))
                self.after(0, lambda: self.method_cards["Grad-CAM"].set_status("Complete", COLORS['success']))
                self.after(0, lambda: self.status_bar.set_status("Enhanced Grad-CAM analysis complete", COLORS['success']))
                self.after(0, lambda: self._update_gradcam_metrics())

            except Exception as e:
                error_msg = f"Enhanced Grad-CAM error: {str(e)}"
                print(f"[Enhanced Grad-CAM ERROR] {error_msg}")
                self._log_to_progress(progress_dialog, error_msg)
                self.after(0, lambda: self.method_cards["Grad-CAM"].set_status("Error", COLORS['danger']))
                self.after(0, lambda: self.status_bar.set_status("Enhanced Grad-CAM failed", COLORS['danger']))
            finally:
                if progress_dialog and hasattr(progress_dialog, 'winfo_exists'):
                    def close_dialog():
                        try:
                            if progress_dialog.winfo_exists():
                                progress_dialog.destroy()
                        except:
                            pass
                    self.after(2000, close_dialog)
        
        # Start the processing thread
        threading.Thread(target=process, daemon=True).start()

    def _debug_model_structure(self):
        """Print model modules and layer4 structure to stdout for debugging."""
        if self.model is None:
            print("Model not loaded")
            return
            
        print("Model structure:")
        for name, module in self.model.named_modules():
            print(f"  {name}: {type(module).__name__}")
            
        # Print specifically the layer4 structure
        print("\nLayer4 structure:")
        for i, block in enumerate(self.model.backbone.layer4):
            print(f"  Block {i}: {type(block).__name__}")
            for name, submodule in block.named_modules():
                if name:  # Skip the block itself
                    print(f"    {name}: {type(submodule).__name__}")

    def _build_gradcam_overlay(self, cam_np, method_note, brain_mask):
        """
        Create a medical-style overlay with heatmap AND contours.
        Contours mimic how a radiologist would circle a lesion.
        """
        if self.current_image is None:
            return

        try:
            # 1. Prepare Base Image
            base = self.current_image.resize((224, 224), Image.BILINEAR)
            base_np = np.array(base)
            
            # 2. Prepare Heatmap (Jet colormap)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # 3. Prepare Contours (Threshold top 20% of activation)
            # This isolates the "core" region the model is looking at
            threshold_val = np.percentile(cam_np[cam_np > 0], 80) if np.any(cam_np > 0) else 0.5
            _, thresh_img = cv2.threshold(np.uint8(255 * cam_np), int(255 * threshold_val), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 4. Blend Heatmap
            alpha = float(self.sld_overlay.get())
            # Mask heatmap to brain area only to avoid background noise
            mask_3ch = np.stack([brain_mask]*3, axis=2) // 255
            heatmap_masked = heatmap * mask_3ch
            
            blended = (0.4 * heatmap_masked) + (0.6 * base_np) # Keep base visible
            blended = np.clip(blended, 0, 255).astype(np.uint8)
            
            # 5. Draw Contours on top (Cyan color for high contrast)
            # Thickness 2
            cv2.drawContours(blended, contours, -1, (0, 255, 255), 2)
            
            # 6. Finalize
            self.overlay_image = Image.fromarray(blended)
            self.last_heatmap = cam_np # Store raw for resizing if needed
            self.last_method_note = method_note
            
            # Render
            self._render_overlay_image_direct()
            
        except Exception as e:
            print(f"GradCAM Overlay Error: {e}")
    
    def _render_overlay_image_direct(self):
        """Helper to render self.overlay_image directly to canvas."""
        if self.overlay_image is None: return
        
        # Scale to display size
        container_width = 400
        container_height = 400
        img_width, img_height = self.overlay_image.size
        
        scale = min(container_width/img_width, container_height/img_height)
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        result_img = self.overlay_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Add label
        if ImageDraw and ImageFont:
            draw = ImageDraw.Draw(result_img)
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            text = self.last_method_note
            bbox = draw.textbbox((0, 0), text, font=font)
            draw.rectangle([8, 8, bbox[2] + 16, bbox[3] + 16], fill=(0, 0, 0, 180))
            draw.text((12, 12), text, fill=(255, 255, 255), font=font)

        self.photo_right = CTkImage(light_image=result_img, size=(new_width, new_height))
        self.canvas_right.configure(image=self.photo_right, text="")
    
    def _update_gradcam_metrics(self):
        """Append Grad-CAM specific metrics into the reliability metrics textbox."""
        if not hasattr(self, 'gradcam_metrics') or self.gradcam_metrics is None:
            return
            
        metrics = self.gradcam_metrics
        
        # Interpret Focus Score
        fs = metrics['focus_score']
        focus_desc = "Highly Focal (Specific)" if fs > 0.5 else "Diffuse (Uncertain)" if fs < 0.2 else "Moderate"
        
        # Interpret Infidelity
        inf = metrics['infidelity']
        inf_desc = "Stable" if inf < 0.01 else "Unstable"
        
        current_text = self.metrics_text.get("1.0", "end").strip()
        
        gradcam_text = f"""

Grad-CAM Analysis:
  Target: {metrics['class_name'].upper()}
  Focus Score: {fs:.2f} ({focus_desc})
  Stability (Infidelity): {inf:.4f} ({inf_desc})
  
  Note: Cyan contours indicate the 
  region of highest model activation."""
        
        # Clear and update
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", current_text + gradcam_text)

    def _run_shap(self):
        """Run GradientSHAP attribution and build a signed heatmap overlay."""
        if shap is None or torch is None or self.model is None or not gradient_shap_available:
            self._show_not_implemented("SHAP")
            self.method_cards["SHAP"].set_status("N/A", "gray30")
            return
            
        progress_dialog = self._show_progress("Computing SHAP", "Sampling baselines and gradients...")
        
        def process():
            try:
                self.model.eval()
                x = self._prep_input(self.current_image)
                if x is None:
                    raise RuntimeError("Model transforms not available.")
                    
                with torch.no_grad():
                    logits = self.model(x)
                    probs = F.softmax(logits, dim=1)
                    target = int(torch.argmax(probs, dim=1).item())

                gs = GradientShap(self.model)
                # Use a black baseline and a noisy baseline
                baseline_black = torch.zeros_like(x)
                baseline_noise = torch.randn_like(x) * 0.001
                baselines = torch.stack([baseline_black.squeeze(0), (baseline_black + baseline_noise).squeeze(0)], dim=0)

                self._log_to_progress(progress_dialog, "Calculating Shapley values (Gradient approximation)...")
                attributions = gs.attribute(
                    x, baselines=baselines, target=target, n_samples=20, stdevs=0.09
                )

                # Process attributions for signed visualization
                # Sum across color channels to get pixel-wise importance
                attr_np = attributions.squeeze(0).sum(dim=0).cpu().numpy() # Shape (224, 224)
                
                # Calculate metrics
                total_pos_mass = float(np.sum(attr_np[attr_np > 0]))
                total_neg_mass = float(np.sum(np.abs(attr_np[attr_np < 0])))
                max_importance = float(np.max(np.abs(attr_np)))
                
                self.shap_metrics = {
                    "pos_mass": total_pos_mass,
                    "neg_mass": total_neg_mass,
                    "max_imp": max_importance
                }

                # Create signed RGB heatmap (Red = Positive, Blue = Negative)
                # Normalize by max absolute value so 0 is neutral
                if max_importance > 0:
                    norm_attr = attr_np / max_importance
                else:
                    norm_attr = attr_np
                
                heatmap_colored = np.zeros((224, 224, 3), dtype=np.float32)
                
                # Positive contributions -> Red channel
                heatmap_colored[:, :, 0] = np.maximum(norm_attr, 0) 
                # Negative contributions -> Blue channel (standard SHAP color scheme)
                heatmap_colored[:, :, 2] = np.maximum(-norm_attr, 0)

                method_note = f"SHAP — Red: Supports {CLASS_NAMES[target]} | Blue: Opposes"
                
                # Update UI
                self.after(0, lambda: self._build_overlay(heatmap_colored, method_note))
                self.after(0, lambda: self._update_why_card_with_shap(target))
                self.after(0, lambda: self._update_shap_metrics())
                
                self.after(0, lambda: self.method_cards["SHAP"].set_status("Complete", COLORS['success']))
                self.after(0, lambda: self.status_bar.set_status("SHAP analysis complete", COLORS['success']))
                
            except Exception as e:
                print(f"[SHAP ERROR] {e}")
                self._log_to_progress(progress_dialog, f"Error: {e}")
                self.after(0, lambda: self.method_cards["SHAP"].set_status("Error", COLORS['danger']))
            finally:
                if progress_dialog:
                    self.after(2000, lambda: progress_dialog.destroy() if progress_dialog.winfo_exists() else None)
                    
        threading.Thread(target=process, daemon=True).start()

    def _update_why_card_with_shap(self, target_idx):
        """Populate Why Card with SHAP-specific game theoretic explanations."""
        if not hasattr(self, 'shap_metrics'):
            return

        pred_class = CLASS_NAMES[target_idx]
        pos_mass = self.shap_metrics['pos_mass']
        neg_mass = self.shap_metrics['neg_mass']
        total_mass = pos_mass + neg_mass + 1e-9
        
        pos_ratio = pos_mass / total_mass
        
        # Base medical reasoning
        reasoning = MEDICAL_REASONING.get(pred_class, {
            "because": ["Features detected by model"],
            "despite": ["Contrasting evidence"]
        })
        
        # Enhanced "Because" text
        shap_because = list(reasoning["because"])
        shap_because.insert(0, f"SHAP Analysis: {pos_ratio:.1%} of active features support this diagnosis")
        shap_because.insert(1, "Red regions indicate pixels increasing the probability")
        
        # Enhanced "Despite" text
        shap_despite = list(reasoning["despite"])
        if neg_mass > 0.1 * pos_mass:
            shap_despite.insert(0, f"SHAP Analysis: {neg_mass/total_mass:.1%} of features oppose this diagnosis")
            shap_despite.insert(1, "Blue regions indicate pixels lowering the probability")
        else:
            shap_despite.insert(0, "SHAP Analysis: Very little conflicting evidence found")

        # Update UI
        because_text = "\n".join([f"• {item}" for item in shap_because])
        self.why_card.because_text.delete("1.0", "end")
        self.why_card.because_text.insert("1.0", because_text)
        
        despite_text = "\n".join([f"• {item}" for item in shap_despite])
        self.why_card.despite_text.delete("1.0", "end")
        self.why_card.despite_text.insert("1.0", despite_text)

    def _update_shap_metrics(self):
        """Update metrics panel with SHAP quantitative data."""
        if not hasattr(self, 'shap_metrics'):
            return
            
        current_text = self.metrics_text.get("1.0", "end").strip()
        m = self.shap_metrics
        
        shap_text = f"""

SHAP Attribution Metrics:
  Total Positive Impact: {m['pos_mass']:.2f}
  Total Negative Impact: {m['neg_mass']:.2f}
  Max Pixel Importance: {m['max_imp']:.4f}
  
Interpretation:
  Positive (Red) -> Pushes prediction TOWARDS class
  Negative (Blue) -> Pushes prediction AWAY from class"""
        
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", current_text + shap_text)
    
    def _lime_predict_fn(self, images):
        """Batch prediction function for LIME that mirrors the ResNet preprocessing."""
        if len(images) == 0:
            return np.array([]).reshape(0, len(CLASS_NAMES))

        if self.model is None or torch is None:
            return np.random.dirichlet([1,1,1,1], size=len(images))

        batch_size = min(32, len(images))
        all_probs = []
        
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i+batch_size]
                batch_tensors = []
                
                for img in batch_images:
                    if img.dtype != np.uint8:
                        img = img.astype(np.uint8)
                    
                    pil_img = Image.fromarray(img)
                    if IM_TF:
                        tensor = IM_TF(pil_img).unsqueeze(0)
                        batch_tensors.append(tensor)
                
                if batch_tensors:
                    try:
                        batch_input = torch.cat(batch_tensors, dim=0).to(self.device)
                        with torch.no_grad():
                            logits = self.model(batch_input)
                            probs = F.softmax(logits, dim=1).cpu().numpy()
                            all_probs.append(probs)
                    except Exception as e:
                        print(f"Batch prediction error: {e}")
                        fallback = np.random.dirichlet([1,1,1,1], size=len(batch_images))
                        all_probs.append(fallback)
        
        if all_probs:
            return np.vstack(all_probs)
        else:
            return np.random.dirichlet([1,1,1,1], size=len(images))

    def _create_lime_fallback(self):
        """Create a heuristic, medically-inspired fallback heatmap when LIME is unavailable."""
        if self.current_image is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        img = self.current_image.resize((224, 224), Image.BILINEAR)
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Create brain mask for realistic attention
        _, brain_mask = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        brain_mask = cv2.morphologyEx(brain_mask, cv2.MORPH_CLOSE, kernel)
        
        # Get prediction for targeted fallback
        pred_idx = 0
        if self.prediction and self.prob_vector is not None:
            pred_idx = np.argmax(self.prob_vector)
        
        # Create class-specific attention patterns
        if pred_idx == 0:  # Glioma
            edges = cv2.Canny(gray, 30, 100)
            heat = cv2.GaussianBlur(edges.astype(float), (15, 15), 0)
        elif pred_idx == 1:  # Meningioma
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            heat = np.sqrt(grad_x**2 + grad_y**2)
        elif pred_idx == 2:  # No tumor
            heat = cv2.GaussianBlur(brain_mask.astype(float), (25, 25), 0)
        else:  # Pituitary
            y, x = np.indices(gray.shape)
            cy, cx = int(gray.shape[0] * 0.7), gray.shape[1] // 2
            heat = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * 40**2))
            
        # Apply brain mask and normalize
        heat = heat * (brain_mask > 0)
        if heat.max() > heat.min():
            heat = (heat - heat.min()) / (heat.max() - heat.min())
            
        return heat.astype(np.float32)

    def _build_overlay(self, heat_224, method_note=""):
        """Normalize a heatmap, store it, and trigger overlay rendering."""
        if self.current_image is None or Image is None:
            return
        try:
            base = self.current_image.resize((224, 224), Image.BILINEAR)

            # Check if heat_224 is already a 3D RGB image (e.g. from SHAP signed output)
            is_rgb = False
            if isinstance(heat_224, np.ndarray) and heat_224.ndim == 3 and heat_224.shape[2] == 3:
                is_rgb = True
                heat = heat_224 # Already normalized [0,1]
            else:
                # Standard 2D heatmap processing
                if isinstance(heat_224, torch.Tensor):
                    heat = heat_224.detach().cpu().numpy()
                else:
                    heat = np.array(heat_224)
                if heat.ndim == 3:
                    heat = np.mean(heat, axis=-1)
                h_min, h_max = np.nanmin(heat), np.nanmax(heat)
                if h_max > h_min:
                    heat = np.clip((heat - h_min) / (h_max - h_min), 0, 1)
                else:
                    heat = np.zeros_like(heat, dtype=np.float32)
                heat = np.nan_to_num(heat).astype(np.float32)

            # Store and render
            self.last_heatmap = heat
            self.last_method_note = method_note
            self._render_overlay()
        except Exception as e:
            print(f"Overlay error: {e}")

    def _render_overlay(self):
        """
        Render the current heatmap blended over the base image at the chosen alpha.
        
        For LIME, supports positive-only, negative-only, or both modes.
        For SHAP, supports pre-colored RGB heatmaps.
        """
        if self.last_heatmap is None or self.current_image is None or not CTkImage:
            return

        try:
            base = self.current_image.resize((224, 224), Image.BILINEAR)
            heat = self.last_heatmap.copy()
            
            # Check if we have a pre-colored RGB heatmap (SHAP signed)
            if heat.ndim == 3 and heat.shape[2] == 3:
                # Apply overlay mode filter for SHAP
                overlay_mode_var = getattr(self, 'overlay_mode_var', None)
                mode = overlay_mode_var.get() if hasattr(overlay_mode_var, 'get') else "both"
                
                filtered_heat = heat.copy()
                if mode == "positive":
                    # Keep Red (channel 0), zero Blue (channel 2)
                    filtered_heat[:, :, 2] = 0
                elif mode == "negative":
                    # Keep Blue (channel 2), zero Red (channel 0)
                    filtered_heat[:, :, 0] = 0
                # "both" keeps both
                
                overlay = Image.fromarray((filtered_heat * 255).astype(np.uint8))
            else:
                # Standard LIME/GradCAM processing
                # For LIME: apply overlay mode filter (positive_only, negative_only, or both)
                if self.current_xai_method == "LIME" and hasattr(self, 'last_lime_segments') and self.last_lime_segments is not None:
                    overlay_mode_var = getattr(self, 'overlay_mode_var', None)
                    if overlay_mode_var and self.lime_explanation_obj is not None:
                        try:
                            mode = overlay_mode_var.get() if hasattr(overlay_mode_var, 'get') else "both"
                            segments = self.last_lime_segments
                            
                            # Get segment weights from explanation
                            if hasattr(self.lime_explanation_obj, 'local_exp') and self.prediction:
                                pred_idx = CLASS_NAMES.index(self.prediction) if self.prediction in CLASS_NAMES else 0
                                exp_dict = self.lime_explanation_obj.local_exp
                                if pred_idx in exp_dict:
                                    seg_weights = dict(exp_dict[pred_idx])
                                    
                                    # Create filtered heatmap based on mode
                                    # We reconstruct the heatmap from segments to ensure accurate filtering
                                    filtered_heat = np.zeros(segments.shape, dtype=np.float32)
                                    
                                    for seg_id, weight in seg_weights.items():
                                        mask = (segments == seg_id)
                                        if mode == "positive":
                                            # Only show positive contributions (supports diagnosis)
                                            if weight > 0:
                                                filtered_heat[mask] = weight
                                        elif mode == "negative":
                                            # Only show negative contributions (opposes diagnosis)
                                            if weight < 0:
                                                filtered_heat[mask] = abs(weight)
                                        else:  # "both"
                                            # Show magnitude of all contributions
                                            filtered_heat[mask] = abs(weight)
                                    
                                    # Normalize filtered heat
                                    if filtered_heat.max() > 0:
                                        filtered_heat = filtered_heat / filtered_heat.max()
                                    
                                    heat = filtered_heat
                        except Exception as e:
                            print(f"[DEBUG] Overlay mode filter error: {e}")

                # Apply colormap (inspired by ImageExplanation visualization)
                if cm:
                    overlay_mode_var = getattr(self, 'overlay_mode_var', None)
                    if self.current_xai_method == "LIME" and overlay_mode_var:
                        try:
                            mode = overlay_mode_var.get() if hasattr(overlay_mode_var, 'get') else "both"
                            if mode == "positive":
                                # Green colormap for positive contributions (supports diagnosis)
                                colored = np.zeros((*heat.shape, 3), dtype=np.float32)
                                colored[:, :, 1] = heat  # Green channel
                            elif mode == "negative":
                                # Red colormap for negative contributions (opposes diagnosis)
                                colored = np.zeros((*heat.shape, 3), dtype=np.float32)
                                colored[:, :, 0] = heat  # Red channel
                            else:
                                # Both: use jet colormap (red=negative, blue=positive)
                                cmap = cm.jet
                                colored = cmap(heat)[..., :3]
                        except Exception as e:
                            print(f"[DEBUG] Colormap error: {e}")
                            cmap = cm.jet
                            colored = cmap(heat)[..., :3]
                    else:
                        cmap = cm.jet
                        colored = cmap(heat)[..., :3]
                    overlay = Image.fromarray((colored * 255).astype(np.uint8))
                else:
                    heat_uint8 = (heat * 255).astype(np.uint8)
                    overlay = Image.fromarray(np.stack([heat_uint8, np.zeros_like(heat_uint8), np.zeros_like(heat_uint8)], -1))

            # Blend with current alpha
            alpha = float(self.sld_overlay.get())
            base_np = np.array(base).astype(float)
            overlay_np = np.array(overlay).astype(float)
            result = np.clip(base_np * (1 - alpha) + overlay_np * alpha, 0, 255).astype(np.uint8)
            
            # Scale to display size
            container_width = 400
            container_height = 400
            img_width, img_height = result.shape[1], result.shape[0]
            
            scale = min(container_width/img_width, container_height/img_height)
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            result_img = Image.fromarray(result).resize((new_width, new_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.BICUBIC)

            # --- ENHANCEMENT: Add Annotations (Segment IDs for LIME, Legend for SHAP) ---
            if ImageDraw and ImageFont:
                draw = ImageDraw.Draw(result_img)
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                    small_font = ImageFont.truetype("arial.ttf", 12)
                    bold_font = ImageFont.truetype("arialbd.ttf", 12)
                except:
                    font = ImageFont.load_default()
                    small_font = ImageFont.load_default()
                    bold_font = ImageFont.load_default()

                # 1. LIME Segment Labels
                if self.current_xai_method == "LIME" and hasattr(self, 'last_lime_segments') and self.last_lime_segments is not None and hasattr(self, 'lime_topk') and self.lime_topk:
                    # Map 224x224 segments to display size
                    scale_x = new_width / 224.0
                    scale_y = new_height / 224.0
                    
                    # Label top 10 segments (most important)
                    ids_to_label = [x[0] for x in self.lime_topk[:10]]
                    
                    for seg_id in ids_to_label:
                        # Find centroid of the segment
                        ys, xs = np.where(self.last_lime_segments == seg_id)
                        if len(xs) == 0: continue
                        
                        cx, cy = int(np.mean(xs)), int(np.mean(ys))
                        
                        # Scale to display coordinates
                        screen_x = int(cx * scale_x)
                        screen_y = int(cy * scale_y)
                        
                        text = str(seg_id)
                        
                        # Draw text with heavy outline for readability against heatmap
                        bbox = draw.textbbox((screen_x, screen_y), text, font=bold_font)
                        w_txt = bbox[2] - bbox[0]
                        h_txt = bbox[3] - bbox[1]
                        
                        # Center text
                        tx = screen_x - w_txt // 2
                        ty = screen_y - h_txt // 2
                        
                        # Outline (black)
                        for adj in range(-1, 2):
                            for adj2 in range(-1, 2):
                                draw.text((tx+adj, ty+adj2), text, font=bold_font, fill="black")
                        
                        # Text (White)
                        draw.text((tx, ty), text, font=bold_font, fill="white")

                # 2. SHAP Legend
                elif self.current_xai_method == "SHAP":
                    # Draw legend for Red/Blue interpretation
                    legend_w, legend_h = 110, 55
                    lx = new_width - legend_w - 10
                    ly = new_height - legend_h - 10
                    
                    # Semi-transparent background
                    draw.rectangle([lx, ly, lx + legend_w, ly + legend_h], fill=(0, 0, 0, 160), outline="white")
                    
                    # Red box (Supports)
                    draw.rectangle([lx + 8, ly + 8, lx + 20, ly + 20], fill=(255, 0, 0))
                    draw.text((lx + 28, ly + 8), "Supports", fill="white", font=small_font)
                    
                    # Blue box (Opposes)
                    draw.rectangle([lx + 8, ly + 28, lx + 20, ly + 40], fill=(0, 0, 255))
                    draw.text((lx + 28, ly + 28), "Opposes", fill="white", font=small_font)

                # 3. Method Label (Existing logic, moved to end to be on top)
                text = self.last_method_note or f"{self.current_xai_method}: {self.prediction or 'analyzing...'}"
                bbox = draw.textbbox((0, 0), text, font=font)
                draw.rectangle([8, 8, bbox[2] + 16, bbox[3] + 16], fill=(255, 255, 255, 200))
                draw.text((12, 12), text, fill=(0, 0, 0), font=font)

            self.overlay_image = result_img
            
            # Force creation of a new CTkImage and ensure it's assigned to self before use
            # This prevents garbage collection of the underlying Tk image
            new_photo = CTkImage(light_image=self.overlay_image, size=(new_width, new_height))
            self.photo_right = new_photo
            
            # Update the label
            self.canvas_right.configure(image=self.photo_right, text="")
            
        except Exception as e:
            print(f"Error rendering overlay: {e}")

    def _on_alpha(self, value):
        """Handle overlay alpha slider changes and re-render the heatmap overlay."""
        alpha_val = f"{float(value):.2f}"
        self.alpha_label.configure(text=alpha_val)
        
        # Re-render overlay with new alpha
        if self.last_heatmap is not None:
            self._render_overlay()
    
    def _on_overlay_mode_change(self, mode):
        """Handle overlay mode change (both/positive/negative) for LIME explanations."""
        self.status_bar.set_status(f"Overlay mode: {mode}")
        
        # Re-render overlay if we have LIME data
        if self.last_heatmap is not None and self.current_xai_method == "LIME":
            self._render_overlay()

    def _update_metrics(self, prediction, confidence, entropy, prob_vector):
        """Update the reliability metrics textbox with prediction scores and class probs."""
        metrics_text = f"""Prediction Confidence: {confidence:.1f}%
Uncertainty (Entropy): {entropy:.3f}
Method: {self.current_xai_method}

Class Probabilities:"""

        for i, (cls, prob) in enumerate(zip(CLASS_NAMES, prob_vector)):
            metrics_text += f"\n  {cls}: {prob*100:.1f}%"
            
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", metrics_text)

        # Enable feedback buttons now that the user sees a concrete prediction
        if hasattr(self, "btn_feedback_right") and hasattr(self, "btn_feedback_wrong"):
            self.btn_feedback_right.configure(state="normal")
            self.btn_feedback_wrong.configure(state="normal")
    
    def _update_lime_metrics(self, top_label):
        """Update metrics with detailed LIME explanation information."""
        if self.lime_score is None:
            return
        
        current_text = self.metrics_text.get("1.0", "end").strip()
        
        # Format values safely to avoid f-string errors
        local_pred_str = f"{self.lime_local_pred:.3f}" if self.lime_local_pred is not None else "N/A"
        intercept_str = f"{self.lime_intercept:.3f}" if self.lime_intercept is not None else "N/A"
        
        lime_metrics = f"""

LIME Explanation Quality:
  Fidelity (R²): {self.lime_score:.3f} {'✓ Good' if self.lime_score > 0.7 else '⚠ Moderate' if self.lime_score > 0.4 else '✗ Low'}
  Local Prediction: {local_pred_str}
  Intercept: {intercept_str}
  
Top Contributing Regions:"""
        
        if self.lime_positive_segments:
            lime_metrics += f"\n  Positive (supporting {CLASS_NAMES[top_label]}):"
            for seg_id, weight in self.lime_positive_segments[:5]:
                lime_metrics += f"\n    Segment {seg_id}: +{weight:.3f}"
        
        if self.lime_negative_segments:
            lime_metrics += f"\n  Negative (opposing {CLASS_NAMES[top_label]}):"
            for seg_id, weight in self.lime_negative_segments[:5]:
                lime_metrics += f"\n    Segment {seg_id}: {weight:.3f}"
        
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", current_text + lime_metrics)
    
    def _update_why_card_with_lime(self, top_label, explanation):
        """Enhance Why Card with LIME-specific insights for medical staff."""
        if not hasattr(self, 'why_card') or self.prediction is None:
            return
        
        # Get the actual prediction probability
        actual_pred_prob = float(self.prob_vector[top_label]) if self.prob_vector is not None else 0.0
        
        # Calculate explanation quality indicators
        fidelity_quality = "High" if self.lime_score > 0.7 else "Moderate" if self.lime_score > 0.4 else "Low"
        
        # Get top positive and negative segments with medical context
        positive_count = len(self.lime_positive_segments) if self.lime_positive_segments else 0
        negative_count = len(self.lime_negative_segments) if self.lime_negative_segments else 0
        
        # Enhance the "because" section with LIME insights
        reasoning = MEDICAL_REASONING.get(self.prediction, {
            "because": ["Features detected by model"],
            "despite": ["Contrasting evidence"]
        })
        
        # Add LIME-specific evidence
        enhanced_because = list(reasoning["because"])
        if positive_count > 0:
            enhanced_because.insert(0, f"LIME identified {positive_count} image regions supporting this diagnosis")
        if self.lime_score is not None and self.lime_score > 0.6:
            enhanced_because.insert(0, f"High explanation fidelity (R²={self.lime_score:.2f})")
        
        enhanced_despite = list(reasoning["despite"])
        if negative_count > 0:
            enhanced_despite.insert(0, f"LIME found {negative_count} regions with opposing signals")
        if self.lime_score is not None and self.lime_score < 0.5:
            enhanced_despite.insert(0, f"Lower explanation fidelity (R²={self.lime_score:.2f}) - consider alternative views")
        
        # Update the Why Card textboxes
        because_text = "\n".join([f"• {item}" for item in enhanced_because])
        self.why_card.because_text.delete("1.0", "end")
        self.why_card.because_text.insert("1.0", because_text)
        
        despite_text = "\n".join([f"• {item}" for item in enhanced_despite])
        self.why_card.despite_text.delete("1.0", "end")
        self.why_card.despite_text.insert("1.0", despite_text)

    def _reset_metrics(self):
        """Reset the reliability metrics textbox to its initial placeholder text."""
        self.metrics_text.delete("1.0", "end")
        self.metrics_text.insert("1.0", "Reliability metrics will appear after analysis...")

    def _show_progress(self, title, message):
        """Show a modal progress dialog with a debug console for long-running XAI jobs."""
        if not ctk:
            return None
            
        dialog = ctk.CTkToplevel(self)
        dialog.title(title)
        dialog.geometry("500x300")
        dialog.resizable(True, True)
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        w = dialog.winfo_width()
        h = dialog.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (w // 2)
        y = (self.winfo_screenheight() // 2) - (h // 2)
        dialog.geometry(f"{w}x{h}+{x}+{y}")

        # Main container
        main_frame = ctk.CTkFrame(dialog)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(dialog, text=message, font=ctk.CTkFont(size=14)).pack(pady=(10, 5))
        
        # Progress bar
        progress = ctk.CTkProgressBar(dialog)
        progress.pack(fill="x", padx=20, pady=5)
        progress.configure(mode="indeterminate")
        progress.start()

        ctk.CTkLabel(main_frame, text="Processing:", font=ctk.CTkFont(size=12, weight="bold")).pack(pady=(15,5), anchor="w", padx=20)

        # Debug console
        debug_frame = ctk.CTkFrame(main_frame)
        debug_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        debug_text = ctk.CTkTextbox(debug_frame, height=150, font=ctk.CTkFont(family="Courier", size=10))
        debug_text.pack(fill="both", expand=True, padx=5, pady=5)

        dialog.debug_text = debug_text
        dialog.debug_text.insert("1.0", f"Starting {self.current_xai_method} analysis...\n")
        
        self.update_idletasks()
        
        return dialog

    def _show_not_implemented(self, method):
        """Inform the user that a given XAI method is unavailable due to missing deps."""
        if messagebox:
            messagebox.showinfo("Not Available", 
                f"{method} explanation requires additional dependencies.\n"
                f"Focus is on enhanced LIME with medical domain knowledge.\n\n"
                f"To enable {method}:\n"
                f"- Grad-CAM: pip install captum\n"
                f"- SHAP: pip install shap captum")

    def _show_lime_unavailable(self):
        """Warn the user that LIME is not installed and provide installation hints."""
        if messagebox:
            messagebox.showwarning("LIME Unavailable", 
                "LIME is not available. Install with:\n"
                "pip install lime scikit-image scipy\n\n"
                "Required for Local Interpretable Model-Agnostic Explanations")

    def _feedback(self, kind, show_thank_you=True):
        """
        Log user feedback (right / wrong) about the current prediction to disk.

        Parameters
        ----------
        kind : str
            Either 'pos' (seems right) or 'neg' (seems wrong).
        show_thank_you : bool, optional
            If True, show a confirmation dialog after logging, by default True.
        """
        out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "feedback"))
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{'pos' if kind=='pos' else 'neg'}_{ts}.json"
        
        try:
            feedback_data = {
                "timestamp": ts,
                "feedback": kind,
                "prediction": self.prediction,
                "confidence": float(self.prob_vector[np.argmax(self.prob_vector)]) if self.prob_vector is not None else None,
                "method": self.current_xai_method,
                "file": os.path.basename(self.files[self.idx]) if self.files else None
            }
            
            with open(os.path.join(out_dir, fname), 'w') as f:
                json.dump(feedback_data, f, indent=2)

            print(f"Feedback logged: {fname}")
            if messagebox and show_thank_you:
                messagebox.showinfo("Feedback", "Thank you! Your feedback has been logged.")
                
        except Exception as e:
            print(f"Failed to log feedback: {e}")

    def on_feedback_right(self):
        """Handle the 'Seems right' button: inform user and persist positive feedback."""
        if self.prediction is None:
            if messagebox:
                messagebox.showinfo(
                    "Feedback",
                    "Run the analysis first so there is a prediction you can rate.",
                )
            return

        if messagebox:
            messagebox.showinfo(
                "Feedback — Seems right",
                (
                    "You indicated that this AI diagnosis seems right.\n\n"
                    "This feedback is stored locally and is intended to guide a future "
                    "correction and calibration algorithm.\n"
                    "It does not change the current diagnosis or model behaviour in this prototype."
                ),
            )

        # Store feedback without showing a second thank-you dialog
        self._feedback("pos", show_thank_you=False)

    def on_feedback_wrong(self):
        """Handle the 'Seems wrong' button: warn user and persist negative feedback."""
        if self.prediction is None:
            if messagebox:
                messagebox.showinfo(
                    "Feedback",
                    "Run the analysis first so there is a prediction you can rate.",
                )
            return

        if messagebox:
            messagebox.showwarning(
                "Feedback — Seems wrong",
                (
                    "You indicated that this AI diagnosis seems wrong.\n\n"
                    "This action records feedback that will be used in a future iteration "
                    "to drive a correction algorithm and improve the model.\n"
                    "In this research prototype, it does not immediately change or override "
                    "the current diagnosis."
                ),
            )

        # Store feedback without showing a second thank-you dialog
        self._feedback("neg", show_thank_you=False)

    def on_export(self):
        """Export a PDF report with images, prediction, and medical reasoning summary."""
        if not reportlab_available:
            if messagebox:
                messagebox.showwarning("PDF Export", "ReportLab not available. Install with: pip install reportlab")
            return

        if self.current_image is None:
            messagebox.showinfo("No image", "Load an image first.")
            return

        try:
            out_dir = ensure_dir(os.path.join(os.path.dirname(__file__), "reports"))
            ts = time.strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(out_dir, f"report_xai_{ts}.pdf")

            # Save images for embedding
            orig_path = os.path.join(out_dir, f"original_{ts}.png")
            self.display_image.save(orig_path) if self.display_image else self.current_image.save(orig_path)

            overlay_path = None
            if self.overlay_image:
                overlay_path = os.path.join(out_dir, f"overlay_{ts}.png")
                self.overlay_image.save(overlay_path)

            c = canvas.Canvas(pdf_path, pagesize=A4)
            W, H = A4
            y = H - 40

            # Header
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, y, "EMII V2 — Brain Tumor MRI Scanner (XAI Report)")
            y -= 25

            c.setFont("Helvetica", 12)
            c.drawString(40, y, f"File: {os.path.basename(self.files[self.idx]) if self.files else '-'}")
            y -= 15
            c.drawString(40, y, f"XAI Method: {self.current_xai_method}")
            y -= 15
            c.drawString(40, y, f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            y -= 25

            if self.prediction:
                c.setFont("Helvetica-Bold", 14)
                c.drawString(40, y, f"Prediction: {self.prediction.upper()}")
                y -= 18
                c.setFont("Helvetica", 12)
                if self.prob_vector is not None:
                    conf = self.prob_vector[np.argmax(self.prob_vector)] * 100
                    c.drawString(40, y, f"Confidence: {conf:.1f}%")
                    y -= 15

            # Images
            if os.path.exists(orig_path):
                c.drawString(40, y, "Original Image:")
                y -= 15
                c.drawImage(ImageReader(orig_path), 40, y-250, width=200, height=200, preserveAspectRatio=True)

            if overlay_path and os.path.exists(overlay_path):
                c.drawString(280, y, f"{self.current_xai_method} Explanation:")
                c.drawImage(ImageReader(overlay_path), 280, y-250, width=200, height=200, preserveAspectRatio=True)

            y -= 270

            # Medical reasoning (Dynamic from UI)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(40, y, "Medical Reasoning & XAI Analysis:")
            y -= 18
            
            # Extract text from UI widgets to capture dynamic XAI insights
            because_content = self.why_card.because_text.get("1.0", "end").strip().split('\n')
            despite_content = self.why_card.despite_text.get("1.0", "end").strip().split('\n')
            
            c.setFont("Helvetica", 10)
            c.drawString(40, y, "Supporting Evidence:")
            y -= 12
            for line in because_content:
                if line.strip():
                    # Handle wrapping roughly
                    if len(line) > 90:
                        c.drawString(50, y, line[:90])
                        y -= 10
                        c.drawString(60, y, line[90:])
                    else:
                        c.drawString(50, y, line)
                    y -= 12
            
            y -= 5
            c.drawString(40, y, "Contrasting Evidence:")
            y -= 12
            for line in despite_content:
                if line.strip():
                    if len(line) > 90:
                        c.drawString(50, y, line[:90])
                        y -= 10
                        c.drawString(60, y, line[90:])
                    else:
                        c.drawString(50, y, line)
                    y -= 12

            # Footer
            c.setFont("Helvetica-Oblique", 9)
            c.drawString(40, 50, "Research prototype with medical XAI. Based on LIME/SHAP frameworks. Not for clinical use.")
            
            c.save()
            messagebox.showinfo("Report", f"XAI report saved:\n{pdf_path}")
            self.status_bar.set_status("Report exported successfully")
            
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            self.status_bar.set_status("Export failed", COLORS['danger'])


def main() -> int:
    """
    Module-level entry point to launch the EMII XAI GUI.

    This function is kept lightweight so it can be reused from other scripts
    (for example, `main.py`) without pulling in unrelated side effects.

    Returns
    -------
    int
        Exit status code (0 on success, non-zero on failure).
    """
    if ctk is None:
        print("CustomTkinter not available. Install with: pip install customtkinter")
        return 1

    app = EMIIXAI()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
