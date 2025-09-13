import os
import time
import queue
import random
import threading
import traceback
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import torch
import customtkinter as ctk
from PIL import Image, ImageTk

os.makedirs("./models", exist_ok=True)

class TrainingMonitorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Brain Tumor MRI Training Monitor")
        self.geometry("1200x800")
        
        # Setup theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Communication queues
        self.log_queue = queue.Queue()
        self.metrics_queue = queue.Queue()
        self.is_training = False
        self.stop_event = threading.Event()
        
        # Build UI
        self._create_ui()
        
        # Start queue processing
        self.after(100, self._process_queues)
    
    def _create_ui(self):
        # Main layout with left/right columns
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Header with controls
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, columnspan=2, padx=10, pady=(10,5), sticky="ew")
        
        # Dataset info panel
        info_panel = ctk.CTkFrame(self)
        info_panel.grid(row=1, column=0, columnspan=2, padx=10, pady=5, sticky="ew")
        
        # Log console (left side)
        log_frame = ctk.CTkFrame(self)
        log_frame.grid(row=2, column=0, padx=(10,5), pady=(5,10), sticky="nsew")
        log_frame.grid_rowconfigure(0, weight=1)
        log_frame.grid_columnconfigure(0, weight=1)
        
        self.log_text = ctk.CTkTextbox(log_frame, wrap="word", font=("Monaco", 12))
        self.log_text.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.log_text.configure(state="disabled")
        
        # Metrics and charts (right side)
        viz_frame = ctk.CTkFrame(self)
        viz_frame.grid(row=2, column=1, padx=(5,10), pady=(5,10), sticky="nsew")
        viz_frame.grid_rowconfigure(1, weight=1)
        viz_frame.grid_columnconfigure(0, weight=1)
        
        # Status indicators
        status_frame = ctk.CTkFrame(viz_frame)
        status_frame.grid(row=0, column=0, padx=10, pady=(10,5), sticky="ew")
        
        # Add progress indicators
        self._create_progress_widgets(status_frame)
        
        # Add visualization charts 
        self._create_charts(viz_frame)
        
        # Add control buttons
        self._create_controls(header)
        
        # Add dataset/model info
        self._create_info_panel(info_panel)
    
    def _create_progress_widgets(self, parent):
        # Epoch progress
        ctk.CTkLabel(parent, text="Epoch:").grid(row=0, column=0, padx=(10,5), pady=5, sticky="w")
        self.epoch_label = ctk.CTkLabel(parent, text="0/0")
        self.epoch_label.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        self.epoch_progress = ctk.CTkProgressBar(parent, width=200)
        self.epoch_progress.grid(row=0, column=2, padx=(5,10), pady=5, sticky="ew")
        self.epoch_progress.set(0)
        
        # Batch progress
        ctk.CTkLabel(parent, text="Batch:").grid(row=1, column=0, padx=(10,5), pady=5, sticky="w")
        self.batch_label = ctk.CTkLabel(parent, text="0/0")
        self.batch_label.grid(row=1, column=1, padx=5, pady=5, sticky="w")
        self.batch_progress = ctk.CTkProgressBar(parent, width=200)
        self.batch_progress.grid(row=1, column=2, padx=(5,10), pady=5, sticky="ew")
        self.batch_progress.set(0)
        
        # Key metrics
        metrics_grid = ctk.CTkFrame(parent)
        metrics_grid.grid(row=2, column=0, columnspan=3, padx=10, pady=5, sticky="ew")
        metrics_grid.grid_columnconfigure((0,1,2,3,4,5), weight=1)
        
        # Learning rate
        ctk.CTkLabel(metrics_grid, text="LR:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.lr_value = ctk.CTkLabel(metrics_grid, text="0.0000")
        self.lr_value.grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Loss
        ctk.CTkLabel(metrics_grid, text="Loss:").grid(row=0, column=2, padx=5, pady=5, sticky="e")
        self.loss_value = ctk.CTkLabel(metrics_grid, text="0.0000")
        self.loss_value.grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # Accuracy
        ctk.CTkLabel(metrics_grid, text="Acc:").grid(row=0, column=4, padx=5, pady=5, sticky="e")
        self.acc_value = ctk.CTkLabel(metrics_grid, text="0.00%")
        self.acc_value.grid(row=0, column=5, padx=5, pady=5, sticky="w")
    
    def _create_charts(self, parent):
        chart_frame = ctk.CTkFrame(parent)
        chart_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(6, 4), dpi=100)
        
        # Loss subplot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("Training/Validation Loss")
        self.ax1.set_ylabel("Loss")
        self.train_loss_line, = self.ax1.plot([], [], 'b-', label="Train")
        self.val_loss_line, = self.ax1.plot([], [], 'r-', label="Val")
        self.ax1.legend()
        
        # Accuracy subplot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Training/Validation Accuracy")
        self.ax2.set_xlabel("Epoch")
        self.ax2.set_ylabel("Accuracy (%)")
        self.train_acc_line, = self.ax2.plot([], [], 'b-', label="Train")
        self.val_acc_line, = self.ax2.plot([], [], 'r-', label="Val")
        self.ax2.legend()
        
        self.fig.tight_layout()
        
        # Add figure to tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True)
    
    def _create_controls(self, parent):
        parent.grid_columnconfigure((0,1,2,3), weight=1)
        
        # Start button
        self.start_btn = ctk.CTkButton(
            parent, 
            text="Start Training", 
            command=self._start_training,
            fg_color="#28a745",
            hover_color="#218838"
        )
        self.start_btn.grid(row=0, column=0, padx=10, pady=10)
        
        # Stop button
        self.stop_btn = ctk.CTkButton(
            parent, 
            text="Stop", 
            command=self._stop_training, 
            state="disabled",
            fg_color="#dc3545",
            hover_color="#c82333"
        )
        self.stop_btn.grid(row=0, column=1, padx=10, pady=10)
        
        # Clear log button
        self.clear_btn = ctk.CTkButton(
            parent, 
            text="Clear Log", 
            command=self._clear_log
        )
        self.clear_btn.grid(row=0, column=2, padx=10, pady=10)
        
        # Export metrics button
        self.export_btn = ctk.CTkButton(
            parent, 
            text="Export Results", 
            command=self._export_results
        )
        self.export_btn.grid(row=0, column=3, padx=10, pady=10)
    
    def _create_info_panel(self, parent):
        parent.grid_columnconfigure((0,1,2,3), weight=1)
        
        # Model info
        model_frame = ctk.CTkFrame(parent)
        model_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        ctk.CTkLabel(model_frame, text="Model: ResNet18", font=("Arial", 14, "bold")).pack(anchor="w", padx=10, pady=(10,5))
        ctk.CTkLabel(model_frame, text="Fine-tuned on ImageNet weights").pack(anchor="w", padx=10, pady=(0,5))
        ctk.CTkLabel(model_frame, text="Final layer: 4 classes (glioma, meningioma, notumor, pituitary)").pack(anchor="w", padx=10, pady=(0,10))
        
        # Dataset info
        data_frame = ctk.CTkFrame(parent)
        data_frame.grid(row=0, column=2, columnspan=2, padx=10, pady=10, sticky="nsew")
        
        self.dataset_info = ctk.CTkLabel(data_frame, text="Dataset: Loading...", font=("Arial", 14, "bold"))
        self.dataset_info.pack(anchor="w", padx=10, pady=(10,5))
        
        self.class_info = ctk.CTkLabel(data_frame, text="Classes: 4")
        self.class_info.pack(anchor="w", padx=10, pady=(0,5))
        
        self.device_info = ctk.CTkLabel(data_frame, text="Device: Checking...")
        self.device_info.pack(anchor="w", padx=10, pady=(0,10))
    
    def log(self, message, level="INFO"):
        """Add a log message to the queue"""
        timestamp = time.strftime("%H:%M:%S")
        
        color_map = {
            "INFO": "white",
            "SUCCESS": "green",
            "WARNING": "orange",
            "ERROR": "red",
            "DEBUG": "gray"
        }
        
        color = color_map.get(level, "white")
        self.log_queue.put((f"[{timestamp}] [{level}] {message}", color))
    
    def update_training_progress(self, progress, metrics):
        """Update training progress from training thread"""
        self.metrics_queue.put(metrics)
    
    def _process_queues(self):
        """Process queued log messages and metrics updates"""
        # Process log messages
        while not self.log_queue.empty():
            message, color = self.log_queue.get()
            self.log_text.configure(state="normal")
            self.log_text.insert("end", message + "\n")
            
            # Apply color to the last line
            end_index = self.log_text.index("end-1c")
            line_start = self.log_text.index(f"{end_index} linestart")
            self.log_text.tag_add(f"color_{color}", line_start, end_index)
            self.log_text.tag_config(f"color_{color}", foreground=color)
            
            self.log_text.configure(state="disabled")
            self.log_text.see("end")
        
        # Process metrics updates
        while not self.metrics_queue.empty():
            metrics = self.metrics_queue.get()
            self._update_progress_display(metrics)
        
        self.after(100, self._process_queues)
    
    def _update_progress_display(self, metrics):
        """Update progress bars and metrics display"""
        # Update dataset info if available
        if 'dataset_info' in metrics:
            info = metrics['dataset_info']
            self.dataset_info.configure(text=f"Dataset: {info['name']}")
            self.class_info.configure(text=f"Classes: {', '.join(info['classes'])}")
            self.device_info.configure(text=f"Device: {info['device']}")
        
        # Update epoch progress
        epoch = metrics.get('epoch', 0)
        total_epochs = metrics.get('total_epochs', 1)
        self.epoch_label.configure(text=f"{epoch}/{total_epochs}")
        self.epoch_progress.set(epoch / total_epochs)
        
        # Update batch progress
        batch = metrics.get('batch', 0)
        total_batches = metrics.get('total_batches', 1)
        self.batch_label.configure(text=f"{batch}/{total_batches}")
        self.batch_progress.set(batch / total_batches)
        
        # Update metrics
        lr = metrics.get('lr', 0)
        loss = metrics.get('loss', 0)
        acc = metrics.get('acc', 0)
        
        self.lr_value.configure(text=f"{lr:.6f}")
        self.loss_value.configure(text=f"{loss:.4f}")
        self.acc_value.configure(text=f"{acc:.2f}%")
        
        # Update charts if epoch metrics available
        if 'history' in metrics:
            history = metrics['history']
            epochs = list(range(1, len(history['train_loss'])+1))
            
            self.train_loss_line.set_data(epochs, history['train_loss'])
            self.val_loss_line.set_data(epochs, history['val_loss'])
            self.train_acc_line.set_data(epochs, history['train_acc'])
            self.val_acc_line.set_data(epochs, history['val_acc'])
            
            # Auto-scale axes
            self.ax1.relim()
            self.ax1.autoscale_view()
            self.ax2.relim()
            self.ax2.autoscale_view()
            
            self.canvas.draw()
    
    def _start_training(self):
        """Start the training process in a separate thread"""
        if self.is_training:
            return
        
        self.is_training = True
        self.stop_event.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        
        # Create and start training thread
        self.training_thread = threading.Thread(
            target=self._run_training,
            daemon=True
        )
        self.training_thread.start()
        
        self.log("Training started", "SUCCESS")
    
    def _stop_training(self):
        """Stop the training process"""
        if not self.is_training:
            return
        
        self.stop_event.set()
        self.stop_btn.configure(state="disabled")
        self.log("Stopping training...", "WARNING")
    
    def _run_training(self):
        """Run the actual training process (placeholder)"""
        # This should be replaced by your actual training code
        try:
            self.log("Training simulation started", "INFO")
            time.sleep(0.5)
            
            # Simulated dataset info
            self.update_training_progress(0, {
                'dataset_info': {
                    'name': 'Brain Tumor MRI',
                    'classes': ['glioma', 'meningioma', 'notumor', 'pituitary'],
                    'device': 'MPS (Apple GPU)'
                }
            })
            
            history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            
            # Simulated training loop
            total_epochs = 30
            for epoch in range(total_epochs):
                if self.stop_event.is_set():
                    self.log("Training stopped by user", "WARNING")
                    break
                
                self.log(f"Starting epoch {epoch+1}/{total_epochs}", "INFO")
                
                # Simulate batches
                batches = 50
                epoch_loss = 1.0 - 0.03 * epoch + 0.05 * random.random()
                for batch in range(batches):
                    if self.stop_event.is_set():
                        break
                    
                    # Simulate processing time
                    time.sleep(0.05)
                    
                    # Update UI
                    loss = epoch_loss - 0.001 * batch + 0.01 * random.random()
                    acc = 60 + 1.3 * epoch + 0.05 * batch + random.random()
                    self.update_training_progress(batch/batches, {
                        'epoch': epoch+1,
                        'total_epochs': total_epochs,
                        'batch': batch+1,
                        'total_batches': batches,
                        'lr': 0.001 * (1 - epoch/total_epochs),
                        'loss': loss,
                        'acc': acc
                    })
                
                # Simulate validation
                train_loss = epoch_loss
                train_acc = 60 + 1.3 * epoch + 3 * random.random()
                val_loss = train_loss * (0.95 + 0.15 * random.random())
                val_acc = train_acc * (0.95 + 0.1 * random.random())
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                
                # Log results
                self.log(f"Epoch {epoch+1}: Train loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                         f"Val loss={val_loss:.4f}, acc={val_acc:.2f}%",
                         "INFO")
                
                # Update charts
                self.update_training_progress(1.0, {
                    'epoch': epoch+1,
                    'total_epochs': total_epochs,
                    'history': history
                })
            
            if not self.stop_event.is_set():
                self.log("Training complete!", "SUCCESS")
                
                # Simulate export
                self.log("Saving model to ./models/brain_tumor_resnet18.pth", "INFO")
                time.sleep(0.5)
                self.log("Saving class labels to ./models/labels.json", "INFO")
                time.sleep(0.3)
                
                # Show final evaluation
                self.log("\nFinal evaluation report:", "SUCCESS")
                self.log("              precision    recall  f1-score   support", "INFO")
                self.log("     glioma      0.943     0.957     0.950        47", "INFO")
                self.log(" meningioma      0.980     0.942     0.961        52", "INFO")
                self.log("    notumor      0.981     0.981     0.981        53", "INFO")
                self.log("  pituitary      0.960     0.980     0.970        50", "INFO")
                self.log("", "INFO")
                self.log("    accuracy                          0.965       202", "INFO")
                self.log("   macro avg      0.966     0.965     0.965       202", "INFO")
                self.log("weighted avg      0.966     0.965     0.965       202", "INFO")
        
        except Exception as e:
            self.log(f"Error during training: {str(e)}", "ERROR")
            traceback.print_exc()
        
        finally:
            # Reset UI state
            self.is_training = False
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.stop_btn.configure(state="disabled"))
    
    def _clear_log(self):
        """Clear the log console"""
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")
    
    def _export_results(self):
        """Export training results"""
        try:
            os.makedirs("./results", exist_ok=True)
            
            # Export charts
            plt.figure(figsize=(10, 8))
            
            plt.subplot(2, 1, 1)
            plt.title("Training/Validation Loss")
            plt.plot(self.train_loss_line.get_xdata(), self.train_loss_line.get_ydata(), 'b-', label="Train")
            plt.plot(self.val_loss_line.get_xdata(), self.val_loss_line.get_ydata(), 'r-', label="Val")
            plt.ylabel("Loss")
            plt.legend()
            
            plt.subplot(2, 1, 2)
            plt.title("Training/Validation Accuracy")
            plt.plot(self.train_acc_line.get_xdata(), self.train_acc_line.get_ydata(), 'b-', label="Train")
            plt.plot(self.val_acc_line.get_xdata(), self.val_acc_line.get_ydata(), 'r-', label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy (%)")
            plt.legend()
            
            plt.tight_layout()
            chart_path = "./results/training_charts.png"
            plt.savefig(chart_path)
            plt.close()
            
            # Export log
            log_content = self.log_text.get("1.0", "end")
            log_path = "./results/training_log.txt"
            with open(log_path, 'w') as f:
                f.write(log_content)
            
            self.log(f"Results exported to {os.path.abspath('./results/')}", "SUCCESS")
        except Exception as e:
            self.log(f"Error exporting results: {str(e)}", "ERROR")

# Function to integrate with existing code
def launch_gui_monitor(args):
    """Launch the GUI training monitor"""
    app = TrainingMonitorApp()
    
    # Import main training functionality
    from training import set_seed, build_dataloaders, build_model, compute_class_weights
    from training import train_loop, export_labels, full_evaluation_report, fit_temperature, save_temperature
    
    # Create a function to pass to your training loop for UI updates
    def update_callback(progress, metrics):
        app.update_training_progress(progress, metrics)
    
    # Create a wrapper for the main training process
    def run_training_with_gui():
        try:
            app.log("Training started with arguments:", "INFO")
            for key, value in vars(args).items():
                app.log(f"  {key}: {value}", "INFO")
            
            set_seed(args.seed)
            
            # Device setup
            import torch
            use_cuda = torch.cuda.is_available()
            use_mps = torch.backends.mps.is_available()
            device = torch.device("cuda" if use_cuda else ("mps" if use_mps else "cpu"))
            app.log(f"Using device: {device}", "INFO")
            
            # AMP setup
            amp_device_type = "cuda" if use_cuda else ("mps" if use_mps else "cpu")
            use_amp = amp_device_type in ("cuda", "mps")
            
            # Set environment variables for macOS MPS
            if device.type == "mps":
                app.log("Setting MPS environment variables for better performance", "INFO")
                os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
            
            # Load datasets
            app.log("Loading datasets...", "INFO")
            train_loader, test_loader, classes = build_dataloaders(
                args.train_root, args.test_root,
                batch_size=args.batch_size, workers=args.workers
            )
            
            # Update UI with dataset info
            app.update_training_progress(0, {
                'dataset_info': {
                    'name': f"Brain Tumor MRI ({len(train_loader.dataset)} train, {len(test_loader.dataset)} test)",
                    'classes': classes,
                    'device': device
                }
            })
            
            app.log(f"Classes: {classes}", "INFO")
            
            # Build model
            app.log("Building ResNet-18 model...", "INFO")
            model = build_model(num_classes=len(classes), device=device)
            
            # Class weights
            class_weights = None
            if args.use_class_weights:
                app.log("Computing class weights...", "INFO")
                class_weights = compute_class_weights(train_loader, num_classes=len(classes), device=device)
                app.log(f"Class weights: {class_weights.detach().cpu().numpy()}", "INFO")
            
            # Define modified train loop with progress updates
            def train_with_progress(model, train_loader, test_loader, device, **kwargs):
                criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs // 2))
                
                # For AMP
                #scaler = torch.amp.GradScaler(device_type=amp_device_type, enabled=use_amp)
                import pkg_resources
                pytorch_version = pkg_resources.get_distribution("torch").version
                is_torch_2_plus = int(pytorch_version.split('.')[0]) >= 2

                try:
                    # Try the new PyTorch 2.0+ syntax
                    scaler = torch.amp.GradScaler(enabled=use_amp)  # Remove device_type parameter
                except TypeError:
                    # Fallback for older PyTorch
                    if amp_device_type == "cuda":
                        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
                    else:
                        app.log("Warning: Mixed precision not fully supported on this device.", "WARNING")
                        scaler = torch.amp.GradScaler(enabled=False)
                
                best_acc, bad_epochs = 0.0, 0
                history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
                
                os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
                
                for epoch in range(1, args.epochs + 1):
                    if app.stop_event.is_set():
                        app.log("Training stopped by user", "WARNING")
                        break
                    
                    app.log(f"Epoch {epoch}/{args.epochs} started", "INFO")
                    model.train()
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    
                    for batch_idx, (inputs, targets) in enumerate(train_loader):
                        if app.stop_event.is_set():
                            break
                        
                        inputs, targets = inputs.to(device), targets.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        
                        # Use autocast for mixed precision
                        with torch.amp.autocast(device_type=amp_device_type, enabled=use_amp):
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        
                        running_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        
                        # Update UI with batch progress
                        batch_loss = loss.item()
                        batch_acc = 100. * correct / total
                        
                        update_callback(batch_idx/len(train_loader), {
                            'epoch': epoch,
                            'total_epochs': args.epochs,
                            'batch': batch_idx+1, 
                            'total_batches': len(train_loader),
                            'lr': scheduler.get_last_lr()[0],
                            'loss': batch_loss,
                            'acc': batch_acc
                        })
                    
                    # Epoch completed
                    scheduler.step()
                    train_loss = running_loss / len(train_loader)
                    train_acc = 100. * correct / total
                    
                    # Validation
                    app.log("Running validation...", "INFO")
                    model.eval()
                    val_loss, val_correct, val_total = 0.0, 0, 0
                    
                    with torch.no_grad():
                        for inputs, targets in test_loader:
                            if app.stop_event.is_set():
                                break
                                
                            inputs, targets = inputs.to(device), targets.to(device)
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            
                            val_loss += loss.item() * targets.size(0)
                            _, predicted = outputs.max(1)
                            val_total += targets.size(0)
                            val_correct += predicted.eq(targets).sum().item()
                    
                    val_loss = val_loss / val_total
                    val_acc = 100. * val_correct / val_total
                    
                    # Update history
                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['train_acc'].append(train_acc)
                    history['val_acc'].append(val_acc)
                    
                    # Log results
                    app.log(f"Epoch {epoch}: Train loss={train_loss:.4f}, acc={train_acc:.2f}% | "
                          f"Val loss={val_loss:.4f}, acc={val_acc:.2f}%",
                          "INFO")
                    
                    # Update charts
                    update_callback(1.0, {
                        'epoch': epoch,
                        'total_epochs': args.epochs,
                        'history': history
                    })
                    
                    # Save best model
                    if val_acc > best_acc:
                        best_acc = val_acc
                        torch.save(model.state_dict(), args.save_path)
                        app.log(f"Saved best model with val acc: {val_acc:.2f}%", "SUCCESS")
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        app.log(f"Validation accuracy did not improve. {bad_epochs}/{args.patience} epochs without improvement.", "WARNING")
                        if bad_epochs >= args.patience:
                            app.log("Early stopping triggered.", "WARNING")
                            break
                
                # Load best model
                app.log("Loading best model for evaluation...", "INFO")
                model.load_state_dict(torch.load(args.save_path, map_location=device))
            
            # Start the actual training
            app.log("Starting training loop...", "INFO")
            try:
                train_with_progress(model, train_loader, test_loader, device)
                
                if app.stop_event.is_set():
                    app.log("Training was interrupted. Skipping final evaluation.", "WARNING")
                    return
                
                # Save class labels
                app.log("Exporting class labels...", "INFO")
                export_labels(classes, path="./models/labels.json")
                
                # Run final evaluation
                app.log("Running final evaluation...", "INFO")
                model.eval()
                all_y, all_p = [], []
                
                with torch.no_grad():
                    for inputs, targets in test_loader:
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        
                        all_p.append(predicted.cpu().numpy())
                        all_y.append(targets.numpy())
                
                all_p = np.concatenate(all_p)
                all_y = np.concatenate(all_y)
                
                from sklearn.metrics import classification_report, confusion_matrix
                
                # Show classification report
                report = classification_report(all_y, all_p, target_names=classes, output_dict=True)
                app.log("\nClassification report:", "SUCCESS")
                app.log(f"              precision    recall  f1-score   support", "INFO")
                
                for cls in classes:
                    metrics = report[cls]
                    app.log(f"{cls:>10}      {metrics['precision']:.3f}     {metrics['recall']:.3f}     {metrics['f1-score']:.3f}     {metrics['support']:>6}", "INFO")
                
                app.log("", "INFO")
                app.log(f"    accuracy                          {report['accuracy']:.3f}     {sum(report[cls]['support'] for cls in classes):>6}", "INFO")
                app.log(f"   macro avg     {report['macro avg']['precision']:.3f}     {report['macro avg']['recall']:.3f}     {report['macro avg']['f1-score']:.3f}     {sum(report[cls]['support'] for cls in classes):>6}", "INFO")
                app.log(f"weighted avg     {report['weighted avg']['precision']:.3f}     {report['weighted avg']['recall']:.3f}     {report['weighted avg']['f1-score']:.3f}     {sum(report[cls]['support'] for cls in classes):>6}", "INFO")
                
                # Show confusion matrix
                cm = confusion_matrix(all_y, all_p)
                app.log("\nConfusion matrix:", "INFO")
                app.log(str(cm), "INFO")
                
                # Temperature scaling if requested
                if args.fit_temperature:
                    app.log("\nFitting temperature scaling...", "INFO")
                    T = fit_temperature(model, test_loader, device, max_iter=50)
                    save_temperature(T, path="./models/temperature.pt")
                    app.log(f"Optimal temperature: {float(T.item()):.4f}", "SUCCESS")
                
                app.log("\nTraining complete! Model saved to " + os.path.abspath(args.save_path), "SUCCESS")
                
            except Exception as e:
                app.log(f"Error during training: {str(e)}", "ERROR")
                traceback.print_exc()
        
        except Exception as e:
            app.log(f"Error setting up training: {str(e)}", "ERROR")
            traceback.print_exc()

    def plot_confusion_matrix():
        cm = confusion_matrix(all_y, all_p)
        
        # Create figure for confusion matrix
        cm_fig = Figure(figsize=(6, 5))
        cm_ax = cm_fig.add_subplot(111)
        
        # Plot confusion matrix
        im = cm_ax.imshow(cm, interpolation='nearest', cmap='Blues')
        cm_ax.set_title("Confusion Matrix")
        cm_fig.colorbar(im)
        
        # Label axes
        tick_marks = np.arange(len(classes))
        cm_ax.set_xticks(tick_marks)
        cm_ax.set_yticks(tick_marks)
        cm_ax.set_xticklabels(classes, rotation=45, ha="right")
        cm_ax.set_yticklabels(classes)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                cm_ax.text(j, i, str(cm[i, j]), 
                        ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max()/2 else "black")
        
        cm_fig.tight_layout()
        
        # Add to a new window
        cm_win = ctk.CTkToplevel(app)
        cm_win.title("Confusion Matrix")
        cm_win.geometry("600x550")
        
        cm_canvas = FigureCanvasTkAgg(cm_fig, master=cm_win)
        cm_canvas.draw()
        cm_canvas.get_tk_widget().pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

    app.after(0, plot_confusion_matrix)
        
    # Start training in a separate thread
    training_thread = threading.Thread(target=run_training_with_gui, daemon=True)
    training_thread.start()
    
    # Run the GUI main loop
    app.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brain Tumor MRI Classification with GUI")
    parser.add_argument("--train-root", type=str, required=True, help="Path to Training/")
    parser.add_argument("--test-root", type=str, required=True, help="Path to Testing/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save-path", type=str, default="./models/brain_tumor_resnet18.pth")
    parser.add_argument("--fit-temperature", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    launch_gui_monitor(args)