
import os
import threading
import pickle
import math
import time
import numpy as np
import torch
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageGrab, ImageTk

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# SNN Imports
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from torchvision import datasets, transforms

# ----------------------- Configuration -----------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# ----------------------- Training Logic -----------------------
def train_snn(epochs: int, subset_size: int, num_neurons: int, update_callback=None):
    """
    Train the SNN and return model dict with weights, labels, firing tracker and test data.
    """
    X_test, y_test = [], []
    
    # Use CUDA if available, but ensure consistency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------------------------
    # 1. Setup Network (Diehl & Cook 2015 variant)
    # --------------------------------------------------------------------------
    network = DiehlAndCook2015(
        n_inpt=28 * 28,
        n_neurons=num_neurons,
        exc=22.5,
        inh=17.5,
        dt=1.0,
        nu=(1e-4, 1e-2),
        wmin=0.0,
        wmax=1.0,
        norm=78.4,
        theta_plus=0.05,
        tc_theta_decay=1e7,
        inpt_shape=(1, 28, 28),
    ).to(device)

    # Monitors
    monitor = Monitor(network.layers["Ae"], state_vars=["s"], time=250)
    network.add_monitor(monitor, name="AeSpikes")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    encoder = PoissonEncoder(time=250, dt=1.0)
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        # Slight augmentation to improve generalization
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform_train)
    mnist_test  = datasets.MNIST(root="./data", train=False, download=True, transform=transform_test)

    # Subset handling
    if subset_size < len(mnist_train):
        mnist_train = torch.utils.data.Subset(mnist_train, range(subset_size))
    if subset_size < len(mnist_test):
        mnist_test = torch.utils.data.Subset(mnist_test, range(subset_size))

    # --------------------------------------------------------------------------
    # 3. Training Loop
    # --------------------------------------------------------------------------
    neuron_spike_sums = np.zeros((num_neurons, 10))
    
    total_iterations = epochs * len(mnist_train)
    current_iteration = 0

    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(epochs):
        for i, (img, label) in enumerate(mnist_train):
            # Reset
            network.reset_state_variables()
            monitor.reset_state_variables()
            
            # Encode & Run
            sample = encoder(img.clone().detach() * 128).to(device) 
            network.run(inputs={"X": sample}, time=250)
            
            # Record Spikes
            spikes = monitor.get("s").view(-1, num_neurons) # [time, neurons]
            if spikes.dim() == 3: # Handle batch dimension if present
                spikes = spikes.squeeze(1)
            spike_counts = spikes.sum(0).cpu().numpy()      # [neurons]
            
            # Accumulate votes for the correct label
            neuron_spike_sums[:, int(label)] += spike_counts
            
            current_iteration += 1
            if update_callback:
                elapsed_time = time.time() - start_time
                update_callback(current_iteration, total_iterations, elapsed_time)

    # --------------------------------------------------------------------------
    # 4. Label Assignment (Soft Voting)
    # --------------------------------------------------------------------------
    neuron_label_map = {}
    for i in range(num_neurons):
        activity = neuron_spike_sums[i]
        if activity.sum() > 0:
            assigned_label = np.argmax(activity)
            neuron_label_map[i] = int(assigned_label)
        else:
            neuron_label_map[i] = -1

    # --------------------------------------------------------------------------
    # 5. Testing
    # --------------------------------------------------------------------------
    print("Running test set evaluation...")
    y_true = []
    y_pred = []
    firing_tracker = neuron_spike_sums 

    correct = 0
    total = 0
    
    for i, (img, label) in enumerate(mnist_test):
        network.reset_state_variables()
        monitor.reset_state_variables()
        
        sample = encoder(img.clone().detach() * 128).to(device)
        network.run(inputs={"X": sample}, time=250)
        
        spikes = monitor.get("s").view(-1, num_neurons)
        if spikes.dim() == 3:
            spikes = spikes.squeeze(1)
            
        spike_counts = spikes.sum(0).cpu().numpy()
        
        # Predict
        class_votes = np.zeros(10)
        for nid, count in enumerate(spike_counts):
            if count > 0 and nid in neuron_label_map and neuron_label_map[nid] != -1:
                class_votes[neuron_label_map[nid]] += count
        
        pred = np.argmax(class_votes)
        
        y_true.append(int(label))
        y_pred.append(int(pred))
        
        if len(X_test) < 1000:
            spk = spikes.cpu().numpy()
            first_spike = np.argmax(spk > 0, axis=0)
            first_spike[np.max(spk, axis=0) == 0] = 250
            X_test.append(np.concatenate([spike_counts, first_spike]))
            y_test.append(int(label))

        if pred == label:
            correct += 1
        total += 1
    
    acc = correct / total if total > 0 else 0
    weights = network.connections["X", "Ae"].w.detach().cpu().numpy().T
    
    return {
        "weights": weights,
        "labels": neuron_label_map,
        "X_test": X_test,
        "y_test": y_test,
        "firing": firing_tracker,
        "accuracy": acc
    }

# ----------------------- Main App (GUI) -----------------------

class SNNModel:
    """
    Manages the BindsNET network, loading, and prediction logic.
    """
    def __init__(self):
        self.network = None
        self.encoder = PoissonEncoder(time=250, dt=1.0)
        self.monitor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model state
        self.weights = None
        self.labels = {}
        self.neuron_count = 0
        self.firing_history = None 

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        self.weights = data["weights"] 
        self.labels = data["labels"]
        self.neuron_count = self.weights.shape[0]
        
        self._build_network()
        
        w_matrix = torch.tensor(self.weights.T, device=self.device, dtype=torch.float)
        self.network.connections["X", "Ae"].w.data = w_matrix
        
        print(f"Model loaded with {self.neuron_count} neurons.")

    def load_from_memory(self, data):
        """Loads model directly from training output dictionary"""
        self.weights = data["weights"]
        self.labels = data["labels"]
        self.neuron_count = self.weights.shape[0]
        
        self._build_network()
        
        w_matrix = torch.tensor(self.weights.T, device=self.device, dtype=torch.float)
        self.network.connections["X", "Ae"].w.data = w_matrix
        print(f"Model loaded from memory with {self.neuron_count} neurons.")

    def _build_network(self):
        self.network = DiehlAndCook2015(
            n_inpt=784,
            n_neurons=self.neuron_count,
            exc=22.5,
            inh=17.5,
            dt=1.0,
            nu=(1e-4, 1e-2),
            wmin=0.0,
            wmax=1.0,
            norm=78.4,
            theta_plus=0.05,
            tc_theta_decay=1e7,
            inpt_shape=(1, 28, 28),
        ).to(self.device)
        
        self.monitor = Monitor(self.network.layers["Ae"], state_vars=["s"], time=250)
        self.network.add_monitor(self.monitor, name="AeSpikes")

    def predict(self, img_array):
        # Prepare input
        # img_array should be 28x28. 
        # Scale to consistent intensity (e.g. 0-128 Hz)
        img_tensor = torch.tensor(img_array, dtype=torch.float, device=self.device)
        if img_tensor.max() <= 1.0:
            img_tensor *= 128.0
        
        # Flatten to (1, 784) because DiehlAndCook2015 expects flattened input or (batch, 1, 28, 28)?
        # The error "mat1 and mat2 shapes cannot be multiplied (28x28 and 784x225)" 
        # suggests it treats the input as [28, 28] and tries to multiply by [784, 225].
        # We need [1, 784] to match [784, 225].
        img_tensor = img_tensor.view(1, 784)
        
        encoded = self.encoder(img_tensor).to(self.device)
        
        # Run
        self.network.reset_state_variables()
        self.monitor.reset_state_variables()
        # BindsNET might expect {"X": [time, batch, input_dim]} or just [time, input_dim]
        # But DiehlAndCook2015 usually takes {"X": ...}
        self.network.run(inputs={"X": encoded}, time=250)
        
        spikes = self.monitor.get("s") 
        if spikes.dim() == 3:
            spikes = spikes.squeeze(1)
            
        spike_counts = spikes.sum(0).cpu().numpy() 
        
        votes = np.zeros(10)
        for nid, count in enumerate(spike_counts):
            if count > 0 and nid in self.labels and self.labels[nid] != -1:
                votes[self.labels[nid]] += count
                
        total_votes = votes.sum()
        if total_votes > 0:
            pred = np.argmax(votes)
            conf = votes[pred] / total_votes
        else:
            pred = -1
            conf = 0.0
            
        spike_times_indices = torch.nonzero(spikes, as_tuple=False).cpu().numpy()
        
        return pred, conf, spike_counts, spike_times_indices

class SNNApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Simulated Neural Network (SNN) - Dashboard")
        self.geometry("1400x900")
        
        self.snn = SNNModel()
        
        # Layout Config
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # UI Components
        self._setup_sidebar()
        self._setup_main_area()
        
    def _setup_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="SNN Control", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.pack(padx=20, pady=(20, 10))
        
        # Training Controls
        self.train_lbl = ctk.CTkLabel(self.sidebar, text="Training Config", anchor="w")
        self.train_lbl.pack(padx=20, pady=(10, 0), anchor="w")
        
        self.subset_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Subset (e.g. 1000)")
        self.subset_entry.pack(padx=20, pady=5)
        self.subset_entry.insert(0, "1000")

        self.epochs_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Epochs")
        self.epochs_entry.pack(padx=20, pady=5)
        self.epochs_entry.insert(0, "1")

        self.neurons_entry = ctk.CTkEntry(self.sidebar, placeholder_text="Neurons (e.g. 225)")
        self.neurons_entry.pack(padx=20, pady=5)
        self.neurons_entry.insert(0, "225")
        
        self.train_btn = ctk.CTkButton(self.sidebar, text="Start Training", command=self.start_training)
        self.train_btn.pack(padx=20, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(self.sidebar)
        self.progress_bar.pack(padx=20, pady=0)
        self.progress_bar.set(0)
        
        self.eta_lbl = ctk.CTkLabel(self.sidebar, text="ETA: --:--", font=("Arial", 10))
        self.eta_lbl.pack(padx=20, pady=0)
        
        self.status_lbl = ctk.CTkLabel(self.sidebar, text="Ready", font=("Arial", 10))
        self.status_lbl.pack(padx=20, pady=5)
        
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=20)
        
        # Model I/O
        self.load_btn = ctk.CTkButton(self.sidebar, text="Load Model", command=self.load_model)
        self.load_btn.pack(padx=20, pady=5)
        
        self.save_btn = ctk.CTkButton(self.sidebar, text="Save Model", command=self.save_model)
        self.save_btn.pack(padx=20, pady=5)

    # ... (Previous code remains, just updating affected methods)

    def _setup_main_area(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=0) # Canvas
        self.main_frame.grid_columnconfigure(1, weight=1) # Dashboard
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # --- Left: Canvas & Input ---
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 20))
        
        self.canvas_lbl = ctk.CTkLabel(self.input_frame, text="Draw Digit (0-9)", font=("Arial", 16))
        self.canvas_lbl.pack(pady=10)
        
        # Canvas (White background)
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.input_frame, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        self.canvas.bind("<B1-Motion>", self._on_draw)
        
        # In-memory image for robust capturing (No screenshots needed)
        from PIL import ImageDraw
        self.pil_image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.pil_draw = ImageDraw.Draw(self.pil_image)
        
        self.predict_btn = ctk.CTkButton(self.input_frame, text="Predict", height=40, font=("Arial", 14, "bold"), command=self.predict)
        self.predict_btn.pack(pady=10, padx=20, fill="x")
        
        self.clear_btn = ctk.CTkButton(self.input_frame, text="Clear Canvas", fg_color="gray", command=self.clear_canvas)
        self.clear_btn.pack(pady=5, padx=20, fill="x")
        
        self.result_frame = ctk.CTkFrame(self.input_frame, fg_color="gray20")
        self.result_frame.pack(pady=20, padx=10, fill="x")
        
        self.pred_lbl = ctk.CTkLabel(self.result_frame, text="?", font=("Arial", 48, "bold"), text_color="#3B8ED0")
        self.pred_lbl.pack(pady=10)
        self.conf_lbl = ctk.CTkLabel(self.result_frame, text="Confidence: --%", font=("Arial", 12))
        self.conf_lbl.pack(pady=(0, 10))

        # --- Right: Visualization Dashboard ---
        self.dash_frame = ctk.CTkTabview(self.main_frame)
        self.dash_frame.grid(row=0, column=1, sticky="nsew")
        
        self.tab_heatmap = self.dash_frame.add("Neuron Heatmap")
        self.tab_spikes = self.dash_frame.add("Spike Raster")
        self.tab_weights = self.dash_frame.add("Weight Patches")
        
        # Heatmap setup
        self.heatmap_frame = ctk.CTkScrollableFrame(self.tab_heatmap, fg_color="transparent")
        self.heatmap_frame.pack(fill="both", expand=True)
        self.neuron_labels = [] # To store grid labels
        
        # Spike Raster setup
        self.fig_spikes = Figure(figsize=(5, 4), dpi=100, facecolor="#2b2b2b")
        self.ax_spikes = self.fig_spikes.add_subplot(111)
        self.ax_spikes.set_facecolor("#2b2b2b")
        self.ax_spikes.tick_params(axis='x', colors='white')
        self.ax_spikes.tick_params(axis='y', colors='white')
        self.canvas_spikes = FigureCanvasTkAgg(self.fig_spikes, master=self.tab_spikes)
        self.canvas_spikes.get_tk_widget().pack(fill="both", expand=True)
        
        # Weights setup
        self.weights_frame = ctk.CTkScrollableFrame(self.tab_weights)
        self.weights_frame.pack(fill="both", expand=True)

    def _on_draw(self, event):
        x, y = event.x, event.y
        r = 10 # Brush radius
        # Draw on UI Canvas
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        # Draw on in-memory PIL image
        self.pil_draw.ellipse([x-r, y-r, x+r, y+r], fill="white", outline="white")

    def clear_canvas(self):
        self.canvas.delete("all")
        # Clear in-memory image
        self.pil_draw.rectangle([0, 0, self.canvas_size, self.canvas_size], fill="black")
        self.pred_lbl.configure(text="?")
        self.conf_lbl.configure(text="Confidence: --%")

    def _get_canvas_image(self):
        # Resize from 280x280 to 28x28
        img = self.pil_image.resize((28, 28), resample=Image.Resampling.LANCZOS)
        return np.array(img) / 255.0

    def start_training(self):
        try:
            epochs = int(self.epochs_entry.get())
            subset = int(self.subset_entry.get())
            neurons = int(self.neurons_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid training parameters")
            return
            
        self.train_btn.configure(state="disabled")
        self.status_lbl.configure(text="Training...")
        self.progress_bar.set(0)
        self.eta_lbl.configure(text="ETA: Calculating...")
        
        def run():
            try:
                # Callback to update UI from thread
                def callback(current_iter, total_iters, elapsed_time):
                    prog = current_iter / total_iters
                    
                    # Calculate ETA
                    if prog > 0:
                        total_time = elapsed_time / prog
                        remaining_time = total_time - elapsed_time
                        
                        # Format as MM:SS
                        m, s = divmod(int(remaining_time), 60)
                        eta_str = f"ETA: {m:02d}:{s:02d} ({int(prog*100)}%)"
                    else:
                        eta_str = "ETA: Calculating..."
                    
                    self.progress_bar.set(prog)
                    self.eta_lbl.configure(text=eta_str)
                
                model_data = train_snn(epochs, subset, neurons, update_callback=callback)
                
                # Auto-save immediately after training as requested
                self.after(0, lambda: self._prompt_save_and_load(model_data))
                
            except Exception as e:
                print(e)
                self.after(0, lambda: self._on_training_complete(False, str(e)))
        
        threading.Thread(target=run, daemon=True).start()

    def _prompt_save_and_load(self, model_data):
        # Notify user and save
        self.status_lbl.configure(text="Saving...")
        self.eta_lbl.configure(text="ETA: 00:00")
        self.progress_bar.set(1.0)
        
        # Load immediately into memory for use
        self.snn.load_from_memory(model_data)
        self._rebuild_heatmap_grid()
        self._visualize_weights()
        
        # Save to file with TIMESTAMP
        timestamp = int(time.time())
        default_name = f"model_{timestamp}.pkl" # User requested integer/time
        
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", 
            initialfile=default_name, 
            title="Save Trained Model"
        )
        
        if path:
            # Fix double extensions if user added one manually or OS weirdness
            if path.endswith(".pkl.pkl"):
                path = path[:-4]
                
            with open(path, "wb") as f:
                pickle.dump(model_data, f)
            messagebox.showinfo("Success", f"Training Complete!\nModel saved to: {path}\nModel is now active.")
        else:
            messagebox.showinfo("Training Complete", "Model trained and loaded into memory.\n(Not saved to disk)")
            
        self._on_training_complete(True)

    def _on_training_complete(self, success, error=None):
        self.train_btn.configure(state="normal")
        if success:
            self.status_lbl.configure(text="Ready") 
        else:
            self.status_lbl.configure(text="Failed")
            self.eta_lbl.configure(text="ETA: --:--")
            messagebox.showerror("Training Error", error)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        if path:
            try:
                self.snn.load(path)
                self._rebuild_heatmap_grid()
                self._visualize_weights()
                messagebox.showinfo("Loaded", f"Model loaded: {path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def save_model(self):
        # This saves the CURRENTLY LOADED model, not retraining
        if self.snn.weights is None:
             messagebox.showwarning("Warning", "No model loaded to save.")
             return
             
        timestamp = int(time.time())
        default_name = f"model_{timestamp}.pkl"

        path = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            initialfile=default_name
        )
        
        if path:
             # Fix double extensions
            if path.endswith(".pkl.pkl"):
                path = path[:-4]
                
            # We reconstruct the dict format
            data = {
                "weights": self.snn.weights,
                "labels": self.snn.labels,
            }
            with open(path, "wb") as f:
                pickle.dump(data, f)
            messagebox.showinfo("Saved", f"Model saved to {path}")

    def predict(self):
        if self.snn.network is None:
            messagebox.showwarning("Warning", "Please train or load a model first.")
            return
            
        img = self._get_canvas_image()
        self.status_lbl.configure(text="Predicting...")
        self.update_idletasks()
        
        try:
            # Fix Shape Error: Flatten image for SNN input
            # img is (28, 28)
            # network expects (1, 1, 28, 28) or flattened depending on BindsNET internals.
            # The encoding happens in SNNModel.predict
            pred, conf, counts, spike_times = self.snn.predict(img)
            
            # Update UI
            self.pred_lbl.configure(text=str(pred) if pred != -1 else "?")
            self.conf_lbl.configure(text=f"Confidence: {conf*100:.1f}%")
            
            # visualizations
            self._update_heatmap(counts)
            self._update_spike_plot(spike_times)
            
            self.status_lbl.configure(text="Done")
        except Exception as e:
            messagebox.showerror("Prediction Error", str(e))
            self.status_lbl.configure(text="Error")

    def _rebuild_heatmap_grid(self):
        # Clear old
        for widget in self.heatmap_frame.winfo_children():
            widget.destroy()
        self.neuron_labels.clear()
        
        n = self.snn.neuron_count
        cols = int(math.ceil(math.sqrt(n)))
        
        for i in range(n):
            lbl = ctk.CTkLabel(self.heatmap_frame, text=f"{i}", width=30, height=30, fg_color="gray30", corner_radius=5)
            lbl.grid(row=i//cols, column=i%cols, padx=2, pady=2)
            
            # Tooltip via bind (simple print for now)
            lbl.bind("<Enter>", lambda e, idx=i: self.status_lbl.configure(text=f"Neuron {idx}"))
            
            self.neuron_labels.append(lbl)

    def _update_heatmap(self, counts):
        if not self.neuron_labels: return
        
        max_spikes = counts.max()
        for i, count in enumerate(counts):
            if i >= len(self.neuron_labels): break
            
            intensity = count / max_spikes if max_spikes > 0 else 0
            # Color map: Blue (low) -> Red (high)
            # Simple interpolation
            r = int(255 * intensity)
            b = int(255 * (1 - intensity))
            color = f"#{r:02x}00{b:02x}"
            
            self.neuron_labels[i].configure(fg_color=color)

    def _update_spike_plot(self, spike_times_indices):
        self.ax_spikes.clear()
        
        # spike_times_indices: [timestep, neuron_id]
        if len(spike_times_indices) > 0:
            times = spike_times_indices[:, 0]
            neurons = spike_times_indices[:, 1]
            self.ax_spikes.scatter(times, neurons, s=5, c="cyan", alpha=0.7)
            
        self.ax_spikes.set_xlim(0, 250)
        self.ax_spikes.set_ylim(0, self.snn.neuron_count)
        self.ax_spikes.set_xlabel("Time (ms)", color="white")
        self.ax_spikes.set_ylabel("Neuron ID", color="white")
        self.ax_spikes.set_title("Spike Raster Plot", color="white")
        
        self.canvas_spikes.draw()

    def _visualize_weights(self):
        # Clear
        for widget in self.weights_frame.winfo_children():
            widget.destroy()
            
        weights = self.snn.weights # (Neurons, 784)
        if weights is None: return
        
        n = weights.shape[0]
        cols = int(math.ceil(math.ceil(math.sqrt(n))))
        
        # Reuse logic from original GUI but adaptable
        # Since generating 200 images might be slow, we show first 50 or lazy load
        # For now, show all but small
        
        for i in range(min(n, 400)):
             # normalized 0-255
            w = weights[i].reshape(28, 28)
            w = (w - w.min()) / (w.max() - w.min() + 1e-5) * 255
            img = Image.fromarray(w.astype(np.uint8))
            img = img.resize((40, 40), Image.Resampling.NEAREST)
            tk_img = ImageTk.PhotoImage(img)
            
            lbl = tk.Label(self.weights_frame, image=tk_img, bd=0)
            lbl.image = tk_img # keep ref
            lbl.grid(row=i//cols, column=i%cols, padx=1, pady=1)


if __name__ == "__main__":
    app = SNNApp()
    app.mainloop()
