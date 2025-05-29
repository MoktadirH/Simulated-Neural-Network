import torch
from torchvision import datasets, transforms
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.learning import PostPre

import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict

# Set seed for reproducibility
torch.manual_seed(0)
# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
n_epochs = 2
update_interval = 1
numNeurons = 64
simTime = 350

# Dataset and encoding
#Manually encodes each image and passes it to the neural network
encoder = PoissonEncoder(time=simTime)
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
subset = 1000
mnist_data = torch.utils.data.Subset(mnist_data, range(subset))

# Initialize the SNN
network = DiehlAndCook2015(n_inpt=28*28, n_neurons=numNeurons)
network.to(device)

# Monitor setup for spikes
spike_monitor = Monitor(network.layers["Ae"], state_vars=["s"], time=simTime)
network.add_monitor(spike_monitor, name="AeSpikes")

print(f"\n[INFO] Starting training loop on {len(mnist_data)} samples!\n")
labels = []
predictions = []

# Firing tracker
firing_tracker = np.zeros((numNeurons, 10), dtype=int)

for i in range(n_epochs * len(mnist_data)):
    #Loops over the data when running with multiple epochs to not create an index error
    image, label = mnist_data[i % len(mnist_data)]
    image = image.view(-1)
    image = image * 255
    spike_input = encoder(image.cpu())
    spike_input = spike_input.to(device)

    if i % update_interval == 0:
        print(f"[INFO] Sample {i} / {n_epochs * len(mnist_data)}")
    network.run(inputs={"X": spike_input}, time=simTime)
    spike_record = spike_monitor.get("s")

    network.reset_state_variables()
    spike_monitor.reset_state_variables()

    spikes = spike_record.sum(0)
    neuron_id = spikes.argmax().item()
    firing_tracker[neuron_id, label] += 1

    labels.append(int(label))
    predictions.append(neuron_id)  # temporary; will decode later

neuron_label_map={}
# Assign each neuron a label based on firing history, with a threshold to avoid labelling neurons that are not functioning or barely firing
for i in range(numNeurons):
    if firing_tracker[i].sum() > 0:
        neuron_label_map[i] = firing_tracker[i].argmax()
final_predictions = [neuron_label_map.get(nid, -1) for nid in predictions]

# Evaluation
correct = sum([p == l for p, l in zip(final_predictions, labels)])
accuracy = (correct / len(labels) * 100) if labels else 0.0
print(f"\nFinal Accuracy: {accuracy:.2f}%")

# Visualization
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

plot_spikes({"Ae": spike_record})
plt.savefig(os.path.join(output_dir, "spike_plot.png"))

#Grab weight matrix from in to ou neurons
#Detach goes from pytorch tensor to numpy array and .t makes them separate
weights = network.connections["X", "Ae"].w.detach().cpu().numpy().T
fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    if i < numNeurons:
        ax.imshow(weights[i].reshape(28, 28), cmap="hot")
        ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "weights_patches.png"))

plot_weights(network.connections["X", "Ae"].w, cmap="hot")
plt.savefig(os.path.join(output_dir, "weights.png"))

print("\n[INFO] Mean connection weight:", network.connections["X", "Ae"].w.mean().item())
print("\n[INFO] Plots saved in outputs/ directory.")
