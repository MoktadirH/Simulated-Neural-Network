import torch
import pickle
from torchvision import datasets, transforms
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_weights
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.learning import PostPre
from collections import Counter, defaultdict


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
numNeurons = 100
simTime = 100

# Dataset and encoding
#Manually encodes each image and passes it to the neural network
encoder = PoissonEncoder(time=simTime)
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
subset = 400
subTest= 400
mnist_data = torch.utils.data.Subset(mnist_data, range(subset))
mnist_test=torch.utils.data.Subset(mnist_test, range(subTest))

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
#training
X_train = []
y_train = []
X_test = []
y_test = []

for i in range(n_epochs * len(mnist_data)):
    #Loops over the data when running with multiple epochs to not create an index error
    image, label = mnist_data[i % len(mnist_data)]
    imagetest,labeltest=mnist_test[i % len(mnist_test)]

    image = image.view(-1)
    image = image * 255
    imagetest = imagetest.view(-1)
    imagetest = imagetest * 255

    spike_input = encoder(image.cpu())
    spike_inputT = encoder(imagetest.cpu())
    spike_input = spike_input.to(device)
    spike_inputT = spike_inputT.to(device)

    if i % update_interval == 0:
        print(f"[INFO] Sample {i} / {n_epochs * len(mnist_data)}")
    network.run(inputs={"X": spike_input}, time=simTime)
    spike_record = spike_monitor.get("s")
    network.reset_state_variables()
    spike_monitor.reset_state_variables()

    network.run(inputs={"X": spike_inputT}, time=simTime)
    spike_recordT = spike_monitor.get("s")
    network.reset_state_variables()
    spike_monitor.reset_state_variables()

    spike_counts = spike_record.sum(0).cpu().numpy()
    X_train.append(spike_counts)
    y_train.append(int(label))

    spike_countsT = spike_recordT.sum(0).cpu().numpy()
    X_test.append(spike_countsT)
    y_test.append(int(label))

    spikes = spike_record.sum(0)
    neuron_id = spikes.argmax().item()
    firing_tracker[neuron_id, label] += 1

    labels.append(int(label))
    predictions.append(neuron_id)


neuron_label_map={}
# Assign each neuron a label based on firing history, with a threshold to avoid labelling neurons that are not functioning or barely firing
#Old system with no confidence threshold to not assign neurons that are not specialized fully
#for i in range(numNeurons):
 #   if firing_tracker[i].sum() > 0:
   #     neuron_label_map[i] = firing_tracker[i].argmax()

for i in range(numNeurons):
    total_spikes = firing_tracker[i].sum()
    if total_spikes > 0:
        label = firing_tracker[i].argmax()
        confidence = firing_tracker[i, label] / total_spikes
        if confidence > 0.15:
            neuron_label_map[i] = label

#Old prediction method
#unlabeled neurons still get compared
final_predictions = [neuron_label_map.get(nid, -1) for nid in predictions]

#final_predictions = []
#true_labels = []

#for nid, label in zip(predictions, labels):
#    if nid in neuron_label_map:
#        final_predictions.append(neuron_label_map[nid])
#        true_labels.append(label)

# Evaluation
correct = sum([p == l for p, l in zip(final_predictions, labels)])
accuracy = (correct / len(labels) * 100) if labels else 0.0
print(f"\nFinal Accuracy: {accuracy:.2f}%")
print(f"Number of labeled neurons: {len(neuron_label_map)}")

# digit accuracy to see what numbers are correct with a high accuracy
correct_per_class = defaultdict(int)
total_per_class = Counter(labels)

for p, l in zip(final_predictions, labels):
    if p == l:
        correct_per_class[l] += 1

print("\nPer-digit accuracy:")
for digit in range(10):
    total = total_per_class[digit]
    correct = correct_per_class[digit]
    acc = (correct / total * 100) if total > 0 else 0.0
    print(f"Digit {digit}: {acc:.2f}%")

# Visualization
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)


#How often each neuron fires off
neuron_activity = [firing_tracker[i].sum() for i in range(numNeurons)]
plt.figure(figsize=(10, 4))
plt.bar(range(numNeurons), neuron_activity)
plt.title("Neuron Firing Frequency")
plt.xlabel("Neuron ID")
plt.ylabel("Total spikes across training")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "neuron_activity.png"))

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


with open("outputs/X_train.pkl", "wb") as f: pickle.dump(X_train, f)
with open("outputs/y_train.pkl", "wb") as f: pickle.dump(y_train, f)
with open("outputs/X_test.pkl", "wb") as f: pickle.dump(X_test, f)
with open("outputs/y_test.pkl", "wb") as f: pickle.dump(y_test, f)
print("Finished training!")



