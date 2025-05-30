import pickle, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier #replacement for logisitic regression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# Load
with open("outputs/X_train.pkl", "rb") as f: X_train = pickle.load(f)
with open("outputs/y_train.pkl", "rb") as f: y_train = pickle.load(f)
with open("outputs/X_test.pkl", "rb") as f: X_test = pickle.load(f)
with open("outputs/y_test.pkl", "rb") as f: y_test = pickle.load(f)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train).ravel()
y_test = np.array(y_test).ravel()

# Reshape to 2D if necessary
if X_train.ndim > 2:
    X_train = X_train.reshape(X_train.shape[0], -1)
if X_test.ndim > 2:
    X_test = X_test.reshape(X_test.shape[0], -1)
    
clf = RandomForestClassifier(n_estimators=100, random_state=0)

#Normalize features for classifier
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Old method
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

clf = SVC(kernel='rbf', C=1, gamma='scale')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"Supervised test accuracy: {accuracy * 100:.2f}%")

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
unique_labels = np.unique(np.concatenate((y_test, y_pred)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)

# Plot and save
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap="Reds")
plt.title("Confusion Matrix")
plt.savefig("outputs/supervised_confusion_matrix.png")
plt.show()
