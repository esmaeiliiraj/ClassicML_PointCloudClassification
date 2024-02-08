# Libraries such as trimesh, tensorflow, numpy, and other scientific libraries need to be installed.
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt

# Data path
data_dir = tf.keras.utils.get_file("modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
data_dir = os.path.join(os.path.dirname(data_dir), "ModelNet10")

# Loading an example in a mesh form
mesh = trimesh.load(os.path.join(data_dir, "table//train//table_0001.off"))
mesh.show()

# Sampling points from the mesh example
points = mesh.sample(2048)

# Plotting the sampled points in 3D according to their coordinates, i.e., x, y, z
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
plt.savefig('point_cloud_sample.png', dpi=300)
plt.show();

# Function for reading data from storage and constructing the train and test sets
def parse_dataset(num_points=2048):
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []
    data = []
    labels = []
    class_map = {}
    folders = glob.glob(os.path.join(data_dir, "[!README]*"))

    for i, folder in enumerate(folders):
        print(f"Processing class: {os.path.basename(folder)}")
        # Store class names and labels so we could use later
        class_map[i] = folder.split("/")[-1]

        # Gathering files
        train_files = glob.glob(os.path.join(folder, "train//*"))
        test_files = glob.glob(os.path.join(folder, "test//*"))

        for f in train_files:
            data.append(trimesh.load(f).sample(num_points))
            labels.append(i)

        for f in test_files:
            data.append(trimesh.load(f).sample(num_points))
            labels.append(i)
    
    return (
        np.array(data),
        np.array(labels),
        class_map
    )


# Reading data
num_points = 1024     # Number of points for each sample

data, labels, class_map = parse_dataset(num_points)


from sklearn.model_selection import train_test_split

# Flatten the 3D point cloud data
data_flat = data.reshape((data.shape[0], -1))
test_size = 0.20

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(data_flat, labels, test_size=test_size)

# Support Vector Machine (SVM): Model 1
from sklearn.svm import SVC

svm_model = SVC(C=2.0, kernel='rbf')
svm_model.fit(X_train, y_train)
svm_accuracy = svm_model.score(X_test, y_test)

print("Support Vector Machine: Model 1")
print("-"*50)
print("Model Parameters:")
print(f"  . C     : {svm_model.C}")
print(f"  . kernel: {svm_model.kernel}")
print("-"*50)
print(f"Accuracy: {svm_accuracy:.3f}")


# Support Vector Machine (SVM): Model 2, trying different hyperparameters

svm_model2 = SVC(C=4, kernel='linear', coef0=0.1)
svm_model2.fit(X_train, y_train)
svm2_accuracy = svm_model2.score(X_test, y_test)

print("Support Vector Machine: Model 2")
print("-"*50)
print("Model Parameters:")
print(f"  . C     : {svm_model2.C}")
print(f"  . kernel: {svm_model2.kernel}")
print("-"*50)
print(f"Accuracy: {svm2_accuracy:.3f}")


# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

cv = ShuffleSplit(n_splits=10)

rf_model = RandomForestClassifier(n_estimators=70, max_depth=10)
cv_results_rf = cross_validate(rf_model, X_train, y_train, cv=cv, scoring='accuracy',
                              return_train_score=True)

rf_model.fit(X_train, y_train)
rf_accuracy = rf_model.score(X_test, y_test)

print("Random Forest")
print(f"Test accuracy: {rf_accuracy:.3f}")
print("Validation accuracy via cross-validation: "
      f"{cv_results_rf['test_score'].mean():.3f} +/- "
      f"{cv_results_rf['test_score'].std():.3f}")
print("Train accuracy via cross-validation: "
      f"{cv_results_rf['train_score'].mean():.3f} +/- "
      f"{cv_results_rf['train_score'].std():.3f}")
print("Average fit time: "
      f"{cv_results_rf['fit_time'].mean():.3f} seconds")
print("Average score time: "
      f"{cv_results_rf['score_time'].mean():.3f} seconds")


from sklearn.model_selection import validation_curve

n_estimators_range = range(1, 50, 2)
rf_train_scores, rf_test_scores = validation_curve(RandomForestClassifier(),
                                                   X_train, y_train,
                                                   param_name='n_estimators',
                                                   param_range=n_estimators_range,
                                                   cv=cv,
                                                   scoring='accuracy')


plt.plot(n_estimators_range, rf_train_scores.mean(axis=1),label='Train Accuracy')
plt.plot(n_estimators_range, rf_test_scores.mean(axis=1),label='Test Accuracy')

plt.legend()
plt.xlabel("Number of Estimators")
plt.ylabel("Classification Accuracy")
plt.title("Validation curve")
plt.savefig('validation_curve_n_estimators.png', dpi=300)
plt.show();


max_depth_range = range(5, 30, 5)

rf_train_scores, rf_test_scores = validation_curve(RandomForestClassifier(n_estimators=30),
                                                   data_flat, labels,
                                                   param_name='max_depth',
                                                   param_range=max_depth_range,
                                                   cv=cv,
                                                   scoring='accuracy')

plt.plot(max_depth_range, rf_train_scores.mean(axis=1),label='Train Accuracy')
plt.plot(max_depth_range, rf_test_scores.mean(axis=1),label='Test Accuracy')

plt.legend()
plt.xlabel("Max Depth")
plt.ylabel("Classification Accuracy")
plt.title("Validation curve")
plt.savefig('validation_curve_max_depth.png', dpi=300)
plt.show();


from sklearn.model_selection import learning_curve

sample_sizes = np.linspace(0.1, 1, 10)

train_sizes_rf, train_scores_rf, test_scores_rf = learning_curve(
    RandomForestClassifier(n_estimators=30, max_depth=20),
    data_flat, labels,
    train_sizes=sample_sizes,
    cv=cv,
    scoring='accuracy')


#plt.plot(sample_sizes, train_scores_rf.mean(axis=1),label='Train Score')
plt.plot(sample_sizes, test_scores_rf.mean(axis=1),label='Test Score')
plt.legend()
plt.xlabel('Train Size')
plt.ylabel('Classification Accuracy')
plt.title('Learning Curve for Random Forest (n_estimators=30, max_depth=20)')
plt.savefig('learning_curve.png', dpi=300)
plt.show();


# Building the random forest model with the optimal parameters and train size

random_forest_model = RandomForestClassifier(n_estimators=30, max_depth=20)
random_forest_model.fit(X_train, y_train)
rf_accuracy = random_forest_model.score(X_test, y_test)

print("Random Forest Model")
print("-"*50)
print("Model Parameters:")
print(f"  . n_estimators: {random_forest_model.n_estimators}")
print(f"  . max_depth   : {random_forest_model.max_depth}")
print(f"  . train size  : {1-test_size}")
print("-"*50)
print(f"Accuracy: {rf_accuracy:.3f}")


from sklearn.model_selection import GridSearchCV

n_estimators_range = range(30, 50, 2)
max_depth_range = range(20, 30, 5)

# Using GridSearchCV to obtain the optimal hyper parameters.
rf_grid = GridSearchCV(rf_model, param_grid={'n_estimators': n_estimators_range,
                                             'max_depth': max_depth_range})

# Getting the optimal hyper parameters.
rf_grid.fit(X_train, y_train);
rf_grid.best_params_

random_forest_opt = RandomForestClassifier(n_estimators=48, max_depth=20)
random_forest_opt.fit(X_train, y_train)
rf_opt_accuracy = random_forest_opt.score(X_test, y_test)

print("Random Forest model with optimal parameters")
print("-"*50)
print("Model Parameters:")
print(f"  . n_estimators: {random_forest_opt.n_estimators}")
print(f"  . max_depth   : {random_forest_opt.max_depth}")
print(f"  . train size  : {1-test_size}")
print("-"*50)
print(f"Accuracy: {rf_opt_accuracy:.3f}")


X_train
from sklearn.preprocessing import StandardScaler

std_scaler = StandardScaler()
X_train_tr = std_scaler.fit_transform(X_train)
X_test_tr = std_scaler.fit_transform(X_test)


# Desicion Tree
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(max_depth=25, min_samples_leaf=10)
decision_tree.fit(X_train, y_train)
dt_accuracy = decision_tree.score(X_test, y_test)

print("Decision Tree Model")
print("-"*50)
print("Model Parameters:")
print(f"  . max_depth       : {decision_tree.max_depth}")
print(f"  . min_samples_leaf: {decision_tree.min_samples_leaf}")
print("-"*50)
print(f"Accuracy: {dt_accuracy:.3f}")

# Logistic Regression
from sklearn.linear_model import LogisticRegression
#solver : {'lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'},  
lr_model = LogisticRegression(max_iter=100, solver='sag')
lr_model.fit(X_train, y_train)
lr_accuracy = lr_model.score(X_test, y_test)

print("Logistic Regression Model")
print("-"*50)
print("Model Parameters:")
print(f"  . max_iter    : {lr_model.max_iter}")
print(f"  . solver      : {lr_model.solver}")
print("-"*50)
print(f"Accuracy: {lr_accuracy:.3f}")

# K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
knn_accuracy = knn_model.score(X_test, y_test)

print("K-Nearest Neighbors (KNN) Model")
print("-"*50)
print("Model Parameters:")
print(f"  . n_neighbors : {knn_model.n_neighbors}")
print("-"*50)
print(f"Accuracy: {knn_accuracy:.3f}")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Evaluate the classifiers
classifiers = {
    'Support Vector Machine (SVM) model 1': svm_model,
    'Support Vector Machine (SVM) model 2': svm_model2,
    'Decision Tree': decision_tree,
    'Random Forest': random_forest_model,
    'Logistic Regression': lr_model,
    'K-Nearest Neighbors (KNN)': knn_model,
}


results = {}
for name, model in classifiers.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    results[name] = {'Accuracy': accuracy, 'Classification Report': report, 'Confusion Matrix': confusion_mat}

# Print the results
for name, result in results.items():
    print(f"\n{name} Results:")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print("Classification Report:")
    print(result["Classification Report"])
    print("Confusion Matrix:")
    print(result["Confusion Matrix"])


