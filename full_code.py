import pandas as pd

# Load the dataset
df = pd.read_csv('[RAW] Mid-TermProject_16-09-27_Data_5percent.csv')
print(df.info())


print(df.head())


print("Original shape:", df.shape)
df = df.drop(['eth.addr', 'eth.addr_oui', 'eth.addr_oui_resolved', 'eth.addr_resolved',
              'eth.dst', 'eth.dst_oui_resolved', 'eth.dst_resolved', 'eth.src', 'eth.src_ig',
              'eth.src_lg', 'eth.src_oui', 'eth.src_oui_resolved', 'eth.src_resolved', 'ip.addr',
              'ip.dst', 'ip.dst_host', 'ip.host', 'ip.src', 'ip.src_host', 'arp.dst_hw_mac',
              'arp.dst_proto_ipv4', 'arp.src_hw_mac', 'arp.src_proto_ipv4'], axis=1)
print("Shape after removing unreliable data:", df.shape)


# Drop duplicate rows
df = df.drop_duplicates(keep="first")
print("Shape after removing duplicate rows:", df.shape)


print("Empty values in the dataset:")
print(df.isna().sum().sort_values())
print("\nTotal:", df.isna().sum().sum())


# Impute missing values
from sklearn.impute import SimpleImputer

# For numerical columns with only 1 type of value (and N/A), replace N/A with zeroes
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for i in df[numerical_cols]:
    if (len(df[i].unique()) < 2):
        df[numerical_cols] = df[numerical_cols].fillna(0)

# For other numerical columns, simply replace empty values with the mean
num_imputer = SimpleImputer(strategy='mean')
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# For string columns with only 1 type of value (and N/A), replace N/A with the string "N/A"
object_cols = df.select_dtypes(include=['object']).columns
for a in df[object_cols]:
    if (len(df[a].unique()) < 2):
        df[a] = df[a].fillna("N/A")
# For boolean columns with only 1 type of value (and N/A), replace N/A with the inverse of that boolean
bool_cols = df.select_dtypes(include=['bool']).columns
for b in df[bool_cols]:
    if (len(df[b].unique()) < 2):
        df[b] = df[b].fillna(not bool(df[b].mode()[0]))
categorical_cols = df.select_dtypes(include=['object', 'bool']).columns

# For other categorical columns, simply replace empty values with the mode
cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("Empty values after imputing:")
print(df.isna().sum())
print("\nTotal:", df.isna().sum().sum())


# Feature encoding for categorical columns
from sklearn.preprocessing import OneHotEncoder

categorical_features = categorical_cols.drop(['DeviceName'])

print("Dropping categorical features with too many different values:")
for i in df[categorical_features]:
    if (len(df[i].unique()) > 50):
        print("\t", i, ": " , len(df[i].unique()), sep="")
        categorical_features = categorical_features.drop([str(i)])


print("Initializing OneHotEncoder...")
encoder = OneHotEncoder(sparse_output=False)
# Fit and transform the categorical features
print("Fitting data...")
encoder.fit(df[categorical_features])
print("Transforming data...")
encoded = encoder.transform(df[categorical_features])
print("Converting into DataFrame...")
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))
print("Encoded columns:", list(df_encoded.columns))


df = df.reset_index(drop=True)
df_encoded = df_encoded.reset_index(drop=True)

print("Concatenating encoded columns with numerical...")
X = pd.concat([df.drop(columns=categorical_cols), df_encoded], axis=1)
print("Features (X):", list(X.columns))


y = df['DeviceName']
print('Label (y):', y.name)
print(f"Classes ({len(y.unique())}):\n", y.unique())


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np

print("Train/test split...", end=" ")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print("Done")


seeds = [1, 2, 7, 54, 241, 675, 1271, 7539, 21418] # Seeds for reproducibility
top_n = 10 # Use 10 most important features
all_importances = np.zeros((len(seeds), len(X_train.columns))) # Store importances with every seed to compute mean later

print("Preparing plot...")

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Feature importances (top 10) with different RandomForestClassifier seeds', fontsize=18)

for i, seed in enumerate(seeds):

    print(f"Iteration {i+1}/{len(seeds)}...")
    
    # Train RandomForest with current seed
    rf = RandomForestClassifier(
        bootstrap=True, max_samples=0.5,
        n_estimators=100, max_depth=10,
        n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_train)

    # Get and sort feature importances
    importances = rf.feature_importances_
    all_importances[i] = importances
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[sorted_idx]
    sorted_feature_names = [X_train.columns[j] for j in sorted_idx]
    
    # Select subplot (3x3 grid)
    ax = axes[i//3, i%3]
    
    # Plot horizontal bar chart
    ax.barh(range(len(sorted_feature_names)), sorted_importances)
    ax.set_yticks(range(len(sorted_feature_names)))
    ax.set_yticklabels(sorted_feature_names)
    ax.invert_yaxis() # Most important at top
    ax.set_xlabel('Importance Score')
    ax.set_title(f'random_state = {seed}')

print("Plotting...")
plt.tight_layout()
plt.show()


# Compute mean importances across seeds
mean_importances = np.mean(all_importances, axis=0)

# Sort features by mean importance (descending)
sorted_idx = np.argsort(mean_importances)[::-1][:top_n]
sorted_importances = mean_importances[sorted_idx]
sorted_feature_names = [X.columns[j] for j in sorted_idx]

# Plot the averaged importances
plt.figure(figsize=(7, 4))
plt.barh(range(len(sorted_feature_names)), sorted_importances)
plt.yticks(range(len(sorted_feature_names)), sorted_feature_names)
plt.gca().invert_yaxis()
plt.xlabel('Average Importance Score')
plt.title('Average top feature importances across 9 seeds')
plt.tight_layout()
plt.show()


print("\nComparing results against test data:")

rf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1, random_state=1)
print("\tTraining classifier...")
rf.fit(X_test, y_test)

print("\tCalculating results...")
test_importances = rf.feature_importances_
test_sorted_idx = np.argsort(test_importances)[::-1][:top_n]
test_sorted_importances = test_importances[test_sorted_idx]
test_sorted_feature_names = [X_test.columns[j] for j in test_sorted_idx]

print("Plotting...")
plt.figure(figsize=(7, 4))
plt.barh(range(len(test_sorted_feature_names)), test_sorted_importances)
plt.yticks(range(len(test_sorted_feature_names)), test_sorted_feature_names)
plt.gca().invert_yaxis()
plt.xlabel('Average Importance Score')
plt.title('Average top feature test_importances (test data)')
plt.tight_layout()
plt.show()


# Feature selection
print("Shape of X:", X.shape)
X = X[sorted_feature_names]
print("Shape of X after feature selection:", X.shape)


# Feature scaling for numerical columns
from sklearn.preprocessing import StandardScaler

print("Scaling...")
scaler = StandardScaler()
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("Scaled dataset:")
print(X.head())


# Combining into the processed dataset
df = pd.concat([y, X], axis=1)
print("Final processed dataset:")
print(df.info())
print(df.head())


print("\nKNN classification")

import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("Execution start...")
start_time = time.time()

# Values to save best hyperparameters
best_accuracy = 0
best_params = None

# Saving values to later visualise against accuracy
data = list()
print("Data processing start...")
data_processing_start = time.time()

print("Test/train split...")

X = df.drop(['DeviceName'], axis=1)
y = df['DeviceName']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=80)

print("Hyperparameter tuning...")

iteration = 0

for k in [1, 3, 7]: # Arbitrary k values
    for weights in ['uniform', 'distance']:
        for p_val in [1, 2]: # Manhattan/Minkowski distance

            iteration += 1

            print(f"\tIteration {iteration}/12: k={k}, weights={weights}, p={p_val}")

            # Initialize and train the KNN classifier with current hyperparameters
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p_val, n_jobs=-1)
            
            # Train the model
            print("\t\tModel training...", end=" ")
            training_start = time.time()
            knn.fit(X_train, y_train)
            training_time = time.time() - training_start
            print(f"{training_time:.4f}sec")

            # Evaluate the model
            print("\t\tModel evaluation...", end=" ")
            evaluation_start = time.time()
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # Check if the current model is the best
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = (k, weights, p_val)
            evaluation_time = time.time() - evaluation_start
            print(f"{evaluation_time:.4f}sec")
            
            print(f"\t\tAccuracy: {accuracy:.8f}")
            if p_val == 1:
                p_str = "Manhattan"
            else:
                p_str = "Minkowski"
            data.append({'k': k, 'weights': weights, 'p': p_str, 'train_time': training_time, 'eval_time': evaluation_time, 'accuracy': accuracy})

data_processing_time = time.time() - data_processing_start
print("Data processing finished")

# Training phase
print("Final model training...", end=" ")
training_start = time.time()
best_knn = KNeighborsClassifier(n_neighbors=best_params[0], weights=best_params[1], p=best_params[2])
best_knn.fit(X_train, y_train)
training_time = time.time() - training_start
print(f"{training_time:.4f}sec")

# Test phase
print("Final model testing & evaluation...", end=" ")
evaluation_start = time.time()
y_pred_final = best_knn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_final)
evaluation_time = time.time() - evaluation_start
print(f"{evaluation_time:.4f}sec")

print("Accuracy:", accuracy)

total_execution_time = time.time() - start_time
print("Execution finished")

# Report results
total_execution_time, data_processing_time, training_time, evaluation_time, best_params, best_accuracy, test_accuracy
print(f"Total execution time: {total_execution_time:.4f}sec")
print(f"Data processing time: {data_processing_time:.4f}sec)")
print(f"Final model training time: {training_time:.4f}sec")
print(f"Final model evaluation time: {evaluation_time:.4f}sec")
print("Model parameters:", best_params)
print("Accuracy score:", test_accuracy)


data_df = pd.DataFrame(data)
print(data_df.head())


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# Convert categorical variables to numerical values
weights_map = {'uniform': 0, 'distance': 1}
p_map = {'Manhattan': 0, 'Minkowski': 1}

# Prepare data for plotting
k_values = np.array(data_df['k'])
weights_values = np.array([weights_map[w] for w in data_df['weights']])
p_values = np.array([p_map[p] for p in data_df['p']])
accuracy_values = np.array(data_df['accuracy'])

# Normalize Accuracy for color mapping
norm = Normalize(vmin=accuracy_values.min(), vmax=accuracy_values.max())

# Use a custom colormap from red to green
cmap = cm.RdYlGn # Reversed RdYlGn colormap (Red to Green)

# Create a 3D scatter plot
fig = plt.figure(figsize=(23, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot with colors based on accuracy, mapped through the colormap
sc = ax.scatter(k_values, weights_values, p_values, c=accuracy_values, cmap=cmap, norm=norm, s=100)

# Set labels
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('Weights', fontsize=14)
ax.set_zlabel('p', fontsize=14)

# Custom ticks for 'weights' axis
ax.set_yticks([0, 1])
ax.set_yticklabels(['uniform', 'distance'])

# Custom ticks for 'p' axis
ax.set_zticks([0, 1])
ax.set_zticklabels(['Manhattan', 'Minkowski'])

# Add a colorbar to reflect the accuracy
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Accuracy')

plt.title('3D Scatter Plot of k, Weights, p vs Accuracy')
plt.show()


import seaborn as sns
from sklearn.metrics import confusion_matrix

# Assuming y_test is the true labels and y_pred_final is the model predictions
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_final)

# Create a custom colormap with bright red for non-TP values
cmap = sns.color_palette("Blues", as_cmap=True)

# Create a figure
plt.figure(figsize=(8, 6))

# Plot the heatmap with a custom colormap
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=False,
            xticklabels=best_knn.classes_, yticklabels=best_knn.classes_)

# Loop over the matrix to manually set non-TP cells to bright red
for i in range(len(cm)):
    for j in range(len(cm[i])):
        if i != j and cm[i, j] != 0: # Non-TP values
            plt.gca().add_patch(plt.Rectangle((j, i), 1, 1, color='red', alpha=0.5))

plt.xlabel('Predicted Labels', fontsize=16)
plt.ylabel('True Labels', fontsize=16)
plt.title('Confusion Matrix\n', fontsize=20, fontweight='bold')
plt.show()
