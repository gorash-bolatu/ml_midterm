
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
    if (len(df[i].unique()) > 30):
        print("\t", i, ": " , len(df[i].unique()), sep="")
        categorical_features = categorical_features.drop([str(i)])
print(df[categorical_features].head())

print("Initializing OneHotEncoder...")
encoder = OneHotEncoder(sparse_output=False)
# Fit and transform the categorical features
print("Fitting data...")
encoder.fit(df[categorical_features])
print("Transforming data...")
encoded = encoder.transform(df[categorical_features])
print("Converting into DataFrame...")
one_hot_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_features))

print(one_hot_df.head())

print("Concatenating encoded columns with numerical...")
X = pd.concat([df, one_hot_df], axis=1).drop(categorical_cols, axis=1)
print("Features (x):")
print(X.head())

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
fig, axes = plt.subplots(3, 3, figsize=(12, 8.5))
fig.suptitle('Feature importances (top 10) with different RandomForestClassifier seeds', fontsize=18)

for i, seed in enumerate(seeds):

    print("Iteration", i)
    
    # Train RandomForest with current seed
    rf = RandomForestClassifier(
        bootstrap=True, max_samples=0.7,
        n_estimators=100, max_depth=10,
        n_jobs=-1, random_state=seed)
    print("\tTraining classifier...")
    rf.fit(X_train, y_train)

    print("\tCalculating results...")
    
    # Get and sort feature importances
    importances = rf.feature_importances_
    all_importances[i] = importances
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    sorted_importances = importances[sorted_idx]
    sorted_feature_names = [X_train.columns[j] for j in sorted_idx]
    
    # Select subplot (3x3 grid)
    ax = axes[i//3, i%3]
    
    # Plot horizontal bar chart
    print("\tLoading for plotting...")
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

# Comparing against test data
rf = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1, random_state=1)
print("Training classifier...")
rf.fit(X_test, y_test)
print("Calculating results...")
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
X = X[sorted_feature_names]
print(X.shape)
print(X.head())

# Saving as a processed CSV file
df_out = pd.concat([y, X], axis=1)
filename = 'PROCESSED.csv'
df_out.to_csv(filename, index=False)
print("Wrote", df_out.shape[0], "rows and", df_out.shape[1], "columns to", filename)