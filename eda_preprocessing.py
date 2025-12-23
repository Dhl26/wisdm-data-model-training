import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# File path
file_path = r'd:\NFSU\DS & Maths lab\WISDM_ar_v1.1_raw.txt'

# Column names based on readme/about files
columns = ['user_id', 'activity', 'timestamp', 'x_axis', 'y_axis', 'z_axis']

print("Loading data...")
# Removing file that has a trailing ;
try:
    df = pd.read_csv(file_path, header=None, names=columns, on_bad_lines='skip')
except Exception as e:
    print(f"Error reading csv: {e}")
    
    
print("Data loaded. Head:")
print(df.head())

print("\nCleaning data...")
# Remove trailing ';' from z_axis and convert to float

if df['z_axis'].dtype == 'O':
    df['z_axis'] = df['z_axis'].astype(str).str.replace(';', '', regex=False)
    # Convert to numeric, coercing errors to NaN
    df['z_axis'] = pd.to_numeric(df['z_axis'], errors='coerce')

print(f"Shape before removing duplicates: {df.shape}")
df.drop_duplicates(inplace=True)
print(f"Shape after removing duplicates: {df.shape}")

print(f"Missing values before filling:\n{df.isnull().sum()}")
df.ffill(inplace=True) 
df.bfill(inplace=True) 
print(f"Missing values after filling:\n{df.isnull().sum()}")

print("\nData Info:")
print(df.info())

print("\nClass Distribution:")
print(df['activity'].value_counts())

plt.figure(figsize=(10, 6))
df['activity'].value_counts().plot(kind='bar')
plt.title('Activity Distribution')
plt.xlabel('Activity')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(r'd:\NFSU\DS & Maths lab\activity_distribution.png')
print("Saved activity_distribution.png")

user_id = df['user_id'].unique()[0]
activity = 'Jogging'
subset = df[(df['user_id'] == user_id) & (df['activity'] == activity)].iloc[:200] 

plt.figure(figsize=(15, 5))
plt.plot(subset['x_axis'], label='x')
plt.plot(subset['y_axis'], label='y')
plt.plot(subset['z_axis'], label='z')
plt.title(f'Time Series for {activity} (User {user_id})')
plt.legend()
plt.savefig(r'd:\NFSU\DS & Maths lab\time_series_example.png')
print("Saved time_series_example.png")

window_size = 200
step_size = 100

segments = []
labels = []

print("\nStarting segmentation (this might take a while)...")
for user in df['user_id'].unique():
    user_df = df[df['user_id'] == user]
    for activity in user_df['activity'].unique():
        activity_df = user_df[user_df['activity'] == activity]
        
        for i in range(0, len(activity_df) - window_size, step_size):
            xs = activity_df['x_axis'].values[i : i + window_size]
            ys = activity_df['y_axis'].values[i : i + window_size]
            zs = activity_df['z_axis'].values[i : i + window_size]
            
         
            label = activity_df['activity'].values[i]
            
            segments.append([xs, ys, zs])
            labels.append(label)

# Reshape segments
reshaped_segments = np.asarray(segments, dtype=np.float32).transpose(0, 2, 1)
labels = np.asarray(labels)

print(f"Segmentation complete. Shape: {reshaped_segments.shape}")


print("Extracting features...")
features = []

for i in range(reshaped_segments.shape[0]):
    segment = reshaped_segments[i]
    
    row_features = []
    for axis in range(3):
        axis_data = segment[axis]
        row_features.append(np.mean(axis_data))
        row_features.append(np.std(axis_data))
        row_features.append(np.max(axis_data))
        row_features.append(np.min(axis_data))
        row_features.append(np.mean(np.absolute(axis_data - np.mean(axis_data))))
    
    features.append(row_features)

features = np.array(features)
feature_names = []
for axis in ['x', 'y', 'z']:
    for stat in ['mean', 'std', 'max', 'min', 'mad']:
        feature_names.append(f"{axis}_{stat}")

print(f"Features extracted. Shape: {features.shape}")

processed_df = pd.DataFrame(features, columns=feature_names)
processed_df['activity'] = labels

output_path = r'd:\NFSU\DS & Maths lab\processed_wisdm_data.csv'
processed_df.to_csv(output_path, index=False)
print(f"Processed data saved to {output_path}")
