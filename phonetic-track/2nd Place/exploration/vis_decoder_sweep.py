import plotly.express as px
import wandb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# 1. Initialize the W&B API
api = wandb.Api()

# 2. Define the exact artifact path based on your URL
# artifact_path = "speech-phonetic-track/decoding_sweep/run-ocfw3qpz-decoder_sweepresults_table-ZF_5nA:v0"
artifact_path = "speech-phonetic-track/decode_sweep/run-im1cmdq1-decoder_sweepresults_table-vYkedw:v0"


try:
    print(f"Fetching artifact: {artifact_path}")
    # Download the artifact
    artifact = api.artifact(artifact_path)
    
    # Get the specific table from the artifact
    # Based on the URL ending in files/decoder_sweep/results_table.table.json
    table = artifact.get("decoder_sweep/results_table")
    
    # 3. Convert the W&B Table to a Pandas DataFrame
    df = pd.DataFrame(data=table.data, columns=table.columns)
    
    print(f"Successfully loaded {len(df)} rows!")
    print("Columns available:", df.columns.tolist())
    print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")
    # Fallback: Sometimes artifact.get() requires the exact file path. 
    # If the above fails, it will trigger this block to download the raw JSON.
    print("Attempting to download raw file...")
    artifact_dir = artifact.download()
    import json
    import os
    with open(os.path.join(artifact_dir, "decoder_sweep/results_table.table.json")) as f:
        raw_data = json.load(f)
        df = pd.DataFrame(data=raw_data["data"], columns=raw_data["columns"])
    print(f"Successfully loaded {len(df)} rows from raw JSON!")

# --- PLOTTING ---

# 1. Use the actual parameters AND include 'per' so it draws as an axis
parameters = [
    'temperature', 
    'blank_penalty', 
    'alpha', 
    'beta', 
    'repeat_penalty',
    'per'  # <-- Added PER so it shows up as a vertical line
] 

target_metric = 'per' 

try:
    # 2. Create the interactive parallel coordinates plot with custom colors
    fig = px.parallel_coordinates(
        df, 
        color=target_metric,          
        dimensions=parameters,        
        color_continuous_scale=["yellow", "purple"], # Yellow = lowest PER, Purple = highest PER
        title="Parallel Coordinates of Decoding Parameters vs PER"
    )

    # 3. Flip the PER axis so the lowest (best) value is at the top
    for dim in fig.data[0].dimensions:
        if dim['label'] == 'per':
            # By reversing the range to [max, min], the lowest value goes to the top of the axis
            dim['range'] = [df['per'].max(), df['per'].min()]

    fig.show()

except KeyError as e:
    print(f"\n[!] Plotting Error: Missing column: {e}")


print("Calculating parameter importance and correlations...")

# 1. Define the parameters (X) and the target metric (y)
features = [
    'temperature', 
    'blank_penalty', 
    'alpha', 
    'beta', 
    'repeat_penalty'
]
X = df[features].copy()
y = df['per'].copy()

# Handle any missing values just in case
X = X.fillna(X.median())
y = y.fillna(y.median())

# 2. Train the Random Forest for Importance
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Extract and sort the feature importances
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=True) # Sort by importance

# 3. Calculate correlations AND align them with importance
raw_correlations = df[features].corrwith(df['per'])

# Force the correlations to use the exact same order as the sorted importances
ordered_features = importances.index
correlations = raw_correlations.reindex(ordered_features)

# Generate colors based on the newly ordered correlations
corr_colors = ['royalblue' if val < 0 else 'crimson' for val in correlations]

# ==========================================
# 4. PLOT BOTH CHARTS SIDE-BY-SIDE
# ==========================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Left Plot: Parameter Importance ---
importances.plot(kind='barh', color='mediumpurple', ax=ax1)
ax1.set_title("Parameter Importance on PER (Random Forest)", fontsize=14)
ax1.set_xlabel("Importance Score (Higher = More Impact)")
ax1.set_ylabel("Hyperparameter")

# --- Right Plot: Correlation ---
correlations.plot(kind='barh', color=corr_colors, ax=ax2)
ax2.axvline(x=0, color='black', linewidth=1, linestyle='--') # 0-reference line
ax2.set_title("Linear Correlation with PER", fontsize=14)
ax2.set_xlabel("Correlation Coefficient\n(Negative / Blue = Improves PER)")

# Remove the y-axis labels on the right plot since they exactly match the left plot
ax2.set_yticks([]) 
ax2.set_ylabel("")

plt.tight_layout()
plt.show()