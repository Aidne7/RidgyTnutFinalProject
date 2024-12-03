import pickle
import numpy as np

# Specify the path to your pickle file
pickle_file_path = "output/u_reference_ens_201101070000_pvalues.pkl"

# Open and load the pickle file
with open(pickle_file_path, 'rb') as f:
    data = pickle.load(f)

# Analyze the structure of the data
print("Type of the data:", type(data))  # Check the data type (e.g., dict, list)
if isinstance(data, dict):
    print("Keys in the dictionary:", data.keys())  # Display keys if it's a dictionary

# Explore specific elements
for key, value in data.items():
    print(f"\nKey: {key}")
    print(f"Type of value: {type(value)}")
    if isinstance(value, (list, np.ndarray)):  # For lists or arrays
        print(f"Shape/Length of value: {len(value) if isinstance(value, list) else value.shape}")
    else:
        print(f"Value: {value}")
