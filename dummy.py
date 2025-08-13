import pandas as pd

# Sample dataset
data = {
    'Location': ['Delhi', 'Mumbai', 'Chennai', 'Delhi', 'Chennai'],
    'Size (sqft)': [1200, 1000, 1100, 1500, 1050],
    'Price (L)': [50, 55, 52, 65, 53]
}

df = pd.DataFrame(data)

# One-hot encode the 'Location' column
df_encoded = pd.get_dummies(df, columns=['Location'], drop_first=True)
df_encoded=df_encoded.astype(int)

print("One-Hot Encoded DataFrame:")
print(df_encoded)
