import pandas as pd

df = pd.read_csv("./data/raw_mbti-indo.csv")

# Get unique values in the subcategory column
unique_subcategories = df['subcategory'].unique()

# Print the unique subcategories
print("Unique Subcategories:")
for subcategory in unique_subcategories:
    print(f"- {subcategory}")
