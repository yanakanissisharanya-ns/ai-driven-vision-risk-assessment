import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of rows
n_rows = 100

# Generate synthetic data
data = pd.DataFrame({
    "Age": np.random.randint(18, 50, size=n_rows),
    "Gender": np.random.choice(["Male", "Female"], size=n_rows),
    "Daily_Screen_Time": np.random.randint(2, 12, size=n_rows),
    "Mobile_Usage_Hours": np.random.randint(1, 8, size=n_rows),
    "Laptop_Usage_Hours": np.random.randint(1, 6, size=n_rows),
    "Night_Screen_Usage": np.random.choice(["Yes", "No"], size=n_rows),
    "Break_Frequency": np.random.randint(10, 60, size=n_rows),
    "Blue_Light_Filter": np.random.choice(["Yes", "No"], size=n_rows),
    "Screen_Distance": np.random.randint(30, 70, size=n_rows),
    "Sleep_Hours": np.random.randint(4, 9, size=n_rows),
    "Outdoor_Activity_Hours": np.random.randint(0, 4, size=n_rows),
    "Eye_Strain": np.random.choice(["Yes", "No"], size=n_rows),
    "Headache_Frequency": np.random.randint(0, 7, size=n_rows),
    "Existing_Eye_Power": np.round(np.random.uniform(0.0, 2.0, size=n_rows), 2),
})

# Create Vision_Risk based on simple logic
conditions = [
    (data["Daily_Screen_Time"] > 8) | (data["Eye_Strain"]=="Yes") | (data["Sleep_Hours"]<5),
    (data["Daily_Screen_Time"].between(5,8)) | (data["Sleep_Hours"].between(5,7)),
    (data["Daily_Screen_Time"] < 5) & (data["Eye_Strain"]=="No") & (data["Sleep_Hours"]>=7)
]
choices = ["High", "Medium", "Low"]
data["Vision_Risk"] = np.select(conditions, choices, default="Medium")

# Save to CSV
data.to_csv("vision_screen_time_dataset.csv", index=False)
print("CSV file 'vision_screen_time_dataset.csv' has been created!")