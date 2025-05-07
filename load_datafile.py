# Load your dataset (Modify file name if needed)
data = pd.read_csv("your_sensor_data.csv")

# Display first few rows to verify
print(data.head())

# Ensure relevant columns exist
required_columns = ['Thermal_Cycles', 'Max_Temperature', 'Delta_Temperature', 'Rds_on', 'Threshold_Voltage_Drift', 'Wear_Condition']
missing_cols = [col for col in required_columns if col not in data.columns]
if missing_cols:
    print(f"Missing columns: {missing_cols}")
else:
    print("All required columns are present.")
