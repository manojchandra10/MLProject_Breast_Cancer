
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load("rfs_model.joblib")

# Load test dataset
test_file = "FinalTestDataset2025.csv"
df_test = pd.read_csv(test_file)

df_test = df_test.rename(columns={'ID': 'PatientID', 'pCR (outcome)': 'PCR', 'RelapseFreeSurvival (outcome)': 'RFS'})

# Replace 999
df_test = df_test.replace(999, np.nan)

patient_ids = df_test['PatientID']

X_test = df_test.drop(columns=['PatientID', 'PCR', 'RFS'], errors='ignore')

# predictions
predicted_rfs = model.predict(X_test)

output = pd.DataFrame({'PatientID': patient_ids, 'RFS': predicted_rfs})
output.to_csv("RFSPrediction.csv", index=False)

print(output.head())