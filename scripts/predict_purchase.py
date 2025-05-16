import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import ColumnTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle

# Load the processed dataset
data_path = 'data/processed/processed_data.csv'
df = pd.read_csv(data_path)

# Select features and target for prediction
X = df.drop(columns=['WillPurchase'])  # Features
y = df['WillPurchase']  # Target (whether the customer will make a purchase)

# Reload the saved Random Forest model
model_path = 'models/random_forest_purchase_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Make predictions
predictions = model.predict(X)

# Add predictions to the original dataframe
df['Predicted Purchase'] = predictions

# Save the predictions to a new CSV file
output_file = 'outputs/predicted_data.csv'
df.to_csv(output_file, index=False)

# Display the first few rows of the predictions
print(df[['Customer ID', 'WillPurchase', 'Predicted Purchase']].head())
