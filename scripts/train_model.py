# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Load the preprocessed dataset
data = pd.read_csv('data/processed/processed_data.csv')

# Define the target variable (WillPurchase)
data['WillPurchase'] = (data['Purchase Amount (USD)'] > 0).astype(int)

# Split features and target
X = data.drop(columns=['WillPurchase', 'Purchase Amount (USD)'])  # Drop target and purchase-related info
y = data['WillPurchase']

# Identify categorical and numerical columns
categorical_columns = [
    'Gender', 'Category', 'Location', 'Size', 'Color', 'Season',
    'Subscription Status', 'Payment Method', 'Shipping Type',
    'Discount Applied', 'Promo Code Used', 'Preferred Payment Method'
]
numerical_columns = ['Age', 'Review Rating', 'Previous Purchases', 'Frequency of Purchases']

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns),
    ],
    remainder='drop'  # Drop any columns not explicitly handled
)

# Define the full pipeline with a RandomForestClassifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Model Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model for later use
import joblib
joblib.dump(pipeline, 'models/random_forest_purchase_model.pkl')
print("Model saved to 'models/random_forest_purchase_model.pkl'.")
