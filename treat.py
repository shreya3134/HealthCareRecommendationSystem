import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error

# Load dataset
df = pd.read_csv("detailed_exercise_disease_dataset.csv")

# Fill missing values in text columns and combine them
df['Disease'] = df['Disease'].fillna('')
df['Past_History'] = df['Past_History'].fillna('')
df['BP_Level'] = df['BP_Level'].fillna('')

# Combine text columns into one for TF-IDF
df['combined_text'] = df['Disease'] + ' ' + df['Past_History'] + ' ' + df['BP_Level']

# Features and targets
text_feature = 'combined_text'
numeric_features = ['Age', 'Heart_Beats', 'Oxygen_Level']
output_labels = ['Medicine_1', 'Medicine_2', 'Medicine_3', 'Medicine_4', 
                 'Exercise 1', 'Exercise 2', 'Exercise 3']
regression_target = 'Recovery_Days'  # updated here

# Preprocessing pipeline: TF-IDF + scaling numeric
preprocessor = ColumnTransformer(transformers=[
    ('text', TfidfVectorizer(), text_feature),
    ('num', StandardScaler(), numeric_features)
])

# Prepare inputs
X = df[[text_feature] + numeric_features]

# Prepare outputs
y_classification = df[output_labels]
y_regression = df[regression_target]

# Split data
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
    X, y_classification, y_regression, test_size=0.2, random_state=42
)

# Build and train classification pipeline
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
])
clf_pipeline.fit(X_train, y_class_train)

# Build and train regression pipeline
reg_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])
reg_pipeline.fit(X_train, y_reg_train)

# Predict classification and regression on test set
y_class_pred = clf_pipeline.predict(X_test)
y_reg_pred = reg_pipeline.predict(X_test)

# Convert classification predictions to DataFrame
y_class_pred_df = pd.DataFrame(y_class_pred, columns=output_labels, index=y_class_test.index)

# Print classification reports and accuracy per label
for label in output_labels:
    print(f"\nClassification report for {label}:")
    print(classification_report(y_class_test[label], y_class_pred_df[label], zero_division=0))
print("\nAccuracy per label:")
for label in output_labels:
    acc = accuracy_score(y_class_test[label], y_class_pred_df[label])
    print(f"{label}: {acc:.4f}")
overall_acc = (y_class_pred_df == y_class_test).all(axis=1).mean()
print(f"\nOverall exact match accuracy: {overall_acc:.4f}")

# Evaluate regression output
mse = mean_squared_error(y_reg_test, y_reg_pred)
rmse = mse ** 0.5
print(f"\nRegression evaluation for '{regression_target}':")
print(f"RMSE: {rmse:.4f}")
print(f"Mean actual days: {y_reg_test.mean():.2f}")

# Optionally, print some predictions vs actuals
print("\nSample predictions for Recovery_Days:")
print(pd.DataFrame({'Actual': y_reg_test, 'Predicted': y_reg_pred}).head())
