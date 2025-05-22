import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
df = pd.read_csv('output.data')

# Step 2: Drop columns with all NaN values
df = df.dropna(axis=1, how='all')

# Step 3: Fill missing values with the most frequent value in each column
imputer = SimpleImputer(strategy='most_frequent')
df_imputed = imputer.fit_transform(df)
df_cleaned = pd.DataFrame(df_imputed, columns=df.columns)

# Step 4: Encode the target column
label_encoder = LabelEncoder()
df_cleaned['Disease'] = label_encoder.fit_transform(df_cleaned['Disease'])

# Step 5: Encode the symptom columns
symptom_cols = ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4', 'Symptom_5', 'Symptom_6']
symptom_encoders = {}

for col in symptom_cols:
    le = LabelEncoder()
    df_cleaned[col] = le.fit_transform(df_cleaned[col])
    symptom_encoders[col] = le

# Step 6: Features and labels
X = df_cleaned[symptom_cols]
y = df_cleaned['Disease']

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Save model and encoders
pickle.dump(model, open('pred.pkl', 'wb'))
pickle.dump(symptom_encoders, open('symptom_encoders.pkl', 'wb'))
pickle.dump(label_encoder, open('disease_encoder.pkl', 'wb'))

# Step 10: Print accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
