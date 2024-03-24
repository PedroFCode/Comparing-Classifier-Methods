from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score



import os
import numpy as np

print(os.listdir('../Assignment 1'))

df1 = pd.read_csv('../Assignment 1/opel_corsa_01.csv', delimiter=';')
df1.dataframeName = 'opel_corsa_01.csv'

df1 = df1.drop(['Unnamed: 0'], axis=1)  # Drop index column

encoder = LabelEncoder()
df1['roadSurface'] = encoder.fit_transform(df1['roadSurface'])
df1['traffic'] = encoder.fit_transform(df1['traffic'])
df1['drivingStyle'] = encoder.fit_transform(df1['drivingStyle'])

X = df1.drop('drivingStyle', axis=1)
y = df1['drivingStyle']

# Define the scoring metric
scorer = make_scorer(f1_score, average='macro')

### SVM with SMOTE

# Splitting the dataset with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1138, stratify=y)

# Using SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Pipeline for numerical and categorical preprocessing
numeric_features = X_train_resampled.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train_resampled.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Integrating preprocessing with model training in a comprehensive pipeline (example with SVM)
full_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', SVC(class_weight='balanced'))])

# Model evaluation with cross-validation
scores = cross_val_score(full_pipeline, X_train_resampled, y_train_resampled, cv=5, scoring='f1_macro')
print("F1 Score with SMOTE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Fit the model and predict
full_pipeline.fit(X_train_resampled, y_train_resampled)
y_pred = full_pipeline.predict(X_test)

# Calculate SVM accuracy score with SMOTE
svm_accuracy_smote = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy with SMOTE: {svm_accuracy_smote}')

# Evaluation metrics
print(classification_report(y_test, y_pred))

### SVM without SMOTE

# Splitting the dataset with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1138, stratify=y)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(class_weight='balanced')
svm_classifier.fit(X_train_preprocessed, y_train)

# Predictions on the test set
y_pred = svm_classifier.predict(X_test_preprocessed)


# Calculate SVM accuracy score
svm_accuracy = accuracy_score(y_test, y_pred)
print(f'SVM Accuracy without SMOTE: {svm_accuracy}')

# Corss-Validation
scores = cross_val_score(svm_classifier, X_train_preprocessed, y_train, cv=5, scoring=scorer)
print("SVM F1 Score without SMOTE: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Evaluation metrics
print(classification_report(y_test, y_pred))

### Logistic Regression

# Splitting the dataset with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1138, stratify=y)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Label encode the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Feature selection using Recursive Feature Elimination (RFE)
logistic_classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
rfe = RFE(estimator=logistic_classifier, n_features_to_select=18)  # Adjust the number of features as needed
X_train_selected = rfe.fit_transform(X_train_preprocessed, y_train_encoded)
X_test_selected = rfe.transform(X_test_preprocessed)

# Train the Logistic Regression classifier
logistic_classifier.fit(X_train_selected, y_train_encoded)

# Predictions on the test set
y_pred = logistic_classifier.predict(X_test_selected)

# Calculate Logistic Regression accuracy score
logistic_accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Logistic Regression Accuracy: {logistic_accuracy}')

# Cross-validation
scores = cross_val_score(logistic_classifier, X_train_selected, y_train_encoded, cv=5, scoring=scorer)
print("Logistic Regression F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Evaluation metrics
print(classification_report(y_test_encoded, y_pred))


### kNN

# Splitting the dataset with stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1138, stratify=y)

# Preprocess the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Train the kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_preprocessed, y_train)

# Predictions on the test set
y_pred = knn_classifier.predict(X_test_preprocessed)

# Calculate kNN accuracy score
knn_accuracy = accuracy_score(y_test, y_pred)
print(f'kNN Accuracy: {knn_accuracy}')

# Cross-validation
scores = cross_val_score(knn_classifier, X_train_preprocessed, y_train, cv=5, scoring=scorer)
print("kNN F1 Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# Evaluation metrics
print(classification_report(y_test, y_pred))


