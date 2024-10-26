# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('mushroom.csv')
df = df.drop(columns=['Unnamed: 0'])  # Drop the unnecessary index column

# Data Overview
print(df.info())
print(df.head())
print(df.isnull().sum())
print(df.nunique())
print(df['class'].value_counts())

# Visualizations
# Histograms
df.hist(bins=20, figsize=(15, 10))
plt.show()

# Box Plots for numerical features
sns.boxplot(data=df[['stalk_height', 'cap_diameter']])
plt.title("Box Plot of Stalk Height and Cap Diameter")
plt.show()

# Pair Plot
sns.pairplot(df, hue='class')
plt.show()

# Encode categorical variables
df_encoded = pd.get_dummies(df.drop('class', axis=1), drop_first=True)
y = df['class'].apply(lambda x: 1 if x == 'poisonous' else 0)  # Binary encoding of target

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(df_encoded, y, test_size=0.2, random_state=42)

# SVM Implementation
svm = SVC(kernel='linear')  # Start with a linear kernel
svm.fit(X_train, y_train)

# Predictions and Evaluation
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1-Score:", f1_score(y_test, y_pred))

# Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, cv=5)
grid.fit(X_train, y_train)
print("Best Parameters:", grid.best_params_)

# Evaluate the best model
best_svm = grid.best_estimator_
y_pred_best = best_svm.predict(X_test)
print("Best Model Accuracy:", accuracy_score(y_test, y_pred_best))
print("Best Model Precision:", precision_score(y_test, y_pred_best))
print("Best Model Recall:", recall_score(y_test, y_pred_best))
print("Best Model F1-Score:", f1_score(y_test, y_pred_best))
