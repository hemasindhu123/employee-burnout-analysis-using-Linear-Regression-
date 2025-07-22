import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load datase

file_path = r'C:\Users\dell\Desktop\emp\employee_burnout_analysis-AI 2.xlsx'
data = pd.read_excel(file_path)

# Display first 5 rows to inspect the data
print(data.head())

# Data inspection
print(data.shape)
print(data.isnull().sum())
print(data.info())

# Convert 'Date of Joining' to datetime format
data['Date of Joining'] = pd.to_datetime(data['Date of Joining'])

# Drop 'Employee ID' column
data = data.drop(['Employee ID'], axis=1)

# Check unique values and value counts for each column
for col in data.columns:
    print(f"\n\n{col} unique values: {data[col].unique()}")
    print(f"{col} value counts:\n{data[col].value_counts()}\n\n")

# Check for skewness and fill missing values
intfloatdata = data.select_dtypes(include=['int64', 'int32', 'float64', 'float32'])
for col in intfloatdata.columns:
    skew_value = intfloatdata[col].skew()
    if skew_value >= 0.1:
        print(f"\n{col} feature is positively skewed with a skew value of {skew_value}")
    elif skew_value <= -0.1:
        print(f"\n{col} feature is negatively skewed with a skew value of {skew_value}")
    else:
        print(f"\n{col} feature is normally distributed with a skew value of {skew_value}")

# Fill missing values with the mean
data['Resource Allocation'].fillna(data['Resource Allocation'].mean(), inplace=True)
data['Mental Fatigue Score'].fillna(data['Mental Fatigue Score'].mean(), inplace=True)
data['Burn Rate'].fillna(data['Burn Rate'].mean(), inplace=True)

# Correlation matrix
numerical_data = data.select_dtypes(include=['number'])
correlation_matrix = numerical_data.corr()
print(correlation_matrix)

# Heatmap plot using Seaborn
plt.figure(figsize=(14,12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Countplot for Gender distribution
plt.figure(figsize=(8,6))
sns.countplot(x="Gender", data=data, palette='magma')
plt.title("Gender Distribution")
plt.show()

# Histograms for numeric columns
numeric_columns = ['Resource Allocation', 'Mental Fatigue Score', 'Burn Rate']
for col in numeric_columns:
    plt.figure(figsize=(8,6))
    sns.histplot(data[col], kde=True, color='indianred')
    plt.title(f"Distribution of {col}")
    plt.show()

# Line plot for Burn Rate by Designation
plt.figure(figsize=(10,6))
sns.lineplot(data=data, x='Date of Joining', y='Burn Rate', hue='Designation', palette='Pastel1')
plt.title("Burn Rate by Designation")
plt.show()

# Line plot for Burn Rate by Gender
plt.figure(figsize=(10,6))
sns.lineplot(data=data, x='Date of Joining', y='Burn Rate', hue='Gender', palette='Pastel1')
plt.title("Burn Rate by Gender")
plt.show()

# Encode categorical columns
label_encoder = preprocessing.LabelEncoder()
data['GenderLabel'] = label_encoder.fit_transform(data['Gender'])
data['Company_TypeLabel'] = label_encoder.fit_transform(data['Company Type'])
data['WFH_Setup_Available'] = label_encoder.fit_transform(data['WFH Setup Available'])

# Selecting features and target
Columns = ['Designation', 'Resource Allocation', 'Mental Fatigue Score', 'GenderLabel', 'Company_TypeLabel', 'WFH_Setup_Available']
X = data[Columns]
y = data['Burn Rate']

# PCA for dimensionality reduction
pca = PCA(0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X)

print("Original shape of X:", X.shape)
print("PCA shape of X:", X_pca.shape)
print("Variance ratio:", pca.explained_variance_ratio_)
print("Number of components selected:", pca.n_components_)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.25, random_state=10)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = r2_score(y_test, y_pred)
print(f"Model accuracy (R²): {accuracy * 100:.2f}%")
