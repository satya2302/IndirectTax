import pandas as pd
import numpy as np
df = pd.read_csv("../data/journal_synthetic.csv")
df.head()


# Feature engineering
df["GrossPerTaxable"] = df["Gross"] / (df["Taxable"].replace(0, np.nan))
df["IsReclass"] = (df["InputSource"] == "reclassin").astype(int)
df["IsBalanceImport"] = (df["InputSource"] == "balanceimport").astype(int)
df["IsReclassOut"] = (df["InputSource"] == "reclassout").astype(int)
#df["MonthName"] = df["Month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
#df["YearMonth"] = df["Year"].astype(str) + "-" + df["Month"].astype(str).str.zfill(2)
df["CityCounty"] = df["City"] + "_" + df["County"]
df["LogGross"] = np.log1p(df["Gross"])
df["LogTaxable"] = np.log1p(df["Taxable"].abs())
df.head()

# Check for null values in df
df_nulls = df.isnull().sum()
print("Null values in df:")
display(df_nulls)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Prepare training data: only use rows where PlaceDetermination == "reclass"
train_df = df[df["PlaceDetermination"] == "reclass"].copy()


# Encode categorical features with persistent encoders
cat_cols = ["Region", "County", "Entity", "District", "CityCounty"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col].astype(str))
    encoders[col] = le
# Target encoding
city_le = LabelEncoder()
train_df["CityEncoded"] = city_le.fit_transform(train_df["City"].astype(str))


# Example feature extraction for df
# 1. Ratio features
df["GrossToTaxableRatio"] = df["Gross"] / (df["Taxable"].replace(0, np.nan))
df["TaxableToGrossRatio"] = df["Taxable"] / (df["Gross"].replace(0, np.nan))
# 2. Log transformations (if not already present)
if "LogGross" not in df.columns:
    df["LogGross"] = np.log1p(df["Gross"])
if "LogTaxable" not in df.columns:
    df["LogTaxable"] = np.log1p(df["Taxable"].abs())
# 3. Date-based features
# 4. Categorical encoding (example: create a combined city-county feature)
if "CityCounty" not in df.columns:
    df["CityCounty"] = df["City"] + "_" + df["County"].astype(str)
# 5. Binary features for InputSource
df["IsReclass"] = (df["InputSource"] == "reclassin").astype(int)
df["IsBalanceImport"] = (df["InputSource"] == "balanceimport").astype(int)
df["IsReclassOut"] = (df["InputSource"] == "reclassout").astype(int)
print("Feature extraction complete. New features added to df.")

import matplotlib.pyplot as plt
import seaborn as sns
# Example: Distribution of Gross and Taxable
# Visualize Gross and Taxable by PlaceDetermination (reclass states)
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x="PlaceDetermination", y="Gross", data=df)
plt.title("Gross by PlaceDetermination")
plt.xticks(rotation=45)
plt.subplot(1, 2, 2)
sns.boxplot(x="PlaceDetermination", y="Taxable", data=df)
plt.title("Taxable by PlaceDetermination")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))
sns.scatterplot(x='Gross', y='Taxable', data=df, alpha=0.6, hue='Entity')
plt.title('Relationship between Gross and Taxable by Entity')
plt.xlabel('Gross')
plt.ylabel('Taxable')
plt.show()

from sklearn.model_selection import GridSearchCV
# Features for training
features = ["Region", "County", "Entity", "District", "Gross", "Taxable", "TaxRate",  "GrossPerTaxable", "IsReclass", "IsBalanceImport", "IsReclassOut", "CityCounty", "LogGross", "LogTaxable"]
X_train = train_df[features]
y_train = train_df["CityEncoded"]
# Define parameter grid for RandomForest
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}
# Train model
rf = RandomForestClassifier( random_state=42, n_jobs=-1)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
print("Best parameters found:", grid_search.best_params_)
print(f"Best cross-validated accuracy: {grid_search.best_score_:.4f}")
clf = grid_search.best_estimator_
# --- TESTING ---
# Read test data
test_df = pd.read_csv("../data/reconcillation_synthetic.csv")
# Only test on rows with UnreportedTax > 0
test_df = test_df[test_df["UnreportedTax"] > 0].copy()
test_df.head()



# For each test row, find all possible reclass rows from training data with the same Region and County
# Prepare test features (encode using training encoders)
for col in ["Region", "County", "Entity", "District"]:
    if col not in test_df.columns:
        test_df[col] = -1  # Fill missing columns with -1
    le = encoders[col]
    test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
#test_df["MonthName"] = test_df["Month"].map({1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"})
#test_df["YearMonth"] = test_df["Year"].astype(str) + "-" + test_df["Month"].astype(str).str.zfill(2)
test_df["CityCounty"] = test_df["City"] + "_" + test_df["County"].astype(str)
for col in ["CityCounty"]:
    le = encoders[col]
    test_df[col] = test_df[col].astype(str).map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
test_df["LogGross"] = np.log1p(test_df["Gross"])
test_df["LogTaxable"] = np.log1p(test_df["Taxable"].abs())
test_df["GrossPerTaxable"] = test_df["Gross"] / (test_df["Taxable"].replace(0, np.nan))
test_df["IsReclass"] = 0
test_df["IsBalanceImport"] = 0
test_df["IsReclassOut"] = 0
# Predict city for each test row, restrict to same region and county (with debug)
predicted_cities = []
for idx, row in test_df.iterrows():
    region_val = row["Region"]
    county_val = row["County"]
    # Find all possible city encodings from the same region and county in the training data
    possible_cities = train_df[(train_df["Region"] == region_val) & (train_df["County"] == county_val)]["CityEncoded"].unique()
    if len(possible_cities) == 0:
        # Debug: print when fallback happens
        print(f"No city found for region={region_val}, county={county_val} in train. Fallback to all cities.")
        possible_cities = train_df["CityEncoded"].unique()
    # Use DataFrame to preserve feature names
    row_df = pd.DataFrame([row[features]], columns=features)
    probs = clf.predict_proba(row_df)[0]
    mask = np.zeros_like(probs, dtype=bool)
    mask[possible_cities] = True
    probs_masked = probs * mask
    if probs_masked.sum() == 0:
        # Debug: print when fallback happens
        print(f"All masked out for region={region_val}, county={county_val}. Fallback to original prediction.")
        pred = np.argmax(probs)
    else:
        pred = np.argmax(probs_masked)
    predicted_cities.append(pred)
test_df["PredictedCity"] = city_le.inverse_transform(predicted_cities)
print("Start Predicted cities for test data:")
print(test_df.head())
test_df.to_csv("../data/reconcillation_synthetic_output.csv", index=False)
print("End Predicted cities for test data:")
# Debug: Check if predicted city is in the same region and county as test row
check = []
for idx, row in test_df.iterrows():
    pred_city = row["PredictedCity"]
    region_val = row["Region"]
    county_val = row["County"]
    # Find all cities in train for this region and county
    valid_cities = train_df[(train_df["Region"] == region_val) & (train_df["County"] == county_val)]["City"].unique()
    if pred_city not in valid_cities:
        check.append((region_val, county_val, row["City"], pred_city))
       
# Debug: Check all rows, not just head()
output_df_full = test_df.copy()


