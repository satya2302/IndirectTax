from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import io
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import  tools_condition
from langchain_groq import ChatGroq
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from IPython.display import Image, display



app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def create_monthly_data():
    # Fetch data from the API
    response = requests.get('http://localhost:5078/api/Usage')
    response.raise_for_status()
    api_data = response.json()

    # Convert API data to DataFrame
    df = pd.DataFrame(api_data)
    # Rename columns to match expected names
    df = df.rename(columns={
        'year': 'Year',
        'month': 'Month',
        'transactions': 'Transactions',
        'taxReturns': 'TaxReturns',
        'eFilings': 'EFilings',
        'users_Transactions': 'Users_Transactions',
        'users_TaxReturns': 'Users_Return',
        'users_EFilings': 'Users_efile'
    })
    # Ensure Year is integer (handles both int and string input)
    df['Year'] = df['Year'].astype(int)
    # Create 'ds' column in format YYYY-MMM-01
    df['ds'] = df['Year'].astype(str) + '-' + df['Month'].str[:3].str.upper() + '-01'
    # Reorder columns
    df = df[['ds', 'Year', 'Month', 'Transactions', 'TaxReturns', 'EFilings', 'Users_Transactions', 'Users_Return', 'Users_efile']]
    return df

def month_to_num(month):
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    month = month.upper()
    for i, m in enumerate(months):
        if m.startswith(month[:3]):
            return i + 1
    return 1

def train_and_forecast_rf(data, column):
    # Prepare features: year, month as numeric
    data = data.copy()
    data['Year'] = data['ds'].str[:4].astype(int)
    data['MonthNum'] = data['Month'].apply(month_to_num)
    # Add lag features (previous 1, 2, 3 months)
    for lag in [1, 2, 3]:
        data[f'lag_{lag}'] = data[column].shift(lag)
    data = data.dropna().reset_index(drop=True)
    feature_cols = ['Year', 'MonthNum', 'lag_1', 'lag_2', 'lag_3']
    X = data[feature_cols]
    y = data[column]
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    # Prepare to predict for 2024
    last_known = data.iloc[-3:][column].tolist()  # last 3 known values
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    preds = []
    for i, m in enumerate(months):
        year = 2024
        month_num = i + 1
        # For first 3 months, use last known + predicted lags
        if i == 0:
            lags = last_known[-1], last_known[-2], last_known[-3]
        elif i == 1:
            lags = preds[-1], last_known[-1], last_known[-2]
        elif i == 2:
            lags = preds[-1], preds[-2], last_known[-1]
        else:
            lags = preds[-1], preds[-2], preds[-3]
        X_pred = pd.DataFrame([[year, month_num, lags[0], lags[1], lags[2]]], columns=['Year', 'MonthNum', 'lag_1', 'lag_2', 'lag_3'])
        pred = model.predict(X_pred)[0]
        preds.append(pred)
    result = pd.DataFrame({
        'ds': [f'2024-{m[:3]}-01' for m in months],
        'yhat': preds
    })
    return result


def reclassTool(query: str)-> str:
    """Reclass of Tax.
       You are a Tax expert in solving Reclass issues
       Please resolve the reclass tax issue.
       Once solve inform the users about the  cities reclassed from where to where
       And the number of cities reclassed.
       
    """
   
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
    
    test_df.to_csv("../data/reconcillation_synthetic_output.csv", index=False)
    
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
    return test_df

def assistant(state: MessagesState):
    llm =ChatGroq(api_key ="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", model="meta-llama/llama-4-scout-17b-16e-instruct")
    tools = [reclassTool]
    llm_with_tools=llm.bind_tools(tools)
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    print(user_message)

    # Graph
    builder = StateGraph(MessagesState)

    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode([reclassTool]))

    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    react_graph = builder.compile()

    messages = [HumanMessage(content=user_message)]
    messages = react_graph.invoke({"messages": messages})
    ai_contents = []
    for m in messages['messages']:
        if isinstance(m, AIMessage):
          print(m.content)
          ai_contents.append({'from' : 'bot','text': m.content})
    print(ai_contents)         
    return jsonify(ai_contents)

@app.route('/reclass', methods=['GET'])
def reclass():
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


@app.route('/predict', methods=['GET'])
def predict():
    data = create_monthly_data()
    metrics = [
        ('Transactions', 'transactions'),
        ('TaxReturns', 'taxReturns'),
        ('EFilings', 'eFilings')
    ]
    # Predict for each metric
    predictions = {}
    for metric, api_key in metrics:
        pred_df = train_and_forecast_rf(data, metric)
        predictions[api_key] = pred_df['yhat'].tolist()
    # Build response in requested format
    months = ['JANUARY', 'FEBRUARY', 'MARCH', 'APRIL', 'MAY', 'JUNE',
              'JULY', 'AUGUST', 'SEPTEMBER', 'OCTOBER', 'NOVEMBER', 'DECEMBER']
    result = []
    for i, month in enumerate(months):
        result.append({
            "year": 2026,
            "month": month,
            "transactions": int(round(predictions['transactions'][i])),
            "taxReturns": int(round(predictions['taxReturns'][i])),
            "eFilings": int(round(predictions['eFilings'][i]))
          
        })
    return jsonify(result)

@app.route('/predict-plot/<metric>', methods=['GET'])
def predict_plot(metric):
    # Supported metrics
    metric_map = {
        'usertransactions': ('Users_Transactions', 'Users Transactions'),
        'usertaxreturns': ('Users_Return', 'Users Tax Returns'),
        'userefilings': ('Users_efile', 'Users EFilings')
    }
    if metric not in metric_map:
        return jsonify({'error': 'Invalid metric'}), 400
    col, label = metric_map[metric]
    data = create_monthly_data()
    # Group by year and month for all years
    data['date'] = pd.to_datetime(data['ds'])
    data = data.sort_values('date')
    # Plot all years
    plt.figure(figsize=(12, 6))
    for year in sorted(data['Year'].unique()):
        year_data = data[data['Year'] == year]
        plt.plot(year_data['date'], year_data[col], marker='o', label=f'{label} ({year})')
    plt.title(f'{label} for All Years')
    plt.xlabel('Month')
    plt.ylabel(label)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
     app.run(debug=True, port=5000, use_reloader=False)
