# IndirectTax ML Service

This Python service uses Flask to expose a prediction API for IndirectTax usage data. It fetches data from the .NET backend and uses Facebook Prophet to forecast metrics for the next year.

## Requirements

- Python 3.8+
- Flask
- pandas
- requests
- prophet
- numpy

Install dependencies:

```
pip install -r requirements.txt
```

## Usage

1. Ensure your .NET backend is running and accessible at `http://localhost:5078/api/Usage`.
2. Start the Flask ML API:

```
python main.py
```

3. Access the prediction endpoint:

```
GET http://localhost:5000/predict
```

This will return a JSON object with 2024 predictions for each metric (Transactions, TaxReturns, EFilings, Users_Transactions, Users_Return, Users_efile).

## Example Response

```
{
  "Transactions": [
    {"ds": "2024-01-01", "yhat": 12345},
    ...
  ],
  ...
}
```

## Notes
- The ML service must be able to reach the backend API.
- The `/predict` endpoint always returns predictions for the next 12 months (2024).

