# Credit Card Transactions EDA

This Streamlit app provides interactive exploratory data analysis for credit card transaction datasets.

## Features

- Upload your own CSV or use the default dataset
- Filter transactions by amount and age
- Preview the dataset in the sidebar
- Visualize age distribution, fraud analysis, and transaction amounts
- Interactive plots for job, category, city, gender, and age

## Usage

1. Install dependencies:
   ```
   pip install streamlit pandas plotly
   ```
2. Run the app:
   ```
   streamlit run app.py
   ```
3. Upload your CSV or use the default.

## Expected CSV Columns

- `amt`: Transaction amount
- `age`: Age of cardholder
- `is_fraud`: Fraud label (0/1)
- `job`, `category`, `city`, `gender`: Categorical features

## File Structure

- `app.py`: Main Streamlit app
- `visualizations.py`: Plotting utilities

---

**Tip:** For large datasets, use the sidebar filters to focus your analysis.