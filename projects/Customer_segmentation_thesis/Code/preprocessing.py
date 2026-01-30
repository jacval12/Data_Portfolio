# preprocessing.py
   
import polars as pl
import pandas as pd
from time import perf_counter
 
def load_and_preprocess_data(filepath: str):
    start = perf_counter()
    retail_polars = pl.read_excel(filepath)
    print(retail_polars.head())
    end = perf_counter()
    print(f"Time: {end - start:.6f} seconds")

    retail_original = retail_polars.to_pandas()
    retail = retail_original.copy()

    if '__UNNAMED__8' in retail.columns:
        retail = retail.drop('__UNNAMED__8', axis=1)

    print(retail.info())

    retail_orders = retail[retail["InvoiceNo"].notnull()]
    print(retail_orders.info())

    stockcode_repeat = retail_orders.dropna(subset=['Description']) \
                                    .groupby('StockCode')['Description'] \
                                    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)

    null_descriptions = retail_orders[retail_orders['Description'].isnull()]
    for i in null_descriptions.index:
        stock_code = retail_orders.at[i, 'StockCode']
        if stock_code in stockcode_repeat.index:
            mode_description = stockcode_repeat[stock_code]
            if mode_description:
                retail_orders.at[i, 'Description'] = mode_description

    retail_orders = retail_orders[retail_orders['Description'].notnull()]
    print(retail_orders.info())

    missing_customer_ids = retail_orders['CustomerID'].isna().sum()
    percentage_unknown_customer = (missing_customer_ids / len(retail)) * 100
    print(f"{round(percentage_unknown_customer, 2)}% of customers are unknown")

    irrelevant_conditions = [
        ((retail_orders['StockCode'] == 'S') & retail_orders['Description'].str.lower().str.contains('samples') & (retail_orders['Quantity'] < 0) & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'M') & (retail_orders['Description'] == 'Manual') & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'POST') & (retail_orders['Description'] == 'POSTAGE') & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'DOT') & (retail_orders['Description'] == 'DOTCOM POSTAGE') & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'BANK CHARGES') & (retail_orders['Description'] == 'Bank Charges') & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'AMAZONFEE') & (retail_orders['Description'] == 'AMAZON FEE') & retail_orders['CustomerID'].isnull()),
        ((retail_orders['StockCode'] == 'B') & (retail_orders['Description'] == 'Adjust bad debt') & retail_orders['CustomerID'].isnull())
    ]

    for condition in irrelevant_conditions:
        retail_orders = retail_orders[~condition]

    print(retail_orders.info())

    non_informative_patterns = ['?', '??', '???', '?missing', '???missing', '?sold as sets?', '??missing', '??', '???lost', '????damages????', '????missing']
    retail_orders = retail_orders[~retail_orders['Description'].astype(str).apply(
        lambda x: any(pattern in x for pattern in non_informative_patterns)
    )]
  
    print(retail_orders.info())

    retail_orders["Total_Price"] = retail_orders["Quantity"] * retail_orders["UnitPrice"]
    retail_orders['Year'] = retail_orders['InvoiceDate'].dt.year
    retail_orders['Month'] = retail_orders['InvoiceDate'].dt.month
    retail_orders['Day'] = retail_orders['InvoiceDate'].dt.day

    retail_new = retail_orders.copy()
    retail_orders_known = retail_new[retail_new['CustomerID'].notnull()]
    retail_orders_unknown = retail_new[retail_new['CustomerID'].isnull()]

    return retail_orders , retail_orders_known, retail_orders_unknown
  