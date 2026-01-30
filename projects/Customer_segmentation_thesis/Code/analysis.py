      
import seaborn as sns
import matplotlib.pyplot as plt
def analyze_and_visualize_data(retail_orders_known, show_plots=True):
    # Number of unique customers by country
    retail_orders_countries_customers = retail_orders_known.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False)
    print(retail_orders_countries_customers)

    top_countries_no_uk = retail_orders_countries_customers.drop('United Kingdom').sort_values(ascending=False).head(10)
     
    if show_plots:
        # Generate a color palette
        colors = sns.color_palette('tab10', len(top_countries_no_uk))

        # Plot
        plt.figure(figsize=(5, 3))
        bars = top_countries_no_uk.sort_values().plot(kind='barh', color=colors)

        # Annotate bars
        for i, (value, name) in enumerate(zip(top_countries_no_uk.sort_values(), top_countries_no_uk.sort_values().index)):
            plt.text(value + 1, i, str(value), va='center')

        plt.title('Top 10 Non-UK Countries by Number of Customers')
        plt.xlabel('Number of Customers')
        plt.ylabel('Country')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Total revenue by country
    retail_orders_countries_total_Sales = retail_orders_known.groupby('Country')['Total_Price'].sum().sort_values(ascending=False)
    print(retail_orders_countries_total_Sales)

    # Number of unique invoices by country
    retail_invoices = retail_orders_known.groupby('Country')['InvoiceNo'].nunique().sort_values(ascending=False)
    print(retail_invoices)

    top_countries_invoices_no_uk = retail_invoices.drop('United Kingdom').sort_values(ascending=False).head(10)

    if show_plots:
        colors = sns.color_palette('tab10', len(top_countries_invoices_no_uk))

        # Plot
        plt.figure(figsize=(5, 3))
        bars = top_countries_invoices_no_uk.sort_values().plot(kind='barh', color=colors)

        # Annotate bars
        for i, (value, name) in enumerate(zip(top_countries_invoices_no_uk.sort_values(), top_countries_invoices_no_uk.sort_values().index)):
            plt.text(value + 1, i, str(value), va='center')

        plt.title('Top 10 Non-UK Countries by Number of Invoices')
        plt.xlabel('Number of Invoices')
        plt.ylabel('Country')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # Number of variety of products by country
    product_variety_by_country = retail_orders_known.groupby('Country')['StockCode'].nunique().sort_values(ascending=False)
    print(product_variety_by_country)

    # Most loyal customers
    loyal_customers = retail_orders_known.groupby('CustomerID')['InvoiceNo'].nunique().sort_values(ascending=False)
    print(loyal_customers.head(10))

    # Highest spending customers
    rich_customers = retail_orders_known.groupby('CustomerID')['Total_Price'].sum().sort_values(ascending=False)
    print(rich_customers.head(10))

    # Describe stats as LaTeX
    print(retail_orders_known.describe().round(2).to_latex())

    # Return computed data
    return {
        "unique_customers_by_country": retail_orders_countries_customers,
        "total_sales_by_country": retail_orders_countries_total_Sales,
        "unique_invoices_by_country": retail_invoices,
        "product_variety_by_country": product_variety_by_country,
        "loyal_customers": loyal_customers,
        "rich_customers": rich_customers
    }
   