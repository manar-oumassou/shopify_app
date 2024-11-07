import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
st.set_page_config(page_title="Sales Data Analysis and Forecasting", layout="wide")

# Load dataset function
@st.cache_data
def load_data(file):
    data = pd.read_csv(file)
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

# App Title and Sidebar
st.sidebar.title("Sales Analysis & Forecasting")
st.sidebar.info("Navigate through the pages to explore and forecast your sales data.")
page = st.sidebar.radio("Select Page", ["Home", "Analysis","promotion analysis","Forecast"], index=0)
data = None

# Home Page for File Upload
if page == "Home":
    st.title("📈 Sales Data Analysis and Forecasting")
    st.write("Upload a CSV file to get started with sales data analysis and forecasting.")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        data = load_data(uploaded_file)
        st.session_state["data"] = data
        st.success("File uploaded and data loaded successfully!")
        st.write("Now, navigate to **Analysis** or **Forecast** to explore the data.")

if "data" not in st.session_state:
    st.session_state["data"] = None  # Initialize it to None
data = st.session_state['data']




# Analysis Page
if page == "Analysis":
    if data is not None:
        # Displaying the Data Analysis Header
        st.header("Data Analysis Dashboard")
        
        # Basic Summary Statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())
        
        # Time Series Plot for Sales Over Time
        st.subheader("Sales Over Time")
        if 'Date' in data.columns and 'Ventes totales' in data.columns:
            # Ensure 'Date' is a datetime type
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            sales_over_time = data.groupby(data['Date'].dt.to_period("M")).sum(numeric_only=True)['Ventes totales']
            sales_over_time.index = sales_over_time.index.to_timestamp()  # Convert period index to timestamp
            fig = px.line(sales_over_time, x=sales_over_time.index, y='Ventes totales', title='Total Sales Over Time')
            st.plotly_chart(fig)
        else:
            st.warning("Required columns 'Date' and 'Ventes totales' not found in data.")
        
        # Top Products by Sales
        st.subheader("Top Products by Sales")
        if 'Produit' in data.columns and 'Ventes totales' in data.columns:
            top_products = data.groupby('Produit').sum(numeric_only=True)['Ventes totales'].nlargest(10)
            fig2 = px.bar(top_products, x=top_products.index, y='Ventes totales', title='Top 10 Products by Sales')
            st.plotly_chart(fig2)
        else:
            st.warning("Required columns 'Produit' and 'Ventes totales' not found in data.")
        
        # Sales by Region
        st.subheader("Sales by Region")
        if 'Pays de facturation' in data.columns and 'Ventes totales' in data.columns:
            sales_by_region = data.groupby('Pays de facturation').sum(numeric_only=True)['Ventes totales'].nlargest(10)
            fig3 = px.bar(sales_by_region, x=sales_by_region.index, y='Ventes totales', title='Top 10 Regions by Sales')
            st.plotly_chart(fig3)
        else:
            st.warning("Region data is not available.")
        
        # New Analysis: Average Order Value (AOV) Over Time
        st.subheader("Average Order Value (AOV) Over Time")
        if 'Ventes totales' in data.columns and 'Référence de commande' in data.columns:
            order_totals = data.groupby(data['Date'].dt.to_period("M")).agg({
                'Ventes totales': 'sum', 
                'Référence de commande': 'nunique'
            })
            order_totals['AOV'] = order_totals['Ventes totales'] / order_totals['Référence de commande']
            fig4 = px.line(order_totals, x=order_totals.index.to_timestamp(), y='AOV', title='Average Order Value Over Time')
            st.plotly_chart(fig4)
        else:
            st.warning("Required columns 'Ventes totales' and 'Référence de commande' not found in data.")

        
        # New Analysis: Sales by Sales Channel
        st.subheader("Sales by Sales Channel")
        if 'Canal de vente' in data.columns and 'Ventes totales' in data.columns:
            sales_by_channel = data.groupby('Canal de vente').sum(numeric_only=True)['Ventes totales']
            fig6 = px.pie(sales_by_channel, values='Ventes totales', names=sales_by_channel.index, title='Sales Distribution by Sales Channel')
            st.plotly_chart(fig6)
        else:
            st.warning("Sales channel data is not available.")
        
        # New Analysis: Discount Impact on Sales
        st.subheader("Discounts Over Time")
        if 'Date' in data.columns and 'Réductions' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            discounts_over_time = data.groupby(data['Date'].dt.to_period("M")).sum(numeric_only=True)['Réductions']
            discounts_over_time.index = discounts_over_time.index.to_timestamp()
            fig7 = px.line(discounts_over_time, x=discounts_over_time.index, y='Réductions', title='Total Discounts Over Time')
            st.plotly_chart(fig7)
        else:
            st.warning("Discount data is not available.")
        
        # New Analysis: Return Rate by Product
        st.subheader("Top Products by Return Rate")

        # Check if necessary columns are in the data
        if 'Produit' in data.columns and 'Quantité nette' in data.columns:
            # Convert Quantité nette to numeric, if it's not already, and filter for negative values
            data['Quantité nette'] = pd.to_numeric(data['Quantité nette'], errors='coerce')
            negative_quantities = data[data['Quantité nette'] < 0]

            # Group by product and sum net quantities to see products with highest return rates
            product_returns = negative_quantities.groupby('Produit').agg({
                'Quantité nette': 'sum'
            })
            
            # Calculate absolute value of Quantité nette to represent it as a positive return rate indicator
            product_returns['Return Rate'] = abs(product_returns['Quantité nette'])

            # Select top 10 products by return rate
            top_returns = product_returns['Return Rate'].nlargest(10)

            # Plot in Streamlit
            st.subheader("Top 10 Products by Return Rate (Negative Quantité Nette)")
            if not top_returns.empty:
                fig = px.bar(top_returns, x=top_returns.index, y=top_returns.values, 
                            labels={'x': 'Product', 'y': 'Return Rate (Absolute Quantité Nette)'},
                            title='Top 10 Products by Return Rate (Negative Quantité Nette)')
                st.plotly_chart(fig)
            else:
                st.write("No products with a negative quantity net were found.")
        else:
            st.warning("Required columns 'Produit' and 'Quantité nette' are not in the data.")
        # New Analysis: Customer Geographic Distribution
        st.subheader("Customer Geographic Distribution")
        if 'Pays de facturation' in data.columns:
            country_counts = data['Pays de facturation'].value_counts().nlargest(10)
            fig8 = px.bar(country_counts, x=country_counts.index, y=country_counts.values, 
                        labels={'x': 'Country', 'y': 'Number of Orders'},
                        title='Top 10 Countries by Order Count')
            st.plotly_chart(fig8)
        else:
            st.warning("Column 'Pays de facturation' is not available for geographic distribution analysis.")

        # New Analysis: Monthly Sales Growth Rate
        st.subheader("Monthly Sales Growth Rate")
        if 'Date' in data.columns and 'Ventes totales' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            monthly_sales = data.groupby(data['Date'].dt.to_period("M")).sum(numeric_only=True)['Ventes totales']
            monthly_sales.index = monthly_sales.index.to_timestamp()
            monthly_growth = monthly_sales.pct_change() * 100  # Percent change to represent growth rate
            fig9 = px.line(monthly_growth, x=monthly_growth.index, y=monthly_growth.values, 
                        labels={'x': 'Month', 'y': 'Growth Rate (%)'},
                        title='Monthly Sales Growth Rate')
            st.plotly_chart(fig9)
        else:
            st.warning("Required columns 'Date' and 'Ventes totales' not found for growth rate analysis.")

        # New Analysis: Shipping Cost Impact Over Time
        st.subheader("Shipping Cost Impact Over Time")
        if 'Date' in data.columns and 'Expédition' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            shipping_costs = data.groupby(data['Date'].dt.to_period("M")).sum(numeric_only=True)['Expédition']
            shipping_costs.index = shipping_costs.index.to_timestamp()
            fig10 = px.line(shipping_costs, x=shipping_costs.index, y=shipping_costs.values, 
                            labels={'x': 'Date', 'y': 'Total Shipping Costs'},
                            title='Total Shipping Costs Over Time')
            st.plotly_chart(fig10)
        else:
            st.warning("Required columns 'Date' and 'Expédition' are not available for shipping cost analysis.")

        # New Analysis: Product-Level Profit Analysis
        st.subheader("Product-Level Profit Analysis")
        if 'Produit' in data.columns and 'Ventes nettes' in data.columns and 'Expédition' in data.columns:
            # Group by product to analyze profitability
            product_profitability = data.groupby('Produit').agg({
                'Ventes nettes': 'sum',
                'Expédition': 'sum'
            })
            product_profitability['Net Profit'] = product_profitability['Ventes nettes'] - product_profitability['Expédition']
            top_profitable_products = product_profitability['Net Profit'].nlargest(10)
            
            # Plot top profitable products
            fig11 = px.bar(top_profitable_products, x=top_profitable_products.index, y=top_profitable_products.values, 
                        labels={'x': 'Product', 'y': 'Net Profit'},
                        title='Top 10 Profitable Products')
            st.plotly_chart(fig11)
        else:
            st.warning("Required columns 'Produit', 'Ventes nettes', and 'Expédition' are not available for profitability analysis.")

        # New Analysis: High Discount Products
        # Display a subheader for context
        st.subheader("Top 10 Products by Discounts Applied")

        # Check if the necessary columns are in the data
        if 'Produit' in data.columns and 'Réductions' in data.columns:
            # Ensure 'Réductions' column is numeric, converting if needed
            data['Réductions'] = pd.to_numeric(data['Réductions'], errors='coerce')
            
            # Take the absolute values of 'Réductions' to interpret discount amounts positively
            data['Réductions'] = data['Réductions'].abs()
            
            # Group by 'Produit' and sum the 'Réductions' for each product
            discount_by_product = data.groupby('Produit').sum(numeric_only=True)['Réductions'].nlargest(10)
            
            # Create a Plotly bar chart
            fig = px.bar(discount_by_product, 
                        x=discount_by_product.index, 
                        y=discount_by_product.values, 
                        labels={'x': 'Product', 'y': 'Total Discounts (€)'},
                        title='Top 10 Products by Discounts Applied')
            
            # Display the chart in Streamlit
            st.plotly_chart(fig)
        else:
            # Show a warning if the required columns are missing
            st.warning("Required columns 'Produit' and 'Réductions' are not in the data for discount analysis.")

        
        # Button to return to Home
        if st.button("Return to Home"):
            st.experimental_rerun()
    else:
        st.info("Please upload a dataset on the Home page.")

# Forecast Page
elif page == "Forecast":
    if data is not None:
        st.header("Sales Forecasting")
        
        # Ensure the 'Date' column is in datetime format without timezone
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)

        # Filter out any missing dates in the Forecast page without affecting other pages
        forecast_data = data.dropna(subset=['Date']).copy()
        forecast_data.set_index('Date', inplace=True)

        # Total Revenue Forecast with SARIMA
        st.subheader("Forecasting Total Sales (CA)")

        if 'Ventes totales' in forecast_data.columns:
            # Data Cleaning
            columns_to_drop = [
                'Emplacements de PDV', 'Pays de facturation', 'Région de facturation', 
                'Ville de facturation', 'Pays d\'expédition', 'Région d\'expédition', 
                'Ville d\'expédition', 'Type de produit'
            ]
            data_cleaned = data.drop(columns=columns_to_drop, errors='ignore')
            data_cleaned = data_cleaned.dropna(subset=['Produit'])
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce')
            data_cleaned = data_cleaned.dropna(subset=['Date'])
            data_cleaned.set_index('Date', inplace=True)

            # Monthly Resampling and Plotting
            st.write("### Monthly Resampled Sales Data")
            monthly_sales = data_cleaned['Ventes totales'].resample('M').sum()
            fig1 = px.line(monthly_sales, title="Monthly Resampled Sales Data", labels={'index': 'Date', 'value': 'Ventes totales'})
            st.plotly_chart(fig1)

            # Log Transformation
            st.write("### Log Transformation of Sales Data")
            log_sales = np.log(monthly_sales.replace(0, np.nan)).dropna()
            fig2 = px.line(log_sales, title="Log Transformation of Sales Data", labels={'index': 'Date', 'value': 'Log(Ventes totales)'})
            st.plotly_chart(fig2)

            # Differencing
            st.write("### Differenced Log-Transformed Sales Data")
            log_sales_diff = log_sales.diff().dropna()
            fig3 = px.line(log_sales_diff, title="Differenced Log-Transformed Sales Data", labels={'index': 'Date', 'value': 'Differenced Log Sales'})
            st.plotly_chart(fig3)

            # Augmented Dickey-Fuller Test
            adf_result_log_diff = adfuller(log_sales_diff)
            adf_log_diff_p_value = adf_result_log_diff[1]
            st.write(f"ADF Test p-value on Differenced Log Sales Data: {adf_log_diff_p_value:.4f}")

            # Seasonal Decomposition
            st.write("### Seasonal Decomposition of Differenced Log Sales Data")
            decomposition = seasonal_decompose(log_sales_diff, model='additive', period=12)
            trend_fig = px.line(decomposition.trend, title="Trend Component")
            seasonal_fig = px.line(decomposition.seasonal, title="Seasonal Component")
            residual_fig = px.line(decomposition.resid, title="Residual Component")
            st.plotly_chart(trend_fig)
            st.plotly_chart(seasonal_fig)
            st.plotly_chart(residual_fig)

            # ADF Test on residuals
            residuals = decomposition.resid.dropna()
            adf_result_residuals = adfuller(residuals)
            adf_residuals_p_value = adf_result_residuals[1]
            st.write(f"ADF Test p-value on Residuals: {adf_residuals_p_value:.4f}")

            # Train-Test Split and Forecast Extension
            train_size = int(len(log_sales_diff) * 0.85)
            train, test = log_sales_diff[:train_size], log_sales_diff[train_size:]
            forecast_extension = st.slider("Select the number of steps to extend the forecast:", min_value=1, max_value=24, value=3)

            # SARIMA Model Fitting and Forecasting
            sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
            sarima_results = sarima_model.fit()
            forecast_steps = len(test) + forecast_extension
            forecast = sarima_results.get_forecast(steps=forecast_steps)
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # Convert forecast back to original scale
            last_log_value_train = log_sales.iloc[train_size - 1]
            forecast_log_values_real_scale = forecast_values.cumsum() + last_log_value_train
            forecast_real_scale = np.exp(forecast_log_values_real_scale)

            # Plot the Original Data and Forecast
            st.write("### SARIMA Model Forecast on Original Sales Data Scale")
            forecast_fig = go.Figure()
            forecast_fig.add_trace(go.Scatter(x=monthly_sales.index, y=monthly_sales, mode='lines', name='Original Sales'))
            forecast_fig.add_trace(go.Scatter(x=forecast_real_scale.index, y=forecast_real_scale, mode='lines', name='Forecast', line=dict(dash='dash', color='red')))
            forecast_fig.update_layout(title='SARIMA Forecast on Original Sales Data Scale', xaxis_title='Date', yaxis_title='Total Sales (CA)')
            st.plotly_chart(forecast_fig)
                        # Download button for the forecasted values
                        # Create DataFrame for Download
                        
            forecast_df = pd.DataFrame({
                "Date": forecast_real_scale.index,
                "Forecasted Sales": forecast_real_scale.values,
            })
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download forecasted values as CSV",
                data=csv,
                file_name="forecasted_sales.csv",
                mime="text/csv"
            )

        else:
            st.warning("Required column 'Ventes totales' not found for revenue forecast.")

        # Product-Level Purchase Forecast with SARIMA
        st.subheader("Forecasting Monthly Purchases per Product")
        best_configs = {
                'adam': (1, 0, 1, 0, 0, 1),
                'archie': (0, 1, 2, 0, 1, 0),
                'benjamin': (1, 1, 0, 1, 0, 1),
                'brody': (1, 0, 0, 0, 1, 0),
                'caleb': (1, 1, 0, 1, 1, 1),
                'cameron': (2, 0, 2, 1, 0, 1),
                'carter-1': (0, 1, 1, 1, 1, 0),
                'louis': (0, 1, 0, 0, 1, 1),
                'lincoln': (2, 1, 0, 1, 0, 0),
                'gift-card': (2, 1, 1, 1, 0, 0),
                'dean': (0, 0, 1, 1, 0, 1),
                'eliot': (1, 0, 0, 0, 1, 0),
                'fabian': (2, 0, 0, 0, 1, 0),
                'filtercappro': (0, 0, 1, 1, 0, 1),
                'hamilton': (2, 1, 1, 0, 1, 1),
                'hugo': (2, 0, 0, 0, 1, 1),
                'jonas': (2, 0, 0, 1, 0, 0),
                'lorie': (2, 0, 2, 1, 0, 0),
                'ryan': (0, 0, 0, 1, 0, 0),
                'vitaminsc': (0, 1, 2, 1, 1, 1),
                'tommy': (0, 1, 0, 1, 0, 0),
                'elvis': (0, 0, 2, 0, 0, 1),
                'harvey': (0, 1, 0, 1, 0, 1),
                'filteruv': (2, 1, 2, 0, 1, 1),
                'ronnie': (0, 1, 0, 0, 1, 0),
                'alfred': (2, 0, 0, 1, 1, 1),
                'frankie': (0, 0, 0, 1, 1, 1),
                'ethan': (2, 0, 0, 1, 0, 1),
                'marcus': (1, 1, 2, 0, 1, 0),
                'logan': (1, 0, 0, 0, 0, 1),
                'liam': (0, 0, 1, 0, 0, 1),
                'geneva': (2, 0, 1, 1, 1, 0),
                'russel': (1, 0, 2, 1, 1, 1),
                'evan': (0, 0, 2, 0, 0, 1),
                'mason': (1, 1, 3, 0, 0, 2),
                'woody': (2, 1, 3, 2, 0, 0),
                'luca2': (2, 1, 3, 1, 0, 1),
                'gerald': (1, 1, 2, 2, 0, 0),
                'gabriel': (2, 1, 3, 2, 0, 2),
                'jaxon': (0, 0, 3, 2, 0, 1),
                'sebastian2': (1, 1, 2, 2, 0, 2),
                'rowan': (2, 1, 3, 1, 1, 1),
                'joseph': (0, 0, 1, 0, 0, 1)
            }
        forecast_results = []
        forecast_steps1 = st.slider("Forecast steps:", min_value=1, max_value=24, value=2)

        # Assuming forecast_data and data have been loaded previously
        if 'Produit' in forecast_data.columns and 'Ventes totales' in forecast_data.columns:
            # Preprocess data
            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            data = data.dropna(subset=['Date', 'Produit', 'Quantité nette'])

            # Agrégation des données par produit et mois
            data['Month'] = data['Date'].dt.to_period('M')
            monthly_sales = data.groupby(['Produit', 'Month'])['Quantité nette'].sum().reset_index()
            monthly_sales['Log_Sales'] = np.log1p(monthly_sales['Quantité nette'])

            # Sélectionner un produit du menu déroulant
            product_list = list(best_configs.keys())
            selected_product = st.selectbox("Select a product for forecasting:", product_list)

            # Obtenir la dernière date des données pour les aligner
            max_date = monthly_sales['Month'].max().to_timestamp()
           
            # For each product, generate forecasts
            # Filter data for the selected product
            product_data = monthly_sales[monthly_sales['Produit'] == selected_product]
            product_series = product_data.set_index('Month')['Log_Sales']
            product_series.index = product_series.index.to_timestamp()  # Convert PeriodIndex to Timestamp

            # Retrieve SARIMA configuration for the selected product
            if selected_product in best_configs:
                order = best_configs[selected_product][:3]
                seasonal_order = best_configs[selected_product][3:] + (12,)
                st.write(f"Using preset SARIMA configuration for {selected_product}: {order}x{seasonal_order}")
            else:
                order = (1, 1, 1)
                seasonal_order = (1, 1, 1, 12)
                st.write(f"No preset found for {selected_product}, using default SARIMA configuration: {order}x{seasonal_order}")

            # Filter data to last three years
            last_three_years = product_series[product_series.index >= (product_series.index.max() - pd.DateOffset(years=3))]

            # Train-test split on last three years of data
            train_size = int(len(last_three_years) * 0.8)
            train, test = last_three_years[:train_size], last_three_years[train_size:]

            # Fit SARIMA model using last three years of training data
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)

            # Forecast with specified steps
            forecast_steps = len(test) + forecast_steps1
            forecast = model_fit.get_forecast(steps=forecast_steps)
            forecast_values = forecast.predicted_mean
            forecast_ci = forecast.conf_int()

            # Revert log transformation
            forecast_sales = np.expm1(forecast_values)
            lower_ci = np.expm1(forecast_ci.iloc[:, 0])
            upper_ci = np.expm1(forecast_ci.iloc[:, 1])
            actual_sales = np.expm1(test) if len(test) > 0 else None
                            # Créer un DataFrame avec les résultats
            forecast_data = pd.DataFrame({
                'Produit': selected_product,
                'Date': forecast_sales.index,
                'Prévision': forecast_sales
            })

            forecast_results.append(forecast_data)
            forecast_table = pd.concat(forecast_results, ignore_index=True)
            pivot_forecast = forecast_table.pivot_table(index='Produit',
                        columns='Date',
                        values='Prévision',  # Only include the 'Prévision' column
                        aggfunc='first'
                    )
            pivot_forecast.columns = [f'Prévision_{col.strftime("%Y-%m-%d")}' for col in pivot_forecast.columns]

            # Plot forecast with Plotly
            fig = go.Figure()

            # Add actual sales if available
            if actual_sales is not None:
                fig.add_trace(go.Scatter(
                    x=actual_sales.index,
                    y=actual_sales,
                    mode='lines',
                    name='Actual Data',
                    line=dict(color='green')
                ))

            # Add forecasted sales
            fig.add_trace(go.Scatter(
                x=forecast_sales.index,
                y=forecast_sales,
                mode='lines',
                name='Forecast',
                line=dict(color='orange')
            ))

            # Add confidence interval as filled area
            fig.add_trace(go.Scatter(
                x=forecast_sales.index.tolist() + forecast_sales.index[::-1].tolist(),
                y=upper_ci.tolist() + lower_ci[::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',  # Light orange fill
                line=dict(color='rgba(255, 165, 0, 0)'),  # No outline
                showlegend=False,
                name='Confidence Interval'
            ))

            # Update layout
            fig.update_layout(
                title=f'SARIMA Forecast for {selected_product}',
                xaxis_title='Date',
                yaxis_title='Sales',
                legend_title='Legend'
            )

            # Show plot in Streamlit
            st.plotly_chart(fig)
           
            # Ajouter les résultats à la liste des résultats
            st.write("### Forecasted Data Table")
            st.write(pivot_forecast)
                # Create DataFrame for download
            forecast_df = pd.DataFrame({
                'Produit': selected_product,
                "Date": forecast_sales.index,
                "Forecasted Sales": forecast_sales.values,
                "Lower CI": lower_ci.values,
                "Upper CI": upper_ci.values
            })
            

            # Add download button for the forecasted values
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Forecasted Values as CSV",
                data=csv,
                file_name=f"{selected_product}_forecasted_sales.csv",
                mime="text/csv"
            )

        else:
            st.write("Please upload a dataset to proceed.")
    else:
        st.info("Please upload a dataset on the Home page.")
elif page == "promotion analysis":
    if data is not None:
        # Convert Date to datetime and create a Promotion column (True if discount is applied)
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
        data['Promotion'] = data['Réductions'] < 0  # Assuming reductions are negative for promotions

        # Group by month to get total sales and revenue during promotion and non-promotion periods
        promotion_impact = data.groupby([data['Date'].dt.to_period("M"), 'Promotion']).sum(numeric_only=True)[['Ventes totales']]
        promotion_impact = promotion_impact.unstack().fillna(0)
        promotion_impact.columns = ['No Promotion', 'With Promotion']

        # Plot sales with and without promotions
        fig = px.bar(promotion_impact, x=promotion_impact.index.to_timestamp(), y=['No Promotion', 'With Promotion'],
                     title='Sales and Revenue During Promotion vs Non-Promotion Periods')
        st.plotly_chart(fig)

        # Quantity Sold with and without Promotion
        quantity_impact = data.groupby([data['Date'].dt.to_period("M"), 'Promotion']).sum(numeric_only=True)[['Quantité nette']]
        quantity_impact = quantity_impact.unstack().fillna(0)
        quantity_impact.columns = ['No Promotion', 'With Promotion']
        fig2 = px.line(quantity_impact, x=quantity_impact.index.to_timestamp(), y=['No Promotion', 'With Promotion'],
                       title='Quantity Sold with vs Without Promotions Over Time')
        st.plotly_chart(fig2)

        # Average Quantity Sold per Product with and without Promotion
        avg_quantity_per_product = data.groupby(['Produit', 'Promotion']).mean(numeric_only=True)['Quantité nette'].unstack().fillna(0)
        avg_quantity_per_product.columns = ['No Promotion', 'With Promotion']
        fig3 = px.bar(avg_quantity_per_product, x=avg_quantity_per_product.index, y=['No Promotion', 'With Promotion'],
                      title='Average Quantity Sold per Product: With vs Without Promotions')
        st.plotly_chart(fig3)
            # Promotion Impact on Average Order Value (AOV)
    # Timing of Purchases During Promotional Periods
        data['Hour'] = data['Date'].dt.hour
        hour_promo_sales = data[data['Promotion']].groupby('Hour').size()
        hour_all_sales = data.groupby('Hour').size()
        promo_purchase_ratio = (hour_promo_sales / hour_all_sales).fillna(0) * 100
        fig8 = px.bar(promo_purchase_ratio, x=promo_purchase_ratio.index, y=promo_purchase_ratio,
                  title='Promotion Purchase Timing (Hourly)', labels={'y': 'Purchase Ratio (%)'})
        st.plotly_chart(fig8)
        # Promotion Impact on Average Order Value (AOV)
        # Ensure we have flattened data with aligned indices for Plotly
        aov_promotion = data.groupby(['Date', 'Promotion']).agg({'Ventes totales': 'sum', 'Référence de commande': 'nunique'})
        aov_promotion['AOV'] = aov_promotion['Ventes totales'] / aov_promotion['Référence de commande']

        # Group by month and promotion status
        aov_monthly = aov_promotion.groupby([aov_promotion.index.get_level_values(0).to_period("M"), 'Promotion']).mean()['AOV']

        # Flatten and prepare the data for Plotly
        aov_monthly_df = aov_monthly.unstack().reset_index()
        aov_monthly_df.columns = ['Month', 'No Promotion', 'With Promotion']
        aov_monthly_df['Month'] = aov_monthly_df['Month'].dt.to_timestamp()  # Convert period index to timestamp

        # Plot with Plotly
        fig7 = px.line(aov_monthly_df, x='Month', y=['No Promotion', 'With Promotion'],
                    title='Promotion Impact on Average Order Value (AOV)')
        st.plotly_chart(fig7)

    else:
        st.write("Please upload a dataset to proceed.")