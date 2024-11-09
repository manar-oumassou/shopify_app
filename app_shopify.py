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

if 'data' not in st.session_state:
    st.session_state['data'] = None  # Initialize data to None if not yet loaded
if 'data2' not in st.session_state:
    st.session_state['data2'] = None  # Initialize data2 similarly

# Function to load data
@st.cache_data
def load_data(file):
    data = pd.read_csv(file, encoding='utf-8')
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data

# Sidebar Navigation
page = st.sidebar.radio("Select Page", ["Home", "Analysis", "Promotion Analysis", "Forecast"], index=0)

# Home Page: Load CSV File
if page == "Home":
    st.title("📈 Sales Data Analysis and Forecasting")
    st.write("Upload a CSV file to get started with sales data analysis and forecasting.")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

    if uploaded_file:
        # Load data and store it in session state
        data = load_data(uploaded_file)
        st.session_state['data'] = data
        st.session_state['data2'] = data.copy()  # Make a copy for data2 if needed separately
        
        st.success("File uploaded and data loaded successfully!")
        st.write("Now, navigate to **Analysis** or **Forecast** to explore the data.")

# Access data from session state on other pages
data = st.session_state.get('data')
data2 = st.session_state.get('data2')


# Analysis Page
if page == "Analysis":
    if data is not None:
        # Displaying the Data Analysis Header
        st.header("Data Analysis Dashboard")
        
        # Basic Summary Statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())
                # Average Quantity Sold per Product with and without Promotion
        data['Produit'] = data['Produit'].str.split('|').str[0].str.strip()
        # Assuming 'Produit' is the column containing the full product names
        data['Produit'] = data['Produit'].str.split('-').str[0].str.strip().str.lower()
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
        st.subheader("Product-orders Analysis")
        if 'Produit' in data.columns and 'Ventes nettes' in data.columns and 'Expédition' in data.columns:
            # Group by product to analyze profitability
            data2 = data[data["Quantité nette"]>0]
            product_profitability = data2.groupby('Produit').agg({
                'Commande': 'count'
            })
            top_profitable_products = product_profitability['Commande'].nlargest(10)
            
            # Plot top profitable products
            fig11 = px.bar(top_profitable_products, x=top_profitable_products.index, y=top_profitable_products.values, 
                        labels={'x': 'Product', 'y': 'Net Profit'},
                        title='Top 10 Products orders of all time')
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

        # Filter out missing dates
        forecast_data = data.dropna(subset=['Date']).copy()
        forecast_data.set_index('Date', inplace=True)

        st.header("Sales Forecasting with Prophet")

        # Total Revenue Forecast with Prophet
        st.subheader("Prophet Model Forecast for Total Sales (CA)")
        if 'Ventes totales' in forecast_data.columns:
            # Data Cleaning
            columns_to_drop = [
                'Emplacements de PDV', 'Pays de facturation', 'Région de facturation',
                'Ville de facturation', 'Pays d\'expédition', 'Région d\'expédition',
                'Ville d\'expédition', 'Type de produit'
            ]
            data_cleaned = data.drop(columns=columns_to_drop, errors='ignore').dropna(subset=['Produit'])
            data_cleaned['Date'] = pd.to_datetime(data_cleaned['Date'], errors='coerce').dt.tz_localize(None)
            data_cleaned.set_index('Date', inplace=True)

            # Monthly Resampling
            monthly_sales = data_cleaned['Ventes totales'].resample('M').sum().reset_index()
            monthly_sales.columns = ['ds', 'y']
            monthly_sales['y'] = np.log1p(monthly_sales['y'])  # Log-transform to stabilize variance

            # Prophet Model Fitting
            model = Prophet(yearly_seasonality=True, daily_seasonality=False)
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.fit(monthly_sales)

            # Forecast period
            forecast_extension = st.slider("Select the number of months to forecast:", min_value=1, max_value=24, value=3)
            future = model.make_future_dataframe(periods=forecast_extension, freq='M')
            forecast = model.predict(future)

            # Reverse log transformation for interpretability
            forecast['yhat'] = np.expm1(forecast['yhat'])
            forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
            forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
            monthly_sales['y'] = np.expm1(monthly_sales['y'])  # Reverse on actual data

            # Plotting Prophet Forecast with Plotly
            fig_prophet = go.Figure()
            fig_prophet.add_trace(go.Scatter(x=monthly_sales['ds'], y=monthly_sales['y'], mode='lines', name='Original Sales'))
            fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecasted Sales',line=dict(dash='dash', color='red')))
            fig_prophet.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0),
                showlegend=False
            ))
            fig_prophet.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0),
                fill='tonexty', name='Confidence Interval', fillcolor='rgba(255,165,0,0.2)'
            ))
            fig_prophet.update_layout(title=f"Forecast for Total Sales (CA) for the next {forecast_extension} weeks", xaxis_title="Date", yaxis_title="Total Sales (CA)")
            st.plotly_chart(fig_prophet)

            # Create Downloadable Forecast CSV
            forecast_df = pd.DataFrame({
                "Date": forecast['ds'],
                "Forecasted Sales": forecast['yhat'],
                "Lower CI": forecast['yhat_lower'],
                "Upper CI": forecast['yhat_upper']
            }).tail(forecast_extension)
            csv = forecast_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download forecasted values as CSV", data=csv, file_name="forecasted_sales.csv", mime="text/csv")

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
            data['Produit'] = data['Produit'].str.split('|').str[0].str.strip()
        # Assuming 'Produit' is the column containing the full product names
            data['Produit'] = data['Produit'].str.split('-').str[0].str.strip().str.lower()
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
            pivot_forecast.columns = pd.to_datetime(pivot_forecast.columns, errors='coerce')
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
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce').dt.tz_localize(None)
           # Filter out any missing dates in the Forecast page without affecting other pages
        forecast_data = data.dropna(subset=['Date']).copy()
        forecast_data.set_index('Date', inplace=True)

        # Prophet Forecasting Section
        st.subheader("Prophet Model - Custom Forecast Period")

        # Data preparation for Prophet
        # Filter data for the last 7 years
        five_years_ago = pd.Timestamp.now().tz_localize(None) - pd.DateOffset(years=7)
        filtered_data = data[data['Date'] >= five_years_ago]

        # Resample to weekly data and reset index for Prophet
        weekly_data = filtered_data.set_index('Date').resample('W')['Quantité nette'].sum().fillna(0).reset_index()

        # Prepare data for Prophet
        prophet_data = weekly_data.rename(columns={'Date': 'ds', 'Quantité nette': 'y'})
        prophet_data['y'] = np.log1p(prophet_data['y'])  # Log-transform to stabilize variance

        # Initialize and fit the Prophet model
        model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Optional: monthly seasonality

        # Fit model on the dataset
        model.fit(prophet_data)

        # Forecast steps slider (in weeks)
        forecast_steps2 = st.slider("Forecast steps (weeks):", min_value=1, max_value=24, value=2)

        # Create future dates DataFrame based on selected forecast steps
        future = model.make_future_dataframe(periods=forecast_steps2, freq='W')

        # Forecast
        forecast = model.predict(future)

        # Reverse log transformation for interpretability
        forecast['yhat'] = np.expm1(forecast['yhat'])
        forecast['yhat_lower'] = np.expm1(forecast['yhat_lower'])
        forecast['yhat_upper'] = np.expm1(forecast['yhat_upper'])
        prophet_data['y'] = np.expm1(prophet_data['y'])  # Reverse on actual data

        # Plot Prophet forecast in Plotly
        fig = go.Figure()

        # Actual data
        fig.add_trace(go.Scatter(
            x=prophet_data['ds'],
            y=prophet_data['y'],
            mode='lines',
            name='Actual Quantité nette',
            line=dict(color='blue')
        ))

        # Forecasted data
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            mode='lines',
            name=f'Prophet Forecast ({forecast_steps2} Weeks)',
            line=dict(color='orange')
        ))

        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast['ds'].tolist() + forecast['ds'][::-1].tolist(),
            y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'][::-1].tolist(),
            fill='toself',
            fillcolor='rgba(255, 165, 0, 0.2)',  # Light orange fill
            line=dict(color='rgba(255, 165, 0, 0)'),  # No outline
            showlegend=False,
            name='Confidence Interval'
        ))

        # Update layout
        fig.update_layout(
            title=f"Forecast for Next {forecast_steps2} Weeks",
            xaxis_title="Date",
            yaxis_title="Quantité nette",
            legend_title="Legend"
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

        # Display forecasted data table for the selected weeks
        st.write("### Forecasted Weekly Sales Data")
        forecast_table = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(forecast_steps2)
        forecast_table.columns = ['Date', 'Forecasted Sales', 'Lower CI', 'Upper CI']
        forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m-%d')  # Format dates
        st.write(forecast_table)

        # Download button for the Prophet forecasted va

    else:
        st.info("Please upload a dataset on the Home page.")
elif page == "Promotion Analysis":
    if data2 is not None:
        # Convert Date to datetime and create a Promotion column (True if discount is applied)
        data2['Date'] = pd.to_datetime(data2['Date'], errors='coerce')
        data2['Promotion'] = data2['Réductions'] < 0  # Assuming reductions are negative for promotions

        # Group by month to get total sales and revenue during promotion and non-promotion periods
        promotion_impact = data2.groupby([data2['Date'].dt.to_period("M"), 'Promotion']).sum(numeric_only=True)[['Ventes totales']]
        promotion_impact = promotion_impact.unstack().fillna(0)
        promotion_impact.columns = ['No Promotion', 'With Promotion']

        # Plot sales with and without promotions
        fig = px.bar(
            promotion_impact, 
            x=promotion_impact.index.to_timestamp(), 
            y=['No Promotion', 'With Promotion'],
            title='Sales and Revenue During Promotion vs Non-Promotion Periods',
            color_discrete_sequence=["blue", "red"]  # Setting colors
        )
        st.plotly_chart(fig)

        # Quantity Sold with and without Promotion
        quantity_impact = data2.groupby([data2['Date'].dt.to_period("M"), 'Promotion']).sum(numeric_only=True)[['Quantité nette']]
        quantity_impact = quantity_impact.unstack().fillna(0)
        quantity_impact.columns = ['No Promotion', 'With Promotion']
        fig2 = px.line(
            quantity_impact, 
            x=quantity_impact.index.to_timestamp(), 
            y=['No Promotion', 'With Promotion'],
            title='Quantity Sold with vs Without Promotions Over Time',
            color_discrete_sequence=["blue", "red"]  # Setting colors
        )
        st.plotly_chart(fig2)

        # Average Quantity Sold per Product with and without Promotion
        data2['Produit'] = data2['Produit'].str.split('|').str[0].str.strip()
        # Assuming 'Produit' is the column containing the full product names
        data2['Produit'] = data2['Produit'].str.split('-').str[0].str.strip().str.lower()

        # Now, 'Produit' will contain only the first word, e.g., "Phoenix"

        avg_quantity_per_product = data2.groupby(['Produit', 'Promotion']).mean(numeric_only=True)['Quantité nette'].unstack().fillna(0)
        avg_quantity_per_product.columns = ['No Promotion', 'With Promotion']
        fig3 = px.bar(
            avg_quantity_per_product, 
            x=avg_quantity_per_product.index, 
            y=['No Promotion', 'With Promotion'],
            title='Average Quantity Sold per Product: With vs Without Promotions',
            color_discrete_sequence=["blue", "red"]  # Setting colors
        )
        st.plotly_chart(fig3)
            # Promotion Impact on Average Order Value (AOV)
    # Timing of Purchases During Promotional Periods
        data2['Hour'] = data2['Date'].dt.hour
        hour_promo_sales = data2[data2['Promotion']].groupby('Hour').size()
        hour_all_sales = data2.groupby('Hour').size()
        promo_purchase_ratio = (hour_promo_sales / hour_all_sales).fillna(0) * 100
        fig8 = px.bar(promo_purchase_ratio, x=promo_purchase_ratio.index, y=promo_purchase_ratio,
                  title='Promotion Purchase Timing (Hourly)', labels={'y': 'Purchase Ratio (%)'})
        st.plotly_chart(fig8)
        # Assuming data2 is your DataFrame with 'Date' and 'Promotion' columns
        data2['Date'] = pd.to_datetime(data2['Date'], errors='coerce')  # Convert Date to datetime
        data2['Day'] = data2['Date'].dt.isocalendar().day  # Extract day of the week (1=Monday, ..., 7=Sunday)

        # Group by day of the week to get the count of sales during promotional and non-promotional periods
        day_promo_sales = data2[data2['Promotion']].groupby('Day').size()
        day_all_sales = data2.groupby('Day').size()

        # Calculate the ratio of promotional purchases to all purchases for each day
        promo_purchase_ratio = (day_promo_sales / day_all_sales).fillna(0) * 100

        # Map day numbers to names for better readability in the chart
        day_name_mapping = {1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday', 7: 'Sunday'}
        promo_purchase_ratio.index = promo_purchase_ratio.index.map(day_name_mapping)

        # Plot the daily promotional purchase ratio
        fig9 = px.bar(
            promo_purchase_ratio, 
            x=promo_purchase_ratio.index, 
            y=promo_purchase_ratio.values,
            title='Promotion Purchase Timing (daily)', 
            labels={'x': 'Day of the Week', 'y': 'Purchase Ratio (%)'}
        )

        # Display the chart in Streamlit
        st.plotly_chart(fig9)
    else:
        st.write("Please upload a dataset to proceed.")
# If data is missing, display a message on each page