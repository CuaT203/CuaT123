# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import joblib
import math
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Business Analytics Dashboard", layout="wide")

# Cache data loading
@st.cache_data
def load_data(file, parse_dates=None):
    try:
        df = pd.read_csv(file, parse_dates=parse_dates)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

# Cache model training
@st.cache_resource
def train_linear_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

@st.cache_resource
def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# Cache route optimization
@st.cache_data
def optimize_routes(customers_df, depot=(10, 10), vehicle_capacity=15, num_vehicles=3):
    locations = [(depot[0], depot[1])] + [(row.x, row.y) for _, row in customers_df.iterrows()]
    def euclid(a, b):
        return int(math.hypot(a[0] - b[0], a[1] - b[1]))
    n = len(locations)
    dist_matrix = [[euclid(locations[i], locations[j]) for j in range(n)] for i in range(n)]
    demands = [0] + customers_df['demand'].tolist()
    
    manager = pywrapcp.RoutingIndexManager(len(dist_matrix), num_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)
    
    def distance_callback(from_index, to_index):
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]
    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)
    
    def demand_callback(from_index):
        return demands[manager.IndexToNode(from_index)]
    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(demand_cb_idx, 0, [vehicle_capacity] * num_vehicles, True, 'Capacity')
    
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.time_limit.FromSeconds(10)
    solution = routing.SolveWithParameters(search_params)
    
    routes = []
    for v in range(num_vehicles):
        idx = routing.Start(v)
        route = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        route.append(manager.IndexToNode(idx))
        routes.append(route)
    return routes

# Main app
st.title("Business Analytics Dashboard")
st.markdown("Upload data files and explore sales forecasts, IoT sensor analytics, and delivery route optimization.")

# Sidebar for file uploads and parameters
st.sidebar.header("Data and Settings")
sales_file = st.sidebar.file_uploader("Upload Sales History (CSV)", type="csv")
iot_file = st.sidebar.file_uploader("Upload IoT Data (CSV)", type="csv")
customers_file = st.sidebar.file_uploader("Upload Customers Data (CSV)", type="csv")
forecast_days = st.sidebar.slider("Forecast Period (Days)", 10, 90, 30)

# Load data
sales_df = None
iot_daily = None
customers_df = None

if sales_file:
    sales_df = load_data(sales_file, parse_dates=['date'])
if iot_file:
    iot_daily = load_data(iot_file, parse_dates=['date'])
if customers_file:
    customers_df = load_data(customers_file)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Sales Forecasting", "IoT Sensor Analytics", "Route Optimization"])

# Sales Forecasting Tab
with tab1:
    st.header("Sales Forecasting")
    if sales_df is not None:
        sales_df = sales_df.sort_values('date').set_index('date')
        sales_df['day_num'] = np.arange(len(sales_df))
        
        # Train Linear Regression
        X = sales_df[['day_num']]
        y = sales_df['sales']
        train_size = int(len(sales_df) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        lin_reg = train_linear_model(X_train, y_train)
        
        # Forecast
        last_day_num = sales_df['day_num'].iloc[-1]
        future_days = np.arange(last_day_num + 1, last_day_num + forecast_days + 1).reshape(-1, 1)
        future_forecast = lin_reg.predict(future_days)
        future_dates = pd.date_range(start=sales_df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({'date': future_dates, 'forecast_sales': future_forecast})
        
        # Metrics
        y_pred_test = lin_reg.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        st.write(f"**MAE**: {mae:.2f} | **RMSE**: {rmse:.2f}")
        
        # Chart.js Line Chart for Sales Forecast
        chart_data = {
            "labels": sales_df.index[-90:].strftime('%Y-%m-%d').tolist() + forecast_df['date'].strftime('%Y-%m-%d').tolist(),
            "datasets": [
                {
                    "label": "Historical Sales",
                    "data": sales_df['sales'][-90:].tolist() + [None] * forecast_days,
                    "borderColor": "#1f77b4",
                    "fill": False
                },
                {
                    "label": "Forecasted Sales",
                    "data": [None] * 90 + forecast_df['forecast_sales'].tolist(),
                    "borderColor": "#ff7f0e",
                    "fill": False
                }
            ]
        }
        st.write("### Sales Trend and Forecast")
        st.write("""
            ```chartjs
            {
                "type": "line",
                "data": {
                    "labels": """ + str(chart_data["labels"]) + """,
                    "datasets": """ + str(chart_data["datasets"]) + """
                },
                "options": {
                    "responsive": true,
                    "plugins": {
                        "legend": {"position": "top"},
                        "title": {"display": true, "text": "Sales History and Forecast"}
                    },
                    "scales": {
                        "x": {"title": {"display": true, "text": "Date"}},
                        "y": {"title": {"display": true, "text": "Sales"}}
                    }
                }
            }
