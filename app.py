import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



# Function to train machine learning models
def train_models(train_data):
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(np.array(train_data["Days Since"]).reshape(-1, 1), np.array(train_data["Confirmed"]).reshape(-1, 1))

    # Polynomial Regression
    poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly_reg.fit(np.array(train_data["Days Since"]).reshape(-1, 1), np.array(train_data["Confirmed"]).reshape(-1, 1))

    # Support Vector Machine
    svm = SVR(C=1, degree=6, kernel='poly', epsilon=0.01)
    svm.fit(np.array(train_data["Days Since"]).reshape(-1, 1), np.array(train_data["Confirmed"]).reshape(-1, 1))

    # Random Forest Regression
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    random_forest.fit(np.array(train_data["Days Since"]).reshape(-1, 1), np.array(train_data["Confirmed"]).reshape(-1, 1))

    # Decision Tree Regression
    decision_tree = DecisionTreeRegressor(random_state=42)
    decision_tree.fit(np.array(train_data["Days Since"]).reshape(-1, 1), np.array(train_data["Confirmed"]).reshape(-1, 1))

    return lin_reg, poly_reg, svm, random_forest, decision_tree

    # Function to make predictions
def make_predictions(models, prediction_date, train_data):
    lin_reg, svm, random_forest, decision_tree, poly_reg = models

    # Convert Timedelta to number of days
    days_since_start = prediction_date.days

    # Linear Regression predictions (with check for extrapolation)
    lr_pred = lin_reg.predict(np.array(days_since_start).reshape(-1, 1))[0][0]

    # Polynomial Regression predictions (with check for extrapolation)
    poly_pred = poly_reg.predict(np.array(days_since_start).reshape(-1, 1))[0]

    # Support Vector Machine predictions (with check for extrapolation)
    svm_pred = svm.predict(np.array(days_since_start).reshape(-1, 1))[0]

    # Decision Tree predictions (with check for extrapolation)
    dt_pred = decision_tree.predict(np.array(days_since_start).reshape(-1, 1))[0]

    # Random Forest predictions (with check for extrapolation)
    rf_pred = random_forest.predict(np.array(days_since_start).reshape(-1, 1))[0]

    # Random Forest predictions (with check for extrapolation)
    if days_since_start <= train_data["Days Since"].max():
        rf_pred = random_forest.predict(np.array(days_since_start).reshape(-1, 1))[0]
    else:
        rf_pred = None

    return lr_pred, poly_pred, svm_pred, dt_pred, rf_pred

# Create the models
lin_reg = LinearRegression()
poly_reg = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
svm = SVR(C=1, degree=6, kernel='poly', epsilon=0.01)
random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
decision_tree = DecisionTreeRegressor(random_state=42)

    # Function to create Streamlit app
def main():
    st.title("COVID-19 Prediction App")

    # Read COVID-19 data
    covid = pd.read_csv("covid_19_data.csv")
    covid["ObservationDate"] = pd.to_datetime(covid["ObservationDate"])
    datewise = covid.groupby(["ObservationDate"]).agg({"Confirmed": 'sum', "Recovered": 'sum', "Deaths": 'sum'})
    datewise["Days Since"] = datewise.index - datewise.index[0]
    datewise["Days Since"] = datewise["Days Since"].dt.days

    # Train models
    train_ml = datewise.iloc[:int(datewise.shape[0] * 0.95)]
    # Store the models in a list
    models = [lin_reg, poly_reg, svm, random_forest, decision_tree]

    # Sidebar for selecting the prediction date
    st.sidebar.subheader("Select Prediction Date:")
    prediction_date = st.sidebar.date_input("Choose a date:", pd.to_datetime("today") + timedelta(days=1))

    # Convert prediction_date to Timestamp
    prediction_date = pd.to_datetime(prediction_date)

    # Ensure that the models are fitted before making predictions
    for model in models:
        model.fit(np.array(train_ml["Days Since"]).reshape(-1, 1), np.array(train_ml["Confirmed"]).reshape(-1, 1))

    # Make predictions
    lr_pred, poly_pred, svm_pred, dt_pred, rf_pred = make_predictions(models, prediction_date - datewise.index[0], train_ml)

    # Display predictions
    st.subheader("Predicted Cases for {}".format(prediction_date))
    if lr_pred is not None:
        st.write("Linear Regression Prediction: {}".format(int(lr_pred)))
    else:
        st.write("Linear Regression Prediction: N/A (Extrapolation)")
    if poly_pred is not None:
        st.write("Polynomial Regression Prediction: {}".format(int(poly_pred)))
    else:
        st.write("Polynomial Regression Prediction: N/A (Extrapolation)")
    if svm_pred is not None:
        st.write("Support Vector Machine Prediction: {}".format(int(svm_pred)))
    else:
        st.write("Support Vector Machine Prediction: N/A (Extrapolation)")
    if dt_pred is not None:
        st.write("Decision Tree Prediction: {}".format(int(dt_pred)))
    else:
        st.write("Decision Tree Prediction: N/A (Extrapolation)")
    if rf_pred is not None:
        st.write("Random Forest Prediction: {}".format(int(rf_pred)))
    else:
        st.write("Random Forest Prediction: N/A (Extrapolation)")

    # Plotting Historical and Predicted Data
    fig_combined = go.Figure()

    # Historical data
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=datewise["Confirmed"], mode='lines', name='Historical Data'))

    # Linear Regression predictions
    lr_pred_values = [lin_reg.predict(np.array(date).reshape(-1, 1))[0][0] for date in datewise["Days Since"]]
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=lr_pred_values, mode='lines', name='Linear Regression Predictions', line=dict(dash='dot', color='red')))

    # Polynomial Regression predictions
    poly_pred_values = [poly_reg.predict(np.array(date).reshape(-1, 1))[0][0] for date in datewise["Days Since"]]
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=poly_pred_values, mode='lines', name='Polynomial Regression Predictions', line=dict(dash='dot', color='orange')))

    # Support Vector Machine predictions
    svm_pred_values = [svm.predict(np.array(date).reshape(-1, 1))[0] for date in datewise["Days Since"]]
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=svm_pred_values, mode='lines', name='SVM Predictions', line=dict(dash='dot', color='green')))
    # Decision Tree predictions
    dt_pred_values = [decision_tree.predict(np.array(date).reshape(-1, 1))[0] for date in datewise["Days Since"]]
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=dt_pred_values, mode='lines', name='Decision Tree Predictions', line=dict(dash='dot', color='blue')))

    # Random Forest predictions
    rf_pred_values = [random_forest.predict(np.array(date).reshape(-1, 1))[0] for date in datewise["Days Since"]]
    fig_combined.add_trace(go.Scatter(x=datewise.index, y=rf_pred_values, mode='lines', name='Random Forest Predictions', line=dict(dash='dot', color='purple')))

    # Updating layout
    fig_combined.update_layout(title="Historical and Predicted Data for Different Models", xaxis_title="Date", yaxis_title="Confirmed Cases", showlegend=True)

    # Displaying the plot in Streamlit
    st.plotly_chart(fig_combined)
    

if __name__ == "__main__":
    main()