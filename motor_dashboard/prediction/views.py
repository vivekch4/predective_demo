

# Create your views here.
from django.shortcuts import render
from django.http import JsonResponse
import joblib
import pandas as pd
import json
import plotly.graph_objs as go
import plotly
from datetime import datetime

# Load model
model = joblib.load("best_model.pkl")

# In-memory history (in production use DB)
historical_predictions = []

def create_charts():
    if len(historical_predictions) < 2:
        return None, None, None

    timestamps = [entry['timestamp'] for entry in historical_predictions]
    temps = [entry['temperature'] for entry in historical_predictions]
    vibs = [entry['vibration'] for entry in historical_predictions]
    rpms = [entry['rpm'] for entry in historical_predictions]
    risks = [entry['risk_percentage'] for entry in historical_predictions]

    # Risk Trend Chart
    risk_chart = go.Figure()
    risk_chart.add_trace(go.Scatter(
        x=timestamps,
        y=risks,
        mode='lines+markers',
        name='Risk %',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    risk_chart.update_layout(
        title='Failure Risk Trend',
        xaxis_title='Prediction Time',
        yaxis_title='Risk Percentage (%)',
        template='plotly_white',
        height=400
    )

    # Parameter Trends
    param_chart = go.Figure()
    param_chart.add_trace(go.Scatter(x=timestamps, y=temps, name='Temperature', line=dict(color='orange', width=2)))
    param_chart.add_trace(go.Scatter(x=timestamps, y=vibs, name='Vibration', line=dict(color='blue', width=2), yaxis='y2'))
    param_chart.add_trace(go.Scatter(x=timestamps, y=rpms, name='RPM', line=dict(color='green', width=2), yaxis='y3'))
    param_chart.update_layout(
        title='Parameter Trends',
        xaxis_title='Prediction Time',
        yaxis=dict(title='Temperature', side='left', color='orange'),
        yaxis2=dict(title='Vibration', side='right', overlaying='y', color='blue'),
        yaxis3=dict(title='RPM', side='right', overlaying='y', anchor='free', position=0.95, color='green'),
        template='plotly_white',
        height=400,
        legend=dict(x=0.01, y=0.99)
    )

    # Gauge Chart
    current_risk = risks[-1] if risks else 0
    gauge_chart = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_risk,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Current Risk Level (%)"},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred" if current_risk > 70 else "orange" if current_risk > 30 else "green"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "lightcoral"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}
        }
    ))
    gauge_chart.update_layout(height=400)

    return (
        json.dumps(risk_chart, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(param_chart, cls=plotly.utils.PlotlyJSONEncoder),
        json.dumps(gauge_chart, cls=plotly.utils.PlotlyJSONEncoder),
    )

def index(request):
    prediction_text = ""
    risk_percentage = 0
    risk_chart_json = None
    param_chart_json = None
    gauge_chart_json = None

    if request.method == "POST":
        try:
            temp = float(request.POST["temperature"])
            vib = float(request.POST["vibration"])
            rpm = float(request.POST["rpm"])

            input_data = pd.DataFrame([[temp, vib, rpm]], columns=["temperature", "vibration", "rpm"])
            prediction = model.predict(input_data)[0]

            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_data)[0]
                risk_percentage = round(probabilities[1] * 100, 2)
            else:
                risk_percentage = 90.0 if prediction == 1 else 10.0

            historical_predictions.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'temperature': temp,
                'vibration': vib,
                'rpm': rpm,
                'prediction': prediction,
                'risk_percentage': risk_percentage
            })

            if len(historical_predictions) > 50:
                historical_predictions.pop(0)

            prediction_text = f"⚠ HIGH FAILURE RISK - {risk_percentage}%" if prediction == 1 else f"✅ LOW FAILURE RISK - {risk_percentage}%"

        except Exception as e:
            prediction_text = f"Error: {e}"
            risk_percentage = 0

    if len(historical_predictions) >= 2:
        risk_chart_json, param_chart_json, gauge_chart_json = create_charts()

    return render(request, "index.html", {
        "prediction": prediction_text,
        "risk_percentage": risk_percentage,
        "risk_chart": risk_chart_json,
        "param_chart": param_chart_json,
        "gauge_chart": gauge_chart_json,
        "total_predictions": len(historical_predictions),
    })

def clear_history(request):
    global historical_predictions
    historical_predictions = []
    return JsonResponse({"status": "success", "message": "History cleared"})
