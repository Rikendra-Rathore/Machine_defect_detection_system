# AI-Powered Textile Machine Defect Detection

## Overview

This project is an **AI-powered industrial monitoring system** designed to detect potential machine defects in textile manufacturing. The system uses **machine learning** to analyze machine parameters and predict whether the machine condition is **Normal or Defective**.

It also provides a **modern industrial dashboard** to visualize predictions, monitor machine metrics, and track prediction history.

The goal of this project is to demonstrate how **AI can be used in manufacturing environments for predictive maintenance and defect detection**.

---

## Features

* Machine defect prediction using Machine Learning
* Industrial monitoring dashboard
* Live machine parameter simulation (Temperature, Speed, Torque)
* Batch prediction using CSV upload
* Prediction history stored in SQLite database
* Interactive charts for monitoring machine trends
* Modern dashboard UI inspired by real industrial control systems

---

## Technologies Used

**Programming Language**

* Python

**Machine Learning**

* Scikit-learn
* Random Forest Classifier
* StandardScaler

**Backend**

* Flask

**Frontend**

* HTML
* CSS
* Bootstrap
* Chart.js

**Database**

* SQLite

---

## System Architecture

Machine Sensor Data → Data Preprocessing → Machine Learning Model → Prediction → Dashboard Visualization

The system takes machine parameters as input:

* Temperature
* Speed
* Torque
* Yarn Type

The trained ML model predicts whether the machine condition is **Normal or Defective**.

---

## Project Structure

app/

app.py

train_model.py

model.pkl

scaler.pkl

predictions.db

templates/

index.html

README.md

---

## Dashboard Features

Industrial AI Dashboard showing:

* Machine Health Indicators
* Defect Risk Prediction
* Live Sensor Monitoring
* Machine Trend Charts
* Batch Prediction Upload
* Prediction History Table

---

## Use Case

This system can be applied in **textile manufacturing plants** to monitor machine conditions and detect potential defects before they affect production.

Benefits include:

* Reduced machine downtime
* Early defect detection
* Improved product quality
* Better production monitoring

---

## Future Improvements

* Integration with real industrial IoT sensors
* Predictive maintenance using time-series models
* Deep learning for anomaly detection
* Cloud deployment for real-time monitoring
* Power BI or advanced analytics integration

---
