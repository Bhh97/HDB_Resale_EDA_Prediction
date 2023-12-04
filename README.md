# HDB Resale Price Prediction

## Overview
This repository contains the code for a predictive model and web application for estimating the resale prices of HDB (Housing & Development Board) flats in Singapore. The model, trained on HDB resale transaction records from 2017 to 2020 and is downloaded from https://www.kaggle.com/datasets/teyang/singapore-hdb-flat-resale-prices-19902020.

## Project Description
The goal of this project is to offer insights into the HDB resale market and provide a tool for predicting resale prices based on various factors such as location, flat type, floor area, and more.

### Exploratory Data Analysis (EDA)
The project includes comprehensive EDA, where the dataset undergoes:
- **Data Cleaning**: Preparation and cleaning of the dataset for analysis.
- **Data Visualization**: Exploration of key variables and their relationships, focusing on resale prices, floor areas, flat types, and storey ranges.
- **Statistical Summaries**: Providing summary statistics to understand data distributions and variations.

### Predictive Modeling
The predictive model is constructed using the `fastai` library's `TabularPandas` and `TabularLearner`. The process involves:
- **Data Preprocessing**: Implementing necessary transformations and encoding for categorical variables.
- **Model Training**: Utilizing a neural network to predict resale prices, with a time-based validation set split.
- **Learning Rate Finder**: Optimizing the learning rate for better model performance.
- **Model Evaluation**: Assessing the model using RMSE and visual comparisons of predicted vs. actual prices.

### Web Application
The model is deployed as a web application using Gradio, allowing users to input relevant details and receive HDB resale price predictions. You can find the model at https://huggingface.co/spaces/bhh97/HDB-Resale-Prediction
