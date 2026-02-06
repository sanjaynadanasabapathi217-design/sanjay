# Advanced Time Series Forecasting with Attention-Based Neural Networks

This project implements and compares two deep learning architectures for multivariate time series forecasting: a baseline **Long Short-Term Memory (LSTM)** network and an **Attention-based Transformer** model. The goal is to predict the future values of a primary variable based on its historical patterns and correlated exogenous features.

---

## ## Project Overview

The repository contains a complete pipeline for time series analysis, including:

* 
**Synthetic Data Generation**: Creation of a complex 5-feature multivariate dataset incorporating trends, seasonality, and noise.


* 
**Deep Learning Architectures**: Implementation of a standard recurrent model (LSTM) and a modern attention-based model (Transformer).


* 
**Performance Benchmarking**: Comparative analysis using standard regression metrics: RMSE, MAE, and MAPE.



---

## ## Technical Architecture

### ### 1. Data Pipeline

The system generates a dataset of 1,500 steps with the following characteristics:

* 
**Features**: Includes linear trends, sinusoidal seasonality ( and -step periods), and correlated features.


* 
**Preprocessing**: Data is scaled using `MinMaxScaler` and split into Training (70%), Validation (15%), and Test (15%) sets.


* 
**Sliding Window**: Uses a sequence length of 30 time steps to predict the next value of the primary feature.



### ### 2. Model Implementations

| Model | Description | Key Components |
| --- | --- | --- |
| **LSTM** | Recurrent baseline designed to capture long-term dependencies.

 | LSTM layer (64 hidden units), Fully Connected output layer.

 |
| **Transformer** | Attention-based model that processes sequences in parallel.

 | Linear embedding, Multi-head attention (4 heads), Encoder layers.

 |

---

## ## Results & Performance

Based on the experimental run, the **LSTM model** outperformed the Transformer on this specific dataset:

| Metric | LSTM | Transformer |
| --- | --- | --- |
| **RMSE** | 0.0265 | 0.1035 |
| **MAE** | 0.0221 | 0.0939 |
| **MAPE** | 2.49% | 10.38% |

> 
> **Note:** The LSTM demonstrated smoother convergence, reaching a final training loss of approximately 0.0007, while the Transformer showed more volatility in its loss curves during the 20-epoch training cycle.
> 
> 

---

## ## Requirements

* 
**Core**: `Python 3.x` 


* 
**Deep Learning**: `PyTorch` 


* 
**Data Processing**: `NumPy`, `Pandas`, `Scikit-Learn` 
