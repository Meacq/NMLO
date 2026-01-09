# From Noise Modeling to Layout Optimization: A Framework for Quantum Circuit Fidelity Enhancement with Machine Learning

This repository contains code and data for **high-fidelity quantum circuit layout selection** on real NISQ devices, with a focus on the **TianYan Quantum Computing Cloud Platform**.


## 🚀 Getting Started

### 1. Apply for Quantum Cloud Access

To run experiments on real quantum hardware:

1. Register an account on the **[TianYan Quantum Computing Cloud Platform](https://qc.zdxlz.com/laboratory/#/computerManage?lang=zh)**.
2. After approval, obtain your **API access key** from the platform dashboard.
3. Replace the placeholder key in `main.py`:

```python
login_key = "YOUR_API_KEY_HERE"  # ← Paste your key here
```

### 2. Repository Structure

```python
unzip qcis_and_execution_data.zip

├── NMLO-ML/                   # Core source code for data collection and layout optimization
├── qcis_and_execution_data/   # Raw experimental data (March 2025 – April 2025)
└── model/                     # Trained machine learning models
```

### 3. Install Dependencies
```python
pip install -r requirements.txt
```


### 4. Run Experiments

```python
python main.py -generate # Collect real-device execution data for training.
python main.py -test # Evaluate model performance on test data.

```
