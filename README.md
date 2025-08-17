# Anomaly Detection in Network Traffic using a Transformer Autoencoder

## 📜 Project Overview

This project implements a **Transformer-based autoencoder** in PyTorch to detect anomalies in network traffic, specifically focusing on identifying Distributed Denial of Service (DDoS) attacks. The model follows an **unsupervised learning** approach, meaning it learns the patterns of "normal" (benign) network behavior exclusively. When confronted with traffic that deviates from this learned normality, the model fails to reconstruct it accurately, resulting in a high reconstruction error. This high error serves as a flag for anomalous activity.

A key feature of this project is its **Explainable AI (XAI)** module, which generates attention heatmaps to provide human-interpretable insights into *why* the model flagged a particular sequence as an anomaly.

-----

## 📁 Project Structure

```
diproject/
│
├── data/                 # -> Place raw CIC-DDoS2019 .csv files here.
│   └── processed/        # -> Stores the preprocessed, sequenced data (.npy file).
│
├── notebooks/            # -> Jupyter notebooks for exploration and prototyping.
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_prototyping.ipynb
│   └── 03_results_visualization.ipynb
│
├── results/              # -> Stores output files like reports and attention heatmaps.
│
├── saved_models/         # -> Stores the trained model weights (.pth file).
│
├── src/                  # -> Main source code for the project.
│   ├── config.py         # -> Central configuration for paths and hyperparameters.
│   ├── data_loader.py    # -> Handles all data loading and preprocessing.
│   ├── model.py          # -> Defines the TransformerAutoencoder architecture.
│   ├── train.py          # -> Script for training the model.
│   ├── evaluate.py       # -> Script for evaluating the model and finding the threshold.
│   ├── explainability.py # -> Contains the XAI logic for generating attention maps.
│   └── utils.py          # -> Helper functions (e.g., for creating sequences).
│
├── requirements.txt      # -> All Python dependencies.
└── README.md             # -> This file.
```

-----

## 📊 Dataset

  - **Name:** CIC-DDoS2019
  - **Source:** Canadian Institute for Cybersecurity
  - **Description:** A large-scale dataset containing a realistic mix of modern benign network traffic and a wide variety of DDoS attack types. This makes it an ideal benchmark for developing robust intrusion detection systems.
  - **Link:** [CIC-DDoS2019 Dataset](https://www.unb.ca/cic/datasets/ddos-2019.html)

-----

## ⚙️ Setup and Installation

Follow these steps to set up your environment and prepare the project for execution.

### Prerequisites

  - Git
  - Python 3.9+

### Step 1: Clone the Repository

```bash
git clone https://github.com/geetikak13/dlproject    
```

### Step 2: Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Mac/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the CIC-DDoS2019 dataset from the link provided above. You will receive a set of `.csv` files. **Place all of these `.csv` files inside the `data/` directory** at the root of the project.

-----

## 🚀 Running the Project

After setting up the environment, run the scripts in the following order.

### Step 1: Preprocess the Data

This script will load the raw CSVs, clean the data, perform feature engineering, isolate benign traffic, and save the final processed sequences to `data/processed/`.

```bash
python src/data_loader.py
```

### Step 2: Train the Model

This script will train the Transformer Autoencoder on the processed benign data and save the model weights to `saved_models/`.

```bash
python src/train.py
```

### Step 3: Evaluate the Model & Generate XAI Results

This script evaluates the model's performance. It first calculates an anomaly threshold, then classifies a simulated test set. Upon finding the first anomaly, it will automatically generate and save an attention heatmap in the `results/` directory.

```bash
python src/evaluate.py
```

-----

## 📓 Jupyter Notebooks for Exploration

The `notebooks/` directory contains Jupyter notebooks for interactive analysis and prototyping. To run them, start the Jupyter server from the project's root directory:

```bash
jupyter notebook
```

  - **`01_data_exploration.ipynb`**: Use this notebook to perform Exploratory Data Analysis (EDA) on the raw dataset. It contains the code to generate the visualizations of class imbalance, feature distributions, and other insights that informed the project's direction.

  - **`02_model_prototyping.ipynb`**: This notebook is for interactively building and debugging the `TransformerAutoencoder`. It allows you to train the model on a small sample of data for a few epochs and visualize a sample reconstruction to ensure the architecture is working correctly before running the full training script.

  - **`03_results_visualization.ipynb`**: After training the full model, use this notebook to dive deeper into the results. You can visualize the distribution of reconstruction errors to fine-tune the anomaly threshold and interactively run the XAI module on different simulated anomalies.

-----

## 📈 Results and Explainability

The primary outputs of the project are:

  - A **trained model** (`transformer_autoencoder.pth`) capable of detecting network anomalies.
  - A **classification report** showing the model's performance (precision, recall, F1-score).
  - An **attention heatmap** (`attention_heatmap.png`) that provides a visual explanation for an anomaly detection.
