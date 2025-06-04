# Evolutionary Game-Theoretic Grey Wolf Optimization (EGT-GWO) Project

This repository implements a novel **Evolutionary Game-Theoretic Grey Wolf Optimization (EGT-GWO)** framework for optimized edge computing and task offloading in **6G networks**.

## 🚀 Project Overview
The goal of this project is to enhance task allocation in **edge computing** environments using a hybrid optimization approach:
- **Grey Wolf Optimizer (GWO)**: A nature-inspired optimization technique based on the hunting behavior of grey wolves.
- **Evolutionary Game Theory (EGT)**: A mathematical framework for strategic decision-making in multi-agent systems.

This combination improves resource utilization, energy efficiency, and overall network performance.

## 📂 Directory Structure
```
EGT-GWO-Project-Final/
│── algorithms/          # Contains the algorithms to be compared
│── data/                # Datasets for testing and evaluation
│── main.py              # Main script to run the optimization
│── app.py               # Streamlit app for visualizing comparisons
│── README.md            # Documentation
│── requirements.txt     # Dependencies
```

## 🔍 Key Features
✔ **Optimized Task Offloading**: Efficiently assigns tasks to edge servers based on resource availability and cost factors.
✔ **Energy-Aware Optimization**: Reduces power consumption in 6G networks.
✔ **Scalability**: Designed to handle large-scale edge computing environments.
✔ **Hybrid AI Approach**: Combines metaheuristic and game-theoretic models for better decision-making.
✔ **Interactive Visualization**: Uses **Streamlit** to showcase algorithm comparisons dynamically.

## 🛠 Installation & Setup
Ensure you have **Python 3.x** installed. Clone the repository and install dependencies:
```sh
git clone https://github.com/jaGaban747/EGT-GWO-Project-Final.git
cd EGT-GWO-Project-Final
pip install -r requirements.txt
```

## ⚙️ Usage Instructions
### Running the Optimization
Run the main script to execute the optimization process:
```sh
python main.py
```
Modify parameters in `config.py` to fine-tune the optimization process.

### Running the Streamlit App
To visualize and compare the optimization algorithms, use **Streamlit**:
```sh
streamlit run app.py
```
This will launch a web-based dashboard displaying the performance metrics and results of different algorithms.

## 📊 Results & Evaluation
- Performance metrics such as **convergence rate, latency reduction, and energy efficiency** are analyzed.
- Results are stored in the `results/` directory.
- Comparisons can be visualized interactively via the **Streamlit** dashboard.

---
Developed by **Leon**.
