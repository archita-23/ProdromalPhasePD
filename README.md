# ProdromalPhasePD
🧠 Prodromal Phase Detection System
Early neurological risk awareness using anomaly detection

This project proposes a data-driven system to identify early abnormal behavioral patterns that may occur during the prodromal phase of neurological diseases such as Parkinson’s disease.
Instead of diagnosing disease, the system learns normal daily behavior and flags unusual deviations across multiple parameters.

🎯 Objective
To develop a non-diagnostic monitoring system that detects early abnormal patterns in longitudinal behavioral data using unsupervised machine learning.

❗ Problem
Neurological diseases are usually detected after visible symptoms appear, by which time significant neuronal damage has already occurred.
Early symptoms exist but are subtle and spread across different domains, making them difficult to recognize manually.

⚙️ Methodology
Collect daily behavioral data (sleep, mood, activity, fatigue)

Preprocess and normalize data

Extract time-series features

Train Isolation+Random Forest anomaly detection model

Detect multi-parameter deviations

Visualize risk windows

🤖 Technologies Used
Python

Pandas, NumPy

Scikit-learn (Isolation Forest)

Matplotlib

Time-series analysis

📊 Output
The system identifies sustained abnormal behavioral trends and marks potential risk windows without providing medical diagnosis.

🚀 How to Run
Install libraries:
nginx
Copy code
pip install numpy pandas matplotlib scikit-learn
Run:
nginx
Copy code
python prodromal.py

📁 Project Structure

prodromal-phase-detection/
│── prodromal.py
│── README.md
│── output_graph.png
│── report.pdf
│── ppt.pptx
⚠️ Disclaimer
This project is for academic and research purposes only.
It does not diagnose or predict any medical condition and should not replace professional medical advice.

👤 Author
Archita Gupta
Course: Project Based Learning
Institution: Manipal University Jaipur

🌟 Future Work
Integrate wearable sensor data

Multi-patient dataset analysis

Real-time monitoring dashboard
