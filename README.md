
# Physics inspired LSTM anomaly detector for time-series data

## Project Summary
This project demonstrates working principles of anomaly detection in a physics inspired LSTM network for a time series data. 

## Dataset
- Simulated simple harmonic oscillator data
- Clean simple harmonic oscillator over limited time window for training
- Longer signal with the same frequency with noise and known anomalies for testing

## Methodology
- Choice of network: LSTM for better preservation of features in time series data
- Implmentation of physics loss function as SHM ODE
- Tune F1 score to achive optimal anomaly detection performance
- Comparison of standard LSTM and PI-LSTM networks

## Results
- Visulization of loss curves for both PI-LSTM and standard LSTM networks
- Visualization of predicted vs actual anomalies
- Comparison of standard LSTM and PI-LSTM network performances

![Anomaly detection performance](results/detected_anomalies_comparison.png)

## How to Run
1. Clone repo  
2. Create environment: `pip install -r requirements.txt`  
3. Run `main.py` for detailed analysis and model training. 

## Future Work
- Implement uncertainty quantification.
- Implement frequency estimation
- Inject more realistic signals with drift, glitches etc. 

## Technologies
Python, pytorch, pandas, scikit-learn, matplotlib

---

Feel free to reach out or check my portfolio: [suchitakulkarni.github.io](https://suchitakulkarni.github.io)
