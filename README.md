# SolarIQ Streamlit App

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload the two Plant 1 CSV files from Kaggle:
- Plant_1_Generation_Data.csv
- Plant_1_Weather_Sensor_Data.csv

Dataset: https://www.kaggle.com/datasets/anikannal/solar-power-generation-data

## Deploy publicly
1. Upload `app.py` and `requirements.txt` to a GitHub repository.
2. Visit https://share.streamlit.io/
3. Connect GitHub and select the repository.
4. Set the main file path to `app.py`.
5. Click Deploy.
6. Open the app URL and upload the two CSV files.

Scope notes: cellular load is simulated. The 8-bit test quantizes input features, not model weights. No physical VLSI chip or FPGA implementation is claimed.
