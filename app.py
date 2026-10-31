import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="SolarIQ", page_icon="☀️", layout="wide")
st.title("☀️ SolarIQ")
st.subheader("AI-Powered Solar Forecasting & Intelligent Energy Management")
st.caption("Solar forecasting + simulated cellular energy management + VLSI-inspired edge analysis.")

with st.expander("What this prototype does"):
    st.write("It predicts the next 15-minute solar output from historical generation and weather data, compares it with a simulated cellular base-station load, and tests an 8-bit input-quantization experiment. Cellular load is simulated, and no physical VLSI chip or FPGA is claimed.")

st.sidebar.header("Simulation settings")
pv_capacity = st.sidebar.number_input("Hypothetical PV capacity (kW)", 0.5, 100.0, 5.0, 0.5)
base_load = st.sidebar.number_input("Base-station baseline load (kW)", 0.1, 20.0, 1.2, 0.1)

st.markdown("### 1. Upload your dataset")
col1, col2 = st.columns(2)
with col1:
    gen_file = st.file_uploader("Plant 1 Generation CSV", type="csv")
with col2:
    weather_file = st.file_uploader("Plant 1 Weather Sensor CSV", type="csv")

if not gen_file or not weather_file:
    st.info("Upload both CSV files from the Kaggle Solar Power Generation Data dataset.")
    st.markdown("[Download dataset from Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)")
    st.stop()

@st.cache_data(show_spinner=False)
def prepare(gen_bytes, weather_bytes):
    g = pd.read_csv(gen_bytes)
    w = pd.read_csv(weather_bytes)
    reqg = {"DATE_TIME", "AC_POWER"}
    reqw = {"DATE_TIME", "IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"}
    if not reqg.issubset(g.columns):
        raise ValueError(f"Generation file missing: {reqg - set(g.columns)}")
    if not reqw.issubset(w.columns):
        raise ValueError(f"Weather file missing: {reqw - set(w.columns)}")
    g["DATE_TIME"] = pd.to_datetime(g["DATE_TIME"], errors="coerce", dayfirst=True)
    w["DATE_TIME"] = pd.to_datetime(w["DATE_TIME"], errors="coerce", dayfirst=True)
    g["AC_POWER"] = pd.to_numeric(g["AC_POWER"], errors="coerce")
    weather_cols = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE"]
    for c in weather_cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    plant = g.dropna(subset=["DATE_TIME", "AC_POWER"]).groupby("DATE_TIME", as_index=False)["AC_POWER"].sum()
    wc = w.dropna(subset=["DATE_TIME"]).groupby("DATE_TIME", as_index=False)[weather_cols].mean()
    d = plant.merge(wc, on="DATE_TIME", how="inner")
    d = d.replace([np.inf, -np.inf], np.nan).dropna().sort_values("DATE_TIME").drop_duplicates("DATE_TIME").reset_index(drop=True)
    d = d[d["AC_POWER"] >= 0].copy()
    hour = d["DATE_TIME"].dt.hour + d["DATE_TIME"].dt.minute / 60
    day = d["DATE_TIME"].dt.dayofweek
    d["hour_sin"] = np.sin(2*np.pi*hour/24)
    d["hour_cos"] = np.cos(2*np.pi*hour/24)
    d["day_sin"] = np.sin(2*np.pi*day/7)
    d["day_cos"] = np.cos(2*np.pi*day/7)
    d["NEXT_TIME"] = d["DATE_TIME"].shift(-1)
    d["NEXT_AC_POWER"] = d["AC_POWER"].shift(-1)
    valid = (d["NEXT_TIME"] - d["DATE_TIME"]) == pd.Timedelta(minutes=15)
    d.loc[~valid, "NEXT_AC_POWER"] = np.nan
    features = ["IRRADIATION", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE",
                "hour_sin", "hour_cos", "day_sin", "day_cos"]
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=features+["NEXT_AC_POWER"]).reset_index(drop=True)
    return d, features

try:
    df, FEATURES = prepare(gen_file.getvalue(), weather_file.getvalue())
except Exception as e:
    st.error(f"Dataset preparation failed: {e}")
    st.stop()

if len(df) < 100:
    st.error("Too few valid records. Check that you uploaded the Plant 1 CSV files.")
    st.stop()

st.success(f"Prepared {len(df):,} usable forecasting records.")
a,b,c = st.columns(3)
a.metric("Records", f"{len(df):,}")
b.metric("From", df["DATE_TIME"].min().strftime("%d %b %Y"))
c.metric("To", df["NEXT_TIME"].max().strftime("%d %b %Y"))

with st.expander("Preview cleaned data"):
    st.dataframe(df.head(10), use_container_width=True)

fig, ax = plt.subplots(figsize=(11,3))
ax.plot(df["DATE_TIME"], df["AC_POWER"], linewidth=.8)
ax.set_title("Historical plant-level AC power")
ax.set_ylabel("Power (kW)")
ax.grid(alpha=.2)
st.pyplot(fig)
plt.close(fig)

if st.button("Train models and run SolarIQ", type="primary", use_container_width=True):
    cut = int(len(df)*.8)
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    Xtr, ytr = train[FEATURES], train["NEXT_AC_POWER"]
    Xte, yte = test[FEATURES], test["NEXT_AC_POWER"]
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=16, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Gradient Boosting": HistGradientBoostingRegressor(max_iter=150, learning_rate=.08, max_leaf_nodes=15, random_state=42)
    }
    preds, rows = {}, []
    progress = st.progress(0)
    for i,(name,model) in enumerate(models.items()):
        t0=time.perf_counter()
        model.fit(Xtr,ytr)
        elapsed=time.perf_counter()-t0
        p=np.clip(model.predict(Xte),0,None)
        preds[name]=p
        rows.append({"Model":name,"MAE (kW)":mean_absolute_error(yte,p),
                     "RMSE (kW)":np.sqrt(mean_squared_error(yte,p)),
                     "R²":r2_score(yte,p),"Training time (s)":elapsed})
        progress.progress((i+1)/3)
    results=pd.DataFrame(rows).sort_values("RMSE (kW)").reset_index(drop=True)
    best_name=results.iloc[0]["Model"]
    st.session_state["run"]={"train":train,"test":test,"Xtr":Xtr,"Xte":Xte,"yte":yte,
                             "models":models,"preds":preds,"results":results,"best":best_name}
    st.success("Model training complete.")

if "run" not in st.session_state:
    st.info("Click the button above to train the models and generate the full dashboard.")
    st.stop()

r=st.session_state["run"]
train,test,Xtr,Xte,yte=r["train"],r["test"],r["Xtr"],r["Xte"],r["yte"]
models,preds,results,best_name=r["models"],r["preds"],r["results"],r["best"]
best_model=models[best_name]
best_pred=preds[best_name]

st.markdown("### 2. Solar AI model comparison")
st.dataframe(results.round(4), use_container_width=True)
st.metric("Best model by RMSE", best_name)
fig,ax=plt.subplots(figsize=(11,4))
n=min(300,len(test))
ax.plot(test["NEXT_TIME"].iloc[:n],yte.iloc[:n],label="Actual")
ax.plot(test["NEXT_TIME"].iloc[:n],best_pred[:n],label="Predicted")
ax.set_title("Actual vs predicted next 15-minute solar output")
ax.set_ylabel("Power (kW)")
ax.legend(); ax.grid(alpha=.2); plt.xticks(rotation=25)
st.pyplot(fig); plt.close(fig)

st.markdown("### 3. Cellular base-station energy simulation")
st.caption("Illustrative simulation; base-station load is not measured telecom data.")
ref=max(float(train["NEXT_AC_POWER"].quantile(.99)),1e-6)
solar=np.clip(best_pred/ref,0,1)*pv_capacity
ts=pd.to_datetime(test["NEXT_TIME"])
hour=ts.dt.hour.to_numpy()+ts.dt.minute.to_numpy()/60
load=base_load+.5*np.exp(-.5*((hour-13)/2.5)**2)+.8*np.exp(-.5*((hour-20)/2.0)**2)
cell=pd.DataFrame({"Timestamp":ts.to_numpy(),"Solar supply (kW)":solar,"Simulated base-station load (kW)":load})
cell["Grid/backup required (kW)"]=np.maximum(cell["Simulated base-station load (kW)"]-cell["Solar supply (kW)"],0)
cell["Solar coverage (%)"]=100*np.minimum(cell["Solar supply (kW)"],cell["Simulated base-station load (kW)"])/cell["Simulated base-station load (kW)"]
cell["Energy shortfall"]=cell["Solar supply (kW)"]<cell["Simulated base-station load (kW)"]
x,y,z=st.columns(3)
x.metric("Average solar coverage",f"{cell['Solar coverage (%)'].mean():.1f}%")
y.metric("Shortfall intervals",f"{int(cell['Energy shortfall'].sum()):,}")
z.metric("Mean grid requirement",f"{cell['Grid/backup required (kW)'].mean():.2f} kW")
fig,ax=plt.subplots(figsize=(11,4))
ax.plot(cell["Timestamp"],cell["Solar supply (kW)"],label="Solar supply")
ax.plot(cell["Timestamp"],cell["Simulated base-station load (kW)"],label="Simulated tower load")
ax.set_title("Solar supply vs cellular demand"); ax.set_ylabel("Power (kW)")
ax.legend(); ax.grid(alpha=.2); plt.xticks(rotation=25)
st.pyplot(fig); plt.close(fig)

st.markdown("### 4. VLSI-inspired edge analysis")
st.caption("Simulated 8-bit input quantization; not model-weight quantization or physical chip synthesis.")
xmin,xmax=Xtr.min(),Xtr.max()
span=(xmax-xmin).replace(0,1)
scaled=((Xte-xmin)/span).clip(0,1)
Xq=(np.round(scaled*255)/255)*span+xmin
pf=np.clip(best_model.predict(Xte),0,None)
pq=np.clip(best_model.predict(Xq),0,None)
maef=mean_absolute_error(yte,pf); maq=mean_absolute_error(yte,pq)
rmsef=np.sqrt(mean_squared_error(yte,pf)); rmseq=np.sqrt(mean_squared_error(yte,pq))
q1,q2,q3=st.columns(3)
q1.metric("Original-input MAE",f"{maef:.4f} kW")
q2.metric("8-bit-input MAE",f"{maq:.4f} kW",delta=f"{maq-maef:+.4f} kW")
lat=[]
for _ in range(100):
    t0=time.perf_counter(); best_model.predict(Xte.iloc[[0]])
    lat.append((time.perf_counter()-t0)*1000)
q3.metric("Median inference latency",f"{np.median(lat):.4f} ms")
st.write(f"Original-input RMSE: **{rmsef:.4f} kW**  |  8-bit-input RMSE: **{rmseq:.4f} kW**")

st.markdown("### 5. Download results")
st.download_button("Download model comparison",results.to_csv(index=False).encode(), "solariq_model_comparison.csv","text/csv")
st.download_button("Download cellular simulation",cell.to_csv(index=False).encode(), "solariq_cellular_simulation.csv","text/csv")
st.info("Prototype limitations: short historical dataset; simulated cellular load; no measured battery model; no RTL/FPGA/ASIC synthesis.")
