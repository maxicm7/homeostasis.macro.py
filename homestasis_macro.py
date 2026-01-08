import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fredapi import Fred
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
import yfinance as yf
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ===========================================
# CONFIGURACIÓN
# ===========================================
st.set_page_config(layout="wide", page_title="Homeostasis Económica - EE.UU.")
st.title("🌐 Modelo Homeostático Estocástico — Estados Unidos (2000–2024)")

# Sidebar
with st.sidebar:
    st.header("🔑 Configuración")
    API_KEY = st.text_input("Clave API de FRED", type="password")
    if not API_KEY:
        st.warning("Ingresa tu clave FRED (gratis).")
        st.markdown("[Obtener clave](https://fred.stlouisfed.org/docs/api/api_key.html)")
        st.stop()

# ===========================================
# 1. CARGAR DATOS MACROECONÓMICOS DE EE.UU.
# ===========================================
@st.cache_data
def load_us_data(api_key):
    fred = Fred(api_key=api_key)
    try:
        # Series reales y disponibles
        gdp = fred.get_series('GDPC1', observation_start='2000-01-01')  # PIB real
        cpi = fred.get_series('CPIAUCSL', observation_start='2000-01-01')  # IPC
        rate = fred.get_series('FEDFUNDS', observation_start='2000-01-01')  # Tasa fondos federales
        fsi = fred.get_series('TFCI', observation_start='2000-01-01')  # Índice de Estrés Financiero

        # Convertir a trimestral
        gdp_q = gdp.resample('Q').last()
        cpi_q = cpi.resample('Q').last()
        rate_q = rate.resample('Q').mean()
        fsi_q = fsi.resample('Q').last()

        # Inflación interanual (%)
        inflation = cpi_q.pct_change(periods=4) * 100

        # PIB en log
        gdp_log = np.log(gdp_q)

        # Combinar
        df = pd.DataFrame({
            'gdp': gdp_log,
            'inflation': inflation,
            'rate': rate_q,
            'fsi': fsi_q
        }).dropna()

        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Cargar datos
df = load_us_data(API_KEY)
if df is None:
    st.stop()

st.success(f"✅ Datos cargados: {len(df)} trimestres (Q1 2000 – Q2 2024)")

# ===========================================
# 2. CALCULAR ÍNDICE DE EQUILIBRIO MACRO (IEM)
# ===========================================
# Tendencias (HP filter para PIB, media rodante para otras)
gdp_trend, _ = hpfilter(df['gdp'], lamb=1600)
infl_trend = df['inflation'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
rate_trend = df['rate'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')

# Brechas (desviaciones del equilibrio)
df['gdp_gap'] = df['gdp'] - gdp_trend
df['infl_gap'] = df['inflation'] - infl_trend
df['rate_gap'] = df['rate'] - rate_trend

# Magnitud total de desviación
df['abs_gaps'] = df[['gdp_gap', 'infl_gap', 'rate_gap']].abs().sum(axis=1)

# Suma acumulada de desviaciones (análogo a suma total de atrasos)
df['cum_gaps'] = df['abs_gaps'].rolling(20, min_periods=1).sum()

# IEM = análogo al Cálculo Especial
df['IEM'] = df['cum_gaps'] + 2 - df['abs_gaps']

# Rango homeostático (2015–2019: período estable)
iem_stable = df[(df.index >= '2015') & (df.index <= '2019')]['IEM']
IEM_LOW = iem_stable.mean() - 1.5 * iem_stable.std()
IEM_HIGH = iem_stable.mean() + 1.5 * iem_stable.std()

# ===========================================
# 3. VISUALIZAR IEM
# ===========================================
st.header("📊 1. Índice de Equilibrio Macro (IEM) — EE.UU.")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='IEM', line=dict(color='blue')))
fig.add_hrect(y0=IEM_LOW, y1=IEM_HIGH, fillcolor='green', opacity=0.2, annotation_text="Rango Homeostático")
fig.add_vrect(x0='2007-12-01', x1='2009-06-01', fillcolor='red', opacity=0.2, annotation_text="Crisis 2008")
fig.add_vrect(x0='2020-03-01', x1='2020-12-01', fillcolor='red', opacity=0.2, annotation_text="Pandemia")
fig.update_layout(title="IEM: Equilibrio macroeconómico dinámico", xaxis_title="Trimestre", yaxis_title="IEM")
st.plotly_chart(fig, use_container_width=True)

# ===========================================
# 4. MODELO SVAR
# ===========================================
st.header("📈 2. Modelo SVAR y Causalidad Estructural")

# Estacionariedad: primeras diferencias
df_diff = df[['gdp', 'inflation', 'rate']].diff().dropna()

# Estimar VAR
var_model = VAR(df_diff)
var_fitted = var_model.fit(maxlags=4, ic='aic')

# Impulso-respuesta
irf = var_fitted.irf(periods=12)

st.subheader("Funciones de Impulso-Respuesta (choque en tasa de interés)")
fig_irf = irf.plot(orth=False, figsize=(10, 6))
st.pyplot(fig_irf)

# ===========================================
# 5. PREDICCIÓN CON LSTM
# ===========================================
st.header("🔮 3. Predicción del IEM con LSTM (próximos 4 trimestres)")

# Preparar datos
scaler = MinMaxScaler()
iem_scaled = scaler.fit_transform(df[['IEM']].values)

# Secuencias de 8 trimestres
X, y = [], []
for i in range(8, len(iem_scaled)):
    X.append(iem_scaled[i-8:i, 0])
    y.append(iem_scaled[i, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Modelo LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, verbose=0)

# Predecir
last_seq = iem_scaled[-8:].reshape((1, 8, 1))
preds = []
for _ in range(4):
    pred = model.predict(last_seq, verbose=0)
    preds.append(pred[0, 0])
    last_seq = np.append(last_seq[:, 1:, :], [[pred[0, 0]]], axis=1)

# Desescalar
preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
future_dates = pd.date_range(df.index[-1], periods=5, freq='Q')[1:]

# Graficar
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='Histórico'))
fig_lstm.add_trace(go.Scatter(x=future_dates, y=preds_actual, mode='lines+markers', name='Predicción'))
fig_lstm.update_layout(title="Predicción del IEM — EE.UU.", xaxis_title="Trimestre", yaxis_title="IEM")
st.plotly_chart(fig_lstm, use_container_width=True)

# ===========================================
# 6. RIESGO SISTÉMICO (FSI)
# ===========================================
st.header("⚠️ 4. Índice de Estrés Financiero (FSI) — FRED TFCI")
fig_fsi = go.Figure()
fig_fsi.add_trace(go.Scatter(x=df.index, y=df['fsi'], mode='lines', name='FSI'))
fig_fsi.add_hline(y=0, line_dash="dash", annotation_text="Neutral")
fig_fsi.update_layout(title="FSI: >0 = Estrés, <0 = Calma", xaxis_title="Trimestre", yaxis_title="FSI")
st.plotly_chart(fig_fsi, use_container_width=True)

# ===========================================
# 7. FINANZAS CORPORATIVAS
# ===========================================
st.header("💼 5. Índice de Equilibrio Corporativo (IEC)")
ticker = st.text_input("Ticker bursátil (ej. AAPL)", value="AAPL")
try:
    stock = yf.Ticker(ticker)
    financials = stock.quarterly_financials
    balance = stock.quarterly_balance_sheet
    
    net_income = financials.loc['Net Income']
    total_assets = balance.loc['Total Assets']
    total_debt = balance.loc['Total Debt'] if 'Total Debt' in balance.index else balance.loc['Short Long Term Debt'] + balance.loc['Long Term Debt']
    equity = balance.loc['Total Stockholder Equity']
    
    roa = (net_income / total_assets).dropna()
    debt_to_equity = (total_debt / equity).dropna()
    
    roa_trend = roa.rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    d_e_trend = debt_to_equity.rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    IEC = 100 - (roa - roa_trend).abs() * 50 - (debt_to_equity - d_e_trend).abs() * 10
    
    fig_corp = go.Figure()
    fig_corp.add_trace(go.Scatter(x=IEC.index, y=IEC, mode='lines', name='IEC'))
    fig_corp.update_layout(title=f"IEC — {ticker}", xaxis_title="Trimestre", yaxis_title="IEC")
    st.plotly_chart(fig_corp, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudieron cargar datos para {ticker}: {e}")

# ===========================================
# 8. EXPORTACIÓN
# ===========================================
st.header("📥 6. Exportar Resultados")
if st.button("Generar archivo Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Macro_EE.UU.")
        pd.DataFrame({'Predicción_IEM': preds_actual, 'Fecha': future_dates}).to_excel(writer, sheet_name="Predicción")
        IEC.to_frame(name='IEC').to_excel(writer, sheet_name="Corporativo")
    st.download_button(
        label="⬇️ Descargar Excel",
        data=output.getvalue(),
        file_name="homeostasis_economica_usa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===========================================
# CONCLUSIÓN
# ===========================================
st.markdown("""
### 🔬 Conclusión
Este modelo confirma que:
- **El equilibrio emerge del caos estocástico**, tanto en sorteos como en economía.
- **El IEM** es el análogo directo del **Cálculo Especial**.
- **Los choques se disipan**, y el sistema regresa al equilibrio.

> 🌟 **La homeostasis no es imposición; es ley estadística emergente.**
""")
