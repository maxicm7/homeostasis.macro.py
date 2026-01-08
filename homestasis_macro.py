import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fredapi import Fred
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.filters.hp_filter import hpfilter
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="Homeostasis Económica — Versión Final")
st.title("🌐 Modelo Homeostático Estocástico — EE.UU. (2000–2024)")

# --- Sidebar: Clave API ---
with st.sidebar:
    st.header("🔑 Configuración")
    API_KEY = st.text_input("Clave API de FRED", type="password")
    st.markdown("[Obtener clave gratis](https://fred.stlouisfed.org/docs/api/api_key.html)")
    if not API_KEY:
        st.warning("Por favor, ingresa tu clave API de FRED.")
        st.stop()

# --- 1. Cargar datos macroeconómicos de EE.UU. ---
@st.cache_data
def load_us_data(api_key):
    fred = Fred(api_key=api_key)
    try:
        # Cargar series reales y disponibles en FRED
        gdp = fred.get_series('GDPC1', observation_start='2000-01-01')
        cpi = fred.get_series('CPIAUCSL', observation_start='2000-01-01')
        rate = fred.get_series('FEDFUNDS', observation_start='2000-01-01')
        fsi = fred.get_series('STLFSI', observation_start='2000-01-01')  # Índice de Estrés Financiero

        # Convertir a trimestral
        gdp_q = gdp.resample('Q').last()
        cpi_q = cpi.resample('Q').last()
        rate_q = rate.resample('Q').mean()
        fsi_q = fsi.resample('Q').last()

        # Calcular inflación interanual (%)
        inflation = cpi_q.pct_change(periods=4) * 100
        gdp_log = np.log(gdp_q)

        # Crear DataFrame
        df = pd.DataFrame({
            'gdp': gdp_log,
            'inflation': inflation,
            'rate': rate_q,
            'fsi': fsi_q
        }).dropna()

        return df
    except Exception as e:
        st.error(f"❌ Error al cargar datos de FRED: {e}")
        st.markdown("💡 Asegúrate de que tu clave API sea correcta.")
        return None

# --- Cargar y validar datos ---
df = load_us_data(API_KEY)
if df is None:
    st.stop()
st.success(f"✅ Datos cargados: {len(df)} trimestres (Q1 2000 – Q2 2024)")

# --- 2. Calcular Índice de Equilibrio Macro (IEM) ---
gdp_trend, _ = hpfilter(df['gdp'], lamb=1600)
infl_trend = df['inflation'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
rate_trend = df['rate'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')

df['gdp_gap'] = df['gdp'] - gdp_trend
df['infl_gap'] = df['inflation'] - infl_trend
df['rate_gap'] = df['rate'] - rate_trend
df['abs_gaps'] = df[['gdp_gap', 'infl_gap', 'rate_gap']].abs().sum(axis=1)
df['cum_gaps'] = df['abs_gaps'].rolling(20, min_periods=1).sum()
df['IEM'] = df['cum_gaps'] + 2 - df['abs_gaps']

# Rango homeostático (período estable 2015–2019)
iem_stable = df[(df.index >= '2015') & (df.index <= '2019')]['IEM']
IEM_LOW = iem_stable.mean() - 1.5 * iem_stable.std()
IEM_HIGH = iem_stable.mean() + 1.5 * iem_stable.std()

# --- 3. Visualizar IEM ---
st.header("📊 Índice de Equilibrio Macro (IEM)")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='IEM'))
fig.add_hrect(y0=IEM_LOW, y1=IEM_HIGH, fillcolor='lightgreen', opacity=0.3, annotation_text="Rango Homeostático")
fig.add_vrect(x0='2007-12-01', x1='2009-06-01', fillcolor='salmon', opacity=0.3, annotation_text="Crisis 2008")
fig.add_vrect(x0='2020-03-01', x1='2020-12-01', fillcolor='salmon', opacity=0.3, annotation_text="Pandemia")
fig.update_layout(title="IEM: Equilibrio macroeconómico (análogo al Cálculo Especial)", xaxis_title="Trimestre", yaxis_title="IEM")
st.plotly_chart(fig, use_container_width=True)

# --- 4. Desarrollo matemático del IEM ---
st.subheader("📘 Fundamento matemático del IEM")
st.markdown(r"""
El **Índice de Equilibrio Macro (IEM)** se define como:

\[
\text{IEM}_t = \underbrace{\sum_{s=t-k}^{t} \left( |y_s - \bar{y}_s| + |\pi_s - \bar{\pi}_s| + |i_s - \bar{i}_s| \right)}_{\text{Suma acumulada de desviaciones (}T\text{)}} + \kappa - \left( |y_t - \bar{y}_t| + |\pi_t - \bar{\pi}_t| + |i_t - \bar{i}_t| \right)
\]

Donde:
- \( y_t, \pi_t, i_t \) = PIB, inflación, tasa de interés.
- \( \bar{y}_t, \bar{\pi}_t, \bar{i}_t \) = tendencias (equilibrio).
- \( \kappa = 2 \) = constante de ajuste.
- El rango **homeostático** es el intervalo donde el sistema es estable (análogo a CE ∈ [300,327]).

> ✅ **Interpretación**: Cuanto más cerca esté el IEM del rango homeostático, mayor es la probabilidad de equilibrio.
""")

# --- 5. Modelo SVAR con Plotly (sin matplotlib) ---
st.header("📈 Modelo SVAR — Causalidad Estructural")

df_diff = df[['gdp', 'inflation', 'rate']].diff().dropna()
var_model = VAR(df_diff)
var_fitted = var_model.fit(maxlags=4, ic='aic')
irf = var_fitted.irf(periods=12)

variables = ['gdp', 'inflation', 'rate']
fig_svar = go.Figure()
for i, shock in enumerate(variables):
    for j, resp in enumerate(variables):
        response = irf.irfs[:, j, i]
        fig_svar.add_trace(go.Scatter(
            x=list(range(len(response))),
            y=response,
            mode='lines',
            name=f'{shock} → {resp}'
        ))
fig_svar.update_layout(title="Funciones de Impulso-Respuesta (IRF)", xaxis_title="Períodos", yaxis_title="Respuesta")
st.plotly_chart(fig_svar, use_container_width=True)

# --- 6. Desarrollo matemático del SVAR ---
st.subheader("📘 Fundamento matemático del SVAR")
st.markdown(r"""
El modelo **SVAR (Structural VAR)** se define como:

\[
\mathbf{B}_0 \mathbf{y}_t = \mathbf{c} + \sum_{i=1}^p \mathbf{B}_i \mathbf{y}_{t-i} + \mathbf{\varepsilon}_t
\]

Donde:
- \( \mathbf{y}_t = [y_t, \pi_t, i_t]' \) = vector de estado.
- \( \mathbf{\varepsilon}_t \) = choques estructurales **no correlacionados**.
- \( \mathbf{B}_0 \) = matriz de impactos contemporáneos.

La **función de impulso-respuesta** es:

\[
\text{IRF}_{ij}(h) = \left[ \mathbf{\Psi}_h \right]_{ij}, \quad \text{donde} \quad \mathbf{\Psi}(L) = \mathbf{B}(L)^{-1}\mathbf{B}_0
\]

> ✅ **Interpretación**: Captura causalidad en el sentido de **Granger estructural**. Ej: un choque en la tasa → afecta PIB e inflación.
""")

# --- 7. Predicción con LSTM (¡CORREGIDO!) ---
st.header("🔮 Predicción del IEM con LSTM")

# Preparar datos
scaler = MinMaxScaler()
iem_scaled = scaler.fit_transform(df[['IEM']].values)

# Crear secuencias de 8 trimestres
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

# Predecir los próximos 4 trimestres
last_seq = iem_scaled[-8:].reshape((1, 8, 1))
preds = []
for _ in range(4):
    pred = model.predict(last_seq, verbose=0)
    # ✅ CORRECCIÓN CLAVE: usar .item() para obtener un escalar
    pred_value = pred.item()
    preds.append(pred_value)
    # Añadir a la secuencia (manteniendo forma (1,8,1))
    last_seq = np.append(last_seq[:, 1:, :], [[[pred_value]]], axis=1)

# Desescalar predicciones
preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
future_dates = pd.date_range(df.index[-1], periods=5, freq='Q')[1:]

# Mostrar gráfico
fig_lstm = go.Figure()
fig_lstm.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='Histórico'))
fig_lstm.add_trace(go.Scatter(x=future_dates, y=preds_actual, mode='markers+lines', name='Predicción'))
fig_lstm.update_layout(title="Predicción del IEM — Próximos 4 trimestres", xaxis_title="Trimestre", yaxis_title="IEM")
st.plotly_chart(fig_lstm, use_container_width=True)

# --- 8. Índice de Estrés Financiero (STLFSI) ---
st.header("⚠️ Índice de Estrés Financiero (STLFSI)")
fig_fsi = go.Figure()
fig_fsi.add_trace(go.Scatter(x=df.index, y=df['fsi'], mode='lines', name='STLFSI'))
fig_fsi.add_hline(y=0, line_dash="dash", line_color="gray")
fig_fsi.update_layout(title="STLFSI: Valores >0 indican estrés financiero", xaxis_title="Trimestre", yaxis_title="Índice")
st.plotly_chart(fig_fsi, use_container_width=True)

# --- 9. Finanzas Corporativas ---
st.header("💼 Equilibrio Corporativo")
ticker = st.text_input("Ticker bursátil (ej. AAPL, MSFT)", value="AAPL")
try:
    import yfinance as yf
    stock = yf.Ticker(ticker)
    financials = stock.quarterly_financials
    balance = stock.quarterly_balance_sheet
    
    net_income = financials.loc['Net Income']
    total_assets = balance.loc['Total Assets']
    total_debt = balance.loc.get('Total Debt', balance.loc.get('Short Long Term Debt', 0) + balance.loc.get('Long Term Debt', 0))
    equity = balance.loc['Total Stockholder Equity']
    
    roa = (net_income / total_assets).dropna()
    debt_to_equity = (total_debt / equity).dropna()
    
    roa_trend = roa.rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    d_e_trend = debt_to_equity.rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    IEC = 100 - (roa - roa_trend).abs() * 50 - (debt_to_equity - d_e_trend).abs() * 10
    
    fig_corp = go.Figure()
    fig_corp.add_trace(go.Scatter(x=IEC.index, y=IEC, mode='lines', name='IEC'))
    fig_corp.update_layout(title=f"Índice de Equilibrio Corporativo — {ticker}", xaxis_title="Trimestre", yaxis_title="IEC")
    st.plotly_chart(fig_corp, use_container_width=True)
except Exception as e:
    st.warning(f"No se pudieron cargar datos para {ticker}: {str(e)[:100]}...")

# --- 10. Exportación ---
st.header("📥 Exportar Resultados")
if st.button("Generar archivo Excel con todos los resultados"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Macro_EE.UU.")
        pd.DataFrame({'Predicción_IEM': preds_actual, 'Fecha': future_dates}).to_excel(writer, sheet_name="Predicción")
        if 'IEC' in locals():
            IEC.to_frame(name='IEC').to_excel(writer, sheet_name="Corporativo")
    st.download_button(
        label="⬇️ Descargar Excel",
        data=output.getvalue(),
        file_name="homeostasis_economica_resultados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.markdown("💡 **Conclusión**: El equilibrio macroeconómico es un fenómeno **homeostático estocástico** — igual que en tu modelo de lotería.")
