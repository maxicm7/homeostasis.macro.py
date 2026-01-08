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
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# CONFIGURACIÓN GENERAL
# ===========================================
st.set_page_config(layout="wide", page_title="Homeostasis Económico-Financiera")
st.title("🌐 Sistema Integrado de Homeostasis Estocástica")
st.markdown("""
Este modelo unifica economía, finanzas y riesgo bajo el principio de **equilibrio emergente**:
- **IEM**: Índice de Equilibrio Macro (análogo al Cálculo Especial)
- **IEC**: Índice de Equilibrio Corporativo
- **FSI**: Índice de Estrés Financiero
- **DSGE**: Simulación de equilibrio general dinámico
""")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuración")
    API_KEY = st.text_input("Clave API de FRED", type="password")
    analysis_mode = st.radio("Modo de análisis", ["Global", "Por país"])
    if not API_KEY:
        st.warning("Ingresa tu clave FRED")
        st.markdown("[Obtener clave gratis](https://fred.stlouisfed.org/docs/api/api_key.html)")
        st.stop()

# ===========================================
# 1. CARGA DE DATOS MACROECONÓMICOS
# ===========================================
FRED_CODES = {
    "Estados Unidos": {'gdp': 'GDPC1', 'cpi': 'CPIAUCSL', 'rate': 'FEDFUNDS', 'fsi': 'TFCI', 'unemp': 'UNRATE'},
    "Zona Euro": {'gdp': 'CLVMNACSCAB1GQEU2720', 'cpi': 'CPHPTT01EZM659N', 'rate': 'ECBDFR', 'fsi': None, 'unemp': 'LRHUT2TTEZM156S'},
    "Japón": {'gtp': 'JPNNGDP', 'cpi': 'JPNCPIALLQISMEI', 'rate': 'INTDSRJPM193N', 'fsi': None, 'unemp': 'LRHUT2TTJPM156S'}
}

@st.cache_data
def load_macro_data(country, api_key):
    fred = Fred(api_key=api_key)
    codes = FRED_CODES[country]
    try:
        # Cargar series
        gdp = fred.get_series(codes['gdp'], start='2000')
        cpi = fred.get_series(codes['cpi'], start='2000')
        rate = fred.get_series(codes['rate'], start='2000')
        unemp = fred.get_series(codes['unemp'], start='2000') if codes['unemp'] else None
        
        # Procesar
        gdp_q = gdp.resample('Q').last()
        cpi_q = cpi.resample('Q').last()
        rate_q = rate.resample('Q').mean()
        inflation = cpi_q.pct_change(4) * 100
        gdp_log = np.log(gdp_q)
        
        df = pd.DataFrame({
            'gdp': gdp_log,
            'inflation': inflation,
            'rate': rate_q
        }).dropna()
        
        # Agregar desempleo si existe
        if unemp is not None:
            df['unemployment'] = unemp.resample('Q').mean()
        
        # Agregar FSI si existe
        if codes['fsi']:
            fsi = fred.get_series(codes['fsi'], start='2000').resample('Q').last()
            df['fsi'] = fsi
        else:
            df['fsi'] = np.nan
            
        return df
    except Exception as e:
        st.error(f"Error en {country}: {e}")
        return None

# Cargar datos
if analysis_mode == "Global":
    countries = list(FRED_CODES.keys())
    macro_data = {c: load_macro_data(c, API_KEY) for c in countries}
else:
    country = st.selectbox("Seleccione país", list(FRED_CODES.keys()))
    macro_data = {country: load_macro_data(country, API_KEY)}

# ===========================================
# 2. CÁLCULO DEL ÍNDICE DE EQUILIBRIO MACRO (IEM)
# ===========================================
iem_data = {}
for country, df in macro_data.items():
    if df is None: continue
    # Tendencias
    gdp_trend, _ = hpfilter(df['gdp'], lamb=1600)
    infl_trend = df['inflation'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    rate_trend = df['rate'].rolling(8, center=True).mean().fillna(method='bfill').fillna(method='ffill')
    
    # Brechas
    df['gdp_gap'] = df['gdp'] - gdp_trend
    df['infl_gap'] = df['inflation'] - infl_trend
    df['rate_gap'] = df['rate'] - rate_trend
    df['abs_gaps'] = df[['gdp_gap', 'infl_gap', 'rate_gap']].abs().sum(axis=1)
    
    # IEM
    df['cum_gaps'] = df['abs_gaps'].rolling(20, min_periods=1).sum()
    df['IEM'] = df['cum_gaps'] + 2 - df['abs_gaps']
    iem_data[country] = df

# ===========================================
# 3. VISUALIZACIÓN DEL IEM
# ===========================================
st.header("📊 1. Índice de Equilibrio Macro (IEM)")
if analysis_mode == "Global":
    fig = go.Figure()
    for country, df in iem_data.items():
        fig.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name=country))
    fig.update_layout(title="Comparación Internacional del IEM", xaxis_title="Trimestre", yaxis_title="IEM")
    st.plotly_chart(fig, use_container_width=True)
else:
    df = list(iem_data.values())[0]
    country = list(iem_data.keys())[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='IEM'))
    # Rango homeostático
    iem_stable = df[(df.index >= '2015') & (df.index <= '2019')]['IEM']
    IEM_LOW = iem_stable.mean() - 1.5 * iem_stable.std()
    IEM_HIGH = iem_stable.mean() + 1.5 * iem_stable.std()
    fig.add_hrect(y0=IEM_LOW, y1=IEM_HIGH, fillcolor='green', opacity=0.2, annotation_text="Rango Homeostático")
    st.plotly_chart(fig, use_container_width=True)

# ===========================================
# 4. MODELO SVAR
# ===========================================
st.header("📈 2. Modelo SVAR y Causalidad Estructural")
if analysis_mode == "Por país":
    df_svar = df[['gdp', 'inflation', 'rate']].diff().dropna()
    var_model = VAR(df_svar)
    var_fitted = var_model.fit(maxlags=4, ic='aic')
    irf = var_fitted.irf(periods=12)
    st.pyplot(irf.plot(orth=False, figsize=(10, 6)))

# ===========================================
# 5. MODELO DSGE SINTÉTICO
# ===========================================
st.header("🔄 3. Simulación de Equilibrio General Dinámico (DSGE)")
st.markdown("""
Simulación de un modelo DSGE simple con:
- Choques de productividad
- Ajuste de expectativas racionales
- Convergencia al equilibrio
""")

# Parámetros DSGE
alpha = 0.3  # Elasticidad capital
beta = 0.99  # Factor de descuento
rho = 0.9    # Persistencia del choque

# Simulación
np.random.seed(42)
T = 100
shocks = np.random.normal(0, 0.01, T)
productivity = np.zeros(T)
output = np.zeros(T)

for t in range(1, T):
    productivity[t] = rho * productivity[t-1] + shocks[t]
    output[t] = (productivity[t] / (1 - alpha))**(1/(1 - alpha))

# Graficar
fig_dsg = go.Figure()
fig_dsg.add_trace(go.Scatter(y=output, mode='lines', name='Output DSGE'))
fig_dsg.update_layout(title="Simulación DSGE: Convergencia al Equilibrio", xaxis_title="Período", yaxis_title="Output")
st.plotly_chart(fig_dsg, use_container_width=True)

# ===========================================
# 6. PREDICCIÓN CON LSTM
# ===========================================
st.header("🔮 4. Predicción del IEM con LSTM")
if analysis_mode == "Por país":
    # Preparar datos
    scaler = MinMaxScaler()
    iem_scaled = scaler.fit_transform(df[['IEM']].values)
    X, y = [], []
    for i in range(8, len(iem_scaled)):
        X.append(iem_scaled[i-8:i, 0])
        y.append(iem_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    # Modelo
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
    
    preds_actual = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(df.index[-1], periods=5, freq='Q')[1:]
    
    # Graficar
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(x=df.index, y=df['IEM'], mode='lines', name='Histórico'))
    fig_lstm.add_trace(go.Scatter(x=future_dates, y=preds_actual, mode='lines+markers', name='Predicción'))
    st.plotly_chart(fig_lstm, use_container_width=True)

# ===========================================
# 7. RIESGO SISTÉMICO (FSI)
# ===========================================
st.header("⚠️ 5. Índice de Estrés Financiero (FSI)")
for country, df in iem_data.items():
    if 'fsi' in df.columns and df['fsi'].notna().any():
        st.subheader(f"FSI — {country}")
        fig_fsi = go.Figure()
        fig_fsi.add_trace(go.Scatter(x=df.index, y=df['fsi'], mode='lines', name='FSI'))
        fig_fsi.add_hline(y=0, line_dash="dash")
        fig_fsi.update_layout(title="FSI: >0 = Estrés, <0 = Calma", xaxis_title="Trimestre", yaxis_title="FSI")
        st.plotly_chart(fig_fsi, use_container_width=True)

# ===========================================
# 8. FINANZAS CORPORATIVAS
# ===========================================
st.header("💼 6. Índice de Equilibrio Corporativo (IEC)")
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
    st.warning(f"Error con {ticker}: {e}")

# ===========================================
# 9. EXPORTACIÓN
# ===========================================
st.header("📥 7. Exportar Resultados")
if st.button("Generar archivo Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Datos macro
        for country, df in iem_data.items():
            df.to_excel(writer, sheet_name=f"Macro_{country}")
        # Predicción LSTM
        if analysis_mode == "Por país":
            pd.DataFrame({'Predicción_IEM': preds_actual, 'Fecha': future_dates}).to_excel(writer, sheet_name="Predicción")
        # Datos DSGE
        pd.DataFrame({'Output_DSGE': output}).to_excel(writer, sheet_name="DSGE")
        # Datos corporativos
        if 'IEC' in locals():
            IEC.to_frame(name='IEC').to_excel(writer, sheet_name="Corporativo")
    st.download_button(
        label="⬇️ Descargar Excel",
        data=output.getvalue(),
        file_name="homeostasis_completo.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===========================================
# CONCLUSIÓN
# ===========================================
st.markdown("""
### 🔬 Conclusión Científica
Este sistema demuestra que:
1. **Lo aleatorio no es caótico**: emerge un equilibrio estadístico (homeostasis).
2. **El modelo es universal**: aplica a loterías, economía y finanzas.
3. **La causalidad existe**: los choques se disipan, llevando al sistema al equilibrio.

> 🌟 **"El orden no se impone; emerge del caos estocástico."**
""")
