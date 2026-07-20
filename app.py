"""
QuantBuffett AI - Plataforma de Análisis Financiero Profesional
Autor: [Tu Nombre]
Versión: 0.1.0 (MVP Inicial)
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="QuantBuffett AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================
st.title("📈 QuantBuffett AI")
st.markdown("""
### Plataforma Profesional de Análisis y Pronóstico Financiero
*Combinando Data Science, Machine Learning y la filosofía de inversión de valor*
""")

st.divider()

# ==============================================================================
# BARRA LATERAL (Panel de Control)
# ==============================================================================
st.sidebar.header("⚙️ Panel de Control")
st.sidebar.markdown(f"**Fecha:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")

modo_analisis = st.sidebar.radio(
    "Modo de Análisis",
    ["🔍 Activo Único", "💼 Portafolio"],
    help="Selecciona si quieres analizar una empresa individual o un portafolio de varias empresas"
)

st.sidebar.divider()

# Input del ticker
if modo_analisis == "🔍 Activo Único":
    ticker_input = st.sidebar.text_input(
        "Ticker de la empresa",
        value="AAPL",
        help="Ingresa el símbolo bursátil (ej: AAPL para Apple, MSFT para Microsoft)"
    ).upper()
else:
    ticker_input = st.sidebar.text_input(
        "Tickers (separados por coma)",
        value="AAPL, MSFT, KO",
        help="Ingresa múltiples tickers separados por coma"
    ).upper()

st.sidebar.divider()

# Información del proyecto
st.sidebar.markdown("---")
st.sidebar.markdown("""
### Sobre QuantBuffett AI
Esta aplicación combina:
- ✅ Análisis fundamental (estilo Warren Buffett)
- ✅ Optimización de portafolios (Markowitz)
- ✅ Pronóstico con Machine Learning (Prophet)
- ✅ Análisis de riesgos con IA (NLP)
""")

# ==============================================================================
# CONTENIDO PRINCIPAL
# ==============================================================================
st.header(f"Análisis en Modo: {modo_analisis}")

if modo_analisis == "🔍 Activo Único":
    st.info(f"🎯 Analizando: **{ticker_input}**")
    
    # Métricas de ejemplo (las reemplazaremos con datos reales en el Paso 2)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Precio Actual",
            value="$0.00",
            delta="Cargando...",
            help="Precio de mercado en tiempo real"
        )
    
    with col2:
        st.metric(
            label="ROIC",
            value="0.0%",
            delta="Cargando...",
            help="Return on Invested Capital - Eficiencia del negocio"
        )
    
    with col3:
        st.metric(
            label="Deuda/EBITDA",
            value="0.0x",
            delta="Cargando...",
            help="Ratio de solvencia - Capacidad de pago de deuda"
        )
    
    with col4:
        st.metric(
            label="Margen de Seguridad",
            value="0.0%",
            delta="Cargando...",
            help="Diferencia entre valor intrínseco y precio de mercado"
        )
    
    st.markdown("""
    ### 📊 Próximos Pasos
    En las siguientes versiones, aquí verás:
    - Gráficos interactivos de evolución histórica
    - Análisis de flujo de caja libre (FCF)
    - Valuación por descuento de flujos (DCF)
    - Pronóstico con Machine Learning
    """)

else:  # Modo Portafolio
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    st.info(f"💼 Analizando portafolio de **{len(tickers)} activos**: {', '.join(tickers)}")
    
    st.markdown("""
    ### 📊 Próximos Pasos
    En las siguientes versiones, aquí verás:
    - Optimización de asignación de capital (Markowitz)
    - Frontera eficiente visualizada
    - Sistema de alertas de rebalanceo
    - Análisis de correlación entre activos
    """)

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>QuantBuffett AI v0.1.0 | Desarrollado con Streamlit + Python</p>
    <p><em>"Es mucho mejor comprar una empresa maravillosa a un precio justo, 
    que una empresa justa a un precio maravilloso." - Warren Buffett</em></p>
</div>
""", unsafe_allow_html=True)
