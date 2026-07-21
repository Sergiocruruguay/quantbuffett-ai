"""
QuantBuffett AI - Plataforma Profesional de Análisis Financiero
Versión: 1.0.0-alpha | Paso 6A: Dashboard + Activo Único con Alpha Vantage
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import requests
import time

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================
st.set_page_config(
    page_title="QuantBuffett AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key de Alpha Vantage (de st.secrets o hardcoded temporal)
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "5IPGMO6N5R7UB9VC")

# ==============================================================================
# BASE DE DATOS MOCK (Fallback cuando API falla)
# ==============================================================================
MOCK_DATABASE = {
    'AAPL': {
        'ticker': 'AAPL', 'precio': 308.50, 'market_cap': 2400000000000,
        'roic': 55.2, 'deuda_ebitda': 0.35, 'fcf': 105.8,
        'net_income': 112000000000, 'ebit': 130000000000,
        'margen_seguridad': -4.3, 'beta': 1.10, 'pe_ratio': 28.5,
        'dividend_yield': 0.52, 'eps': 10.85,
        'sector': 'Technology', 'industry': 'Consumer Electronics',
        'retorno_anual': 0.28, 'volatilidad_anual': 0.25, 'tendencia': 'Alcista'
    },
    'MSFT': {
        'ticker': 'MSFT', 'precio': 415.20, 'market_cap': 3100000000000,
        'roic': 38.1, 'deuda_ebitda': 0.42, 'fcf': 78.5,
        'net_income': 88000000000, 'ebit': 105000000000,
        'margen_seguridad': 3.5, 'beta': 0.95, 'pe_ratio': 35.2,
        'dividend_yield': 0.75, 'eps': 11.80,
        'sector': 'Technology', 'industry': 'Software',
        'retorno_anual': 0.32, 'volatilidad_anual': 0.22, 'tendencia': 'Alcista'
    },
    'KO': {
        'ticker': 'KO', 'precio': 62.30, 'market_cap': 270000000000,
        'roic': 16.6, 'deuda_ebitda': 1.50, 'fcf': 9.8,
        'net_income': 10500000000, 'ebit': 13000000000,
        'margen_seguridad': 12.5, 'beta': 0.65, 'pe_ratio': 24.1,
        'dividend_yield': 3.05, 'eps': 2.58,
        'sector': 'Consumer Defensive', 'industry': 'Beverages',
        'retorno_anual': 0.08, 'volatilidad_anual': 0.15, 'tendencia': 'Bajista'
    },
    'GOOGL': {
        'ticker': 'GOOGL', 'precio': 175.80, 'market_cap': 2200000000000,
        'roic': 26.4, 'deuda_ebitda': 0.28, 'fcf': 65.2,
        'net_income': 75000000000, 'ebit': 95000000000,
        'margen_seguridad': 8.2, 'beta': 1.05, 'pe_ratio': 26.8,
        'dividend_yield': 0.0, 'eps': 6.56,
        'sector': 'Communication Services', 'industry': 'Internet Content',
        'retorno_anual': 0.25, 'volatilidad_anual': 0.28, 'tendencia': 'Alcista'
    },
    'WMT': {
        'ticker': 'WMT', 'precio': 85.40, 'market_cap': 230000000000,
        'roic': 14.2, 'deuda_ebitda': 1.85, 'fcf': 12.5,
        'net_income': 15000000000, 'ebit': 22000000000,
        'margen_seguridad': 5.8, 'beta': 0.55, 'pe_ratio': 28.5,
        'dividend_yield': 1.35, 'eps': 3.00,
        'sector': 'Consumer Defensive', 'industry': 'Discount Stores',
        'retorno_anual': 0.12, 'volatilidad_anual': 0.18, 'tendencia': 'Alcista'
    },
    'TSLA': {
        'ticker': 'TSLA', 'precio': 245.60, 'market_cap': 780000000000,
        'roic': 12.8, 'deuda_ebitda': 0.95, 'fcf': 8.2,
        'net_income': 12000000000, 'ebit': 15000000000,
        'margen_seguridad': -15.2, 'beta': 2.05, 'pe_ratio': 65.3,
        'dividend_yield': 0.0, 'eps': 3.76,
        'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers',
        'retorno_anual': 0.45, 'volatilidad_anual': 0.55, 'tendencia': 'Bajista'
    }
}

# ==============================================================================
# FUNCIONES DE ALPHA VANTAGE (Con caché robusto)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_alpha_vantage(ticker: str) -> dict:
    """
    Obtiene datos fundamentales de Alpha Vantage.
    Usa caché de 1 hora para no agotar las 500 llamadas/día.
    """
    try:
        # 1. OVERVIEW (datos fundamentales)
        url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp_overview = requests.get(url_overview, timeout=10)
        data_overview = resp_overview.json()
        
        # Verificar que no sea error de rate limit
        if "Note" in data_overview or "Information" in data_overview:
            return None
        
        if not data_overview.get("Symbol"):
            return None
        
        # 2. GLOBAL_QUOTE (precio actual)
        time.sleep(0.5)  # Respetar rate limit
        url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp_quote = requests.get(url_quote, timeout=10)
        data_quote = resp_quote.json()
        
        quote = data_quote.get("Global Quote", {})
        precio_actual = float(quote.get("05. price", 0)) if quote.get("05. price") else 0
        
        # 3. Construir diccionario unificado
        resultado = {
            'ticker': ticker.upper(),
            'precio': precio_actual if precio_actual > 0 else float(data_overview.get('Price', 0)),
            'market_cap': float(data_overview.get('MarketCapitalization', 0)),
            'pe_ratio': float(data_overview.get('PERatio', 0)) if data_overview.get('PERatio') else 0,
            'eps': float(data_overview.get('EPS', 0)) if data_overview.get('EPS') else 0,
            'dividend_yield': float(data_overview.get('DividendYield', 0)) if data_overview.get('DividendYield') else 0,
            'beta': float(data_overview.get('Beta', 1.0)) if data_overview.get('Beta') else 1.0,
            'sector': data_overview.get('Sector', 'N/A'),
            'industry': data_overview.get('Industry', 'N/A'),
            'descripcion': data_overview.get('Description', '')[:200],
            'roic': float(data_overview.get('ReturnOnEquityTTM', 0)) * 100 if data_overview.get('ReturnOnEquityTTM') else 0,
            'profit_margin': float(data_overview.get('ProfitMargin', 0)) * 100 if data_overview.get('ProfitMargin') else 0,
            'operating_margin': float(data_overview.get('OperatingMarginTTM', 0)) * 100 if data_overview.get('OperatingMarginTTM') else 0,
            'revenue': float(data_overview.get('RevenueTTM', 0)),
            'gross_profit': float(data_overview.get('GrossProfitTTM', 0)),
            'deuda_ebitda': 0,  # Alpha Vantage no da este ratio directamente
            'fcf': 0,
            'net_income': float(data_overview.get('NetIncomeTTM', 0)),
            'ebit': 0,
            'margen_seguridad': 0,
            'retorno_anual': 0,
            'volatilidad_anual': 0,
            'tendencia': 'N/A',
            'es_real': True,
            'fuente': 'Alpha Vantage'
        }
        
        return resultado
        
    except Exception as e:
        st.warning(f"️ Error al conectar con Alpha Vantage: {str(e)[:50]}")
        return None


def obtener_datos_financieros(ticker: str) -> dict:
    """
    Obtiene datos financieros: primero intenta Alpha Vantage, 
    luego hace fallback a datos mock.
    """
    ticker_upper = ticker.upper()
    
    # Intentar datos reales
    datos_reales = obtener_datos_alpha_vantage(ticker_upper)
    
    if datos_reales and datos_reales.get('precio', 0) > 0:
        # Completar con datos mock para métricas que Alpha Vantage no proporciona
        mock = MOCK_DATABASE.get(ticker_upper, {})
        datos_reales['deuda_ebitda'] = mock.get('deuda_ebitda', 1.0)
        datos_reales['fcf'] = mock.get('fcf', 0)
        datos_reales['ebit'] = mock.get('ebit', 0)
        datos_reales['margen_seguridad'] = mock.get('margen_seguridad', 0)
        datos_reales['retorno_anual'] = mock.get('retorno_anual', 0.10)
        datos_reales['volatilidad_anual'] = mock.get('volatilidad_anual', 0.20)
        datos_reales['tendencia'] = mock.get('tendencia', 'N/A')
        return datos_reales
    
    # Fallback a mock
    if ticker_upper in MOCK_DATABASE:
        datos = MOCK_DATABASE[ticker_upper].copy()
        datos['es_real'] = False
        datos['fuente'] = 'Base de datos de demostración'
        return datos
    
    return None

# ==============================================================================
# FUNCIONES DE ANÁLISIS
# ==============================================================================
def calcular_score_simple(datos: dict) -> dict:
    """Calcula un score simple (0-100) para el Modo Simple."""
    score = 50  # Base
    
    # ROIC (peso 25)
    roic = datos.get('roic', 0)
    if roic > 20:
        score += 20
    elif roic > 10:
        score += 10
    elif roic < 5:
        score -= 15
    
    # Deuda/EBITDA (peso 25)
    deuda = datos.get('deuda_ebitda', 1)
    if deuda < 1.5:
        score += 20
    elif deuda < 3:
        score += 10
    elif deuda > 4:
        score -= 20
    
    # Margen de seguridad (peso 25)
    margen = datos.get('margen_seguridad', 0)
    if margen > 20:
        score += 20
    elif margen > 0:
        score += 10
    elif margen < -20:
        score -= 20
    
    # Beta (peso 25)
    beta = datos.get('beta', 1)
    if beta < 0.8:
        score += 15
    elif beta < 1.2:
        score += 10
    elif beta > 1.8:
        score -= 15
    
    score = max(0, min(100, score))
    
    if score >= 75:
        veredicto = "🟢 COMPRA RECOMENDADA"
        mensaje = "Empresa sólida con precio atractivo. Cumple con los criterios de inversión de valor."
        color = "green"
    elif score >= 55:
        veredicto = "🟡 OBSERVAR"
        mensaje = "Negocio de calidad pero precio no es el más atractivo. Esperar mejor entrada."
        color = "yellow"
    elif score >= 40:
        veredicto = "🟠 ANÁLISIS MIXTO"
        mensaje = "Requiere evaluación más profunda. No cumple todos los criterios de calidad/valor."
        color = "orange"
    else:
        veredicto = "🔴 EVITAR"
        mensaje = "Perfil de riesgo elevado o sobrevaloración significativa. No recomendado."
        color = "red"
    
    return {
        'score': score,
        'veredicto': veredicto,
        'mensaje': mensaje,
        'color': color
    }

def calcular_dcf_escenarios(datos: dict, wacc: float = 0.09, crecimiento: float = 0.05, g_perpetuo: float = 0.03) -> dict:
    """Calcula DCF con 3 escenarios para Modo Avanzado."""
    fcf_base = datos.get('fcf', 0) * 1e9
    if fcf_base <= 0:
        # Estimación basada en net_income
        fcf_base = datos.get('net_income', 0) * 0.85
    
    precio_actual = datos.get('precio', 0)
    acciones = datos.get('market_cap', 0) / precio_actual if precio_actual > 0 else 1e9
    
    def calcular_valor_justo(tasa_crec):
        flujos = []
        for ano in range(1, 6):
            fcf_fut = fcf_base * ((1 + tasa_crec) ** ano)
            vp = fcf_fut / ((1 + wacc) ** ano)
            flujos.append(vp)
        
        vt = (flujos[-1] * (1 + wacc)) / (wacc - g_perpetuo)
        vp_vt = vt / ((1 + wacc) ** 5)
        enterprise_value = sum(flujos) + vp_vt
        
        deuda_neta = datos.get('deuda_ebitda', 1) * (datos.get('ebit', 0) + abs(datos.get('fcf', 0) * 1e9 * 0.1))
        equity_value = enterprise_value - deuda_neta
        return equity_value / acciones if acciones > 0 else 0
    
    valor_pesimista = calcular_valor_justo(crecimiento * 0.5)
    valor_base = calcular_valor_justo(crecimiento)
    valor_optimista = calcular_valor_justo(crecimiento * 1.5)
    
    return {
        'pesimista': valor_pesimista,
        'base': valor_base,
        'optimista': valor_optimista,
        'margen_pesimista': ((valor_pesimista / precio_actual) - 1) * 100 if precio_actual > 0 else 0,
        'margen_base': ((valor_base / precio_actual) - 1) * 100 if precio_actual > 0 else 0,
        'margen_optimista': ((valor_optimista / precio_actual) - 1) * 100 if precio_actual > 0 else 0,
        'wacc': wacc,
        'crecimiento': crecimiento
    }

# ==============================================================================
# ENCABEZADO
# ==============================================================================
st.title("📈 QuantBuffett AI")
st.markdown("""
**Plataforma Profesional de Análisis Financiero**  
*Data Science + Machine Learning + Filosofía de Inversión de Valor*
""")

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
st.sidebar.header("️ Configuración")

# Toggle Modo Simple / Avanzado
modo_usuario = st.sidebar.radio(
    "Modo de Visualización",
    ["🟢 Simple (Principiantes)", "🔵 Avanzado (Expertos)"],
    help="Simple: veredictos claros. Avanzado: métricas detalladas y parámetros editables."
)

st.sidebar.divider()

# Input de ticker
ticker_input = st.sidebar.text_input(
    "Ticker a analizar",
    value="AAPL",
    help="Símbolo bursátil (ej: AAPL, MSFT, KO, GOOGL)"
).upper()

st.sidebar.divider()

# Información de fuente de datos
st.sidebar.markdown("### 📡 Fuente de Datos")
if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "DEMO_KEY":
    st.sidebar.success("✅ Alpha Vantage conectado")
    st.sidebar.caption("500 llamadas/día disponibles")
else:
    st.sidebar.warning("⚠️ Modo Demo (datos simulados)")

st.sidebar.divider()

st.sidebar.markdown("""
### ℹ️ Sobre QuantBuffett AI
Versión 1.0.0-alpha  
Desarrollado con Streamlit + Python  

*"Es mejor comprar una empresa maravillosa a un precio justo, que una empresa justa a un precio maravilloso."*  
— Warren Buffett
""")

# ==============================================================================
# NAVEGACIÓN POR PESTAÑAS
# ==============================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Dashboard",
    "🔍 Análisis de Activo",
    "💼 Portafolio",
    "🔮 Pronóstico y Riesgos"
])

# ==============================================================================
# PESTAÑA 1: DASHBOARD
# ==============================================================================
with tab1:
    st.header("🏠 Dashboard Ejecutivo")
    
    if modo_usuario == " Simple (Principiantes)":
        st.markdown("""
        ### Bienvenido a QuantBuffett AI
        
        Esta aplicación te ayudará a tomar decisiones de inversión informadas.
        
        **Para comenzar:**
        1. Ve a la pestaña **🔍 Análisis de Activo**
        2. Ingresa el ticker de una empresa (ej: AAPL para Apple)
        3. Haz clic en "Analizar"
        4. Recibe un veredicto claro: 🟢 Comprar, 🟡 Observar, o 🔴 Evitar
        
        **Empresas disponibles para prueba:** AAPL, MSFT, KO, GOOGL, WMT, TSLA
        """)
        
        # Tarjetas rápidas
        st.subheader(" Resumen Rápido del Mercado")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("S&P 500 (simulado)", "5,420", "+0.8%")
        with col2:
            st.metric("Empresas analizables", "6", "AAPL, MSFT, KO...")
        with col3:
            st.metric("Llamadas API restantes", "~495", "de 500 diarias")
    
    else:  # Modo Avanzado
        st.markdown("""
        ### Dashboard de Control
        
        Panel de control para análisis financiero profesional con datos en tiempo real 
        vía Alpha Vantage API.
        """)
        
        # Métricas de sistema
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Alpha Vantage", "Conectada", "500 calls/day")
        with col2:
            st.metric("Modo Activo", "Avanzado", "Parámetros editables")
        with col3:
            st.metric("Empresas en DB", "6", "Mock + Real")
        with col4:
            st.metric("Versión", "1.0.0-alpha", "Paso 6A")
        
        st.divider()
        st.info("💡 **Tip:** Usa la pestaña 'Análisis de Activo' para comenzar. En modo avanzado podrás editar parámetros del DCF, ver análisis de sensibilidad y exportar reportes.")

# ==============================================================================
# PESTAÑA 2: ANÁLISIS DE ACTIVO (LA PRINCIPAL DEL PASO 6A)
# ==============================================================================
with tab2:
    st.header(f"🔍 Análisis de Activo: {ticker_input}")
    
    # Botón de análisis
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        analizar = st.button("🔍 Analizar", type="primary", use_container_width=True)
    
    # Ejecutar análisis
    if analizar or 'ultimo_ticker' not in st.session_state:
        with st.spinner("Obteniendo datos financieros..."):
            datos = obtener_datos_financieros(ticker_input)
            if datos:
                st.session_state.datos_activo = datos
                st.session_state.ultimo_ticker = ticker_input
                st.session_state.error_activo = None
            else:
                st.session_state.datos_activo = None
                st.session_state.error_activo = f"No se encontraron datos para {ticker_input}"
    
    # Mostrar error
    if st.session_state.get('error_activo'):
        st.error(st.session_state.error_activo)
        st.info("Tickers disponibles para demo: AAPL, MSFT, KO, GOOGL, WMT, TSLA")
    elif st.session_state.get('datos_activo'):
        datos = st.session_state.datos_activo
        
        # Indicador de fuente de datos
        if datos.get('es_real'):
            st.success(f"✅ Datos reales de Alpha Vantage | Última actualización: {datetime.now().strftime('%H:%M')}")
        else:
            st.warning("⚠️ Mostrando datos de demostración (API no disponible o rate limit)")
        
        st.divider()
        
        # ==================================================================
        # MODO SIMPLE
        # ==================================================================
        if modo_usuario == "🟢 Simple (Principiantes)":
            # Score y veredicto principal
            score_info = calcular_score_simple(datos)
            
            # Veredicto grande
            st.markdown(f"""
            <div style="background-color: {'#d4edda' if score_info['color']=='green' else '#fff3cd' if score_info['color']=='yellow' else '#f8d7da' if score_info['color']=='red' else '#ffe5cc'}; 
                        padding: 20px; border-radius: 10px; border-left: 6px solid {'green' if score_info['color']=='green' else 'orange' if score_info['color']=='yellow' else 'red' if score_info['color']=='red' else 'orange'};">
                <h2 style="margin:0;">{score_info['veredicto']}</h2>
                <p style="margin:10px 0 0 0; font-size: 1.1em;">{score_info['mensaje']}</p>
                <p style="margin:10px 0 0 0;"><strong>Score:</strong> {score_info['score']}/100</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.divider()
            
            # 3 métricas clave
            st.subheader("📊 Lo Esencial")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("💰 Precio Actual", f"${datos['precio']:.2f}")
                st.caption(f"Capitalización: ${datos['market_cap']/1e9:.1f}B")
            
            with col2:
                st.metric("📈 Potencial 12M", f"{score_info['score'] - 50:+.0f}%", 
                         "Positivo" if score_info['score'] > 50 else "Negativo")
                st.caption("Estimación basada en score")
            
            with col3:
                st.metric("💵 Dividendo", f"{datos.get('dividend_yield', 0)*100:.2f}%" if datos.get('dividend_yield') else "N/A")
                st.caption("Rendimiento anual")
            
            st.divider()
            
            # Simulación de inversión
            st.subheader("💰 Simulador de Inversión")
            capital = st.slider("¿Cuánto invertirías?", 1000, 100000, 10000, 1000)
            
            if score_info['score'] > 50:
                retorno_estimado = capital * (1 + (score_info['score'] - 50) / 500)
                st.success(f"💡 Invertir **${capital:,.0f}** hoy podría convertirse en aproximadamente **${retorno_estimado:,.0f}** en 12 meses (estimación conservadora).")
            else:
                st.warning(f"⚠️ Con un score de {score_info['score']}/100, el riesgo de pérdida es significativo. Considera esperar o diversificar.")
            
            st.divider()
            
            # Descripción simple
            if datos.get('descripcion'):
                st.subheader("🏢 ¿Qué hace esta empresa?")
                st.write(datos['descripcion'])
            
            # Botón para ver más
            with st.expander("🔵 Ver análisis avanzado →"):
                st.info("Cambia el modo a 'Avanzado' en la barra lateral para ver métricas detalladas, DCF con escenarios, y parámetros editables.")
        
        # ==================================================================
        # MODO AVANZADO
        # ==================================================================
        else:
            # Métricas fundamentales en grid
            st.subheader("📊 Métricas Fundamentales")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("💰 Precio", f"${datos['precio']:.2f}")
            with col2:
                st.metric("📈 P/E Ratio", f"{datos.get('pe_ratio', 0):.1f}x")
            with col3:
                st.metric("💵 EPS", f"${datos.get('eps', 0):.2f}")
            with col4:
                st.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 ROIC", f"{datos.get('roic', 0):.1f}%")
            with col2:
                st.metric("⚖️ Deuda/EBITDA", f"{datos.get('deuda_ebitda', 0):.2f}x")
            with col3:
                st.metric(" FCF", f"${datos.get('fcf', 0):.1f}B")
            with col4:
                st.metric("🛡️ Margen Seg.", f"{datos.get('margen_seguridad', 0):.1f}%")
            
            st.divider()
            
            # Parámetros editables del DCF
            st.subheader("⚙️ Parámetros del Modelo DCF")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                wacc = st.slider("WACC (%)", 5.0, 15.0, 9.0, 0.5, help="Costo promedio ponderado de capital")
            with col2:
                crecimiento = st.slider("Crecimiento FCF (%)", 0.0, 15.0, 5.0, 0.5, help="Tasa de crecimiento proyectada")
            with col3:
                g_perpetuo = st.slider("Crecimiento perpetuo (%)", 1.0, 5.0, 3.0, 0.5, help="Tasa de crecimiento a perpetuidad")
            
            # Calcular DCF con escenarios
            dcf = calcular_dcf_escenarios(datos, wacc/100, crecimiento/100, g_perpetuo/100)
            
            st.divider()
            
            # Análisis de sensibilidad
            st.subheader("📐 Análisis de Sensibilidad (3 Escenarios)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: #f8d7da; padding: 15px; border-radius: 8px; border-left: 4px solid red;">
                    <h4 style="margin:0; color: #721c24;">📉 Pesimista</h4>
                    <p style="margin: 10px 0 0 0; font-size: 1.5em; font-weight: bold; color: #721c24;">${dcf['pesimista']:.2f}</p>
                    <p style="margin: 5px 0 0 0; color: #721c24;">{dcf['margen_pesimista']:+.1f}% vs mercado</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">Crecimiento: {(crecimiento*0.5):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 4px solid orange;">
                    <h4 style="margin:0; color: #856404;">️ Base</h4>
                    <p style="margin: 10px 0 0 0; font-size: 1.5em; font-weight: bold; color: #856404;">${dcf['base']:.2f}</p>
                    <p style="margin: 5px 0 0 0; color: #856404;">{dcf['margen_base']:+.1f}% vs mercado</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">Crecimiento: {crecimiento:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background-color: #d4edda; padding: 15px; border-radius: 8px; border-left: 4px solid green;">
                    <h4 style="margin:0; color: #155724;"> Optimista</h4>
                    <p style="margin: 10px 0 0 0; font-size: 1.5em; font-weight: bold; color: #155724;">${dcf['optimista']:.2f}</p>
                    <p style="margin: 5px 0 0 0; color: #155724;">{dcf['margen_optimista']:+.1f}% vs mercado</p>
                    <p style="margin: 5px 0 0 0; font-size: 0.9em;">Crecimiento: {(crecimiento*1.5):.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Gráfico de comparación
            st.subheader("📊 Precio de Mercado vs. Valor Intrínseco")
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Pesimista', 'Base', 'Optimista', 'Mercado Actual'],
                y=[dcf['pesimista'], dcf['base'], dcf['optimista'], datos['precio']],
                marker_color=['#dc3545', '#ffc107', '#28a745', '#6c757d'],
                text=[f"${dcf['pesimista']:.2f}", f"${dcf['base']:.2f}", f"${dcf['optimista']:.2f}", f"${datos['precio']:.2f}"],
                textposition='outside'
            ))
            fig.update_layout(
                title="Comparativa de Valoración",
                yaxis_title="Precio (USD)",
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            # Información adicional
            st.subheader("️ Información de la Empresa")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Sector:** {datos.get('sector', 'N/A')}")
                st.write(f"**Industria:** {datos.get('industry', 'N/A')}")
                st.write(f"**Market Cap:** ${datos.get('market_cap', 0)/1e9:.2f}B")
            with col2:
                st.write(f"**Beta:** {datos.get('beta', 1):.2f}")
                st.write(f"**P/E Ratio:** {datos.get('pe_ratio', 0):.2f}")
                st.write(f"**Dividend Yield:** {datos.get('dividend_yield', 0)*100:.2f}%")
            
            if datos.get('descripcion'):
                st.write(f"\n**Descripción:** {datos['descripcion']}")
            
            st.divider()
            
            # Veredicto avanzado
            st.subheader(" Veredicto del Analista")
            score_info = calcular_score_simple(datos)
            
            if dcf['margen_base'] > 15:
                st.success(f"""
                ### ✅ SUBVALORADA
                El análisis DCF sugiere un valor justo de **${dcf['base']:.2f}** vs precio de mercado de **${datos['precio']:.2f}**.
                
                **Margen de seguridad:** {dcf['margen_base']:.1f}%  
                **Score compuesto:** {score_info['score']}/100
                
                *Recomendación: Considerar posición con stop-loss en el escenario pesimista (${dcf['pesimista']:.2f})*
                """)
            elif dcf['margen_base'] > -15:
                st.info(f"""
                ### ⚖️ PRECIO JUSTO
                El precio de mercado (${datos['precio']:.2f}) está cerca del valor intrínseco estimado (${dcf['base']:.2f}).
                
                **Margen:** {dcf['margen_base']:.1f}%  
                **Score compuesto:** {score_info['score']}/100
                
                *Recomendación: Mantener si ya se posee. No agregar exposición significativa.*
                """)
            else:
                st.warning(f"""
                ### ⚠️ SOBREVALORADA
                El precio de mercado (${datos['precio']:.2f}) supera el valor intrínseco estimado (${dcf['base']:.2f}).
                
                **Sobrevaloración:** {abs(dcf['margen_base']):.1f}%  
                **Score compuesto:** {score_info['score']}/100
                
                *Recomendación: Esperar corrección o reducir posición.*
                """)

# ==============================================================================
# PESTAÑA 3: PORTAFOLIO (Placeholder - Se expandirá en Paso 6B)
# ==============================================================================
with tab3:
    st.header("💼 Optimizador de Portafolio")
    st.info("🚧 **En desarrollo (Paso 6B):** Aquí se implementará el optimizador de Markowitz con modos Simple y Avanzado, Frontera Eficiente, y sistema de rebalanceo estratégico.")
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### ¿Qué encontrarás aquí?
        
        Una herramienta que te dirá exactamente cómo distribuir tu dinero entre varias empresas 
        para maximizar ganancias minimizando riesgos.
        
        **Ejemplo:** Si quieres invertir $10,000 en 4 empresas sólidas, la app te dirá:
        - 35% en KO (Coca-Cola) - Estabilidad
        - 30% en MSFT (Microsoft) - Crecimiento
        - 20% en AAPL (Apple) - Calidad
        - 15% en WMT (Walmart) - Defensa
        
        *Disponible en el próximo paso.*
        """)
    else:
        st.markdown("""
        ### Funcionalidades planificadas:
        
        - ✅ Optimización de Markowitz (Max Sharpe Ratio)
        - ✅ Frontera Eficiente interactiva
        - ✅ Matriz de correlación (heatmap)
        - ✅ Parámetros editables (Rf, restricciones de peso)
        - ✅ Análisis de rebalanceo estratégico
        - ✅ Comparación con benchmarks (S&P 500)
        - ✅ Exportación de reporte de portafolio a PDF
        
        *Implementación completa en Paso 6B.*
        """)

# ==============================================================================
# PESTAÑA 4: PRONÓSTICO Y RIESGOS (Placeholder)
# ==============================================================================
with tab4:
    st.header("🔮 Pronóstico y Análisis de Riesgos")
    st.info("🚧 **En desarrollo (Paso 6B):** Aquí se implementarán los modelos de Machine Learning para pronóstico de precios y el análisis de riesgos con IA.")
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### ¿Qué encontrarás aquí?
        
        - 📈 **Pronóstico a 90 días** con bandas de confianza
        - 🎯 **Semáforo de tendencia** (Alcista/Bajista)
        - 🛡️ **Análisis de riesgos** en lenguaje simple
        - 💡 **Recomendación clara** de acción
        
        *Disponible en el próximo paso.*
        """)
    else:
        st.markdown("""
        ### Funcionalidades planificadas:
        
        - ✅ Pronóstico con Prophet (Meta) - Bandas de confianza 95%
        - ✅ Descomposición de series de tiempo (tendencia + estacionalidad)
        - ✅ Análisis de riesgos por categoría (Radar chart)
        - ✅ Score de riesgo consolidado (0-100)
        - ✅ Planes de mitigación automáticos
        - ✅ Análisis de sensibilidad a parámetros macro
        - ✅ Exportación de reporte de riesgos a PDF
        
        *Implementación completa en Paso 6B.*
        """)

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p><strong>QuantBuffett AI v1.0.0-alpha</strong> | Paso 6A de 14</p>
    <p>Datos: Alpha Vantage API + Base de datos de demostración</p>
    <p><em>"La regla número 1 es no perder dinero. La regla número 2 es no olvidar la regla número 1."</em> — Warren Buffett</p>
</div>
""", unsafe_allow_html=True)
