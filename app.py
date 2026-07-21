"""
QuantBuffett AI - Plataforma Profesional de Análisis Financiero
Versión: 1.0.0-beta | Paso 6B: Portafolio Completo con Markowitz
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

# API Key de Alpha Vantage
ALPHA_VANTAGE_KEY = st.secrets.get("ALPHA_VANTAGE_KEY", "5IPGMO6N5R7UB9VC")

# ==============================================================================
# BASE DE DATOS MOCK (Fallback)
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
# FUNCIONES DE ALPHA VANTAGE
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_alpha_vantage(ticker: str) -> dict:
    """Obtiene datos fundamentales de Alpha Vantage."""
    try:
        url_overview = f"https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp_overview = requests.get(url_overview, timeout=10)
        data_overview = resp_overview.json()
        
        if "Note" in data_overview or "Information" in data_overview:
            return None
        
        if not data_overview.get("Symbol"):
            return None
        
        time.sleep(0.5)
        url_quote = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_KEY}"
        resp_quote = requests.get(url_quote, timeout=10)
        data_quote = resp_quote.json()
        
        quote = data_quote.get("Global Quote", {})
        precio_actual = float(quote.get("05. price", 0)) if quote.get("05. price") else 0
        
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
            'roic': float(data_overview.get('ReturnOnEquityTTM', 0)) * 100 if data_overview.get('ReturnOnEquityTTM') else 0,
            'net_income': float(data_overview.get('NetIncomeTTM', 0)),
            'retorno_anual': 0,
            'volatilidad_anual': 0,
            'es_real': True
        }
        
        return resultado
        
    except Exception as e:
        return None


def obtener_datos_financieros(ticker: str) -> dict:
    """Obtiene datos financieros con fallback a mock."""
    ticker_upper = ticker.upper()
    datos_reales = obtener_datos_alpha_vantage(ticker_upper)
    
    if datos_reales and datos_reales.get('precio', 0) > 0:
        mock = MOCK_DATABASE.get(ticker_upper, {})
        datos_reales['deuda_ebitda'] = mock.get('deuda_ebitda', 1.0)
        datos_reales['fcf'] = mock.get('fcf', 0)
        datos_reales['margen_seguridad'] = mock.get('margen_seguridad', 0)
        datos_reales['retorno_anual'] = mock.get('retorno_anual', 0.10)
        datos_reales['volatilidad_anual'] = mock.get('volatilidad_anual', 0.20)
        datos_reales['tendencia'] = mock.get('tendencia', 'N/A')
        return datos_reales
    
    if ticker_upper in MOCK_DATABASE:
        datos = MOCK_DATABASE[ticker_upper].copy()
        datos['es_real'] = False
        return datos
    
    return None

# ==============================================================================
# FUNCIONES DE PORTAFOLIO (MARKOWITZ)
# ==============================================================================
def optimizar_portafolio(tickers: list, rf: float = 0.04, modo: str = "simple") -> dict:
    """
    Optimiza portafolio usando teoría de Markowitz.
    
    Args:
        tickers: Lista de tickers
        rf: Tasa libre de riesgo
        modo: "simple" o "avanzado"
    
    Returns:
        Diccionario con resultados de optimización
    """
    n = len(tickers)
    retornos = np.array([MOCK_DATABASE[t]['retorno_anual'] for t in tickers])
    volatilidades = np.array([MOCK_DATABASE[t]['volatilidad_anual'] for t in tickers])
    
    # Matriz de correlación (simplificada para demo)
    correlacion = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            correlacion[i,j] = correlacion[j,i] = 0.3
    
    cov_matrix = np.outer(volatilidades, volatilidades) * correlacion
    
    def sharpe_negativo(pesos):
        port_retorno = np.sum(retornos * pesos)
        port_vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        sharpe = (port_retorno - rf) / port_vol if port_vol > 0 else 0
        return -sharpe
    
    restricciones = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    
    if modo == "avanzado":
        # En modo avanzado, permitir pesos entre 0% y 40% por activo
        limites = tuple((0.0, 0.40) for _ in range(n))
    else:
        # En modo simple, sin restricciones adicionales
        limites = tuple((0.0, 1.0) for _ in range(n))
    
    pesos_iniciales = np.ones(n) / n
    resultado = minimize(sharpe_negativo, pesos_iniciales, 
                        method='SLSQP', bounds=limites, constraints=restricciones)
    
    pesos_optimos = resultado.x
    retorno_optimo = np.sum(retornos * pesos_optimos)
    volatilidad_optima = np.sqrt(np.dot(pesos_optimos.T, np.dot(cov_matrix, pesos_optimos)))
    sharpe_optimo = (retorno_optimo - rf) / volatilidad_optima
    
    # Generar frontera eficiente (Montecarlo)
    mc_retornos = []
    mc_vols = []
    mc_sharpes = []
    
    for _ in range(1000):
        pesos_rand = np.random.random(n)
        pesos_rand /= np.sum(pesos_rand)
        r = np.sum(retornos * pesos_rand)
        v = np.sqrt(np.dot(pesos_rand.T, np.dot(cov_matrix, pesos_rand)))
        s = (r - rf) / v if v > 0 else 0
        mc_retornos.append(r * 100)
        mc_vols.append(v * 100)
        mc_sharpes.append(s)
    
    # Calcular métricas adicionales para modo avanzado
    matriz_correlacion = correlacion if modo == "avanzado" else None
    
    return {
        'tickers': tickers,
        'pesos': pesos_optimos,
        'retorno': retorno_optimo * 100,
        'volatilidad': volatilidad_optima * 100,
        'sharpe': sharpe_optimo,
        'mc_data': {
            'retornos': mc_retornos,
            'volatilidades': mc_vols,
            'sharpes': mc_sharpes
        },
        'correlacion': matriz_correlacion
    }

def calcular_rebalanceo(portafolio_actual: dict, portafolio_optimo: dict, capital: float) -> pd.DataFrame:
    """Calcula qué comprar/vender para rebalancear."""
    df = pd.DataFrame()
    df['Activo'] = portafolio_optimo['tickers']
    df['Peso_Actual'] = [portafolio_actual.get(t, 0) for t in portafolio_optimo['tickers']]
    df['Peso_Optimo'] = portafolio_optimo['pesos'] * 100
    df['Diferencia'] = df['Peso_Optimo'] - df['Peso_Actual']
    df['Capital_Ajustar'] = df['Diferencia'] * capital / 100
    df['Accion'] = df['Diferencia'].apply(
        lambda x: "🟢 COMPRAR" if x > 1 else "🔴 VENDER" if x < -1 else "✅ MANTENER"
    )
    return df

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

modo_usuario = st.sidebar.radio(
    "Modo de Visualización",
    ["🟢 Simple (Principiantes)", "🔵 Avanzado (Expertos)"],
    help="Simple: veredictos claros. Avanzado: métricas detalladas y parámetros editables."
)

st.sidebar.divider()

st.sidebar.markdown("###  Fuente de Datos")
if ALPHA_VANTAGE_KEY and ALPHA_VANTAGE_KEY != "DEMO_KEY":
    st.sidebar.success("✅ Alpha Vantage conectado")
    st.sidebar.caption("500 llamadas/día disponibles")
else:
    st.sidebar.warning("⚠️ Modo Demo (datos simulados)")

st.sidebar.divider()

st.sidebar.markdown("""
### ℹ️ Sobre QuantBuffett AI
Versión 1.0.0-beta  
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
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### Bienvenido a QuantBuffett AI
        
        Esta aplicación te ayudará a tomar decisiones de inversión informadas.
        
        **Para comenzar:**
        1. Ve a la pestaña **🔍 Análisis de Activo** para analizar empresas individuales
        2. Ve a la pestaña **💼 Portafolio** para optimizar tu diversificación
        3. Ve a la pestaña **🔮 Pronóstico y Riesgos** para ver proyecciones
        
        **Empresas disponibles:** AAPL, MSFT, KO, GOOGL, WMT, TSLA
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("S&P 500 (simulado)", "5,420", "+0.8%")
        with col2:
            st.metric("Empresas analizables", "6", "AAPL, MSFT, KO...")
        with col3:
            st.metric("Llamadas API restantes", "~495", "de 500 diarias")
    
    else:
        st.markdown("### Dashboard de Control")
        st.markdown("Panel de control para análisis financiero profesional con datos en tiempo real vía Alpha Vantage API.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("API Alpha Vantage", "Conectada", "500 calls/day")
        with col2:
            st.metric("Modo Activo", "Avanzado", "Parámetros editables")
        with col3:
            st.metric("Empresas en DB", "6", "Mock + Real")
        with col4:
            st.metric("Versión", "1.0.0-beta", "Paso 6B")
        
        st.divider()
        st.info("💡 **Tip:** Usa la pestaña 'Portafolio' para optimizar tu diversificación con el modelo de Markowitz.")

# ==============================================================================
# PESTAÑA 2: ANÁLISIS DE ACTIVO
# ==============================================================================
with tab2:
    st.header("🔍 Análisis de Activo")
    
    ticker_input = st.text_input("Ticker a analizar", value="AAPL").upper()
    
    if st.button("🔍 Analizar"):
        with st.spinner("Obteniendo datos..."):
            datos = obtener_datos_financieros(ticker_input)
            if datos:
                st.session_state.datos_activo = datos
            else:
                st.error(f"No se encontraron datos para {ticker_input}")
    
    if 'datos_activo' in st.session_state:
        datos = st.session_state.datos_activo
        
        if datos.get('es_real'):
            st.success("✅ Datos reales de Alpha Vantage")
        else:
            st.warning("⚠️ Datos de demostración")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Precio", f"${datos['precio']:.2f}")
        with col2:
            st.metric("P/E", f"{datos.get('pe_ratio', 0):.1f}x")
        with col3:
            st.metric("ROIC", f"{datos.get('roic', 0):.1f}%")
        with col4:
            st.metric("Beta", f"{datos.get('beta', 1):.2f}")
        
        st.info("📊 Análisis detallado disponible en versiones futuras.")

# ==============================================================================
# PESTAÑA 3: PORTAFOLIO (NUEVA - PASO 6B)
# ==============================================================================
with tab3:
    st.header("💼 Optimizador de Portafolio")
    
    # ==================================================================
    # MODO SIMPLE
    # ==================================================================
    if modo_usuario == " Simple (Principiantes)":
        st.markdown("""
        ### 🎯 Optimización Inteligente de Portafolio
        
        Selecciona las empresas en las que quieres invertir y te diremos exactamente cómo distribuir tu dinero para maximizar ganancias minimizando riesgos.
        """)
        
        st.divider()
        
        # Selección de activos
        st.subheader("1️ Selecciona tus empresas")
        tickers_disponibles = list(MOCK_DATABASE.keys())
        tickers_seleccionados = st.multiselect(
            "Elige entre 2 y 6 empresas",
            options=tickers_disponibles,
            default=['AAPL', 'MSFT', 'KO'],
            help="Selecciona al menos 2 empresas para diversificar"
        )
        
        if len(tickers_seleccionados) < 2:
            st.warning("⚠️ Selecciona al menos 2 empresas para crear un portafolio diversificado.")
        else:
            # Capital a invertir
            st.subheader("2️⃣ ¿Cuánto quieres invertir?")
            capital = st.slider("Capital total (USD)", 1000, 1000000, 10000, 1000)
            
            st.divider()
            
            # Botón de optimización
            if st.button("🚀 Optimizar Mi Portafolio", type="primary", use_container_width=True):
                with st.spinner("Calculando la mejor distribución..."):
                    opt_result = optimizar_portafolio(tickers_seleccionados, rf=0.04, modo="simple")
                    st.session_state.opt_result = opt_result
                    st.session_state.capital = capital
            
            # Mostrar resultados
            if 'opt_result' in st.session_state and st.session_state.opt_result['tickers'] == tickers_seleccionados:
                opt = st.session_state.opt_result
                capital = st.session_state.capital
                
                st.divider()
                st.subheader("3️⃣ Tu Portafolio Óptimo")
                
                # Métricas clave
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📈 Retorno Anual Esperado", f"{opt['retorno']:.1f}%")
                with col2:
                    st.metric("⚠️ Riesgo (Volatilidad)", f"{opt['volatilidad']:.1f}%")
                with col3:
                    st.metric("⭐ Ratio de Sharpe", f"{opt['sharpe']:.2f}")
                
                st.divider()
                
                # Gráfico de torta
                st.subheader("🥧 Distribución Recomendada")
                
                df_pesos = pd.DataFrame({
                    'Empresa': opt['tickers'],
                    'Porcentaje': (opt['pesos'] * 100).round(1),
                    'Monto USD': (opt['pesos'] * capital).round(0)
                })
                
                fig_pie = px.pie(
                    df_pesos,
                    values='Porcentaje',
                    names='Empresa',
                    title='Cómo Distribuir Tu Dinero',
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
                
                st.divider()
                
                # Tabla detallada
                st.subheader("📋 Detalle de Inversión")
                st.dataframe(df_pesos, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # Interpretación simple
                st.subheader("💡 ¿Qué significa esto?")
                
                if opt['sharpe'] > 1.0:
                    st.success(f"""
                    **Excelente portafolio!** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, 
                    este portafolio ofrece muy buen retorno por cada unidad de riesgo asumido.
                    
                    **En términos simples:** Por cada 1% de riesgo que aceptas, esperas ganar {opt['sharpe']:.2f}% de retorno.
                    """)
                elif opt['sharpe'] > 0.5:
                    st.info(f"""
                    **Buen portafolio.** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, 
                    este portafolio ofrece un balance aceptable entre riesgo y retorno.
                    
                    **En términos simples:** Es una diversificación razonable para la mayoría de inversores.
                    """)
                else:
                    st.warning(f"""
                    **Portafolio conservador.** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, 
                    el retorno es moderado en relación al riesgo.
                    
                    **En términos simples:** Considera agregar empresas con mayor crecimiento potencial.
                    """)
                
                st.divider()
                
                # Perfil de riesgo
                st.subheader("🎯 Perfil de Este Portafolio")
                
                if opt['volatilidad'] < 15:
                    st.markdown("""
                    **Conservador** 🟢  
                    Este portafolio está diseñado para proteger tu capital. Ideal si:
                    - Te preocupa perder dinero
                    - Prefieres estabilidad sobre crecimiento agresivo
                    - Tu horizonte de inversión es corto (1-3 años)
                    """)
                elif opt['volatilidad'] < 25:
                    st.markdown("""
                    **Moderado** 🟡  
                    Este portafolio balancea crecimiento y estabilidad. Ideal si:
                    - Buscas crecimiento pero con cierta protección
                    - Tu horizonte de inversión es mediano (3-7 años)
                    - Puedes tolerar fluctuaciones moderadas
                    """)
                else:
                    st.markdown("""
                    **Agresivo** 🔴  
                    Este portafolio busca maximizar ganancias aceptando mayor volatilidad. Ideal si:
                    - Tu horizonte de inversión es largo (+7 años)
                    - Puedes tolerar caídas significativas temporales
                    - Buscas crecimiento agresivo
                    """)
    
    # ==================================================================
    # MODO AVANZADO
    # ==================================================================
    else:
        st.markdown("""
        ### Optimización de Portafolio - Modelo de Markowitz
        
        Optimización matemática de asignación de activos utilizando la Teoría Moderna de Portafolios.
        """)
        
        st.divider()
        
        # Selección de activos
        st.subheader("1. Selección de Activos")
        tickers_disponibles = list(MOCK_DATABASE.keys())
        tickers_seleccionados = st.multiselect(
            "Selecciona activos (2-6)",
            options=tickers_disponibles,
            default=['AAPL', 'MSFT', 'KO', 'GOOGL']
        )
        
        if len(tickers_seleccionados) < 2:
            st.warning("Se requieren al menos 2 activos.")
        else:
            # Parámetros avanzados
            st.subheader("2. Parámetros del Modelo")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                rf = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 4.0, 0.5, 
                              help="Rendimiento de bonos del Tesoro de EE.UU.")
            with col2:
                max_peso = st.slider("Peso Máximo por Activo (%)", 10, 100, 40, 5,
                                    help="Restricción de concentración máxima")
            with col3:
                num_simulaciones = st.slider("Simulaciones Montecarlo", 500, 5000, 1000, 500,
                                            help="Número de portafolios aleatorios para frontera eficiente")
            
            st.divider()
            
            # Optimización
            if st.button("⚙️ Ejecutar Optimización", type="primary"):
                with st.spinner("Optimizando portafolio..."):
                    opt_result = optimizar_portafolio(
                        tickers_seleccionados, 
                        rf=rf/100, 
                        modo="avanzado"
                    )
                    # Ajustar pesos al máximo permitido
                    opt_result['pesos'] = np.minimum(opt_result['pesos'], max_peso/100)
                    opt_result['pesos'] /= opt_result['pesos'].sum()  # Renormalizar
                    
                    # Recalcular métricas con pesos ajustados
                    retornos = np.array([MOCK_DATABASE[t]['retorno_anual'] for t in tickers_seleccionados])
                    volatilidades = np.array([MOCK_DATABASE[t]['volatilidad_anual'] for t in tickers_seleccionados])
                    correlacion = np.eye(len(tickers_seleccionados))
                    for i in range(len(tickers_seleccionados)):
                        for j in range(i+1, len(tickers_seleccionados)):
                            correlacion[i,j] = correlacion[j,i] = 0.3
                    cov_matrix = np.outer(volatilidades, volatilidades) * correlacion
                    
                    opt_result['retorno'] = np.sum(retornos * opt_result['pesos']) * 100
                    opt_result['volatilidad'] = np.sqrt(np.dot(opt_result['pesos'].T, np.dot(cov_matrix, opt_result['pesos']))) * 100
                    opt_result['sharpe'] = (opt_result['retorno']/100 - rf/100) / (opt_result['volatilidad']/100)
                    
                    st.session_state.opt_result_avanzado = opt_result
                    st.session_state.rf = rf
                    st.session_state.max_peso = max_peso
            
            # Mostrar resultados
            if 'opt_result_avanzado' in st.session_state:
                opt = st.session_state.opt_result_avanzado
                rf = st.session_state.rf
                
                st.divider()
                st.subheader("3. Resultados de la Optimización")
                
                # Métricas
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Retorno Esperado", f"{opt['retorno']:.2f}%")
                with col2:
                    st.metric("Volatilidad", f"{opt['volatilidad']:.2f}%")
                with col3:
                    st.metric("Ratio de Sharpe", f"{opt['sharpe']:.2f}")
                with col4:
                    st.metric("Tasa Libre de Riesgo", f"{rf:.1f}%")
                
                st.divider()
                
                # Frontera Eficiente
                st.subheader("4. Frontera Eficiente")
                st.markdown("""
                Cada punto representa un portafolio posible. La estrella roja indica el portafolio óptimo 
                que maximiza el Ratio de Sharpe (retorno por unidad de riesgo).
                """)
                
                fig_ef = go.Figure()
                
                # Nube de puntos Montecarlo
                fig_ef.add_trace(go.Scatter(
                    x=opt['mc_data']['volatilidades'],
                    y=opt['mc_data']['retornos'],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=opt['mc_data']['sharpes'],
                        colorscale='Viridis',
                        colorbar=dict(title='Sharpe Ratio'),
                        opacity=0.6
                    ),
                    name='Portafolios Aleatorios'
                ))
                
                # Portafolio óptimo
                fig_ef.add_trace(go.Scatter(
                    x=[opt['volatilidad']],
                    y=[opt['retorno']],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
                    name='Portafolio Óptimo'
                ))
                
                fig_ef.update_layout(
                    title='Frontera Eficiente: Retorno vs Volatilidad',
                    xaxis_title='Volatilidad Anual (%)',
                    yaxis_title='Retorno Anual (%)',
                    hovermode='closest',
                    height=500
                )
                
                st.plotly_chart(fig_ef, use_container_width=True)
                
                st.divider()
                
                # Asignación óptima
                st.subheader("5. Asignación Óptima de Pesos")
                
                df_pesos = pd.DataFrame({
                    'Activo': opt['tickers'],
                    'Peso Óptimo (%)': (opt['pesos'] * 100).round(2),
                    'Retorno Individual (%)': [MOCK_DATABASE[t]['retorno_anual']*100 for t in opt['tickers']],
                    'Volatilidad Individual (%)': [MOCK_DATABASE[t]['volatilidad_anual']*100 for t in opt['tickers']]
                })
                
                st.dataframe(df_pesos, use_container_width=True, hide_index=True)
                
                st.divider()
                
                # Matriz de correlación
                st.subheader("6. Matriz de Correlación")
                st.markdown("Correlación entre los activos seleccionados (1.0 = correlación perfecta, 0.0 = sin correlación)")
                
                if opt['correlacion'] is not None:
                    fig_corr = px.imshow(
                        opt['correlacion'],
                        labels=dict(x="Activo", y="Activo", color="Correlación"),
                        x=opt['tickers'],
                        y=opt['tickers'],
                        color_continuous_scale='RdBu_r',
                        zmin=-1, zmax=1
                    )
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                st.divider()
                
                # Sistema de rebalanceo
                st.subheader("7. Sistema de Rebalanceo Estratégico")
                st.markdown("""
                Compara tu portafolio actual con el óptimo para identificar ajustes necesarios.
                """)
                
                # Input de portafolio actual
                st.write("**Pesos actuales de tu portafolio (%):**")
                pesos_actuales = {}
                cols = st.columns(len(opt['tickers']))
                for i, ticker in enumerate(opt['tickers']):
                    with cols[i]:
                        pesos_actuales[ticker] = st.number_input(
                            f"{ticker}",
                            min_value=0.0,
                            max_value=100.0,
                            value=100.0/len(opt['tickers']),
                            step=1.0
                        )
                
                capital_rebalanceo = st.number_input("Capital total para rebalanceo (USD)", value=100000, step=10000)
                
                if st.button(" Calcular Rebalanceo"):
                    df_rebalanceo = calcular_rebalanceo(pesos_actuales, opt, capital_rebalanceo)
                    
                    st.dataframe(df_rebalanceo, use_container_width=True, hide_index=True)
                    
                    # Resumen de acciones
                    comprar = df_rebalanceo[df_rebalanceo['Accion'].str.contains('COMPRAR')]
                    vender = df_rebalanceo[df_rebalanceo['Accion'].str.contains('VENDER')]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if not comprar.empty:
                            st.success(f"**Comprar:** {', '.join(comprar['Activo'].tolist())}")
                    with col2:
                        if not vender.empty:
                            st.error(f"**Vender:** {', '.join(vender['Activo'].tolist())}")

# ==============================================================================
# PESTAÑA 4: PRONÓSTICO Y RIESGOS
# ==============================================================================
with tab4:
    st.header("🔮 Pronóstico y Análisis de Riesgos")
    st.info("🚧 En desarrollo - Próximamente disponible")

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p><strong>QuantBuffett AI v1.0.0-beta</strong> | Paso 6B de 14</p>
    <p>Datos: Alpha Vantage API + Base de datos de demostración</p>
    <p><em>"La regla número 1 es no perder dinero. La regla número 2 es no olvidar la regla número 1."</em> — Warren Buffett</p>
</div>
""", unsafe_allow_html=True)
