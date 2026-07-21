"""
QuantBuffett AI - Plataforma de Análisis Financiero Profesional
Versión: 0.4.0 (Con Optimización de Portafolio Markowitz)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize

# ==============================================================================
# BASE DE DATOS DE EJEMPLO (Datos realistas)
# ==============================================================================
MOCK_DATABASE = {
    'AAPL': {
        'ticker': 'AAPL', 'precio': 308.50, 'market_cap': 2400000000000,
        'roic': 55.2, 'deuda_ebitda': 0.35, 'fcf': 105.8,
        'net_income': 112000000000, 'ebit': 130000000000,
        'margen_seguridad': -4.3, 'beta': 1.10,
        'sector': 'Technology', 'industry': 'Consumer Electronics',
        'retorno_anual': 0.28, 'volatilidad_anual': 0.25
    },
    'MSFT': {
        'ticker': 'MSFT', 'precio': 415.20, 'market_cap': 3100000000000,
        'roic': 38.1, 'deuda_ebitda': 0.42, 'fcf': 78.5,
        'net_income': 88000000000, 'ebit': 105000000000,
        'margen_seguridad': 3.5, 'beta': 0.95,
        'sector': 'Technology', 'industry': 'Software',
        'retorno_anual': 0.32, 'volatilidad_anual': 0.22
    },
    'KO': {
        'ticker': 'KO', 'precio': 62.30, 'market_cap': 270000000000,
        'roic': 16.6, 'deuda_ebitda': 1.50, 'fcf': 9.8,
        'net_income': 10500000000, 'ebit': 13000000000,
        'margen_seguridad': 12.5, 'beta': 0.65,
        'sector': 'Consumer Defensive', 'industry': 'Beverages',
        'retorno_anual': 0.08, 'volatilidad_anual': 0.15
    },
    'GOOGL': {
        'ticker': 'GOOGL', 'precio': 175.80, 'market_cap': 2200000000000,
        'roic': 26.4, 'deuda_ebitda': 0.28, 'fcf': 65.2,
        'net_income': 75000000000, 'ebit': 95000000000,
        'margen_seguridad': 8.2, 'beta': 1.05,
        'sector': 'Communication Services', 'industry': 'Internet Content',
        'retorno_anual': 0.25, 'volatilidad_anual': 0.28
    },
    'WMT': {
        'ticker': 'WMT', 'precio': 85.40, 'market_cap': 230000000000,
        'roic': 14.2, 'deuda_ebitda': 1.85, 'fcf': 12.5,
        'net_income': 15000000000, 'ebit': 22000000000,
        'margen_seguridad': 5.8, 'beta': 0.55,
        'sector': 'Consumer Defensive', 'industry': 'Discount Stores',
        'retorno_anual': 0.12, 'volatilidad_anual': 0.18
    },
    'TSLA': {
        'ticker': 'TSLA', 'precio': 245.60, 'market_cap': 780000000000,
        'roic': 12.8, 'deuda_ebitda': 0.95, 'fcf': 8.2,
        'net_income': 12000000000, 'ebit': 15000000000,
        'margen_seguridad': -15.2, 'beta': 2.05,
        'sector': 'Consumer Cyclical', 'industry': 'Auto Manufacturers',
        'retorno_anual': 0.45, 'volatilidad_anual': 0.55
    }
}

# ==============================================================================
# FUNCIONES AUXILIARES
# ==============================================================================
def obtener_datos_financieros(ticker: str):
    """Obtiene datos financieros del ticker."""
    ticker_upper = ticker.upper()
    if ticker_upper in MOCK_DATABASE:
        return MOCK_DATABASE[ticker_upper].copy()
    return None

def optimizar_portafolio(tickers: list, rf: float = 0.04) -> dict:
    """
    Optimiza portafolio usando teoría de Markowitz.
    
    Args:
        tickers: Lista de tickers
        rf: Tasa libre de riesgo (4%)
    
    Returns:
        Diccionario con resultados de optimización
    """
    # Datos de ejemplo para simulación
    n = len(tickers)
    retornos = np.array([MOCK_DATABASE[t]['retorno_anual'] for t in tickers])
    volatilidades = np.array([MOCK_DATABASE[t]['volatilidad_anual'] for t in tickers])
    
    # Matriz de correlación simplificada (ejemplo)
    correlacion = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            correlacion[i,j] = correlacion[j,i] = 0.3  # Correlación moderada
    
    # Matriz de covarianza
    cov_matrix = np.outer(volatilidades, volatilidades) * correlacion
    
    # Función objetivo: maximizar Sharpe Ratio
    def sharpe_negativo(pesos):
        port_retorno = np.sum(retornos * pesos)
        port_vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        sharpe = (port_retorno - rf) / port_vol if port_vol > 0 else 0
        return -sharpe
    
    # Restricciones: pesos suman 1, todos positivos
    restricciones = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    limites = tuple((0.0, 1.0) for _ in range(n))
    
    # Optimización
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
        }
    }

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
        value="AAPL, MSFT, KO, GOOGL",
        help="Ingresa múltiples tickers separados por coma"
    ).upper()

st.sidebar.divider()

# Botón para ejecutar análisis
ejecutar_analisis = st.sidebar.button(" Analizar Ahora", type="primary", use_container_width=True)

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

**Modo Demostración:** Mostrando datos de ejemplo.
""")

# ==============================================================================
# CONTENIDO PRINCIPAL
# ==============================================================================
st.header(f"Análisis en Modo: {modo_analisis}")

if modo_analisis == "🔍 Activo Único":
    st.info(f"🎯 Analizando: **{ticker_input}**")
    
    if ejecutar_analisis or 'datos_cache' not in st.session_state:
        with st.spinner(f"📊 Extrayendo datos financieros de {ticker_input}..."):
            try:
                datos = obtener_datos_financieros(ticker_input)
                
                if datos and datos.get('precio', 0) > 0:
                    st.session_state.datos_cache = datos
                    st.session_state.error = None
                else:
                    st.session_state.datos_cache = None
                    st.session_state.error = f"No se encontraron datos para {ticker_input}. Verifica el ticker."
                    
            except Exception as e:
                st.session_state.datos_cache = None
                st.session_state.error = f"Error al extraer datos: {str(e)}"
    
    # Mostrar resultados o error
    if st.session_state.error:
        st.error(st.session_state.error)
        st.info("💡 Consejo: Tickers disponibles: AAPL, MSFT, KO, GOOGL, WMT, TSLA")
    elif st.session_state.datos_cache:
        datos = st.session_state.datos_cache
        
        # Mostrar aviso de modo demostración
        st.warning("""
        ⚠️ **Modo Demostración**: Mostrando datos de ejemplo para desarrollo. 
        Los datos reales de Yahoo Finance se integrarán en la próxima versión.
        """)
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            market_cap_text = f"${datos['market_cap']/1e9:.1f}B" if datos.get('market_cap', 0) else "N/A"
            st.metric(
                label="💰 Precio Actual",
                value=f"${datos['precio']:.2f}",
                delta=market_cap_text,
                help="Precio de mercado en tiempo real"
            )
        
        with col2:
            roic_val = datos.get('roic', 0)
            roic_delta = "Excelente" if roic_val > 15 else "Bueno" if roic_val > 10 else "Regular"
            st.metric(
                label="📊 ROIC",
                value=f"{roic_val:.1f}%",
                delta=roic_delta,
                help="Return on Invested Capital - Eficiencia del negocio. >15% es excelente"
            )
        
        with col3:
            deuda_val = datos.get('deuda_ebitda', 0)
            deuda_status = "✅ Sólido" if deuda_val < 2 else "️ Moderado" if deuda_val < 4 else "🔴 Alto"
            st.metric(
                label="📉 Deuda/EBITDA",
                value=f"{deuda_val:.2f}x",
                delta=deuda_status,
                help="Ratio de solvencia - Capacidad de pago de deuda. <2x es sólido"
            )
        
        with col4:
            margen_val = datos.get('margen_seguridad', 0)
            margen_status = "🟢 Atractivo" if margen_val > 20 else " Justo" if margen_val > 0 else "🔴 Sobrevalorado"
            st.metric(
                label="🎯 Margen de Seguridad",
                value=f"{margen_val:.1f}%",
                delta=margen_status,
                help="Diferencia entre valor intrínseco (DCF) y precio de mercado"
            )
        
        # Análisis detallado
        st.divider()
        st.subheader("📈 Análisis Detallado")
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("### 💵 Flujo de Caja Libre")
            st.metric("FCF (Anual)", f"${datos.get('fcf', 0):.2f}B")
            st.caption("Free Cash Flow - Dinero real que genera la empresa")
        
        with col_b:
            st.markdown("### 📄 Utilidad Neta")
            net_income_billones = datos.get('net_income', 0) / 1e9
            st.metric("Net Income", f"${net_income_billones:.2f}B")
            st.caption("Beneficio contable después de impuestos")
        
        # Veredicto estilo Buffett
        st.divider()
        st.subheader("🎯 Veredicto del Agente")
        
        roic_ok = datos.get('roic', 0) > 15
        deuda_ok = datos.get('deuda_ebitda', 99) < 2.0
        margen_ok = datos.get('margen_seguridad', 0) > 0
        
        if roic_ok and deuda_ok and margen_ok:
            st.success("""
            ### ✅ COMPRA POTENCIAL
            **Empresa maravillosa a precio justo:**
            - ✅ ROIC excelente (>15%) - Negocio con foso competitivo
            - ✅ Deuda controlada (<2x EBITDA) - Salud financiera sólida
            - ✅ Margen de seguridad positivo - Precio atractivo
            
            *Cumple con los criterios de Warren Buffett*
            """)
        elif roic_ok and deuda_ok:
            st.warning("""
            ### ⏳ OBSERVAR / ESPERAR MEJOR PRECIO
            **Negocio de calidad pero precio elevado:**
            - ✅ ROIC excelente - Negocio maravilloso
            - ✅ Deuda controlada - Gestión conservadora
            - ️ Margen de seguridad negativo - Esperar corrección
            
            *Recomendación: Agregar a watchlist y esperar mejor entrada*
            """)
        else:
            st.info("""
            ### 🔍 ANÁLISIS MIXTO
            **Requiere análisis más profundo:**
            - Revisar tendencias históricas
            - Analizar ventajas competitivas
            - Evaluar catalizadores futuros
            
            *No cumple todos los criterios de calidad/valor*
            """)
        
        # Información adicional
        with st.expander("ℹ️ Información de la Empresa"):
            st.write(f"**Sector:** {datos.get('sector', 'N/A')}")
            st.write(f"**Industria:** {datos.get('industry', 'N/A')}")
            st.write(f"**Beta:** {datos.get('beta', 'N/A')}")
            st.write(f"**Ticker:** {datos.get('ticker', 'N/A')}")
    
    else:
        # Estado inicial (sin datos)
        st.markdown("""
        ### 👈 Ingresa un ticker y haz clic en "Analizar Ahora"
        
        **Tickers disponibles:** AAPL, MSFT, KO, GOOGL, WMT, TSLA
        
        La aplicación mostrará:
        - Precio actual y capitalización de mercado
        - ROIC (eficiencia del negocio)
        - Ratio Deuda/EBITDA (solvencia)
        - Margen de seguridad (valoración DCF)
        - Flujo de caja libre
        """)

else:  # Modo Portafolio
    tickers = [t.strip() for t in ticker_input.split(",") if t.strip()]
    st.info(f"💼 Analizando portafolio de **{len(tickers)} activos**: {', '.join(tickers)}")
    
    if ejecutar_analisis:
        # Validar que todos los tickers existen
        tickers_validos = [t for t in tickers if t in MOCK_DATABASE]
        
        if len(tickers_validos) < 2:
            st.error("Se necesitan al menos 2 tickers válidos para optimizar el portafolio.")
            st.info(f"Tickers disponibles: {', '.join(MOCK_DATABASE.keys())}")
        else:
            with st.spinner("📊 Optimizando portafolio..."):
                opt_result = optimizar_portafolio(tickers_validos)
            
            # Métricas del portafolio óptimo
            st.subheader("🎯 Portafolio Óptimo (Maximiza Ratio de Sharpe)")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Retorno Anual Esperado", f"{opt_result['retorno']:.1f}%")
            with col2:
                st.metric("Volatilidad (Riesgo)", f"{opt_result['volatilidad']:.1f}%")
            with col3:
                st.metric("Ratio de Sharpe", f"{opt_result['sharpe']:.2f}")
            
            st.divider()
            
            # Gráfico de torta con asignación óptima
            st.subheader("🥧 Asignación de Capital Óptima")
            
            df_pesos = pd.DataFrame({
                'Activo': opt_result['tickers'],
                'Peso (%)': (opt_result['pesos'] * 100).round(1)
            })
            
            fig_pie = px.pie(
                df_pesos, 
                values='Peso (%)', 
                names='Activo',
                title='Distribución Óptima del Portafolio',
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.divider()
            
            # Frontera Eficiente
            st.subheader("📊 Frontera Eficiente de Markowitz")
            st.markdown("""
            Cada punto representa un portafolio posible. El **portafolio óptimo** (estrella roja) 
            maximiza el retorno por unidad de riesgo (Ratio de Sharpe).
            """)
            
            fig_ef = go.Figure()
            
            # Nube de puntos de Montecarlo
            fig_ef.add_trace(go.Scatter(
                x=opt_result['mc_data']['volatilidades'],
                y=opt_result['mc_data']['retornos'],
                mode='markers',
                marker=dict(
                    size=6,
                    color=opt_result['mc_data']['sharpes'],
                    colorscale='Viridis',
                    colorbar=dict(title='Sharpe Ratio'),
                    opacity=0.6
                ),
                name='Portafolios Aleatorios'
            ))
            
            # Portafolio óptimo
            fig_ef.add_trace(go.Scatter(
                x=[opt_result['volatilidad']],
                y=[opt_result['retorno']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='black')),
                name='Portafolio Óptimo'
            ))
            
            fig_ef.update_layout(
                title='Frontera Eficiente: Retorno vs Volatilidad',
                xaxis_title='Volatilidad Anual (%)',
                yaxis_title='Retorno Anual (%)',
                hovermode='closest',
                showlegend=True
            )
            
            st.plotly_chart(fig_ef, use_container_width=True)
            
            st.divider()
            
            # Tabla detallada
            st.subheader(" Detalle de Asignación")
            st.dataframe(df_pesos, use_container_width=True, hide_index=True)
    
    else:
        st.markdown("""
        ### 👈 Ingresa tickers y haz clic en "Analizar Ahora"
        
        **Tickers disponibles:** AAPL, MSFT, KO, GOOGL, WMT, TSLA
        
        La aplicación calculará:
        - Asignación óptima de capital (Markowitz)
        - Frontera eficiente visualizada
        - Ratio de Sharpe máximo
        - Retorno y volatilidad esperados
        """)

# ==============================================================================
# PIE DE PÁGINA
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>QuantBuffett AI v0.4.0 | Desarrollado con Streamlit + Python + Plotly</p>
    <p><em>"Es mucho mejor comprar una empresa maravillosa a un precio justo, 
    que una empresa justa a un precio maravilloso." - Warren Buffett</em></p>
</div>
""", unsafe_allow_html=True)





