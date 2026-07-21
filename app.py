"""
QuantBuffett AI - Plataforma de Análisis Financiero Profesional
Versión: 0.3.0 (Auto-contenida - Sin dependencias externas)
"""

import streamlit as st
import pandas as pd
from datetime import datetime

# ==============================================================================
# BASE DE DATOS DE EJEMPLO (Datos realistas)
# ==============================================================================
MOCK_DATABASE = {
    'AAPL': {
        'ticker': 'AAPL',
        'precio': 308.50,
        'market_cap': 2400000000000,
        'roic': 55.2,
        'deuda_ebitda': 0.35,
        'fcf': 105.8,
        'net_income': 112000000000,
        'ebit': 130000000000,
        'margen_seguridad': -4.3,
        'beta': 1.10,
        'sector': 'Technology',
        'industry': 'Consumer Electronics'
    },
    'MSFT': {
        'ticker': 'MSFT',
        'precio': 415.20,
        'market_cap': 3100000000000,
        'roic': 38.1,
        'deuda_ebitda': 0.42,
        'fcf': 78.5,
        'net_income': 88000000000,
        'ebit': 105000000000,
        'margen_seguridad': 3.5,
        'beta': 0.95,
        'sector': 'Technology',
        'industry': 'Software'
    },
    'KO': {
        'ticker': 'KO',
        'precio': 62.30,
        'market_cap': 270000000000,
        'roic': 16.6,
        'deuda_ebitda': 1.50,
        'fcf': 9.8,
        'net_income': 10500000000,
        'ebit': 13000000000,
        'margen_seguridad': 12.5,
        'beta': 0.65,
        'sector': 'Consumer Defensive',
        'industry': 'Beverages'
    },
    'GOOGL': {
        'ticker': 'GOOGL',
        'precio': 175.80,
        'market_cap': 2200000000000,
        'roic': 26.4,
        'deuda_ebitda': 0.28,
        'fcf': 65.2,
        'net_income': 75000000000,
        'ebit': 95000000000,
        'margen_seguridad': 8.2,
        'beta': 1.05,
        'sector': 'Communication Services',
        'industry': 'Internet Content'
    },
    'WMT': {
        'ticker': 'WMT',
        'precio': 85.40,
        'market_cap': 230000000000,
        'roic': 14.2,
        'deuda_ebitda': 1.85,
        'fcf': 12.5,
        'net_income': 15000000000,
        'ebit': 22000000000,
        'margen_seguridad': 5.8,
        'beta': 0.55,
        'sector': 'Consumer Defensive',
        'industry': 'Discount Stores'
    },
    'TSLA': {
        'ticker': 'TSLA',
        'precio': 245.60,
        'market_cap': 780000000000,
        'roic': 12.8,
        'deuda_ebitda': 0.95,
        'fcf': 8.2,
        'net_income': 12000000000,
        'ebit': 15000000000,
        'margen_seguridad': -15.2,
        'beta': 2.05,
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers'
    }
}

def obtener_datos_financieros(ticker: str):
    """Obtiene datos financieros del ticker."""
    ticker_upper = ticker.upper()
    if ticker_upper in MOCK_DATABASE:
        return MOCK_DATABASE[ticker_upper].copy()
    return None

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
st.sidebar.header("️ Panel de Control")
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

# Botón para ejecutar análisis
ejecutar_analisis = st.sidebar.button("🔍 Analizar Ahora", type="primary", use_container_width=True)

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
        with st.spinner(f" Extrayendo datos financieros de {ticker_input}..."):
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
        ️ **Modo Demostración**: Mostrando datos de ejemplo para desarrollo. 
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
                label=" ROIC",
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
            margen_status = "🟢 Atractivo" if margen_val > 20 else "⚪ Justo" if margen_val > 0 else "🔴 Sobrevalorado"
            st.metric(
                label="🎯 Margen de Seguridad",
                value=f"{margen_val:.1f}%",
                delta=margen_status,
                help="Diferencia entre valor intrínseco (DCF) y precio de mercado"
            )
        
        # Análisis detallado
        st.divider()
        st.subheader(" Análisis Detallado")
        
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
            - ⚠️ Margen de seguridad negativo - Esperar corrección
            
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
    st.info(f" Analizando portafolio de **{len(tickers)} activos**: {', '.join(tickers)}")
    
    if ejecutar_analisis:
        with st.spinner("📊 Extrayendo datos de todos los activos..."):
            resultados = []
            for ticker in tickers:
                datos = obtener_datos_financieros(ticker)
                if datos:
                    resultados.append(datos)
            
            if resultados:
                st.success(f"✅ Datos extraídos para {len(resultados)} de {len(tickers)} activos")
                
                # Tabla comparativa
                df = pd.DataFrame(resultados)
                st.dataframe(df[['ticker', 'precio', 'roic', 'deuda_ebitda', 'margen_seguridad']], use_container_width=True)
            else:
                st.error("No se pudieron extraer datos de ningún activo")
    
    st.markdown("""
    ### 📊 Próximos Pasos (en desarrollo)
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
    <p>QuantBuffett AI v0.3.0 | Desarrollado con Streamlit + Python</p>
    <p><em>"Es mucho mejor comprar una empresa maravillosa a un precio justo, 
    que una empresa justa a un precio maravilloso." - Warren Buffett</em></p>
</div>
""", unsafe_allow_html=True)




