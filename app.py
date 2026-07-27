"""
QuantBuffett AI - Plataforma Profesional de Análisis Financiero
Versión: 1.2.0 | yfinance Robusto + Watchlist + Alertas + Corrección ROIC
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.optimize import minimize
import yfinance as yf
import time
import os
import tempfile
from fpdf import FPDF

# ==============================================================================
# CONFIGURACIÓN INICIAL
# ==============================================================================
st.set_page_config(
    page_title="QuantBuffett AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar Watchlist en sesión
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'KO']

# ==============================================================================
# DATOS MOCK - SOLO COMO FALLBACK DE ÚLTIMO RECURSO
# ==============================================================================
MOCK_DATABASE = {
    'AAPL': {
        'ticker': 'AAPL', 'precio': 327.74, 'market_cap': 2500000000000,
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
        'ticker': 'KO', 'precio': 81.97, 'market_cap': 355000000000,
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
        'ticker': 'WMT', 'precio': 85.40, 'market_cap': 690000000000,
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
# FUNCIONES DE DATOS ROBUSTAS (yfinance + Fallback)
# ==============================================================================
@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_yfinance(ticker: str) -> dict:
    """Obtiene datos REALES de yfinance con normalización estricta."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or ('currentPrice' not in info and 'regularMarketPrice' not in info):
            return None
        
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        market_cap = info.get('marketCap', 0) or 0
        
        # NORMALIZACIÓN ESTRICTA: Si es decimal (< 1), multiplicar por 100
        roe_raw = info.get('returnOnEquity', 0) or 0
        roic = (roe_raw * 100) if roe_raw < 1 else roe_raw
        
        div_raw = info.get('dividendYield', 0) or 0
        div_yield = (div_raw * 100) if div_raw < 1 else div_raw
        
        return {
            'ticker': ticker.upper(),
            'precio': float(price),
            'market_cap': float(market_cap),
            'pe_ratio': float(info.get('trailingPE', 0) or 0),
            'eps': float(info.get('trailingEps', 0) or 0),
            'beta': float(info.get('beta', 1.0) or 1.0),
            'roic': float(roic),
            'dividend_yield': float(div_yield),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'descripcion': (info.get('longBusinessSummary', '') or '')[:300],
            'es_real': True,
            'fuente': 'Yahoo Finance (yfinance)'
        }
    except Exception:
        return None


def obtener_datos_financieros(ticker: str) -> dict:
    """Orquestador: yfinance (Primario) -> Mock (Complemento/Fallback)."""
    ticker_upper = ticker.upper()
    datos_yf = obtener_datos_yfinance(ticker_upper)
    
    if datos_yf and datos_yf.get('precio', 0) > 0:
        mock = MOCK_DATABASE.get(ticker_upper, {})
        datos_yf['deuda_ebitda'] = mock.get('deuda_ebitda', 1.0)
        datos_yf['fcf'] = mock.get('fcf', 0)
        datos_yf['margen_seguridad'] = mock.get('margen_seguridad', 0)
        datos_yf['ebit'] = mock.get('ebit', 0)
        datos_yf['retorno_anual'] = mock.get('retorno_anual', 0.12)
        datos_yf['volatilidad_anual'] = mock.get('volatilidad_anual', 0.25)
        datos_yf['tendencia'] = mock.get('tendencia', 'Alcista' if datos_yf['roic'] > 15 else 'Bajista')
        
        MOCK_DATABASE[ticker_upper] = datos_yf
        return datos_yf
    
    if ticker_upper in MOCK_DATABASE:
        datos = MOCK_DATABASE[ticker_upper].copy()
        datos['es_real'] = False
        datos['fuente'] = 'Base de datos de demostración (Fallback)'
        return datos
    
    return None
    # ==============================================================================
# FUNCIONES DE ANÁLISIS DE RIESGOS (Corregidas)
# ==============================================================================
def analizar_riesgos_ia(ticker: str) -> dict:
    """Análisis de riesgos basado en métricas fundamentales normalizadas."""
    datos = MOCK_DATABASE.get(ticker, {})
    riesgos = []
    
    # 1. Riesgo Financiero (Deuda)
    deuda_ebitda = datos.get('deuda_ebitda', 1.0)
    if deuda_ebitda > 3.0: severidad, nivel = 90, "Crítico"
    elif deuda_ebitda > 2.0: severidad, nivel = 70, "Alto"
    elif deuda_ebitda > 1.0: severidad, nivel = 50, "Moderado"
    else: severidad, nivel = 20, "Bajo"
    
    riesgos.append({
        'categoria': 'Financiero', 
        'descripcion': f"Ratio Deuda/EBITDA de {deuda_ebitda:.2f}x", 
        'severidad': severidad, 'nivel': nivel, 
        'mitigacion': 'Mantener política de deuda conservadora' if deuda_ebitda <= 2 else 'Reducir deuda mediante generación de caja'
    })
    
    # 2. Riesgo Operativo (ROIC) - USA EL VALOR NORMALIZADO
    roic = datos.get('roic', 15)
    if roic < 10: severidad, nivel = 85, "Crítico"
    elif roic < 15: severidad, nivel = 60, "Alto"
    elif roic < 25: severidad, nivel = 40, "Moderado"
    else: severidad, nivel = 15, "Bajo"
    
    riesgos.append({
        'categoria': 'Operativo', 
        'descripcion': f"ROIC del {roic:.1f}%", 
        'severidad': severidad, 'nivel': nivel, 
        'mitigacion': 'Continuar con estrategia rentable' if roic >= 15 else 'Optimizar asignación de capital'
    })
    
    # 3. Riesgo de Mercado (Margen de Seguridad)
    margen = datos.get('margen_seguridad', 0)
    if margen < -20: severidad, nivel = 80, "Crítico"
    elif margen < 0: severidad, nivel = 60, "Alto"
    elif margen < 15: severidad, nivel = 40, "Moderado"
    else: severidad, nivel = 20, "Bajo"
    
    riesgos.append({
        'categoria': 'Mercado', 
        'descripcion': f"Margen de seguridad del {margen:.1f}%", 
        'severidad': severidad, 'nivel': nivel, 
        'mitigacion': 'Precio ofrece protección' if margen >= 0 else 'Esperar corrección del mercado'
    })
    
    # 4. Riesgo Sistemático (Beta)
    beta = datos.get('beta', 1.0)
    if beta > 1.5: severidad, nivel = 75, "Alto"
    elif beta > 1.0: severidad, nivel = 50, "Moderado"
    else: severidad, nivel = 25, "Bajo"
    
    riesgos.append({
        'categoria': 'Sistemático', 
        'descripcion': f"Beta de {beta:.2f}", 
        'severidad': severidad, 'nivel': nivel, 
        'mitigacion': 'Beta aceptable' if beta <= 1.5 else 'Diversificar portafolio para reducir exposición'
    })
    
    # 5. Riesgo Sectorial
    sector = datos.get('sector', 'Technology')
    sectores_riesgo = {
        'Technology': {'severidad': 55, 'nivel': 'Moderado', 'descripcion': 'Sector tecnológico con rápida obsolescencia'},
        'Consumer Cyclical': {'severidad': 65, 'nivel': 'Alto', 'descripcion': 'Sector cíclico sensible a recesiones'},
        'Consumer Defensive': {'severidad': 30, 'nivel': 'Bajo', 'descripcion': 'Sector defensivo con demanda estable'},
        'Communication Services': {'severidad': 50, 'nivel': 'Moderado', 'descripcion': 'Sector con riesgos regulatorios'},
        'Financials': {'severidad': 60, 'nivel': 'Alto', 'descripcion': 'Sector expuesto a tasas de interés'}
    }
    riesgo_sec = sectores_riesgo.get(sector, {'severidad': 50, 'nivel': 'Moderado', 'descripcion': 'Perfil de riesgo sectorial estándar'})
    
    riesgos.append({
        'categoria': 'Sectorial', 
        'descripcion': riesgo_sec['descripcion'], 
        'severidad': riesgo_sec['severidad'], 'nivel': riesgo_sec['nivel'], 
        'mitigacion': 'Diversificar entre sectores para reducir concentración'
    })
    
    # Score consolidado
    score_riesgo = np.mean([r['severidad'] for r in riesgos])
    
    if score_riesgo < 30: perfil, rec = "Bajo Riesgo - Conservador", "Ideal para preservación de capital. Apto para inversores estilo Buffett."
    elif score_riesgo < 50: perfil, rec = "Riesgo Moderado", "Balance adecuado entre riesgo y retorno. Apto para la mayoría de inversores."
    elif score_riesgo < 70: perfil, rec = "Riesgo Moderado-Alto", "Requiere diversificación. Apto para inversores con tolerancia al riesgo."
    else: perfil, rec = "Alto Riesgo - Agresivo", "No recomendado para inversores conservadores. Solo perfiles agresivos."
    
    return {
        'ticker': ticker, 'riesgos': riesgos, 
        'score_riesgo': round(score_riesgo, 1), 
        'perfil_riesgo': perfil, 'recomendacion': rec
    }

# ==============================================================================
# FUNCIONES DE PRONÓSTICO
# ==============================================================================
def pronosticar_precio(ticker: str, dias_pronostico: int = 90) -> dict:
    """Pronóstico basado en regresión lineal + bandas de confianza."""
    datos = MOCK_DATABASE.get(ticker, {})
    precio_actual = datos.get('precio', 100)
    volatilidad = datos.get('volatilidad_anual', 0.25)
    
    # Generar histórico simulado para el modelo (para no saturar la API con 1 año de datos cada vez)
    np.random.seed(42)
    fechas = pd.date_range(end=datetime.now(), periods=365, freq='D')
    retornos_diarios = np.random.normal(0.0008, volatilidad / np.sqrt(252), 365)
    precios = [precio_actual]
    for i in range(364, 0, -1):
        precios.append(precios[-1] * (1 - retornos_diarios[i]))
    precios.reverse()
    historico = pd.DataFrame({'Fecha': fechas, 'Precio': precios})
    
    # Regresión lineal
    x = np.arange(len(historico))
    y = historico['Precio'].values
    coeficientes = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coeficientes)
    
    # Proyección
    x_futuro = np.arange(len(historico), len(historico) + dias_pronostico)
    precios_pronosticados = tendencia(x_futuro)
    
    # Bandas de confianza
    error_estandar = volatilidad * precio_actual / np.sqrt(252)
    intervalo_confianza = 1.96 * error_estandar * np.sqrt(np.arange(1, dias_pronostico + 1))
    
    precio_inicial = historico['Precio'].iloc[-30]
    cambio_30d = ((precio_actual - precio_inicial) / precio_inicial) * 100
    cambio_pronostico = ((precios_pronosticados[-1] - precio_actual) / precio_actual) * 100
    
    return {
        'ticker': ticker, 'historico': historico, 
        'precios_pronosticados': precios_pronosticados,
        'limite_superior': precios_pronosticados + intervalo_confianza,
        'limite_inferior': precios_pronosticados - intervalo_confianza,
        'dias_pronostico': dias_pronostico, 
        'cambio_30d': cambio_30d, 'cambio_pronostico': cambio_pronostico,
        'volatilidad_diaria': volatilidad / np.sqrt(252) * 100, 
        'tendencia': datos.get('tendencia', 'Alcista')
    }

# ==============================================================================
# FUNCIONES DE PORTAFOLIO (MARKOWITZ)
# ==============================================================================
def optimizar_portafolio(tickers: list, rf: float = 0.04) -> dict:
    """Optimización de Markowitz para maximizar el Ratio de Sharpe."""
    n = len(tickers)
    retornos = np.array([MOCK_DATABASE.get(t, {}).get('retorno_anual', 0.12) for t in tickers])
    volatilidades = np.array([MOCK_DATABASE.get(t, {}).get('volatilidad_anual', 0.25) for t in tickers])
    
    # Matriz de correlación (estimada para el ejemplo)
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
    limites = tuple((0.0, 1.0) for _ in range(n))
    
    pesos_iniciales = np.ones(n) / n
    resultado = minimize(sharpe_negativo, pesos_iniciales, 
                        method='SLSQP', bounds=limites, constraints=restricciones)
    
    pesos_optimos = resultado.x
    retorno_optimo = np.sum(retornos * pesos_optimos)
    volatilidad_optima = np.sqrt(np.dot(pesos_optimos.T, np.dot(cov_matrix, pesos_optimos)))
    sharpe_optimo = (retorno_optimo - rf) / volatilidad_optima
    
    # Simulación Montecarlo para la Frontera Eficiente
    mc_retornos, mc_vols, mc_sharpes = [], [], []
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
        'tickers': tickers, 'pesos': pesos_optimos,
        'retorno': retorno_optimo * 100, 'volatilidad': volatilidad_optima * 100,
        'sharpe': sharpe_optimo,
        'mc_data': {'retornos': mc_retornos, 'volatilidades': mc_vols, 'sharpes': mc_sharpes},
        'correlacion': correlacion
    }

# ==============================================================================
# GENERADOR DE PDF (ROBUSTO Y NORMALIZADO)
# ==============================================================================
class PDFReport(FPDF):
    """Clase base para reportes PDF profesionales."""
    def header(self):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, 'QuantBuffett AI', 0, 1, 'L')
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Plataforma Profesional de Analisis Financiero', 0, 1, 'L')
        self.ln(5)
        self.set_draw_color(26, 54, 93)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def section_title(self, title):
        self.set_font('Helvetica', 'B', 14)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def metric_row(self, label, value):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(90, 8, label, 0, 0, 'L')
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, str(value), 0, 1, 'R')

def generar_pdf_activo(ticker: str, datos: dict, pronostico: dict, riesgos: dict) -> str:
    """Genera PDF de análisis de activo individual."""
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 20, f'Analisis: {ticker}', 0, 1, 'C')
    
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 10, f'Fuente: {datos.get("fuente", "No disponible")}', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.section_title('Metricas Fundamentales')
    pdf.metric_row('Precio Actual:', f'${datos.get("precio", 0):.2f}')
    pdf.metric_row('P/E Ratio:', f'{datos.get("pe_ratio", 0):.1f}x')
    pdf.metric_row('EPS:', f'${datos.get("eps", 0):.2f}')
    pdf.metric_row('Beta:', f'{datos.get("beta", 1.0):.2f}')
    pdf.metric_row('ROIC/ROE:', f'{datos.get("roic", 0):.1f}%') # Normalizado
    pdf.metric_row('Deuda/EBITDA:', f'{datos.get("deuda_ebitda", 0):.2f}x')
    pdf.metric_row('Market Cap:', f'${datos.get("market_cap", 0)/1e9:.1f}B')
    pdf.metric_row('Dividend Yield:', f'{datos.get("dividend_yield", 0):.2f}%') # Normalizado
    pdf.ln(5)
    
    pdf.section_title('Pronostico a 90 Dias')
    pdf.metric_row('Cambio 30 dias:', f'{pronostico["cambio_30d"]:.1f}%')
    pdf.metric_row('Pronostico 90 dias:', f'{pronostico["cambio_pronostico"]:.1f}%')
    pdf.metric_row('Tendencia:', pronostico['tendencia'])
    pdf.metric_row('Volatilidad diaria:', f'{pronostico["volatilidad_diaria"]:.2f}%')
    pdf.ln(5)
    
    pdf.section_title('Analisis de Riesgos')
    pdf.metric_row('Score de Riesgo:', f'{riesgos["score_riesgo"]}/100')
    pdf.metric_row('Perfil:', riesgos['perfil_riesgo'])
    pdf.ln(3)
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Riesgos Identificados:', 0, 1)
    for i, riesgo in enumerate(riesgos['riesgos'], 1):
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f'{i}. {riesgo["categoria"]} - {riesgo["nivel"]} ({riesgo["severidad"]}/100)', 0, 1)
        pdf.set_font('Helvetica', 'I', 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, f'   {riesgo["descripcion"][:80]}', 0, 1)
    
    pdf.ln(5)
    pdf.section_title('Veredicto Final')
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    veredicto = "PERFIL CONSERVADOR" if riesgos['score_riesgo'] < 30 else "PERFIL BALANCEADO" if riesgos['score_riesgo'] < 50 else "PERFIL AGRESIVO"
    pdf.cell(0, 10, veredicto, 0, 1, 'C')
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 6, riesgos['recomendacion'])
    
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'DISCLAIMER: Reporte generado automaticamente con fines informativos. No constituye asesoramiento financiero.')
    
    temp_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_path)
    return temp_path

def generar_pdf_portafolio(tickers: list, opt_result: dict, capital: float) -> str:
    """Genera PDF de análisis de portafolio."""
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 20, 'Analisis de Portafolio', 0, 1, 'C')
    
    pdf.set_font('Helvetica', '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 10, f'Activos: {", ".join(tickers)}', 0, 1, 'C')
    pdf.cell(0, 10, f'Capital: ${capital:,.0f}', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.section_title('Metricas del Portafolio Optimo')
    pdf.metric_row('Retorno Anual Esperado:', f'{opt_result["retorno"]:.2f}%')
    pdf.metric_row('Volatilidad (Riesgo):', f'{opt_result["volatilidad"]:.2f}%')
    pdf.metric_row('Ratio de Sharpe:', f'{opt_result["sharpe"]:.2f}')
    pdf.ln(5)
    
    pdf.section_title('Asignacion Optima de Capital')
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 8, 'Activo', 1, 0, 'C', True)
    pdf.cell(35, 8, 'Peso (%)', 1, 0, 'C', True)
    pdf.cell(40, 8, 'Monto ($)', 1, 0, 'C', True)
    pdf.cell(35, 8, 'Retorno', 1, 0, 'C', True)
    pdf.cell(30, 8, 'Riesgo', 1, 1, 'C', True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 9)
    
    for i, ticker in enumerate(opt_result['tickers']):
        peso = opt_result['pesos'][i] * 100
        monto = opt_result['pesos'][i] * capital
        datos_ticker = MOCK_DATABASE.get(ticker, {})
        retorno_indiv = datos_ticker.get('retorno_anual', 0.12) * 100
        volatilidad_indiv = datos_ticker.get('volatilidad_anual', 0.25) * 100
        
        if i % 2 == 0: pdf.set_fill_color(240, 240, 240)
        else: pdf.set_fill_color(255, 255, 255)
        
        pdf.cell(50, 8, ticker, 1, 0, 'L', True)
        pdf.cell(35, 8, f'{peso:.1f}%', 1, 0, 'C', True)
        pdf.cell(40, 8, f'${monto:,.0f}', 1, 0, 'R', True)
        pdf.cell(35, 8, f'{retorno_indiv:.1f}%', 1, 0, 'R', True)
        pdf.cell(30, 8, f'{volatilidad_indiv:.1f}%', 1, 1, 'R', True)
    
    pdf.ln(5)
    pdf.section_title('Perfil de Riesgo del Portafolio')
    volatilidad = opt_result['volatilidad']
    if volatilidad < 15: perfil = "CONSERVADOR - Proteccion de capital"
    elif volatilidad < 25: perfil = "MODERADO - Balance crecimiento/estabilidad"
    else: perfil = "AGRESIVO - Maximizacion de ganancias"
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, perfil, 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'DISCLAIMER: La optimizacion de Markowitz se basa en datos historicos que no garantizan resultados futuros.')
    
    temp_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_path)
    return temp_path
    # ==============================================================================
# INTERFAZ DE USUARIO
# ==============================================================================
st.title("📈 QuantBuffett AI")
st.markdown("""
**Plataforma Profesional de Análisis Financiero**  
*Data Science + Machine Learning + Filosofía de Inversión de Valor*
""")

# ==============================================================================
# BARRA LATERAL (Con Watchlist)
# ==============================================================================
st.sidebar.header("⚙️ Configuración")

modo_usuario = st.sidebar.radio(
    "Modo de Visualización",
    ["🟢 Simple (Principiantes)", " Avanzado (Expertos)"],
    help="Simple: veredictos claros. Avanzado: métricas detalladas y parámetros editables."
)

st.sidebar.divider()

# WATCHLIST INTERACTIVA
st.sidebar.markdown("### ⭐ Mi Watchlist")

if not st.session_state.watchlist:
    st.sidebar.info("Agrega tickers desde la pestaña de Análisis")
else:
    for ticker in st.session_state.watchlist:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(f"• **{ticker}**")
        with col2:
            if st.button("🗑️", key=f"rm_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()

st.sidebar.divider()

st.sidebar.markdown("### 📡 Fuente de Datos")
st.sidebar.success("✅ Yahoo Finance (yfinance)")
st.sidebar.caption("Datos en tiempo real + Fallback automático")

st.sidebar.divider()

st.sidebar.markdown("""
### ℹ️ Sobre QuantBuffett AI
Versión 1.2.0  
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
# PESTAÑA 1: DASHBOARD (Con Alertas de Watchlist)
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
        
        **Empresas de ejemplo:** AAPL, MSFT, KO, GOOGL, WMT, TSLA
        """)
        
        st.divider()
        st.subheader("📊 Resumen de tu Watchlist")
        
        if not st.session_state.watchlist:
            st.info("Agrega tickers a tu Watchlist desde la barra lateral o la pestaña de Análisis.")
        else:
            cols = st.columns(len(st.session_state.watchlist))
            for i, ticker in enumerate(st.session_state.watchlist):
                with cols[i]:
                    datos = obtener_datos_financieros(ticker)
                    if datos:
                        st.metric(ticker, f"${datos['precio']:.2f}", delta=f"{datos['margen_seguridad']:+.1f}% Margen")
                        
                        # ALERTAS VISUALES
                        if datos['margen_seguridad'] > 15:
                            st.success("🟢 Oportunidad de Compra")
                        elif datos['margen_seguridad'] < -10:
                            st.error("🔴 Sobrevalorado")
                        else:
                            st.info("⚪ Precio Justo")
                    else:
                        st.error(f"Error: {ticker}")
    
    else:
        st.markdown("### Dashboard de Control")
        st.markdown("Panel de control para análisis financiero profesional con datos en tiempo real vía Yahoo Finance.")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fuente de Datos", "Yahoo Finance", "Tiempo real")
        with col2:
            st.metric("Modo Activo", "Avanzado", "Parámetros editables")
        with col3:
            st.metric("Watchlist", f"{len(st.session_state.watchlist)} activos", "Monitoreados")
        with col4:
            st.metric("Versión", "1.2.0", "Paso 8")
        
        st.divider()
        st.subheader("📊 Análisis Rápido de Watchlist")
        
        if st.session_state.watchlist:
            for ticker in st.session_state.watchlist:
                datos = obtener_datos_financieros(ticker)
                if datos:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric(ticker, f"${datos['precio']:.2f}")
                    col2.metric("P/E", f"{datos.get('pe_ratio', 0):.1f}x")
                    col3.metric("ROIC", f"{datos.get('roic', 0):.1f}%")
                    col4.metric("Beta", f"{datos.get('beta', 1):.2f}")
                    
                    if datos['margen_seguridad'] > 15:
                        col5.success("🟢 Compra")
                    elif datos['margen_seguridad'] < -10:
                        col5.error("🔴 Evitar")
                    else:
                        col5.info("⚪ Observar")
        else:
            st.info("Agrega tickers a tu Watchlist desde la barra lateral.")

# ==============================================================================
# PESTAÑA 2: ANÁLISIS DE ACTIVO
# ==============================================================================
with tab2:
    st.header("🔍 Análisis de Activo")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_input = st.text_input("Ticker a analizar", value="AAPL").upper()
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Agregar a Watchlist"):
            if ticker_input not in st.session_state.watchlist:
                st.session_state.watchlist.append(ticker_input)
                st.success(f"✅ {ticker_input} agregado a Watchlist!")
                st.rerun()
            else:
                st.warning("Ya está en la Watchlist")
    
    if st.button("🔍 Analizar", type="primary"):
        with st.spinner("Obteniendo datos en tiempo real..."):
            datos = obtener_datos_financieros(ticker_input)
            if datos:
                st.session_state.datos_activo = datos
            else:
                st.error(f"No se encontraron datos para {ticker_input}")
    
    if 'datos_activo' in st.session_state:
        datos = st.session_state.datos_activo
        
        if datos.get('es_real'):
            st.success(f"✅ Datos REALES de {datos.get('fuente', 'Yahoo Finance')}")
        else:
            st.warning(f"⚠️ {datos.get('fuente', 'Datos de demostración')}")
        
        # MODO SIMPLE
        if modo_usuario == "🟢 Simple (Principiantes)":
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio", f"${datos['precio']:.2f}")
            col2.metric("📈 P/E", f"{datos.get('pe_ratio', 0):.1f}x")
            col3.metric("📊 ROIC", f"{datos.get('roic', 0):.1f}%")
            col4.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
            st.divider()
            
            # Veredicto simple
            roic = datos.get('roic', 0)
            deuda = datos.get('deuda_ebitda', 1)
            margen = datos.get('margen_seguridad', 0)
            
            if roic > 15 and deuda < 2 and margen > 0:
                st.success("""
                ### ✅ COMPRA POTENCIAL
                **Empresa maravillosa a precio justo:**
                - ✅ ROIC excelente (>15%)
                - ✅ Deuda controlada (<2x EBITDA)
                - ✅ Margen de seguridad positivo
                
                *Cumple con los criterios de Warren Buffett*
                """)
            elif roic > 15 and deuda < 2:
                st.warning("""
                ### ⏳ OBSERVAR
                **Negocio de calidad pero precio elevado:**
                - ✅ ROIC excelente
                - ✅ Deuda controlada
                - ⚠️ Margen de seguridad negativo
                
                *Recomendación: Agregar a watchlist y esperar mejor entrada*
                """)
            else:
                st.info("""
                ### 🔍 ANÁLISIS MIXTO
                **Requiere análisis más profundo:**
                - Revisar tendencias históricas
                - Analizar ventajas competitivas
                - Evaluar catalizadores futuros
                """)
            
            st.divider()
            
            if st.button("📥 Descargar PDF del Análisis"):
                with st.spinner("Generando PDF..."):
                    try:
                        pronostico = pronosticar_precio(ticker_input, 90)
                        riesgos = analizar_riesgos_ia(ticker_input)
                        pdf_path = generar_pdf_activo(ticker_input, datos, pronostico, riesgos)
                        
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Descargar PDF",
                                data=f.read(),
                                file_name=f"Analisis_{ticker_input}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                type="primary"
                            )
                        st.success("✅ PDF generado correctamente")
                    except Exception as e:
                        st.error(f"Error al generar PDF: {str(e)}")
        
        # MODO AVANZADO
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio", f"${datos['precio']:.2f}")
            col2.metric("📈 P/E", f"{datos.get('pe_ratio', 0):.1f}x")
            col3.metric("📊 ROIC", f"{datos.get('roic', 0):.1f}%")
            col4.metric(" Beta", f"{datos.get('beta', 1):.2f}")
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💵 EPS", f"${datos.get('eps', 0):.2f}")
                st.metric("🏢 Sector", datos.get('sector', 'N/A'))
            with col2:
                st.metric("💼 Market Cap", f"${datos.get('market_cap', 0)/1e9:.1f}B")
                st.metric(" Industria", datos.get('industry', 'N/A'))
            
            if datos.get('descripcion'):
                st.divider()
                st.subheader(" Descripción de la Empresa")
                st.write(datos['descripcion'])
            
            st.divider()
            
            if st.button("📥 Descargar PDF del Análisis"):
                with st.spinner("Generando PDF..."):
                    try:
                        pronostico = pronosticar_precio(ticker_input, 90)
                        riesgos = analizar_riesgos_ia(ticker_input)
                        pdf_path = generar_pdf_activo(ticker_input, datos, pronostico, riesgos)
                        
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="️ Descargar PDF",
                                data=f.read(),
                                file_name=f"Analisis_{ticker_input}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                type="primary"
                            )
                        st.success("✅ PDF generado correctamente")
                    except Exception as e:
                        st.error(f"Error al generar PDF: {str(e)}")

# ==============================================================================
# PESTAÑA 3: PORTAFOLIO
# ==============================================================================
with tab3:
    st.header("💼 Optimizador de Portafolio")
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### 🎯 Optimización Inteligente de Portafolio
        
        Selecciona las empresas en las que quieres invertir y te diremos exactamente cómo distribuir tu dinero.
        """)
        
        st.divider()
        
        st.subheader("1️⃣ Selecciona tus empresas")
        
        tickers_recomendados = ['AAPL', 'MSFT', 'KO', 'GOOGL', 'WMT', 'TSLA']
        tickers_seleccionados = st.multiselect(
            "Elige entre 2 y 6 empresas",
            options=tickers_recomendados,
            default=['AAPL', 'MSFT', 'KO'],
            help="Selecciona al menos 2 empresas para diversificar"
        )
        
        st.divider()
        
        if len(tickers_seleccionados) < 2:
            st.warning("️ Selecciona al menos 2 empresas para crear un portafolio diversificado.")
        else:
            st.subheader("2️⃣ ¿Cuánto quieres invertir?")
            capital = st.slider("Capital total (USD)", 1000, 1000000, 10000, 1000)
            
            st.divider()
            
            if st.button("🚀 Optimizar Mi Portafolio", type="primary", use_container_width=True):
                with st.spinner("Calculando la mejor distribución..."):
                    opt_result = optimizar_portafolio(tickers_seleccionados, rf=0.04)
                    st.session_state.opt_result = opt_result
                    st.session_state.capital = capital
                    st.session_state.tickers_portafolio = tickers_seleccionados
            
            if 'opt_result' in st.session_state and st.session_state.get('tickers_portafolio') == tickers_seleccionados:
                opt = st.session_state.opt_result
                capital = st.session_state.capital
                
                st.divider()
                st.subheader("3️ Tu Portafolio Óptimo")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 Retorno Anual Esperado", f"{opt['retorno']:.1f}%")
                col2.metric("⚠️ Riesgo (Volatilidad)", f"{opt['volatilidad']:.1f}%")
                col3.metric("⭐ Ratio de Sharpe", f"{opt['sharpe']:.2f}")
                
                st.divider()
                
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
                
                st.subheader("📋 Detalle de Inversión")
                st.dataframe(df_pesos, use_container_width=True, hide_index=True)
                
                st.divider()
                
                st.subheader("💡 ¿Qué significa esto?")
                
                if opt['sharpe'] > 1.0:
                    st.success(f"**Excelente portafolio!** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, este portafolio ofrece muy buen retorno por cada unidad de riesgo asumido.")
                elif opt['sharpe'] > 0.5:
                    st.info(f"**Buen portafolio.** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, este portafolio ofrece un balance aceptable entre riesgo y retorno.")
                else:
                    st.warning(f"**Portafolio conservador.** Con un Ratio de Sharpe de {opt['sharpe']:.2f}, el retorno es moderado en relación al riesgo.")
                
                st.divider()
                
                st.subheader("🎯 Perfil de Este Portafolio")
                
                if opt['volatilidad'] < 15:
                    st.markdown("**Conservador** 🟢\nEste portafolio está diseñado para proteger tu capital. Ideal si tu horizonte de inversión es corto (1-3 años).")
                elif opt['volatilidad'] < 25:
                    st.markdown("**Moderado** \nEste portafolio balancea crecimiento y estabilidad. Ideal si tu horizonte de inversión es mediano (3-7 años).")
                else:
                    st.markdown("**Agresivo** 🔴\nEste portafolio busca maximizar ganancias aceptando mayor volatilidad. Ideal si tu horizonte de inversión es largo (+7 años).")
                
                st.divider()
                
                if st.button("📥 Descargar PDF del Portafolio"):
                    with st.spinner("Generando PDF..."):
                        try:
                            pdf_path = generar_pdf_portafolio(st.session_state.tickers_portafolio, opt, capital)
                            
                            with open(pdf_path, 'rb') as f:
                                st.download_button(
                                    label="️ Descargar PDF",
                                    data=f.read(),
                                    file_name=f"Portafolio_{datetime.now().strftime('%Y%m%d')}.pdf",
                                    mime="application/pdf",
                                    type="primary"
                                )
                            st.success("✅ PDF generado correctamente")
                        except Exception as e:
                            st.error(f"Error al generar PDF: {str(e)}")
    
    else:
        st.markdown("""
        ### Optimización de Portafolio - Modelo de Markowitz
        
        Optimización matemática de asignación de activos utilizando la Teoría Moderna de Portafolios.
        """)
        
        st.divider()
        
        st.subheader("1. Selección de Activos")
        
        tickers_disponibles = ['AAPL', 'MSFT', 'KO', 'GOOGL', 'WMT', 'TSLA']
        tickers_seleccionados = st.multiselect(
            "Selecciona activos base (2-6)",
            options=tickers_disponibles,
            default=['AAPL', 'MSFT', 'KO', 'GOOGL']
        )
        
        st.divider()
        
        if len(tickers_seleccionados) < 2:
            st.warning("Se requieren al menos 2 activos.")
        else:
            st.subheader("2. Parámetros del Modelo")
            
            col1, col2 = st.columns(2)
            with col1:
                rf = st.slider("Tasa Libre de Riesgo (%)", 0.0, 10.0, 4.0, 0.5, help="Rendimiento de bonos del Tesoro")
            with col2:
                max_peso = st.slider("Peso Máximo por Activo (%)", 10, 100, 40, 5, help="Restricción de concentración")
            
            st.divider()
            
            if st.button("⚙️ Ejecutar Optimización", type="primary"):
                with st.spinner("Optimizando portafolio..."):
                    opt_result = optimizar_portafolio(tickers_seleccionados, rf=rf/100)
                    
                    opt_result['pesos'] = np.minimum(opt_result['pesos'], max_peso/100)
                    opt_result['pesos'] /= opt_result['pesos'].sum()
                    
                    retornos = np.array([MOCK_DATABASE[t].get('retorno_anual', 0.12) for t in tickers_seleccionados])
                    volatilidades = np.array([MOCK_DATABASE[t].get('volatilidad_anual', 0.25) for t in tickers_seleccionados])
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
                    st.session_state.tickers_portafolio_av = tickers_seleccionados
            
            if 'opt_result_avanzado' in st.session_state:
                opt = st.session_state.opt_result_avanzado
                rf = st.session_state.rf
                
                st.divider()
                st.subheader("3. Resultados de la Optimización")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Esperado", f"{opt['retorno']:.2f}%")
                col2.metric("Volatilidad", f"{opt['volatilidad']:.2f}%")
                col3.metric("Ratio de Sharpe", f"{opt['sharpe']:.2f}")
                
                st.divider()
                
                st.subheader("4. Frontera Eficiente")
                st.markdown("Cada punto representa un portafolio posible. La estrella roja indica el portafolio óptimo.")
                
                fig_ef = go.Figure()
                
                fig_ef.add_trace(go.Scatter(
                    x=opt['mc_data']['volatilidades'],
                    y=opt['mc_data']['retornos'],
                    mode='markers',
                    marker=dict(size=6, color=opt['mc_data']['sharpes'], colorscale='Viridis', colorbar=dict(title='Sharpe Ratio'), opacity=0.6),
                    name='Portafolios Aleatorios'
                ))
                
                fig_ef.add_trace(go.Scatter(
                    x=[opt['volatilidad']],
                    y=[opt['retorno']],
                    mode='markers',
                    marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='black')),
                    name='Portafolio Óptimo'
                ))
                
                fig_ef.update_layout(title='Frontera Eficiente: Retorno vs Volatilidad', xaxis_title='Volatilidad Anual (%)', yaxis_title='Retorno Anual (%)', height=500)
                st.plotly_chart(fig_ef, use_container_width=True)
                
                st.divider()
                
                st.subheader("5. Asignación Óptima de Pesos")
                
                df_pesos = pd.DataFrame({
                    'Activo': opt['tickers'],
                    'Peso Óptimo (%)': (opt['pesos'] * 100).round(2),
                    'Retorno Individual (%)': [MOCK_DATABASE[t].get('retorno_anual', 0.12)*100 for t in opt['tickers']],
                    'Volatilidad Individual (%)': [MOCK_DATABASE[t].get('volatilidad_anual', 0.25)*100 for t in opt['tickers']]
                })
                
                st.dataframe(df_pesos, use_container_width=True, hide_index=True)
                
                st.divider()
                
                st.subheader("6. Matriz de Correlación")
                
                if opt['correlacion'] is not None:
                    fig_corr = px.imshow(opt['correlacion'], labels=dict(x="Activo", y="Activo", color="Correlación"), x=opt['tickers'], y=opt['tickers'], color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
                    fig_corr.update_layout(height=400)
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                st.divider()
                
                st.subheader("7. Sistema de Rebalanceo Estratégico")
                
                st.write("**Pesos actuales de tu portafolio (%):**")
                pesos_actuales = {}
                cols = st.columns(len(opt['tickers']))
                for i, ticker in enumerate(opt['tickers']):
                    with cols[i]:
                        pesos_actuales[ticker] = st.number_input(f"{ticker}", min_value=0.0, max_value=100.0, value=100.0/len(opt['tickers']), step=1.0)
                
                capital_rebalanceo = st.number_input("Capital total para rebalanceo (USD)", value=100000, step=10000)
                
                if st.button(" Calcular Rebalanceo"):
                    df = pd.DataFrame()
                    df['Activo'] = opt['tickers']
                    df['Peso_Actual'] = [pesos_actuales.get(t, 0) for t in opt['tickers']]
                    df['Peso_Optimo'] = opt['pesos'] * 100
                    df['Diferencia'] = df['Peso_Optimo'] - df['Peso_Actual']
                    df['Capital_Ajustar'] = df['Diferencia'] * capital_rebalanceo / 100
                    df['Accion'] = df['Diferencia'].apply(lambda x: " COMPRAR" if x > 1 else "🔴 VENDER" if x < -1 else "✅ MANTENER")
                    
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    
                    comprar = df[df['Accion'].str.contains('COMPRAR')]
                    vender = df[df['Accion'].str.contains('VENDER')]
                    
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
    
    ticker_input_raw = st.text_input("Ticker para pronóstico y análisis de riesgos", value="AAPL")
    ticker_pronostico = ticker_input_raw.strip().upper()
    
    if st.button("🔮 Analizar Pronóstico y Riesgos", type="primary"):
        with st.spinner("Ejecutando modelos..."):
            try:
                datos = obtener_datos_financieros(ticker_pronostico)
                
                if datos and datos.get('precio', 0) > 0:
                    if ticker_pronostico not in MOCK_DATABASE:
                        MOCK_DATABASE[ticker_pronostico] = datos
                    
                    pronostico = pronosticar_precio(ticker_pronostico, dias_pronostico=90)
                    analisis_riesgos = analizar_riesgos_ia(ticker_pronostico)
                    
                    st.session_state.pronostico = pronostico
                    st.session_state.riesgos = analisis_riesgos
                    st.session_state.ticker_analizado = ticker_pronostico
                    st.session_state.error_pronostico = None
                    
                    st.success(f"✅ Análisis completado para {ticker_pronostico}")
                else:
                    st.session_state.pronostico = None
                    st.session_state.riesgos = None
                    st.session_state.error_pronostico = f"No se encontraron datos para {ticker_pronostico}"
                    
            except Exception as e:
                st.session_state.pronostico = None
                st.session_state.riesgos = None
                st.session_state.error_pronostico = f"Error en el análisis: {str(e)}"
    
    if st.session_state.get('error_pronostico'):
        st.error(st.session_state.error_pronostico)
    
    elif st.session_state.get('pronostico') and st.session_state.get('riesgos'):
        pron = st.session_state.pronostico
        riesgos = st.session_state.riesgos
        ticker_analizado = st.session_state.get('ticker_analizado', ticker_pronostico)
        
        datos = MOCK_DATABASE.get(ticker_analizado, {})
        
        if datos.get('es_real'):
            st.success(f"✅ Datos REALES de {datos.get('fuente', 'Yahoo Finance')}")
        else:
            st.warning(f"⚠️ {datos.get('fuente', 'Datos de demostración')}")
        
        st.divider()
        
        if modo_usuario == "🟢 Simple (Principiantes)":
            st.subheader(" Pronóstico de Precio a 90 Días")
            
            col1, col2, col3 = st.columns(3)
            col1.metric(" Precio Actual", f"${pron['historico']['Precio'].iloc[-1]:.2f}")
            col2.metric("📊 Cambio 30 días", f"{pron['cambio_30d']:.1f}%", "↗️" if pron['cambio_30d'] > 0 else "↘️")
            col3.metric(" Pronóstico 90 días", f"{pron['cambio_pronostico']:.1f}%", "↗️" if pron['cambio_pronostico'] > 0 else "↘️")
            
            st.divider()
            
            st.subheader("📊 Proyección Visual")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pron['historico']['Fecha'], y=pron['historico']['Precio'], mode='lines', name='Histórico', line=dict(color='blue', width=2)))
            
            fechas_futuras = pd.date_range(start=pron['historico']['Fecha'].iloc[-1], periods=pron['dias_pronostico'] + 1, freq='D')[1:]
            
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['precios_pronosticados'], mode='lines', name='Pronóstico', line=dict(color='orange', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_superior'], mode='lines', name='Límite Superior (95%)', line=dict(width=0), showlegend=True))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_inferior'], mode='lines', name='Límite Inferior (95%)', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=True))
            
            fig.update_layout(title=f'Pronóstico de {ticker_analizado} a 90 días', xaxis_title='Fecha', yaxis_title='Precio (USD)', hovermode='x unified', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("🎯 Veredicto del Modelo")
            
            if pron['cambio_pronostico'] > 10:
                st.success(f"###  SEÑAL ALCISTA FUERTE\nEl modelo predice un crecimiento del **{pron['cambio_pronostico']:.1f}%** en 90 días.")
            elif pron['cambio_pronostico'] > 0:
                st.info(f"### 📈 SEÑAL ALCISTA MODERADA\nEl modelo predice un crecimiento del **{pron['cambio_pronostico']:.1f}%** en 90 días.")
            elif pron['cambio_pronostico'] > -10:
                st.warning(f"### 📉 SEÑAL BAJISTA MODERADA\nEl modelo predice una caída del **{abs(pron['cambio_pronostico']):.1f}%** en 90 días.")
            else:
                st.error(f"### 📉 SEÑAL BAJISTA FUERTE\nEl modelo predice una caída del **{abs(pron['cambio_pronostico']):.1f}%** en 90 días.")
            
            st.divider()
            
            st.subheader("🛡️ Análisis de Riesgos")
            st.markdown(f"**Perfil de Riesgo:** {riesgos['perfil_riesgo']}  \n**Score:** {riesgos['score_riesgo']}/100\n\n**Recomendación:** {riesgos['recomendacion']}")
            
            st.markdown("### ️ Principales Riesgos Identificados")
            riesgos_ordenados = sorted(riesgos['riesgos'], key=lambda x: x['severidad'], reverse=True)[:3]
            
            for i, riesgo in enumerate(riesgos_ordenados, 1):
                icono = "🔴" if riesgo['nivel'] == 'Crítico' else "🟠" if riesgo['nivel'] == 'Alto' else "🟡" if riesgo['nivel'] == 'Moderado' else "🟢"
                st.markdown(f"**{i}. {icono} Riesgo {riesgo['categoria']}** ({riesgo['nivel']})\n- {riesgo['descripcion']}\n- **Mitigación:** {riesgo['mitigacion']}")
            
            st.divider()
            
            if st.button("📥 Descargar PDF de Pronóstico y Riesgos"):
                with st.spinner("Generando PDF..."):
                    try:
                        pdf_path = generar_pdf_activo(ticker_analizado, datos, pron, riesgos)
                        
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="️ Descargar PDF",
                                data=f.read(),
                                file_name=f"Pronostico_{ticker_analizado}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                type="primary"
                            )
                        st.success("✅ PDF generado correctamente")
                    except Exception as e:
                        st.error(f"Error al generar PDF: {str(e)}")
        
        else:
            st.subheader("📈 Pronóstico de Precio con Bandas de Confianza (95%)")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio Actual", f"${pron['historico']['Precio'].iloc[-1]:.2f}")
            col2.metric("📊 Cambio 30 días", f"{pron['cambio_30d']:.1f}%", "↗️" if pron['cambio_30d'] > 0 else "↘️")
            col3.metric("🔮 Pronóstico 90 días", f"{pron['cambio_pronostico']:.1f}%", "↗️" if pron['cambio_pronostico'] > 0 else "↘️")
            col4.metric("⚠️ Volatilidad Diaria", f"{pron['volatilidad_diaria']:.2f}%")
            
            st.divider()
            
            st.subheader(" Proyección con Intervalos de Confianza")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pron['historico']['Fecha'], y=pron['historico']['Precio'], mode='lines', name='Histórico', line=dict(color='blue', width=2)))
            
            fechas_futuras = pd.date_range(start=pron['historico']['Fecha'].iloc[-1], periods=pron['dias_pronostico'] + 1, freq='D')[1:]
            
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['precios_pronosticados'], mode='lines', name='Pronóstico (Regresión Lineal)', line=dict(color='orange', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_superior'], mode='lines', name='Límite Superior (95%)', line=dict(width=0), showlegend=True))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_inferior'], mode='lines', name='Límite Inferior (95%)', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=True))
            
            fig.update_layout(title=f'Pronóstico de {ticker_analizado} - Modelo de Regresión Lineal', xaxis_title='Fecha', yaxis_title='Precio (USD)', hovermode='x unified', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.subheader(" Análisis de Tendencia")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Tendencia Detectada: **{pron['tendencia']}**\n\nEl modelo de regresión lineal indica una tendencia **{pron['tendencia'].lower()}** basada en los últimos 365 días.")
            with col2:
                st.markdown(f"### Bandas de Confianza (95%)\n\nLas bandas naranja muestran el rango donde el precio tiene **95% de probabilidad** de estar en cada fecha futura.")
            
            st.divider()
            
            st.subheader("️ Análisis de Riesgos Consolidado")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=riesgos['score_riesgo'],
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Score de Riesgo (0-100)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 50], 'color': "yellow"},
                            {'range': [50, 70], 'color': "orange"},
                            {'range': [70, 100], 'color': "red"}
                        ],
                        'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 70}
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col2:
                st.markdown(f"### Perfil de Riesgo: **{riesgos['perfil_riesgo']}**\n\n**Score:** {riesgos['score_riesgo']}/100\n\n**Recomendación:**\n{riesgos['recomendacion']}")
            
            st.divider()
            
            st.subheader("🕸️ Mapa de Riesgos por Categoría")
            
            categorias = [r['categoria'] for r in riesgos['riesgos']]
            severidades = [r['severidad'] for r in riesgos['riesgos']]
            
            categorias_closed = categorias + [categorias[0]]
            severidades_closed = severidades + [severidades[0]]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(r=severidades_closed, theta=categorias_closed, fill='toself', name='Nivel de Riesgo', line=dict(color='red', width=2), fillcolor='rgba(255, 0, 0, 0.2)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, title='Mapa de Riesgos por Categoría (0-100)', height=500)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.divider()
            
            st.subheader("🔍 Detalle de Riesgos Identificados")
            
            for i, riesgo in enumerate(riesgos['riesgos'], 1):
                with st.expander(f"{i}. Riesgo {riesgo['categoria']} - Severidad: {riesgo['nivel']} ({riesgo['severidad']}/100)"):
                    st.markdown(f"**Descripción:**\n{riesgo['descripcion']}\n\n**Nivel de Severidad:** {riesgo['nivel']} ({riesgo['severidad']}/100)\n\n**Plan de Mitigación:**\n{riesgo['mitigacion']}")
            
            st.divider()
            
            st.subheader("🎯 Veredicto Final del Agente IA")
            
            if riesgos['score_riesgo'] < 30:
                st.success(f"### ✅ PERFIL CONSERVADOR - APTO PARA BUFFETT\n\n**{ticker_analizado}** presenta un perfil de riesgo **{riesgos['perfil_riesgo'].lower()}** con score de {riesgos['score_riesgo']}/100.\n\n*Cumple con el principio de 'primero, no perder dinero' de Warren Buffett*")
            elif riesgos['score_riesgo'] < 50:
                st.info(f"### ⚖️ PERFIL BALANCEADO - ACEPTABLE CON DIVERSIFICACIÓN\n\n**{ticker_analizado}** presenta un perfil de riesgo **{riesgos['perfil_riesgo'].lower()}** con score de {riesgos['score_riesgo']}/100.\n\n*Balance adecuado entre riesgo y retorno*")
            else:
                st.warning(f"### ⚠️ PERFIL AGRESIVO - REQUIERE ANÁLISIS PROFUNDO\n\n**{ticker_analizado}** presenta un perfil de riesgo **{riesgos['perfil_riesgo'].lower()}** con score de {riesgos['score_riesgo']}/100.\n\n*Solo considerar si el potencial de retorno justifica el riesgo asumido*")
            
            st.divider()
            
            if st.button("📥 Descargar PDF de Pronóstico y Riesgos"):
                with st.spinner("Generando PDF..."):
                    try:
                        pdf_path = generar_pdf_activo(ticker_analizado, datos, pron, riesgos)
                        
                        with open(pdf_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Descargar PDF",
                                data=f.read(),
                                file_name=f"Pronostico_{ticker_analizado}_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf",
                                type="primary"
                            )
                        st.success("✅ PDF generado correctamente")
                    except Exception as e:
                        st.error(f"Error al generar PDF: {str(e)}")

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p><strong>QuantBuffett AI v1.2.0</strong> | Paso 8 de 14</p>
    <p>Datos: Yahoo Finance (yfinance) + Fallback inteligente</p>
    <p><em>"La regla número 1 es no perder dinero. La regla número 2 es no olvidar la regla número 1."</em> — Warren Buffett</p>
</div>
""", unsafe_allow_html=True)
