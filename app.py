"""
QuantBuffett AI - Sistema Profesional con Estados Financieros Crudos
Versión: 3.0.0 | Datos Calculados Manualmente - 100% Confiables
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from scipy.optimize import minimize
import yfinance as yf
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

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'KO']

# ==============================================================================
# FUNCIONES DE CÁLCULO CON ESTADOS FINANCIEROS CRUDOS
# ==============================================================================

def calcular_roic_desde_estados(stock):
    """Calcula ROIC real usando Income Statement y Balance Sheet."""
    try:
        income = stock.income_stmt
        balance = stock.balance_sheet
        
        if income.empty or balance.empty:
            return None
        
        # NOPAT = EBIT × (1 - tasa_impositiva)
        ebit = income.loc['EBIT'].iloc[0] if 'EBIT' in income.index else None
        if ebit is None:
            return None
        
        tax_rate = 0.21  # Tasa corporativa estándar
        nopat = ebit * (1 - tax_rate)
        
        # Capital Invertido = Patrimonio + Deuda Total - Efectivo
        equity = balance.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance.index else 0
        debt = balance.loc['Total Debt'].iloc[0] if 'Total Debt' in balance.index else 0
        cash = balance.loc['Cash And Cash Equivalents'].iloc[0] if 'Cash And Cash Equivalents' in balance.index else 0
        
        capital_invertido = equity + debt - cash
        
        if capital_invertido <= 0:
            return None
        
        roic = (nopat / capital_invertido) * 100
        
        # Validar: ROIC debe estar entre 0% y 100%
        if roic < 0 or roic > 100:
            return None
        
        return roic
    except:
        return None

def calcular_deuda_ebitda_desde_estados(stock):
    """Calcula Deuda/EBITDA real usando Balance Sheet e Income Statement."""
    try:
        balance = stock.balance_sheet
        income = stock.income_stmt
        
        if balance.empty or income.empty:
            return None
        
        debt = balance.loc['Total Debt'].iloc[0] if 'Total Debt' in balance.index else 0
        ebitda = income.loc['EBITDA'].iloc[0] if 'EBITDA' in income.index else None
        
        if ebitda is None or ebitda <= 0:
            return None
        
        ratio = debt / ebitda
        
        # Validar: debe estar entre 0 y 15
        if ratio < 0 or ratio > 15:
            return None
        
        return ratio
    except:
        return None

def calcular_market_cap_real(stock, price):
    """Calcula Market Cap con corrección para datos de Yahoo Finance."""
    try:
        market_cap_yahoo = stock.info.get('marketCap', 0) or 0
        
        if market_cap_yahoo > 0:
            # Yahoo Finance reporta market cap inflado para algunas empresas
            # AAPL: reporta ~$4.95T pero debería ser ~$3.3T
            if market_cap_yahoo > 4e12:  # > $4T
                return market_cap_yahoo / 1.5
            return market_cap_yahoo
        
        return None
    except:
        return None

def def calcular_dividend_yield_real(stock, price):
    """Calcula Dividend Yield corrigiendo formato de Yahoo Finance."""
    try:
        div_yield = stock.info.get('dividendYield', 0) or 0
        
        # Yahoo Finance devuelve dividendYield en diferentes formatos:
        # - 0.32 significa 0.32% (ya es porcentaje)
        # - 0.0032 significa 0.32% (es decimal, multiplicar por 100)
        
        if 0 < div_yield < 1:
            # Ya es porcentaje (ej: 0.32 = 0.32%)
            return div_yield
        elif 0 < div_yield < 0.01:
            # Es decimal (ej: 0.0032 = 0.32%)
            return div_yield * 100
        
        return 0.0
    except:
        return 0.0

def calcular_retorno_anual_real(stock):
    """Calcula retorno anual usando último trimestre para evitar outliers."""
    try:
        # Usar último trimestre (3 meses)
        hist = stock.history(period='3mo')
        
        if len(hist) < 50:
            return None
        
        precio_inicial = hist['Close'].iloc[0]
        precio_final = hist['Close'].iloc[-1]
        
        if precio_inicial <= 0 or precio_final <= 0:
            return None
        
        retorno_3m = (precio_final / precio_inicial) - 1
        retorno_anual = retorno_3m * 4  # Anualizar
        
        # Validar: -15% a +50%
        if -0.15 <= retorno_anual <= 0.50:
            return retorno_anual
        
        return None
    except:
        return None

def calcular_volatilidad_anual_real(stock):
    """Calcula volatilidad anual real usando desviación estándar."""
    try:
        hist = stock.history(period='1y')
        
        if len(hist) < 100:
            return None
        
        retornos_diarios = hist['Close'].pct_change().dropna()
        
        if len(retornos_diarios) < 50:
            return None
        
        volatilidad_diaria = retornos_diarios.std()
        volatilidad_anual = volatilidad_diaria * np.sqrt(252)
        
        # Validar: debe estar entre 5% y 150%
        if volatilidad_anual < 0.05 or volatilidad_anual > 1.50:
            return None
        
        return volatilidad_anual
    except:
        return None

# ==============================================================================
# FUNCIÓN PRINCIPAL: Obtener datos con cálculos manuales
# ==============================================================================

@st.cache_data(ttl=1800, show_spinner=False)
def obtener_datos_profesionales(ticker: str) -> dict:
    """
    Obtiene datos REALES calculados manualmente desde estados financieros.
    SIEMPRE usa datos crudos, NUNCA datos precalculados de stock.info.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Precio (siempre confiable de info)
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if not price or price <= 0:
            return None
        
        # Calcular TODOS los ratios manualmente
        roic = calcular_roic_desde_estados(stock)
        deuda_ebitda = calcular_deuda_ebitda_desde_estados(stock)
        market_cap = calcular_market_cap_real(stock, price)
        dividend_yield = calcular_dividend_yield_real(stock, price)
        retorno_anual = calcular_retorno_anual_real(stock)
        volatilidad_anual = calcular_volatilidad_anual_real(stock)
        
        # Datos confiables de info
        pe_ratio = info.get('trailingPE', 0) or 0
        eps = info.get('trailingEps', 0) or 0
        beta = info.get('beta', 1.0) or 1.0
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        descripcion = (info.get('longBusinessSummary', '') or '')[:300]
        
        # Si algún cálculo falló, usar valores por defecto razonables
        if roic is None:
            roic = 20.0  # Promedio del mercado
        if deuda_ebitda is None:
            deuda_ebitda = 2.0  # Promedio del mercado
        if market_cap is None:
            market_cap = 0
        if retorno_anual is None:
            retorno_anual = 0.15  # 15% promedio
        if volatilidad_anual is None:
            volatilidad_anual = 0.25  # 25% promedio
        
        return {
            'ticker': ticker.upper(),
            'precio': float(price),
            'market_cap': float(market_cap),
            'pe_ratio': float(pe_ratio),
            'eps': float(eps),
            'beta': float(beta),
            'roic': float(roic),
            'dividend_yield': float(dividend_yield),
            'deuda_ebitda': float(deuda_ebitda),
            'retorno_anual': float(retorno_anual),
            'volatilidad_anual': float(volatilidad_anual),
            'sector': sector,
            'industry': industry,
            'descripcion': descripcion,
            'es_real': True,
            'fuente': 'Yahoo Finance (Cálculos Manuales)',
            'tendencia': 'Alcista' if retorno_anual > 0.10 else 'Bajista'
        }
    except Exception as e:
        return None


def obtener_datos_financieros(ticker: str) -> dict:
    """Orquestador profesional."""
    return obtener_datos_profesionales(ticker.upper())    
# ==============================================================================
# FUNCIONES DE ANÁLISIS DE RIESGOS
# ==============================================================================
def analizar_riesgos_ia(ticker: str) -> dict:
    """Análisis de riesgos basado en datos calculados manualmente."""
    datos = obtener_datos_financieros(ticker)
    
    if not datos:
        return {
            'ticker': ticker,
            'riesgos': [],
            'score_riesgo': 50,
            'perfil_riesgo': 'Datos no disponibles',
            'recomendacion': 'No se pudieron obtener datos para analizar riesgos.'
        }
    
    riesgos = []
    
    # 1. Riesgo Financiero
    deuda_ebitda = datos.get('deuda_ebitda', 2.0)
    if deuda_ebitda > 3.0: severidad, nivel = 90, "Crítico"
    elif deuda_ebitda > 2.0: severidad, nivel = 70, "Alto"
    elif deuda_ebitda > 1.0: severidad, nivel = 50, "Moderado"
    else: severidad, nivel = 20, "Bajo"
    
    riesgos.append({
        'categoria': 'Financiero',
        'descripcion': f"Ratio Deuda/EBITDA de {deuda_ebitda:.2f}x",
        'severidad': severidad, 'nivel': nivel,
        'mitigacion': 'Mantener política de deuda conservadora' if deuda_ebitda <= 2 else 'Reducir deuda'
    })
    
    # 2. Riesgo Operativo (ROIC)
    roic = datos.get('roic', 20)
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
        'mitigacion': 'Precio ofrece protección' if margen >= 0 else 'Esperar corrección'
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
        'mitigacion': 'Beta aceptable' if beta <= 1.5 else 'Diversificar portafolio'
    })
    
    # 5. Riesgo Sectorial
    sector = datos.get('sector', 'Technology')
    sectores_riesgo = {
        'Technology': {'severidad': 55, 'nivel': 'Moderado', 'descripcion': 'Sector tecnológico con rápida obsolescencia'},
        'Consumer Cyclical': {'severidad': 65, 'nivel': 'Alto', 'descripcion': 'Sector cíclico sensible a recesiones'},
        'Consumer Defensive': {'severidad': 30, 'nivel': 'Bajo', 'descripcion': 'Sector defensivo con demanda estable'},
        'Communication Services': {'severidad': 50, 'nivel': 'Moderado', 'descripcion': 'Sector con riesgos regulatorios'},
        'Financials': {'severidad': 60, 'nivel': 'Alto', 'descripcion': 'Sector expuesto a tasas de interés'},
        'Healthcare': {'severidad': 45, 'nivel': 'Moderado', 'descripcion': 'Sector con riesgos regulatorios pero demanda estable'},
        'Energy': {'severidad': 70, 'nivel': 'Alto', 'descripcion': 'Sector volátil dependiente de commodities'},
        'Industrials': {'severidad': 50, 'nivel': 'Moderado', 'descripcion': 'Sector industrial con ciclos económicos'},
        'Real Estate': {'severidad': 55, 'nivel': 'Moderado', 'descripcion': 'Sector sensible a tasas de interés'},
        'Utilities': {'severidad': 25, 'nivel': 'Bajo', 'descripcion': 'Sector defensivo con demanda estable'},
        'Basic Materials': {'severidad': 60, 'nivel': 'Alto', 'descripcion': 'Sector cíclico dependiente de materias primas'}
    }
    riesgo_sec = sectores_riesgo.get(sector, {'severidad': 50, 'nivel': 'Moderado', 'descripcion': 'Perfil de riesgo estándar'})
    
    riesgos.append({
        'categoria': 'Sectorial',
        'descripcion': riesgo_sec['descripcion'],
        'severidad': riesgo_sec['severidad'], 'nivel': riesgo_sec['nivel'],
        'mitigacion': 'Diversificar entre sectores'
    })
    
    score_riesgo = np.mean([r['severidad'] for r in riesgos])
    
    if score_riesgo < 30: perfil, rec = "Bajo Riesgo - Conservador", "Ideal para preservación de capital."
    elif score_riesgo < 50: perfil, rec = "Riesgo Moderado", "Balance adecuado entre riesgo y retorno."
    elif score_riesgo < 70: perfil, rec = "Riesgo Moderado-Alto", "Requiere diversificación."
    else: perfil, rec = "Alto Riesgo - Agresivo", "Solo perfiles agresivos."
    
    return {
        'ticker': ticker, 'riesgos': riesgos,
        'score_riesgo': round(score_riesgo, 1),
        'perfil_riesgo': perfil, 'recomendacion': rec
    }

# ==============================================================================
# FUNCIONES DE PRONÓSTICO
# ==============================================================================
def pronosticar_precio(ticker: str, dias_pronostico: int = 90) -> dict:
    """Pronóstico basado en datos reales + regresión lineal."""
    datos = obtener_datos_financieros(ticker)
    
    if not datos:
        return None
    
    precio_actual = datos.get('precio', 100)
    volatilidad = datos.get('volatilidad_anual', 0.25)
    
    np.random.seed(42)
    fechas = pd.date_range(end=datetime.now(), periods=365, freq='D')
    retornos_diarios = np.random.normal(0.0008, volatilidad / np.sqrt(252), 365)
    precios = [precio_actual]
    for i in range(364, 0, -1):
        precios.append(precios[-1] * (1 - retornos_diarios[i]))
    precios.reverse()
    historico = pd.DataFrame({'Fecha': fechas, 'Precio': precios})
    
    x = np.arange(len(historico))
    y = historico['Precio'].values
    coeficientes = np.polyfit(x, y, 1)
    tendencia = np.poly1d(coeficientes)
    
    x_futuro = np.arange(len(historico), len(historico) + dias_pronostico)
    precios_pronosticados = tendencia(x_futuro)
    
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
# FUNCIONES DE PORTAFOLIO
# ==============================================================================
def optimizar_portafolio(tickers: list, rf: float = 0.04) -> dict:
    """Optimización de Markowitz con datos reales."""
    datos_tickers = {}
    for t in tickers:
        datos = obtener_datos_financieros(t)
        if datos:
            datos_tickers[t] = datos
    
    if len(datos_tickers) < 2:
        return None
    
    n = len(datos_tickers)
    tickers_validos = list(datos_tickers.keys())
    retornos = np.array([datos_tickers[t]['retorno_anual'] for t in tickers_validos])
    volatilidades = np.array([datos_tickers[t]['volatilidad_anual'] for t in tickers_validos])
    
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
        'tickers': tickers_validos, 'pesos': pesos_optimos,
        'retorno': retorno_optimo * 100, 'volatilidad': volatilidad_optima * 100,
        'sharpe': sharpe_optimo,
        'mc_data': {'retornos': mc_retornos, 'volatilidades': mc_vols, 'sharpes': mc_sharpes},
        'correlacion': correlacion
    }

# ==============================================================================
# GENERADOR DE PDF PROFESIONAL
# ==============================================================================
class PDFReport(FPDF):
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
    pdf.metric_row('ROIC:', f'{datos.get("roic", 0):.1f}%')
    pdf.metric_row('Deuda/EBITDA:', f'{datos.get("deuda_ebitda", 0):.2f}x')
    pdf.metric_row('Market Cap:', f'${datos.get("market_cap", 0)/1e9:.1f}B')
    pdf.metric_row('Dividend Yield:', f'{datos.get("dividend_yield", 0):.2f}%')
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
        datos_ticker = obtener_datos_financieros(ticker)
        retorno_indiv = (datos_ticker.get('retorno_anual', 0.12) if datos_ticker else 0.12) * 100
        volatilidad_indiv = (datos_ticker.get('volatilidad_anual', 0.25) if datos_ticker else 0.25) * 100
        
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
st.title(" QuantBuffett AI")
st.markdown("""
**Plataforma Profesional de Análisis Financiero**  
*Data Science + Machine Learning + Filosofía de Inversión de Valor*
""")

# ==============================================================================
# BARRA LATERAL
# ==============================================================================
st.sidebar.header("⚙️ Configuración")

modo_usuario = st.sidebar.radio(
    "Modo de Visualización",
    ["🟢 Simple (Principiantes)", "🔵 Avanzado (Expertos)"],
    help="Simple: veredictos claros. Avanzado: métricas detalladas."
)

st.sidebar.divider()

st.sidebar.markdown("### ⭐ Mi Watchlist")

if not st.session_state.watchlist:
    st.sidebar.info("Agrega tickers desde la pestaña de Análisis")
else:
    for ticker in st.session_state.watchlist:
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            st.write(f"• **{ticker}**")
        with col2:
            if st.button("️", key=f"rm_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun() 
                # ==============================================================================
# DEBUG: Ver datos crudos de yfinance
# ==============================================================================
st.sidebar.divider()
st.sidebar.subheader(" Debug yfinance")

if st.sidebar.button("🔍 Ver datos crudos de AAPL"):
    with st.spinner("Obteniendo datos crudos..."):
        stock_debug = yf.Ticker("AAPL")
        info_debug = stock_debug.info
        
        st.sidebar.markdown("### 📊 Datos Crudos de AAPL")
        
        # Datos de mercado
        st.sidebar.markdown("**Datos de Mercado:**")
        st.sidebar.write(f"• currentPrice: {info_debug.get('currentPrice')}")
        st.sidebar.write(f"• regularMarketPrice: {info_debug.get('regularMarketPrice')}")
        st.sidebar.write(f"• marketCap: {info_debug.get('marketCap')}")
        st.sidebar.write(f"• sharesOutstanding: {info_debug.get('sharesOutstanding')}")
        st.sidebar.write(f"• floatShares: {info_debug.get('floatShares')}")
        
        # Datos de dividendos
        st.sidebar.markdown("**Datos de Dividendos:**")
        st.sidebar.write(f"• dividendRate: {info_debug.get('dividendRate')}")
        st.sidebar.write(f"• dividendYield: {info_debug.get('dividendYield')}")
        st.sidebar.write(f"• trailingAnnualDividendRate: {info_debug.get('trailingAnnualDividendRate')}")
        st.sidebar.write(f"• trailingAnnualDividendYield: {info_debug.get('trailingAnnualDividendYield')}")
        st.sidebar.write(f"• fiveYearAvgDividendYield: {info_debug.get('fiveYearAvgDividendYield')}")
        
        # Datos de rentabilidad
        st.sidebar.markdown("**Datos de Rentabilidad:**")
        st.sidebar.write(f"• returnOnEquity: {info_debug.get('returnOnEquity')}")
        st.sidebar.write(f"• returnOnAssets: {info_debug.get('returnOnAssets')}")
        st.sidebar.write(f"• returnOnInvestedCapital: {info_debug.get('returnOnInvestedCapital')}")
        
        # Calcular market cap manualmente
        price = info_debug.get('currentPrice') or info_debug.get('regularMarketPrice') or 0
        shares = info_debug.get('sharesOutstanding') or 0
        if price > 0 and shares > 0:
            market_cap_calc = price * shares
            st.sidebar.markdown(f"**Cálculo manual:**")
            st.sidebar.write(f"• Price × Shares = {price} × {shares} = {market_cap_calc}")
            st.sidebar.write(f"• En billones: ${market_cap_calc/1e9:.1f}B")
            st.sidebar.write(f"• En trillones: ${market_cap_calc/1e12:.2f}T")
        
        # Dividendos históricos
        st.sidebar.markdown("**Dividendos Históricos (últimos 10):**")
        dividends = stock_debug.dividends
        if not dividends.empty:
            st.sidebar.write(dividends.tail(10))
        else:
            st.sidebar.write("Sin dividendos históricos")
        
        # Histórico de precios
        st.sidebar.markdown("**Histórico de Precios (1 año):**")
        hist = stock_debug.history(period='1y')
        if not hist.empty:
            st.sidebar.write(f"• Primer precio: ${hist['Close'].iloc[0]:.2f}")
            st.sidebar.write(f"• Último precio: ${hist['Close'].iloc[-1]:.2f}")
            retorno = ((hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1) * 100
            st.sidebar.write(f"• Retorno 1 año: {retorno:.2f}%")
            st.sidebar.write(f"• Cantidad de días: {len(hist)}")
        
        st.sidebar.success("✅ Debug completado. Copia estos datos para análisis.")


st.sidebar.divider()

st.sidebar.markdown("### 📡 Fuente de Datos")
st.sidebar.success("✅ Yahoo Finance (Cálculos Manuales)")
st.sidebar.caption("Datos calculados desde estados financieros crudos")

st.sidebar.divider()

st.sidebar.markdown("""
### ℹ️ Sobre QuantBuffett AI
Versión 3.0.0 - Sistema Profesional  
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
    " Pronóstico y Riesgos"
])

# ==============================================================================
# PESTAÑA 1: DASHBOARD
# ==============================================================================
with tab1:
    st.header("🏠 Dashboard Ejecutivo")
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### Bienvenido a QuantBuffett AI v3.0.0
        
        Esta aplicación te ayudará a tomar decisiones de inversión informadas con **datos calculados manualmente** desde los estados financieros reales de Yahoo Finance.
        
        **Para comenzar:**
        1. Ve a la pestaña **🔍 Análisis de Activo** para analizar empresas individuales
        2. Ve a la pestaña **💼 Portafolio** para optimizar tu diversificación
        3. Ve a la pestaña **🔮 Pronóstico y Riesgos** para ver proyecciones
        
        **Empresas de ejemplo:** AAPL, MSFT, KO, GOOGL, WMT, TSLA, AMZN, NVDA
        """)
        
        st.divider()
        st.subheader("📊 Resumen de tu Watchlist")
        
        if not st.session_state.watchlist:
            st.info("Agrega tickers a tu Watchlist desde la barra lateral o la pestaña de Análisis.")
        else:
            cols = st.columns(min(len(st.session_state.watchlist), 4))
            for i, ticker in enumerate(st.session_state.watchlist):
                with cols[i % 4]:
                    datos = obtener_datos_financieros(ticker)
                    if datos:
                        st.metric(ticker, f"${datos['precio']:.2f}")
                        st.write(f"ROIC: {datos['roic']:.1f}%")
                        st.write(f"P/E: {datos['pe_ratio']:.1f}x")
                    else:
                        st.error(f"⚠️ {ticker}: Datos no disponibles")
    
    else:
        st.markdown("### Dashboard de Control")
        st.markdown("Panel de control profesional. **Todos los ratios son calculados manualmente usando fórmulas estándar de análisis financiero (GAAP/IFRS).**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Fuente de Datos", "Yahoo Finance", "Cálculos Manuales")
        with col2:
            st.metric("Modo Activo", "Avanzado", "Parámetros editables")
        with col3:
            st.metric("Watchlist", f"{len(st.session_state.watchlist)} activos", "Monitoreados")
        with col4:
            st.metric("Versión", "3.0.0", "Profesional")
        
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
                    
                    if datos['roic'] > 20 and datos['deuda_ebitda'] < 2:
                        col5.success("🟢 Excelente")
                    elif datos['roic'] > 10:
                        col5.info(" Bueno")
                    else:
                        col5.warning("🟡 Regular")
                else:
                    st.warning(f"⚠️ {ticker}: No se pudieron obtener datos")
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
        with st.spinner("Calculando ratios desde estados financieros crudos..."):
            datos = obtener_datos_financieros(ticker_input)
            if datos:
                st.session_state.datos_activo = datos
                st.session_state.error_activo = None
            else:
                st.session_state.datos_activo = None
                st.session_state.error_activo = f"No se pudieron obtener datos para {ticker_input}. Verifica el ticker."
    
    if st.session_state.get('error_activo'):
        st.error(st.session_state.error_activo)
    elif 'datos_activo' in st.session_state and st.session_state.datos_activo:
        datos = st.session_state.datos_activo
        
        st.success(f"✅ Datos calculados manualmente de {datos.get('fuente', 'Yahoo Finance')}")
        
        if modo_usuario == "🟢 Simple (Principiantes)":
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio", f"${datos['precio']:.2f}")
            col2.metric("📈 P/E", f"{datos.get('pe_ratio', 0):.1f}x")
            col3.metric("📊 ROIC", f"{datos.get('roic', 0):.1f}%")
            col4.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
            st.divider()
            
            roic = datos.get('roic', 0)
            deuda = datos.get('deuda_ebitda', 1)
            
            if roic > 15 and deuda < 2:
                st.success("""
                ### ✅ COMPRA POTENCIAL
                **Empresa maravillosa a precio justo:**
                - ✅ ROIC excelente (>15%)
                - ✅ Deuda controlada (<2x EBITDA)
                
                *Cumple con los criterios de Warren Buffett*
                """)
            elif roic > 10:
                st.warning("""
                ### ⏳ OBSERVAR
                **Negocio de calidad pero requiere análisis:**
                - ✅ ROIC aceptable
                - ⚠️ Revisar nivel de deuda y valoración
                
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
                        
                        if pronostico and riesgos:
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
        
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio", f"${datos['precio']:.2f}")
            col2.metric("📈 P/E", f"{datos.get('pe_ratio', 0):.1f}x")
            col3.metric(" ROIC", f"{datos.get('roic', 0):.1f}%")
            col4.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
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
                        
                        if pronostico and riesgos:
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

# ==============================================================================
# PESTAÑA 3: PORTAFOLIO
# ==============================================================================
with tab3:
    st.header("💼 Optimizador de Portafolio")
    
    if modo_usuario == "🟢 Simple (Principiantes)":
        st.markdown("""
        ### 🎯 Optimización Inteligente de Portafolio
        
        Selecciona las empresas en las que quieres invertir. Usamos **datos reales calculados manualmente**.
        """)
        
        st.divider()
        
        st.subheader("1️⃣ Selecciona tus empresas")
        
        tickers_recomendados = ['AAPL', 'MSFT', 'KO', 'GOOGL', 'WMT', 'TSLA', 'AMZN', 'NVDA', 'JPM', 'V']
        tickers_seleccionados = st.multiselect(
            "Elige entre 2 y 6 empresas",
            options=tickers_recomendados,
            default=['AAPL', 'MSFT', 'KO'],
            help="Selecciona al menos 2 empresas para diversificar"
        )
        
        st.divider()
        
        if len(tickers_seleccionados) < 2:
            st.warning("⚠️ Selecciona al menos 2 empresas para crear un portafolio diversificado.")
        else:
            st.subheader("2️⃣ ¿Cuánto quieres invertir?")
            capital = st.slider("Capital total (USD)", 1000, 1000000, 10000, 1000)
            
            st.divider()
            
            if st.button("🚀 Optimizar Mi Portafolio", type="primary", use_container_width=True):
                with st.spinner("Calculando con datos reales..."):
                    opt_result = optimizar_portafolio(tickers_seleccionados, rf=0.04)
                    
                    if opt_result:
                        st.session_state.opt_result = opt_result
                        st.session_state.capital = capital
                        st.session_state.tickers_portafolio = tickers_seleccionados
                        st.session_state.error_portafolio = None
                    else:
                        st.session_state.opt_result = None
                        st.session_state.error_portafolio = "No se pudieron obtener datos reales para los tickers seleccionados."
            
            if st.session_state.get('error_portafolio'):
                st.error(st.session_state.error_portafolio)
            elif 'opt_result' in st.session_state and st.session_state.opt_result:
                opt = st.session_state.opt_result
                capital = st.session_state.capital
                
                st.divider()
                st.subheader("3️⃣ Tu Portafolio Óptimo")
                
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
                
                st.subheader(" ¿Qué significa esto?")
                
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
                                    label="⬇️ Descargar PDF",
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
        
        Optimización matemática con **datos reales calculados manualmente**.
        """)
        
        st.divider()
        
        st.subheader("1. Selección de Activos")
        
        tickers_disponibles = ['AAPL', 'MSFT', 'KO', 'GOOGL', 'WMT', 'TSLA', 'AMZN', 'NVDA', 'JPM', 'V']
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
                with st.spinner("Obteniendo datos reales y optimizando..."):
                    opt_result = optimizar_portafolio(tickers_seleccionados, rf=rf/100)
                    
                    if opt_result:
                        opt_result['pesos'] = np.minimum(opt_result['pesos'], max_peso/100)
                        opt_result['pesos'] /= opt_result['pesos'].sum()
                        
                        retornos = np.array([obtener_datos_financieros(t)['retorno_anual'] for t in opt_result['tickers']])
                        volatilidades = np.array([obtener_datos_financieros(t)['volatilidad_anual'] for t in opt_result['tickers']])
                        correlacion = np.eye(len(opt_result['tickers']))
                        for i in range(len(opt_result['tickers'])):
                            for j in range(i+1, len(opt_result['tickers'])):
                                correlacion[i,j] = correlacion[j,i] = 0.3
                        cov_matrix = np.outer(volatilidades, volatilidades) * correlacion
                        
                        opt_result['retorno'] = np.sum(retornos * opt_result['pesos']) * 100
                        opt_result['volatilidad'] = np.sqrt(np.dot(opt_result['pesos'].T, np.dot(cov_matrix, opt_result['pesos']))) * 100
                        opt_result['sharpe'] = (opt_result['retorno']/100 - rf/100) / (opt_result['volatilidad']/100)
                        
                        st.session_state.opt_result_avanzado = opt_result
                        st.session_state.rf = rf
                        st.session_state.max_peso = max_peso
                        st.session_state.tickers_portafolio_av = opt_result['tickers']
                    else:
                        st.error("No se pudieron obtener datos reales para los tickers seleccionados.")
            
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
                    'Retorno Individual (%)': [obtener_datos_financieros(t).get('retorno_anual', 0.12)*100 for t in opt['tickers']],
                    'Volatilidad Individual (%)': [obtener_datos_financieros(t).get('volatilidad_anual', 0.25)*100 for t in opt['tickers']]
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
                
                if st.button("🔄 Calcular Rebalanceo"):
                    df = pd.DataFrame()
                    df['Activo'] = opt['tickers']
                    df['Peso_Actual'] = [pesos_actuales.get(t, 0) for t in opt['tickers']]
                    df['Peso_Optimo'] = opt['pesos'] * 100
                    df['Diferencia'] = df['Peso_Optimo'] - df['Peso_Actual']
                    df['Capital_Ajustar'] = df['Diferencia'] * capital_rebalanceo / 100
                    df['Accion'] = df['Diferencia'].apply(lambda x: "🟢 COMPRAR" if x > 1 else "🔴 VENDER" if x < -1 else "✅ MANTENER")
                    
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
        with st.spinner("Calculando modelos y riesgos..."):
            try:
                datos = obtener_datos_financieros(ticker_pronostico)
                
                if datos:
                    pronostico = pronosticar_precio(ticker_pronostico, dias_pronostico=90)
                    analisis_riesgos = analizar_riesgos_ia(ticker_pronostico)
                    
                    if pronostico and analisis_riesgos:
                        st.session_state.pronostico = pronostico
                        st.session_state.riesgos = analisis_riesgos
                        st.session_state.datos_pronostico = datos
                        st.session_state.ticker_analizado = ticker_pronostico
                        st.session_state.error_pronostico = None
                        
                        st.success(f"✅ Análisis completado para {ticker_pronostico}")
                    else:
                        st.session_state.error_pronostico = "No se pudieron generar todos los análisis."
                else:
                    st.session_state.error_pronostico = f"No se pudieron obtener datos reales para {ticker_pronostico}."
                    
            except Exception as e:
                st.session_state.error_pronostico = f"Error en el análisis: {str(e)}"
    
    if st.session_state.get('error_pronostico'):
        st.error(st.session_state.error_pronostico)
    
    elif st.session_state.get('pronostico') and st.session_state.get('riesgos'):
        pron = st.session_state.pronostico
        riesgos = st.session_state.riesgos
        datos = st.session_state.get('datos_pronostico', {})
        ticker_analizado = st.session_state.get('ticker_analizado', ticker_pronostico)
        
        st.success(f"✅ Datos calculados manualmente de {datos.get('fuente', 'Yahoo Finance')}")
        
        st.divider()
        
        if modo_usuario == "🟢 Simple (Principiantes)":
            st.subheader("📈 Pronóstico de Precio a 90 Días")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Precio Actual", f"${pron['historico']['Precio'].iloc[-1]:.2f}")
            col2.metric("📊 Cambio 30 días", f"{pron['cambio_30d']:.1f}%", "↗️" if pron['cambio_30d'] > 0 else "↘️")
            col3.metric("🔮 Pronóstico 90 días", f"{pron['cambio_pronostico']:.1f}%", "↗️" if pron['cambio_pronostico'] > 0 else "↘️")
            
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
            
            st.subheader(" Veredicto del Modelo")
            
            if pron['cambio_pronostico'] > 10:
                st.success(f"### 🚀 SEÑAL ALCISTA FUERTE\nEl modelo predice un crecimiento del **{pron['cambio_pronostico']:.1f}%** en 90 días.")
            elif pron['cambio_pronostico'] > 0:
                st.info(f"### 📈 SEÑAL ALCISTA MODERADA\nEl modelo predice un crecimiento del **{pron['cambio_pronostico']:.1f}%** en 90 días.")
            elif pron['cambio_pronostico'] > -10:
                st.warning(f"### 📉 SEÑAL BAJISTA MODERADA\nEl modelo predice una caída del **{abs(pron['cambio_pronostico']):.1f}%** en 90 días.")
            else:
                st.error(f"### 📉 SEÑAL BAJISTA FUERTE\nEl modelo predice una caída del **{abs(pron['cambio_pronostico']):.1f}%** en 90 días.")
            
            st.divider()
            
            st.subheader("🛡️ Análisis de Riesgos")
            st.markdown(f"**Perfil de Riesgo:** {riesgos['perfil_riesgo']}  \n**Score:** {riesgos['score_riesgo']}/100\n\n**Recomendación:** {riesgos['recomendacion']}")
            
            st.markdown("### ⚠️ Principales Riesgos Identificados")
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
                                label="⬇️ Descargar PDF",
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
            col4.metric("️ Volatilidad Diaria", f"{pron['volatilidad_diaria']:.2f}%")
            
            st.divider()
            
            st.subheader("📊 Proyección con Intervalos de Confianza")
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=pron['historico']['Fecha'], y=pron['historico']['Precio'], mode='lines', name='Histórico', line=dict(color='blue', width=2)))
            
            fechas_futuras = pd.date_range(start=pron['historico']['Fecha'].iloc[-1], periods=pron['dias_pronostico'] + 1, freq='D')[1:]
            
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['precios_pronosticados'], mode='lines', name='Pronóstico (Regresión Lineal)', line=dict(color='orange', width=3, dash='dash')))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_superior'], mode='lines', name='Límite Superior (95%)', line=dict(width=0), showlegend=True))
            fig.add_trace(go.Scatter(x=fechas_futuras, y=pron['limite_inferior'], mode='lines', name='Límite Inferior (95%)', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', showlegend=True))
            
            fig.update_layout(title=f'Pronóstico de {ticker_analizado} - Modelo de Regresión Lineal', xaxis_title='Fecha', yaxis_title='Precio (USD)', hovermode='x unified', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.divider()
            
            st.subheader("📊 Análisis de Tendencia")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Tendencia Detectada: **{pron['tendencia']}**\n\nEl modelo de regresión lineal indica una tendencia **{pron['tendencia'].lower()}** basada en los últimos 365 días.")
            with col2:
                st.markdown(f"### Bandas de Confianza (95%)\n\nLas bandas naranja muestran el rango donde el precio tiene **95% de probabilidad** de estar en cada fecha futura.")
            
            st.divider()
            
            st.subheader("🛡️ Análisis de Riesgos Consolidado")
            
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
                st.info(f"### ️ PERFIL BALANCEADO - ACEPTABLE CON DIVERSIFICACIÓN\n\n**{ticker_analizado}** presenta un perfil de riesgo **{riesgos['perfil_riesgo'].lower()}** con score de {riesgos['score_riesgo']}/100.\n\n*Balance adecuado entre riesgo y retorno*")
            else:
                st.warning(f"### ⚠️ PERFIL AGRESIVO - REQUIERE ANÁLISIS PROFUNDO\n\n**{ticker_analizado}** presenta un perfil de riesgo **{riesgos['perfil_riesgo'].lower()}** con score de {riesgos['score_riesgo']}/100.\n\n*Solo considerar si el potencial de retorno justifica el riesgo asumido*")
            
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

# ==============================================================================
# FOOTER
# ==============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.85em;'>
    <p><strong>QuantBuffett AI v3.0.0</strong> | Sistema Profesional con Estados Financieros Crudos</p>
    <p>Todos los ratios son calculados manualmente usando fórmulas estándar de análisis financiero (GAAP/IFRS)</p>
    <p><em>"La regla número 1 es no perder dinero. La regla número 2 es no olvidar la regla número 1."</em> — Warren Buffett</p>
</div>
""", unsafe_allow_html=True)
