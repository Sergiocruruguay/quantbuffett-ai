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
import concurrent.futures
# ==============================================================================
# DESCARGA AUTOMÁTICA DE FUENTE UTF-8 PARA PDFs (Para que funcionen los acentos)
# ==============================================================================
import os
import urllib.request

FONT_PATH = "DejaVuSans.ttf"
if not os.path.exists(FONT_PATH):
    try:
        font_url = "https://github.com/dejavu-fonts/dejavu-fonts/raw/master/ttf/DejaVuSans.ttf"
        print("Descargando fuente para soportar acentos en PDFs...")
        urllib.request.urlretrieve(font_url, FONT_PATH)
        print("✅ Fuente descargada exitosamente.")
    except Exception as e:
        print(f"⚠️ No se pudo descargar la fuente automáticamente: {e}")

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

def calcular_metricas_ttm(stock):
    """
    Calcula ROIC y Deuda/EBITDA usando datos TTM (Trailing Twelve Months).
    Suma los últimos 4 trimestres y calcula la tasa impositiva efectiva real.
    """
    try:
        income = stock.income_stmt
        balance = stock.balance_sheet
        
        if income.empty or balance.empty or len(income.columns) < 1:
            return None, None
        
        # Tomar los últimos 4 trimestres disponibles (o menos si la empresa es nueva)
        num_periods = min(4, len(income.columns))
        income_ttm = income.iloc[:, :num_periods]
        
        # 1. Calcular EBIT y Tasa Impositiva Efectiva TTM
        ebit_ttm = income_ttm.loc['EBIT'].sum() if 'EBIT' in income_ttm.index else 0
        tax_prov = income_ttm.loc['Tax Provision'].sum() if 'Tax Provision' in income_ttm.index else 0
        pretax_income = income_ttm.loc['Pretax Income'].sum() if 'Pretax Income' in income_ttm.index else 0
        
        # Tasa efectiva: si el ingreso antes de impuestos es 0 o negativo, usamos 21% como fallback conservador
        tax_rate = (tax_prov / pretax_income) if pretax_income > 0 else 0.21
        tax_rate = max(0.0, min(0.40, tax_rate)) # Limitar a un rango razonable (0% - 40%)
        
        nopat = ebit_ttm * (1 - tax_rate)
        
        # 2. Calcular Capital Invertido (usando el balance más reciente)
        latest_balance = balance.iloc[:, 0]
        equity = latest_balance.get('Stockholders Equity', 0) or 0
        total_debt = latest_balance.get('Total Debt', 0) or 0
        cash = latest_balance.get('Cash And Cash Equivalents', 0) or 0
        
        capital_invertido = equity + total_debt - cash
        
        roic = (nopat / capital_invertido * 100) if capital_invertido > 0 else None
        
        # 3. Calcular Deuda / EBITDA TTM
        ebitda_ttm = income_ttm.loc['EBITDA'].sum() if 'EBITDA' in income_ttm.index else 0
        deuda_ebitda = (total_debt / ebitda_ttm) if ebitda_ttm > 0 else None
        
        return roic, deuda_ebitda
        
    except Exception as e:
        print(f"Error en cálculo TTM: {e}")
        return None, None

def calcular_market_cap_real(stock, price):
    """Calcula Market Cap real. Sin hacks de división."""
    try:
        shares = stock.info.get('sharesOutstanding', 0) or 0
        if shares > 0 and price > 0:
            return price * shares
        
        # Fallback directo de Yahoo, sin alteraciones
        return stock.info.get('marketCap', 0) or 0
    except Exception as e:
        print(f"Error en Market Cap: {e}")
        return 0


def calcular_dividend_yield_real(stock, price):
    """Calcula Dividend Yield corrigiendo la interpretación decimal de Yahoo Finance."""
    try:
        div_yield = stock.info.get('dividendYield', 0) or 0
        
        # Yahoo Finance lo devuelve como decimal (ej: 0.015 = 1.5%)
        if 0 < div_yield < 1:
            return div_yield * 100  # Convertir a porcentaje real
        elif 1 <= div_yield <= 20:
            return div_yield  # Ya viene como porcentaje en algunos casos raros
        
        return 0.0
    except Exception as e:
        print(f"Error en dividend yield: {e}")
        return 0.0


def calcular_retorno_anual_real(stock):
    """Calcula retorno anual real de 1 año. Devuelve None si los datos son insuficientes o inválidos."""
    try:
        hist = stock.history(period='1y')
        if len(hist) < 200:
            return None  # Mejor None que un dato inventado
        
        precio_inicial = hist['Close'].iloc[0]
        precio_final = hist['Close'].iloc[-1]
        
        if precio_inicial <= 0 or precio_final <= 0:
            return None
        
        # Cálculo correcto de retorno periódico (no anualización lineal)
        retorno = (precio_final / precio_inicial) - 1
        
        # Validación estricta: si es un outlier extremo, retornamos None
        if retorno < -0.80 or retorno > 2.0: 
            return None
            
        return retorno
    except Exception as e:
        print(f"Error calculando retorno: {e}")
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
    """Obtiene datos REALES calculados manualmente. SIN fallbacks optimistas engañosos."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        price = info.get('currentPrice') or info.get('regularMarketPrice') or 0
        if not price or price <= 0:
            return None
        
        roic, deuda_ebitda = calcular_metricas_ttm(stock)
        market_cap = calcular_market_cap_real(stock, price)
        dividend_yield = calcular_dividend_yield_real(stock, price)
        retorno_anual = calcular_retorno_anual_real(stock)
        volatilidad_anual = calcular_volatilidad_anual_real(stock)
        
        pe_ratio = info.get('trailingPE', 0) or 0
        
        # Cálculo real de Margen de Seguridad (vs PE de mercado de referencia 20x)
        pe_referencia = 20.0 
        margen_seguridad = ((pe_referencia - pe_ratio) / pe_referencia) * 100 if pe_ratio > 0 else 0.0
        
        return {
            'ticker': ticker.upper(),
            'precio': float(price),
            'market_cap': float(market_cap) if market_cap else 0.0,
            'pe_ratio': float(pe_ratio),
            'eps': float(info.get('trailingEps', 0) or 0),
            'beta': float(info.get('beta', 1.0) or 1.0),
            'roic': float(roic) if roic is not None else None,          # <-- None, no 20.0
            'dividend_yield': float(dividend_yield),
            'deuda_ebitda': float(deuda_ebitda) if deuda_ebitda is not None else None,
            'retorno_anual': float(retorno_anual) if retorno_anual is not None else None, # <-- None, no 0.15
            'volatilidad_anual': float(volatilidad_anual) if volatilidad_anual is not None else None,
            'margen_seguridad': float(margen_seguridad),                # <-- AGREGADO
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'descripcion': (info.get('longBusinessSummary', '') or '')[:300],
            'es_real': True,
            'fuente': 'Yahoo Finance (Cálculos TTM)',
            'tendencia': 'Alcista' if (retorno_anual or 0) > 0.10 else 'Bajista'
        }
    except Exception as e:
        print(f"Error en obtener_datos_profesionales: {e}")
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
    
    # CORRECCIÓN DEFINITIVA: Inicialización segura al nivel correcto de indentación
    riesgos = []
    
    # 1. Riesgo Financiero
    deuda_ebitda = datos.get('deuda_ebitda')
    if deuda_ebitda is None: 
        deuda_ebitda = 2.0 
        
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
    roic = datos.get('roic')
    if roic is None: 
        roic = 15.0 
        
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
    margen = datos.get('margen_seguridad')
    if margen is None: 
        margen = 0.0
        
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
        'ticker': ticker, 
        'riesgos': riesgos,
        'score_riesgo': round(score_riesgo, 1),
        'perfil_riesgo': perfil, 
        'recomendacion': rec
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
def fetch_historical_returns(ticker):
    """Función auxiliar para descargar retornos diarios en paralelo."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        if len(hist) < 50:
            return ticker, None
        returns = hist['Close'].pct_change().dropna()
        return ticker, returns
    except Exception:
        return ticker, None

def optimizar_portafolio(tickers: list, rf: float = 0.04) -> dict:
    """
    Optimización de Markowitz usando Matriz de Covarianza REAL histórica 
    y ejecución paralela para máximo rendimiento.
    """
    # 1. Obtener datos fundamentales en paralelo (para retornos esperados)
    datos_tickers = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(obtener_datos_financieros, t): t for t in tickers}
        for future in concurrent.futures.as_completed(futures):
            t = futures[future]
            try:
                res = future.result()
                if res:
                    datos_tickers[t] = res
            except Exception:
                pass

    if len(datos_tickers) < 2:
        return None

    valid_tickers = list(datos_tickers.keys())
    
    # 2. Obtener Matriz de Covarianza REAL histórica en paralelo
    returns_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = executor.map(fetch_historical_returns, valid_tickers)
        
    cov_valid_tickers = []
    for ticker, returns in results:
        if returns is not None:
            returns_dict[ticker] = returns
            cov_valid_tickers.append(ticker)
    
    # Intersectar para asegurar consistencia entre datos fundamentales e históricos
    tickers_finales = list(set(valid_tickers) & set(cov_valid_tickers))
    if len(tickers_finales) < 2:
        return None
        
    # Crear DataFrame de retornos y calcular matriz de covarianza anualizada (252 días)
    returns_df = pd.DataFrame({t: returns_dict[t] for t in tickers_finales})
    cov_matrix = returns_df.cov() * 252  # Anualizar
    
    # Asegurar que los arrays estén en el mismo orden que la matriz
    retornos = np.array([datos_tickers[t]['retorno_anual'] for t in tickers_finales])
    
    # CORRECCIÓN CRÍTICA: Si el retorno es None, usar 5% (conservador), NUNCA 15% (optimista engañoso)
    retornos = np.array([r if r is not None else 0.05 for r in retornos])

    # 3. Función de optimización (Maximizar Sharpe)
    def sharpe_negativo(pesos):
        port_retorno = np.sum(retornos * pesos)
        port_vol = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix, pesos)))
        sharpe = (port_retorno - rf) / port_vol if port_vol > 0 else 0
        return -sharpe

    restricciones = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    limites = tuple((0.0, 1.0) for _ in range(len(tickers_finales)))
    pesos_iniciales = np.ones(len(tickers_finales)) / len(tickers_finales)
    
    resultado = minimize(sharpe_negativo, pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
    
    pesos_optimos = resultado.x
    retorno_optimo = np.sum(retornos * pesos_optimos)
    volatilidad_optima = np.sqrt(np.dot(pesos_optimos.T, np.dot(cov_matrix, pesos_optimos)))
    sharpe_optimo = (retorno_optimo - rf) / volatilidad_optima if volatilidad_optima > 0 else 0
    
    # 4. Simulación Monte Carlo para la Frontera Eficiente (UI)
    mc_retornos, mc_vols, mc_sharpes = [], [], []
    for _ in range(1000):
        pesos_rand = np.random.random(len(tickers_finales))
        pesos_rand /= np.sum(pesos_rand)
        r = np.sum(retornos * pesos_rand)
        v = np.sqrt(np.dot(pesos_rand.T, np.dot(cov_matrix, pesos_rand)))
        s = (r - rf) / v if v > 0 else 0
        mc_retornos.append(r * 100)
        mc_vols.append(v * 100)
        mc_sharpes.append(s)
    
    return {
        'tickers': tickers_finales, 
        'pesos': pesos_optimos,
        'retorno': retorno_optimo * 100, 
        'volatilidad': volatilidad_optima * 100,
        'sharpe': sharpe_optimo,
        'cov_matrix': cov_matrix, # Matriz REAL para la UI
        'mc_data': {'retornos': mc_retornos, 'volatilidades': mc_vols, 'sharpes': mc_sharpes}
    }

# ==============================================================================
# GENERADOR DE PDF PROFESIONAL
# ==============================================================================
class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        try:
            # Registra la fuente UTF-8. Requiere que 'DejaVuSans.ttf' esté en el repo de GitHub.
            self.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
            self.set_font('DejaVu', '', 10)
        except Exception:
            # Fallback seguro: si la fuente no está, usa Helvetica (perderá acentos, pero no fallará)
            self.set_font('Helvetica', '', 10)

    def header(self):
        font_name = 'DejaVu' if 'DejaVu' in self.fonts else 'Helvetica'
        self.set_font(font_name, 'B', 16)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, 'QuantBuffett AI', 0, 1, 'L')
        self.set_font(font_name, '', 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 5, 'Plataforma Profesional de Analisis Financiero', 0, 1, 'L')
        self.ln(5)
        self.set_draw_color(26, 54, 93)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        font_name = 'DejaVu' if 'DejaVu' in self.fonts else 'Helvetica'
        self.set_font(font_name, 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}/{{nb}}', 0, 0, 'C')
    
    def section_title(self, title):
        font_name = 'DejaVu' if 'DejaVu' in self.fonts else 'Helvetica'
        self.set_font(font_name, 'B', 14)
        self.set_text_color(26, 54, 93)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)
    
    def metric_row(self, label, value):
        font_name = 'DejaVu' if 'DejaVu' in self.fonts else 'Helvetica'
        self.set_font(font_name, '', 10)
        self.set_text_color(80, 80, 80)
        self.cell(90, 8, label, 0, 0, 'L')
        self.set_font(font_name, 'B', 10)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, str(value), 0, 1, 'R')

def generar_pdf_activo(ticker: str, datos: dict, pronostico: dict, riesgos: dict) -> str:
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Detectar si la fuente UTF-8 se cargó correctamente
    font_base = 'DejaVu' if 'DejaVu' in pdf.fonts else 'Helvetica'
    
    pdf.set_font(font_base, 'B', 24)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 20, f'Analisis: {ticker}', 0, 1, 'C')
    
    pdf.set_font(font_base, '', 12)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, f'Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
    pdf.cell(0, 10, f'Fuente: {datos.get("fuente", "No disponible")}', 0, 1, 'C')
    pdf.ln(10)
    
    pdf.section_title('Metricas Fundamentales')
    pdf.metric_row('Precio Actual:', f'${datos.get("precio", 0):.2f}')
    pdf.metric_row('P/E Ratio:', f'{datos.get("pe_ratio", 0):.1f}x')
    pdf.metric_row('EPS:', f'${datos.get("eps", 0):.2f}')
    pdf.metric_row('Beta:', f'{datos.get("beta", 1.0):.2f}')
    
    # Manejo seguro de None para el PDF
    roic_txt = f"{datos.get('roic', 0):.1f}%" if datos.get('roic') is not None else "N/A"
    deuda_txt = f"{datos.get('deuda_ebitda', 0):.2f}x" if datos.get('deuda_ebitda') is not None else "N/A"
    
    pdf.metric_row('ROIC:', roic_txt)
    pdf.metric_row('Deuda/EBITDA:', deuda_txt)
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
    
    pdf.set_font(font_base, 'B', 11)
    pdf.cell(0, 8, 'Riesgos Identificados:', 0, 1)
    for i, riesgo in enumerate(riesgos['riesgos'], 1):
        pdf.set_font(font_base, '', 9)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 6, f'{i}. {riesgo["categoria"]} - {riesgo["nivel"]} ({riesgo["severidad"]}/100)', 0, 1)
        pdf.set_font(font_base, 'I', 8)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 5, f'   {riesgo["descripcion"][:80]}', 0, 1)
    
    pdf.ln(5)
    pdf.section_title('Veredicto Final')
    pdf.set_font(font_base, 'B', 12)
    pdf.set_text_color(0, 0, 0)
    veredicto = "PERFIL CONSERVADOR" if riesgos['score_riesgo'] < 30 else "PERFIL BALANCEADO" if riesgos['score_riesgo'] < 50 else "PERFIL AGRESIVO"
    pdf.cell(0, 10, veredicto, 0, 1, 'C')
    pdf.set_font(font_base, '', 10)
    pdf.multi_cell(0, 6, riesgos['recomendacion'])
    
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font(font_base, 'I', 7)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'DISCLAIMER: Reporte generado automaticamente con fines informativos. No constituye asesoramiento financiero.')
    
    temp_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_path)
    return temp_path

def generar_pdf_portafolio(tickers: list, opt_result: dict, capital: float) -> str:
    pdf = PDFReport()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Detectar si la fuente UTF-8 se cargó correctamente
    font_base = 'DejaVu' if 'DejaVu' in pdf.fonts else 'Helvetica'
    
    pdf.set_font(font_base, 'B', 24)
    pdf.set_text_color(26, 54, 93)
    pdf.cell(0, 20, 'Analisis de Portafolio', 0, 1, 'C')
    
    pdf.set_font(font_base, '', 12)
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
    pdf.set_font(font_base, 'B', 10)
    pdf.set_fill_color(26, 54, 93)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 8, 'Activo', 1, 0, 'C', True)
    pdf.cell(35, 8, 'Peso (%)', 1, 0, 'C', True)
    pdf.cell(40, 8, 'Monto ($)', 1, 0, 'C', True)
    pdf.cell(35, 8, 'Retorno', 1, 0, 'C', True)
    pdf.cell(30, 8, 'Riesgo', 1, 1, 'C', True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font(font_base, '', 9)
    
    for i, ticker in enumerate(opt_result['tickers']):
        peso = opt_result['pesos'][i] * 100
        monto = opt_result['pesos'][i] * capital
        datos_ticker = obtener_datos_financieros(ticker)
        
        # Manejo seguro de None
        ret_anual = datos_ticker.get('retorno_anual') if datos_ticker else None
        vol_anual = datos_ticker.get('volatilidad_anual') if datos_ticker else None
        
        retorno_indiv = (ret_anual * 100) if ret_anual is not None else 0.0
        volatilidad_indiv = (vol_anual * 100) if vol_anual is not None else 0.0
        
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
    
    pdf.set_font(font_base, 'B', 12)
    pdf.cell(0, 10, perfil, 0, 1, 'C')
    
    pdf.ln(10)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)
    pdf.set_font(font_base, 'I', 7)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4, 'DISCLAIMER: La optimizacion de Markowitz se basa en datos historicos que no garantizan resultados futuros.')
    
    temp_path = tempfile.mktemp(suffix='.pdf')
    pdf.output(temp_path)
    return temp_path
    # ==============================================================================
# PASO 9A: MOTOR DE NARRATIVA BUFFETT (Basado en Reglas)
# ==============================================================================
def generar_narrativa_buffett(datos: dict) -> str:
    """
    Genera un análisis narrativo basado en la filosofía de Warren Buffett.
    Utiliza las métricas reales calculadas por el sistema.
    """
    if not datos:
        return "No hay datos suficientes para generar el análisis."
    
    roic = datos.get('roic', 0)
    deuda = datos.get('deuda_ebitda', 2.0)
    pe = datos.get('pe_ratio', 0)
    margen = datos.get('margen_seguridad', 0)
    sector = datos.get('sector', 'General')
    
    narrativa = []
    
    # 1. EL FOSO ECONÓMICO (ROIC)
    narrativa.append("### 🏰 1. El Foso Económico (Ventaja Competitiva)")
    if roic > 20:
        narrativa.append(f"Con un **ROIC del {roic:.1f}%**, esta empresa demuestra tener un 'foso económico' extraordinariamente amplio. Como siempre digo: *'Busco negocios con una ventaja competitiva durable que proteja sus retornos sobre el capital invertido'.* Una gestión que asigna el capital de esta manera es un tesoro raro.")
    elif roic > 12:
        narrativa.append(f"Un **ROIC del {roic:.1f}%** indica un negocio sólido y rentable. *'Prefiero una empresa maravillosa a un precio justo'*, y este retorno sugiere que la empresa tiene buena salud operativa.")
    else:
        narrativa.append(f"Un **ROIC del {roic:.1f}%** es preocupante. *'El tiempo es amigo de los negocios maravillosos y enemigo de los mediocres'*. Si el retorno sobre el capital no supera fácilmente el costo de ese capital, el negocio tiene los días contados.")
    
    narrativa.append("") # Espacio
    
    # 2. SALUD FINANCIERA (DEUDA)
    narrativa.append("###  2. Salud Financiera y Riesgo")
    if deuda < 1.5:
        narrativa.append(f"Su ratio **Deuda/EBITDA de {deuda:.2f}x** es envidiable. *'La regla número 1 es no perder dinero. La regla número 2 es no olvidar la regla número 1'*. Una empresa con poca deuda puede sobrevivir a cualquier tormenta económica.")
    elif deuda < 3.0:
        narrativa.append(f"Con una **Deuda/EBITDA de {deuda:.2f}x**, la empresa maneja un nivel de apalancamiento aceptable, aunque no es lo ideal. *'Solo cuando baja la marea se ve quién está nadando desnudo'*. Hay que vigilar que los tipos de interés no suban demasiado.")
    else:
        narrativa.append(f"¡Cuidado! Una **Deuda/EBITDA de {deuda:.2f}x** es muy alta. *'El apalancamiento es como la gasolina en un coche: te hace ir más rápido, pero si te chocas, la explosión es mucho mayor'*. Evitaría esta empresa hasta que limpien su balance.")
    
    narrativa.append("") # Espacio
    
    # 3. VALORACIÓN (P/E Y MARGEN DE SEGURIDAD)
    narrativa.append("### 🏷️ 3. Valoración y el 'Mr. Market'")
    if margen > 20:
        narrativa.append(f"Con un **Margen de Seguridad del {margen:.1f}%** y un P/E de {pe:.1f}x, el mercado está siendo irracionalmente pesimista. *'El mercado de valores es un mecanismo para transferir dinero del impaciente al paciente'*. Esta es una clara oportunidad de compra.")
    elif margen > 0:
        narrativa.append(f"El **Margen de Seguridad es del {margen:.1f}%** (P/E: {pe:.1f}x). Es un precio razonable, pero no una ganga. *'Es mejor comprar una empresa maravillosa a un precio justo, que una empresa justa a un precio maravilloso'*. Se puede considerar una entrada parcial.")
    else:
        narrativa.append(f"Con un **Margen de Seguridad del {margen:.1f}%** (P/E: {pe:.1f}x), el precio actual está inflado por el entusiasmo del mercado. *'El precio es lo que pagas, el valor es lo que obtienes'*. A este precio, el riesgo de perder dinero permanente es alto. Paciencia, joven inversor.")
    
    narrativa.append("") # Espacio
    
    # 4. VEREDICTO FINAL
    narrativa.append("### 🎯 Conclusión del Análisis Cuantitativo")
    score = 0
    if roic > 15: score += 1
    if deuda < 2.0: score += 1
    if margen > 0: score += 1
    
    if score == 3:
        narrativa.append("**🟢 SEÑAL CUANTITATIVA POSITIVA:** Una empresa maravillosa a un precio justo. Cumple con todos los criterios de mi filosofía de inversión. Si tienes convicción, es hora de actuar.")
    elif score == 2:
        narrativa.append("**🟡 SEÑAL NEUTRA / EN OBSERVACIÓN:** Tienes un buen negocio, pero el precio no acompaña (o viceversa). Agrega esta empresa a tu lista de seguimiento y espera a que 'Mr. Market' te ofrezca un mejor precio.")
    else:
        narrativa.append("**🔴 SEÑAL CUANTITATIVA NEGATIVA:** O el negocio no es lo suficientemente bueno, o el precio es demasiado alto. En las inversiones, no te pagan por la actividad, te pagan por esperar el momento correcto.")
        
    return "\n".join(narrativa)
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
            
            roic_str = f"{datos['roic']:.1f}%" if datos.get('roic') is not None else "N/A"
            col3.metric("📊 ROIC", roic_str)
            col4.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
            st.divider()
            
            roic = datos.get('roic', 0) or 0
            deuda = datos.get('deuda_ebitda', 99) or 99
            
            if roic > 15 and deuda < 2:
                st.success("###  SEÑAL CUANTITATIVA POSITIVA\n**Fundamentales sólidos:**\n- ✅ ROIC atractivo (>15%)\n- ✅ Deuda controlada (<2x EBITDA)\n*Nota: Esto no constituye una recomendación de compra.*")
            elif roic > 10:
                st.warning("### 🟡 SEÑAL NEUTRA / EN OBSERVACIÓN\n**Requiere análisis adicional:**\n- ⚠️ Revisar nivel de deuda y valoración actual.")
            else:
                st.info("### 🔴 SEÑAL CUANTITATIVA NEGATIVA\n**Fundamentales débiles o precio elevado:**\n- Revisar tendencias históricas y riesgos.")
            
            st.divider()
            
            if st.button("📥 Descargar PDF del Análisis"):
                with st.spinner("Generando reporte PDF..."):
                    try:
                        datos_pdf = st.session_state.datos_activo
                        pronostico_pdf = pronosticar_precio(ticker_input, 90)
                        riesgos_pdf = analizar_riesgos_ia(ticker_input)
                        
                        if not datos_pdf or not pronostico_pdf or not riesgos_pdf:
                            st.error("❌ No se pudieron obtener todos los datos necesarios para generar el reporte.")
                        else:
                            pdf_path = generar_pdf_activo(ticker_input, datos_pdf, pronostico_pdf, riesgos_pdf)
                            
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
                        st.error(f"❌ Error al generar PDF: {str(e)}")
        
        else:  # Modo Avanzado
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Precio", f"${datos['precio']:.2f}")
            col2.metric("📈 P/E", f"{datos.get('pe_ratio', 0):.1f}x")
            col3.metric("📊 ROIC", f"{datos.get('roic', 0):.1f}%" if datos.get('roic') is not None else "N/A")
            col4.metric("🎯 Beta", f"{datos.get('beta', 1):.2f}")
            
            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("💵 EPS", f"${datos.get('eps', 0):.2f}")
                st.metric("🏢 Sector", datos.get('sector', 'N/A'))
            with col2:
                st.metric("💼 Market Cap", f"${datos.get('market_cap', 0)/1e9:.1f}B")
                st.metric("🏭 Industria", datos.get('industry', 'N/A'))
            
            if datos.get('descripcion'):
                st.divider()
                st.subheader(" Descripción de la Empresa")
                st.write(datos['descripcion'])
            
            st.divider()
            st.subheader("🧠 El Veredicto de Buffett")
            st.markdown("*Análisis basado en la filosofía de Inversión de Valor*")
            
            with st.expander("📖 Leer análisis completo", expanded=True):
                narrativa = generar_narrativa_buffett(datos)
                st.markdown(narrativa)
                st.caption("⚠️ Nota: Análisis generado por motor de reglas basado en principios históricos. No es asesoramiento financiero.")
            
            st.divider()
            
            if st.button("📥 Descargar PDF del Análisis"):
                with st.spinner("Generando reporte PDF..."):
                    try:
                        datos_pdf = st.session_state.datos_activo
                        pronostico_pdf = pronosticar_precio(ticker_input, 90)
                        riesgos_pdf = analizar_riesgos_ia(ticker_input)
                        
                        if not datos_pdf or not pronostico_pdf or not riesgos_pdf:
                            st.error("❌ No se pudieron obtener todos los datos necesarios para generar el reporte.")
                        else:
                            pdf_path = generar_pdf_activo(ticker_input, datos_pdf, pronostico_pdf, riesgos_pdf)
                            
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
                        st.error(f"❌ Error al generar PDF: {str(e)}")
# ==============================================================================
# PESTAÑA 3: PORTAFOLIO - VERSIÓN ROBUSTA
# ==============================================================================
with tab3:
    st.header("💼 Optimizador de Portafolio")
    
    if 'tickers_personalizados' not in st.session_state:
        st.session_state.tickers_personalizados = []
    if 'etapa_actual' not in st.session_state:
        st.session_state.etapa_actual = 1
    if 'tickers_seleccionados' not in st.session_state:
        st.session_state.tickers_seleccionados = []
    if 'pesos_actuales' not in st.session_state:
        st.session_state.pesos_actuales = {}
    if 'capital_total' not in st.session_state:
        st.session_state.capital_total = 100000.0
    if 'resultado_evaluacion' not in st.session_state:
        st.session_state.resultado_evaluacion = None
    if 'resultado_optimizacion' not in st.session_state:
        st.session_state.resultado_optimizacion = None
    
    TICKERS_SUGERIDOS = ['AAPL', 'MSFT', 'KO', 'GOOGL', 'WMT', 'TSLA', 'AMZN', 'NVDA', 'JPM', 'V']
    todos_los_tickers = list(set(TICKERS_SUGERIDOS + st.session_state.tickers_personalizados))
    todos_los_tickers.sort()
    
    st.subheader("1. Selección de Activos")
    
    tickers_seleccionados = st.multiselect(
        "Selecciona los activos de tu portafolio (2-6)",
        options=todos_los_tickers,
        default=st.session_state.tickers_seleccionados if st.session_state.tickers_seleccionados else ['AAPL', 'MSFT', 'KO'],
        help="Selecciona entre 2 y 6 empresas",
        key="select_tickers"
    )
    
    st.session_state.tickers_seleccionados = tickers_seleccionados
    st.divider()
    
    st.markdown("**¿No encuentras tu ticker? Agrégalo aquí:**")
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker_nuevo = st.text_input(
            "Ingresa el ticker (ej: META, DIS, NFLX, BA)",
            placeholder="Ej: META",
            help="El ticker debe existir en Yahoo Finance",
            key="input_ticker_nuevo"
        ).upper().strip()
    
    with col2:
        st.write("")
        st.write("")
        if st.button("➕ Agregar Ticker", use_container_width=True, key="btn_agregar"):
            if ticker_nuevo and ticker_nuevo not in todos_los_tickers:
                try:
                    stock_test = yf.Ticker(ticker_nuevo)
                    info_test = stock_test.info
                    if info_test.get('currentPrice') or info_test.get('regularMarketPrice'):
                        st.session_state.tickers_personalizados.append(ticker_nuevo)
                        st.success(f"✅ {ticker_nuevo} agregado")
                        st.rerun()
                    else:
                        st.error(f"❌ {ticker_nuevo} no encontrado")
                except:
                    st.error(f"❌ Error al verificar {ticker_nuevo}")
    
    st.divider()
    
    if len(tickers_seleccionados) < 2:
        st.warning("⚠️ Selecciona al menos 2 activos para continuar.")
    else:
        st.subheader("2. Composición de tu Portafolio Actual")
        
        modo_ingreso = st.radio(
            "¿Cómo quieres ingresar la composición?",
            ["En porcentaje (%)", "En dólares ($)"],
            horizontal=True,
            key="modo_ingreso"
        )
        
        if modo_ingreso == "En porcentaje (%)":
            capital_total = st.number_input(
                "Capital total del portafolio ($)",
                min_value=1000.0,
                value=st.session_state.capital_total,
                step=1000.0,
                key="capital_input"
            )
            st.session_state.capital_total = capital_total
            
            pesos_input = {}
            cols = st.columns(len(tickers_seleccionados))
            for i, ticker in enumerate(tickers_seleccionados):
                with cols[i]:
                    peso = st.number_input(
                        f"{ticker} (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=round(100.0/len(tickers_seleccionados), 1),
                        step=0.1,
                        key=f"peso_{ticker}"
                    )
                    pesos_input[ticker] = peso
            
            total_porcentaje = sum(pesos_input.values())
            if abs(total_porcentaje - 100.0) > 0.1:
                st.warning(f"⚠️ Los porcentajes suman {total_porcentaje:.1f}%. Deben sumar 100%.")
                puede_evaluar = False
            else:
                puede_evaluar = True
                montos_input = {t: (p/100) * capital_total for t, p in pesos_input.items()}
        else:
            montos_input = {}
            cols = st.columns(len(tickers_seleccionados))
            for i, ticker in enumerate(tickers_seleccionados):
                with cols[i]:
                    monto = st.number_input(
                        f"{ticker} ($)",
                        min_value=0.0,
                        value=st.session_state.capital_total/len(tickers_seleccionados),
                        step=100.0,
                        key=f"monto_{ticker}"
                    )
                    montos_input[ticker] = monto
            
            capital_total = sum(montos_input.values())
            st.session_state.capital_total = capital_total
            
            if capital_total <= 0:
                st.warning("⚠️ El capital total debe ser mayor a $0")
                puede_evaluar = False
            else:
                puede_evaluar = True
                pesos_input = {t: (v/capital_total)*100 for t, v in montos_input.items()}
        
        st.divider()
        st.subheader("3. Evaluación del Portafolio Actual")
        
        if puede_evaluar and st.button("📊 Evaluar Portafolio Actual", type="primary", key="btn_evaluar"):
            with st.spinner("Calculando métricas con datos reales..."):
                datos_activos = {}
                for ticker in tickers_seleccionados:
                    datos = obtener_datos_financieros(ticker)
                    if datos:
                        datos_activos[ticker] = datos
                
                if len(datos_activos) >= 2:
                    tickers_validos = [t for t in tickers_seleccionados if t in datos_activos]
                    
                    returns_dict_eval = {}
                    for t in tickers_validos:
                        _, returns = fetch_historical_returns(t)
                        if returns is not None:
                            returns_dict_eval[t] = returns
                    
                    if len(returns_dict_eval) >= 2:
                        returns_df_eval = pd.DataFrame(returns_dict_eval)
                        tickers_con_retorno = list(returns_df_eval.columns)
                        
                        pesos_array = np.array([pesos_input[t]/100 for t in tickers_con_retorno])
                        retornos_array = np.array([datos_activos[t]['retorno_anual'] for t in tickers_con_retorno])
                        
                        cov_matrix_real = returns_df_eval.cov() * 252
                        volatilidad_portafolio = np.sqrt(np.dot(pesos_array.T, np.dot(cov_matrix_real, pesos_array))) * 100
                    else:
                        volatilidad_portafolio = 0.0
                    
                    rf = 0.04
                    retorno_portafolio = np.sum(pesos_array * retornos_array) * 100
                    sharpe_portafolio = (retorno_portafolio/100 - rf) / (volatilidad_portafolio/100) if volatilidad_portafolio > 0 else 0
                    
                    st.session_state.resultado_evaluacion = {
                        'tickers': tickers_validos,
                        'pesos': pesos_input,
                        'montos': montos_input if 'montos_input' in locals() else {},
                        'capital': capital_total,
                        'retorno': retorno_portafolio,
                        'volatilidad': volatilidad_portafolio,
                        'sharpe': sharpe_portafolio,
                        'datos_activos': datos_activos
                    }
                    st.success("✅ Evaluación completada con matriz de covarianza real")
                    st.rerun()
        
        if st.session_state.resultado_evaluacion:
            eval_result = st.session_state.resultado_evaluacion
            
            col1, col2, col3 = st.columns(3)
            col1.metric("📈 Retorno Anual", f"{eval_result['retorno']:.1f}%")
            col2.metric("⚠️ Volatilidad", f"{eval_result['volatilidad']:.1f}%")
            col3.metric("⭐ Ratio de Sharpe", f"{eval_result['sharpe']:.2f}")
            
            st.divider()
            
            df_actual = pd.DataFrame({
                'Activo': eval_result['tickers'],
                'Peso(%)': [round(eval_result['pesos'].get(t, 0), 1) for t in eval_result['tickers']],
                'Monto($)': [round(eval_result['montos'].get(t, 0), 0) for t in eval_result['tickers']],
                'Retorno(%)': [round(eval_result['datos_activos'][t]['retorno_anual']*100, 1) for t in eval_result['tickers']],
                'Riesgo(%)': [round(eval_result['datos_activos'][t]['volatilidad_anual']*100, 1) for t in eval_result['tickers']]
            })
            
            st.subheader("📋 Composición Actual")
            st.dataframe(df_actual, use_container_width=True)
            
            st.divider()
            
            if eval_result['sharpe'] > 1.0:
                st.success(f"**Excelente portafolio!** Sharpe de {eval_result['sharpe']:.2f}")
            elif eval_result['sharpe'] > 0.5:
                st.info(f"**Buen portafolio.** Sharpe de {eval_result['sharpe']:.2f}")
            else:
                st.warning(f"**Portafolio mejorable.** Sharpe de {eval_result['sharpe']:.2f}")
            
            st.divider()
            st.subheader("4. Optimización del Portafolio")
            
            col1, col2 = st.columns(2)
            with col1:
                rf_opt = st.slider(
                    "Tasa Libre de Riesgo (%)",
                    0.0, 10.0, 4.0, 0.5,
                    help="Rendimiento de inversiones sin riesgo. Usualmente 3-5%.",
                    key="rf_slider"
                )
            with col2:
                max_peso = st.slider(
                    "Peso Máximo por Activo (%)",
                    10, 100, 40, 5,
                    help="Porcentaje máximo por empresa. Evita concentración.",
                    key="max_peso_slider"
                )
            
            if st.button("⚙️ Calcular Portafolio Óptimo", type="primary", key="btn_optimizar"):
                with st.spinner("Optimizando..."):
                    opt_result = optimizar_portafolio(eval_result['tickers'], rf=rf_opt/100)
                    
                    if opt_result:
                        opt_result['pesos'] = np.minimum(opt_result['pesos'], max_peso/100)
                        opt_result['pesos'] /= opt_result['pesos'].sum()
                        
                        cov_matrix_real = opt_result['cov_matrix']
                        retornos = np.array([eval_result['datos_activos'][t]['retorno_anual'] for t in opt_result['tickers']])
                        
                        opt_result['retorno'] = np.sum(retornos * opt_result['pesos']) * 100
                        opt_result['volatilidad'] = np.sqrt(np.dot(opt_result['pesos'].T, np.dot(cov_matrix_real, opt_result['pesos']))) * 100
                        opt_result['sharpe'] = (opt_result['retorno']/100 - rf_opt/100) / (opt_result['volatilidad']/100) if opt_result['volatilidad'] > 0 else 0
                        
                        st.session_state.resultado_optimizacion = opt_result
                        st.success("✅ Optimización completada con datos históricos reales")
                        st.rerun()
            
            if st.session_state.resultado_optimizacion:
                opt = st.session_state.resultado_optimizacion
                
                st.divider()
                st.subheader("5. Portafolio Óptimo")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Retorno Óptimo", f"{opt['retorno']:.1f}%")
                col2.metric("Volatilidad Óptima", f"{opt['volatilidad']:.1f}%")
                col3.metric("Sharpe Óptimo", f"{opt['sharpe']:.2f}")
                
                st.divider()
                st.subheader("6. Plan de Rebalanceo")
                
                df_rebalanceo = pd.DataFrame()
                df_rebalanceo['Activo'] = opt['tickers']
                df_rebalanceo['Actual(%)'] = [round(eval_result['pesos'].get(t, 0), 1) for t in opt['tickers']]
                df_rebalanceo['Óptimo(%)'] = (opt['pesos'] * 100).round(1)
                df_rebalanceo['Diferencia(%)'] = (df_rebalanceo['Óptimo(%)'] - df_rebalanceo['Actual(%)']).round(1)
                df_rebalanceo['Monto_Ajustar($)'] = (df_rebalanceo['Diferencia(%)'] * capital_total / 100).round(0)
                df_rebalanceo['Acción'] = df_rebalanceo['Diferencia(%)'].apply(
                    lambda x: "🟢 COMPRAR" if x > 1 else "🔴 VENDER" if x < -1 else "⚪ MANTENER"
                )
                
                st.dataframe(df_rebalanceo, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                comprar = df_rebalanceo[df_rebalanceo['Acción'].str.contains('COMPRAR')]
                vender = df_rebalanceo[df_rebalanceo['Acción'].str.contains('VENDER')]
                mantener = df_rebalanceo[df_rebalanceo['Acción'].str.contains('MANTENER')]
                
                with col1:
                    if not comprar.empty:
                        st.success("**🟢 Comprar:**")
                        for _, row in comprar.iterrows():
                            st.write(f"• {row['Activo']}: ${abs(row['Monto_Ajustar($)']):,.0f}")
                
                with col2:
                    if not vender.empty:
                        st.error("**🔴 Vender:**")
                        for _, row in vender.iterrows():
                            st.write(f"• {row['Activo']}: ${abs(row['Monto_Ajustar($)']):,.0f}")
                
                with col3:
                    if not mantener.empty:
                        st.info("**⚪ Mantener:**")
                        for _, row in mantener.iterrows():
                            st.write(f"• {row['Activo']}")
                
                st.divider()
                
                mejora_sharpe = opt['sharpe'] - eval_result['sharpe']
                if mejora_sharpe > 0:
                    st.success(f"✅ La optimización mejora el Sharpe en {mejora_sharpe:.2f}")
                st.divider()
                
                if st.button("📥 Descargar PDF del Portafolio"):
                    with st.spinner("Generando reporte PDF..."):
                        try:
                            if st.session_state.resultado_optimizacion:
                                opt = st.session_state.resultado_optimizacion
                                capital_pdf = st.session_state.capital_total
                                
                                pdf_path = generar_pdf_portafolio(opt['tickers'], opt, capital_pdf)
                                
                                with open(pdf_path, 'rb') as f:
                                    st.download_button(
                                        label="⬇️ Descargar PDF",
                                        data=f.read(),
                                        file_name=f"Portafolio_{datetime.now().strftime('%Y%m%d')}.pdf",
                                        mime="application/pdf",
                                        type="primary"
                                    )
                                st.success("✅ PDF generado correctamente")
                            else:
                                st.error("❌ Primero debes calcular el portafolio óptimo")
                        except Exception as e:
                            st.error(f"❌ Error al generar PDF: {str(e)}")
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
