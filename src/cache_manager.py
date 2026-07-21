"""
Gestor de Caché y Manejo de Rate Limiting
Evita bloqueos de Yahoo Finance en Streamlit Cloud
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
import time
import random

logger = logging.getLogger(__name__)

# Configuración de headers para simular navegador real
SESSION_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_con_cache(ticker: str) -> Optional[Dict]:
    """
    Obtiene datos financieros con caché de 1 hora.
    Esto reduce drásticamente las solicitudes a Yahoo Finance.
    """
    try:
        # Delay aleatorio para evitar detección de bots
        time.sleep(random.uniform(1.0, 2.5))
        
        stock = yf.Ticker(ticker)
        
        # Forzar descarga con headers personalizados
        stock.session.headers.update(SESSION_HEADERS)
        
        info = stock.info
        
        # Validar que tenemos datos mínimos
        if not info or not info.get('regularMarketPrice') and not info.get('currentPrice'):
            logger.warning(f"No hay precio para {ticker}")
            return None
        
        # Descargar estados financieros con delays
        time.sleep(1)
        financials = stock.financials
        
        time.sleep(1)
        balance = stock.balance_sheet
        
        time.sleep(1)
        cashflow = stock.cashflow
        
        return {
            'info': info,
            'financials': financials,
            'balance': balance,
            'cashflow': cashflow
        }
        
    except Exception as e:
        logger.error(f"Error al obtener datos con caché para {ticker}: {e}")
        return None


# Datos de respaldo para cuando Yahoo Finance esté bloqueado
MOCK_DATA = {
    'AAPL': {
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
    }
}


def obtener_datos_mock(ticker: str) -> Optional[Dict]:
    """
    Retorna datos de ejemplo cuando Yahoo Finance está bloqueado.
    Útil para demostraciones y desarrollo.
    """
    ticker_upper = ticker.upper()
    if ticker_upper in MOCK_DATA:
        data = MOCK_DATA[ticker_upper].copy()
        data['ticker'] = ticker_upper
        data['es_mock'] = True  # Flag para indicar que son datos de ejemplo
        return data
    return None
