"""
Módulo de Extracción de Datos Financieros - Versión Demo
Usa datos de ejemplo para evitar rate limiting de Yahoo Finance
"""

import streamlit as st
from typing import Dict, Optional

# Base de datos de ejemplo con valores realistas
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
        'industry': 'Consumer Electronics',
        'es_mock': True
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
        'industry': 'Software',
        'es_mock': True
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
        'industry': 'Beverages',
        'es_mock': True
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
        'industry': 'Internet Content',
        'es_mock': True
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
        'industry': 'Discount Stores',
        'es_mock': True
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
        'industry': 'Auto Manufacturers',
        'es_mock': True
    }
}

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_financieros(ticker: str) -> Optional[Dict]:
    """
    Obtiene datos financieros de la base de datos de ejemplo.
    
    Args:
        ticker: Símbolo bursátil (ej: 'AAPL', 'MSFT')
    
    Returns:
        Diccionario con métricas financieras o None si no existe
    """
    try:
        ticker_upper = ticker.upper()
        
        # Buscar en la base de datos mock
        if ticker_upper in MOCK_DATABASE:
            return MOCK_DATABASE[ticker_upper].copy()
        else:
            # Si no existe, retornar None
            return None
            
    except Exception as e:
        print(f"Error en obtener_datos_financieros: {str(e)}")
        return None



