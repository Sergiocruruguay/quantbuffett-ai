"""
Módulo de Extracción de Datos Financieros - Versión Simplificada
"""

import yfinance as yf
import pandas as pd
from typing import Dict, Optional
import streamlit as st
import time

@st.cache_data(ttl=3600, show_spinner=False)
def obtener_datos_financieros(ticker: str) -> Optional[Dict]:
    """
    Obtiene datos financieros de Yahoo Finance con caché.
    Retorna datos de ejemplo si la API está bloqueada.
    """
    try:
        ticker = ticker.upper()
        
        # Datos de respaldo (se usan si Yahoo Finance falla)
        mock_data = {
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
            }
        }
        
        # Intentar obtener datos reales
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validar que tenemos precio
            precio = info.get('currentPrice') or info.get('regularMarketPrice')
            if not precio or precio <= 0:
                raise ValueError("No hay precio válido")
            
            # Esperar un poco para evitar rate limiting
            time.sleep(2)
            
            # Obtener estados financieros
            financials = stock.financials
            balance = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Extraer métricas básicas
            market_cap = info.get('marketCap', 0) or 0
            beta = float(info.get('beta', 1.0)) if info.get('beta') else 1.0
            sector = info.get('sector', 'N/A')
            industry = info.get('industry', 'N/A')
            
            # Calcular métricas simples
            roic = 0.0
            deuda_ebitda = 0.0
            fcf = 0.0
            net_income = 0.0
            ebit = 0.0
            
            # Intentar calcular ROIC si hay datos
            try:
                if not financials.empty and not balance.empty:
                    # Buscar EBIT
                    ebit_row = [idx for idx in financials.index if 'EBIT' in idx or 'Operating Income' in idx]
                    if ebit_row:
                        ebit = float(financials.loc[ebit_row[0]].iloc[0])
                    
                    # Buscar Net Income
                    ni_row = [idx for idx in financials.index if 'Net Income' in idx and 'Common' not in idx]
                    if ni_row:
                        net_income = float(financials.loc[ni_row[0]].iloc[0])
                    
                    # Calcular ROIC básico
                    if ebit != 0:
                        total_assets_row = [idx for idx in balance.index if 'Total Assets' in idx]
                        if total_assets_row:
                            total_assets = float(balance.loc[total_assets_row[0]].iloc[0])
                            if total_assets > 0:
                                roic = ((ebit * 0.79) / total_assets) * 100
            except:
                pass
            
            # Intentar calcular Deuda/EBITDA
            try:
                if not balance.empty and not cashflow.empty:
                    # Buscar deudas
                    st_debt_row = [idx for idx in balance.index if 'Short Term Debt' in idx or 'Current Debt' in idx]
                    lt_debt_row = [idx for idx in balance.index if 'Long Term Debt' in idx]
                    
                    st_debt = float(balance.loc[st_debt_row[0]].iloc[0]) if st_debt_row else 0
                    lt_debt = float(balance.loc[lt_debt_row[0]].iloc[0]) if lt_debt_row else 0
                    
                    deuda_total = abs(st_debt) + abs(lt_debt)
                    
                    # Buscar cash
                    cash_row = [idx for idx in balance.index if 'Cash' in idx]
                    cash = float(balance.loc[cash_row[0]].iloc[0]) if cash_row else 0
                    
                    deuda_neta = deuda_total - cash
                    
                    # Calcular EBITDA aproximado
                    ebitda = ebit * 1.1 if ebit != 0 else 1
                    
                    if ebitda > 0:
                        deuda_ebitda = deuda_neta / ebitda
            except:
                pass
            
            # Intentar calcular FCF
            try:
                if not cashflow.empty:
                    ocf_row = [idx for idx in cashflow.index if 'Operating Cash Flow' in idx]
                    capex_row = [idx for idx in cashflow.index if 'Capital Expenditure' in idx]
                    
                    ocf = float(cashflow.loc[ocf_row[0]].iloc[0]) if ocf_row else 0
                    capex = float(cashflow.loc[capex_row[0]].iloc[0]) if capex_row else 0
                    
                    fcf = (ocf + capex) / 1e9  # En billones
            except:
                pass
            
            # Calcular margen de seguridad simple
            margen_seguridad = 0.0
            try:
                if fcf > 0 and precio > 0:
                    shares = info.get('sharesOutstanding', 1e9) or 1e9
                    fcf_per_share = (fcf * 1e9) / shares
                    valor_justo = fcf_per_share * 15  # Múltiplo simple de 15x FCF
                    margen_seguridad = ((valor_justo / precio) - 1) * 100
            except:
                pass
            
            return {
                'ticker': ticker,
                'precio': float(precio),
                'market_cap': market_cap,
                'roic': round(roic, 2),
                'deuda_ebitda': round(deuda_ebitda, 2),
                'fcf': round(fcf, 2),
                'net_income': net_income,
                'ebit': ebit,
                'margen_seguridad': round(margen_seguridad, 2),
                'beta': beta,
                'sector': sector,
                'industry': industry,
                'es_mock': False
            }
            
        except Exception as e:
            # Si falla Yahoo Finance, usar datos mock
            print(f"Usando datos mock para {ticker}: {str(e)}")
            if ticker in mock_data:
                return mock_data[ticker]
            return None
            
    except Exception as e:
        print(f"Error general en obtener_datos_financieros: {str(e)}")
        return None


