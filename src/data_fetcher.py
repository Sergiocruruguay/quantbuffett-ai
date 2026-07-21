"""
Módulo de Extracción de Datos Financieros - Versión con Caché
Usa sistema de caché para evitar rate limiting de Yahoo Finance
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging
from src.cache_manager import obtener_datos_con_cache, obtener_datos_mock

logger = logging.getLogger(__name__)


class FinancialDataFetcher:
    """Clase para extraer y procesar datos financieros de una empresa."""
    
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.raw_data = None
        
    def fetch_data(self) -> Optional[Dict]:
        """Extrae todos los datos financieros necesarios."""
        try:
            logger.info(f"Extrayendo datos para {self.ticker}...")
            
            # Intentar obtener datos reales con caché
            self.raw_data = obtener_datos_con_cache(self.ticker)
            
            if self.raw_data and self.raw_data.get('info'):
                # Procesar datos reales
                data = self._procesar_datos_reales()
                data['es_mock'] = False
                logger.info(f"Datos reales extraídos para {self.ticker}")
                return data
            else:
                # Fallback a datos mock
                logger.warning(f"Usando datos de ejemplo para {self.ticker} (API bloqueada)")
                mock_data = obtener_datos_mock(self.ticker)
                if mock_data:
                    return mock_data
                return None
                
        except Exception as e:
            logger.error(f"Error al extraer datos para {self.ticker}: {str(e)}")
            # Último recurso: datos mock
            return obtener_datos_mock(self.ticker)
    
    def _procesar_datos_reales(self) -> Dict:
        """Procesa los datos crudos de Yahoo Finance."""
        info = self.raw_data['info']
        financials = self.raw_data['financials']
        balance = self.raw_data['balance']
        cashflow = self.raw_data['cashflow']
        
        # Precio y market cap
        precio = float(info.get('currentPrice') or info.get('regularMarketPrice') or 0)
        market_cap = info.get('marketCap', 0) or 0
        
        # Calcular métricas
        roic = self._calcular_roic(financials, balance)
        deuda_ebitda = self._calcular_deuda_ebitda(financials, balance, cashflow)
        fcf = self._calcular_fcf(cashflow)
        net_income = self._get_financial_item(financials, ['Net Income'], 0)
        ebit = self._get_financial_item(financials, ['EBIT', 'Operating Income'], 0)
        margen_seguridad = self._calcular_margen_seguridad(precio, fcf, balance, info)
        
        return {
            'ticker': self.ticker,
            'precio': precio,
            'market_cap': market_cap,
            'roic': roic,
            'deuda_ebitda': deuda_ebitda,
            'fcf': fcf,
            'net_income': net_income,
            'ebit': ebit,
            'margen_seguridad': margen_seguridad,
            'beta': float(info.get('beta', 1.0)) if info.get('beta') else 1.0,
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A')
        }
    
    def _get_financial_item(self, df, keywords: list, default: float = 0.0) -> float:
        """Busca un item en estados financieros."""
        try:
            if df is None or df.empty:
                return default
            for kw in keywords:
                matches = [idx for idx in df.index if kw.lower() in idx.lower()]
                if matches:
                    val = df.loc[matches[0]].iloc[0]
                    return float(val) if not pd.isna(val) else default
        except:
            pass
        return default
    
    def _get_balance_item(self, df, keywords: list, default: float = 0.0) -> float:
        """Busca un item en el balance."""
        return self._get_financial_item(df, keywords, default)
    
    def _get_cashflow_item(self, df, keywords: list, default: float = 0.0) -> float:
        """Busca un item en el cash flow."""
        return self._get_financial_item(df, keywords, default)
    
    def _calcular_roic(self, financials, balance) -> float:
        """Calcula ROIC."""
        try:
            ebit = self._get_financial_item(financials, ['EBIT', 'Operating Income'], 0)
            if ebit == 0:
                return 0.0
            
            nopat = ebit * (1 - 0.21)
            total_assets = self._get_balance_item(balance, ['Total Assets'], 0)
            current_liabilities = self._get_balance_item(balance, ['Total Current Liabilities', 'Current Liabilities'], 0)
            cash = self._get_balance_item(balance, ['Cash And Cash Equivalents', 'Cash'], 0)
            
            capital_invertido = total_assets - current_liabilities - cash
            if capital_invertido <= 0:
                return 0.0
            
            return round((nopat / capital_invertido) * 100, 2)
        except:
            return 0.0
    
    def _calcular_deuda_ebitda(self, financials, balance, cashflow) -> float:
        """Calcula Deuda/EBITDA."""
        try:
            st_debt = self._get_balance_item(balance, ['Short Term Debt', 'Current Debt'], 0)
            lt_debt = self._get_balance_item(balance, ['Long Term Debt'], 0)
            deuda_total = abs(st_debt) + abs(lt_debt)
            
            cash = self._get_balance_item(balance, ['Cash And Cash Equivalents'], 0)
            deuda_neta = deuda_total - cash
            
            ebit = self._get_financial_item(financials, ['EBIT', 'Operating Income'], 0)
            depreciation = self._get_cashflow_item(cashflow, ['Depreciation'], 0)
            ebitda = ebit + abs(depreciation)
            
            if ebitda <= 0:
                return 99.0
            
            return round(deuda_neta / ebitda, 2)
        except:
            return 0.0
    
    def _calcular_fcf(self, cashflow) -> float:
        """Calcula Free Cash Flow."""
        try:
            ocf = self._get_cashflow_item(cashflow, ['Operating Cash Flow', 'Cash Flow From Operating Activities'], 0)
            capex = self._get_cashflow_item(cashflow, ['Capital Expenditure', 'Purchase Of Property'], 0)
            fcf = ocf + capex
            return round(fcf / 1e9, 2)
        except:
            return 0.0
    
    def _calcular_margen_seguridad(self, precio_actual, fcf_billones, balance, info) -> float:
        """Calcula margen de seguridad con DCF simplificado."""
        try:
            if precio_actual <= 0 or fcf_billones <= 0:
                return 0.0
            
            fcf = fcf_billones * 1e9
            wacc = 0.09
            g = 0.03
            
            flujos = [fcf * ((1 + 0.05) ** ano) / ((1 + wacc) ** ano) for ano in range(1, 6)]
            vt = (flujos[-1] * (1 + wacc)) / (wacc - g)
            vp_vt = vt / ((1 + wacc) ** 5)
            
            enterprise_value = sum(flujos) + vp_vt
            
            deuda_neta = (abs(self._get_balance_item(balance, ['Short Term Debt', 'Current Debt'], 0)) + 
                         abs(self._get_balance_item(balance, ['Long Term Debt'], 0)) - 
                         self._get_balance_item(balance, ['Cash And Cash Equivalents'], 0))
            
            equity_value = enterprise_value - deuda_neta
            shares = info.get('sharesOutstanding', 1e9) or 1e9
            precio_justo = equity_value / shares
            
            return round(((precio_justo / precio_actual) - 1) * 100, 2)
        except:
            return 0.0


def obtener_datos_financieros(ticker: str) -> Optional[Dict]:
    """Función rápida para obtener datos financieros."""
    fetcher = FinancialDataFetcher(ticker)
    return fetcher.fetch_data()

