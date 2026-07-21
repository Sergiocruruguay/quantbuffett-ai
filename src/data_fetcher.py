"""
Módulo de Extracción de Datos Financieros
Extrae y calcula métricas fundamentales de Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class FinancialDataFetcher:
    """Clase para extraer y procesar datos financieros de una empresa."""
    
    def __init__(self, ticker: str):
        """
        Inicializa el extractor de datos.
        
        Args:
            ticker: Símbolo bursátil (ej: 'AAPL', 'MSFT')
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.info = None
        self.financials = None
        self.balance = None
        self.cashflow = None
        
    def fetch_data(self) -> Optional[Dict]:
        """
        Extrae todos los datos financieros necesarios.
        
        Returns:
            Diccionario con métricas calculadas o None si falla
        """
        try:
            logger.info(f"Extrayendo datos para {self.ticker}...")
            
            # Descargar estados financieros
            self.info = self.stock.info
            self.financials = self.stock.financials
            self.balance = self.stock.balance_sheet
            self.cashflow = self.stock.cashflow
            
            # Validar que tenemos datos
            if not self.info or not self.financials or not self.balance:
                logger.error(f"No se encontraron datos para {self.ticker}")
                return None
            
            # Extraer métricas individuales
            data = {
                'ticker': self.ticker,
                'precio': self._get_precio(),
                'market_cap': self._get_market_cap(),
                'roic': self._calcular_roic(),
                'deuda_ebitda': self._calcular_deuda_ebitda(),
                'fcf': self._calcular_fcf(),
                'net_income': self._get_net_income(),
                'ebit': self._get_ebit(),
                'margen_seguridad': self._calcular_margen_seguridad(),
                'beta': self.info.get('beta', 1.0),
                'sector': self.info.get('sector', 'N/A'),
                'industry': self.info.get('industry', 'N/A')
            }
            
            logger.info(f"Datos extraídos exitosamente para {self.ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error al extraer datos para {self.ticker}: {str(e)}")
            return None
    
    def _get_precio(self) -> float:
        """Obtiene el precio actual de mercado."""
        return (self.info.get('currentPrice') or 
                self.info.get('regularMarketPrice') or 
                0.0)
    
    def _get_market_cap(self) -> float:
        """Obtiene la capitalización de mercado."""
        return self.info.get('marketCap', 0)
    
    def _get_net_income(self) -> float:
        """Obtiene la utilidad neta más reciente."""
        try:
            label = next((idx for idx in self.financials.index 
                         if 'Net Income' in idx and 'Common' not in idx), None)
            if label:
                return self.financials.loc[label].iloc[0]
        except:
            pass
        return 0.0
    
    def _get_ebit(self) -> float:
        """Obtiene el EBIT (Operating Income) más reciente."""
        try:
            label = next((idx for idx in self.financials.index 
                         if 'EBIT' in idx or 'Operating Income' in idx), None)
            if label:
                return self.financials.loc[label].iloc[0]
        except:
            pass
        return 0.0
    
    def _calcular_roic(self) -> float:
        """
        Calcula el ROIC (Return on Invested Capital).
        
        Fórmula: NOPAT / Capital Invertido
        Donde:
        - NOPAT = EBIT * (1 - tasa_impositiva)
        - Capital Invertido = Activos Totales - Pasivos Corrientes - Efectivo
        """
        try:
            # Obtener EBIT
            ebit = self._get_ebit()
            if ebit == 0:
                return 0.0
            
            # Tasa impositiva estimada (21% para empresas de EE.UU.)
            tasa_impuestos = 0.21
            nopat = ebit * (1 - tasa_impuestos)
            
            # Obtener componentes del balance
            total_assets = self._get_balance_item(['Total Assets'], 0)
            current_liabilities = self._get_balance_item(['Total Current Liabilities', 'Current Liabilities'], 0)
            cash = self._get_balance_item(['Cash And Cash Equivalents', 'Cash'], 0)
            
            # Capital invertido
            capital_invertido = total_assets - current_liabilities - cash
            
            if capital_invertido <= 0:
                return 0.0
            
            roic = (nopat / capital_invertido) * 100
            return round(roic, 2)
            
        except Exception as e:
            logger.warning(f"No se pudo calcular ROIC para {self.ticker}: {e}")
            return 0.0
    
    def _calcular_deuda_ebitda(self) -> float:
        """
        Calcula el ratio Deuda Neta / EBITDA.
        
        Fórmula: (Deuda Total - Efectivo) / EBITDA
        """
        try:
            # Deuda total
            st_debt = self._get_balance_item(['Short Term Debt', 'Current Debt'], 0)
            lt_debt = self._get_balance_item(['Long Term Debt'], 0)
            deuda_total = abs(st_debt) + abs(lt_debt)
            
            # Efectivo
            cash = self._get_balance_item(['Cash And Cash Equivalents', 'Cash'], 0)
            
            # Deuda neta
            deuda_neta = deuda_total - cash
            
            # EBITDA = EBIT + Depreciación
            ebit = self._get_ebit()
            depreciation = self._get_cashflow_item(['Depreciation', 'Depreciation And Amortization'], 0)
            ebitda = ebit + abs(depreciation)
            
            if ebitda <= 0:
                return 99.0  # Valor alto si no hay EBITDA positivo
            
            ratio = deuda_neta / ebitda
            return round(ratio, 2)
            
        except Exception as e:
            logger.warning(f"No se pudo calcular Deuda/EBITDA para {self.ticker}: {e}")
            return 0.0
    
    def _calcular_fcf(self) -> float:
        """
        Calcula el Free Cash Flow (FCF).
        
        Fórmula: Operating Cash Flow + Capital Expenditure
        (Nota: CapEx viene como negativo en yfinance)
        """
        try:
            ocf = self._get_cashflow_item(['Operating Cash Flow', 'Cash Flow From Operating Activities'], 0)
            capex = self._get_cashflow_item(['Capital Expenditure', 'Purchase Of Property'], 0)
            
            # FCF = OCF + CapEx (porque CapEx es negativo)
            fcf = ocf + capex
            return round(fcf / 1e9, 2)  # Retornar en billones
            
        except Exception as e:
            logger.warning(f"No se pudo calcular FCF para {self.ticker}: {e}")
            return 0.0
    
    def _calcular_margen_seguridad(self) -> float:
        """
        Calcula el margen de seguridad usando un DCF simplificado.
        
        Returns:
            Porcentaje de descuento/premio sobre el valor intrínseco
        """
        try:
            precio_actual = self._get_precio()
            if precio_actual <= 0:
                return 0.0
            
            # DCF simplificado
            fcf = self._calcular_fcf() * 1e9  # Convertir a dólares
            if fcf <= 0:
                return 0.0
            
            wacc = 0.09  # 9% tasa de descuento
            g = 0.03     # 3% crecimiento perpetuo
            
            # Proyección a 5 años
            flujos = [fcf * ((1 + 0.05) ** ano) / ((1 + wacc) ** ano) for ano in range(1, 6)]
            
            # Valor terminal
            vt = (flujos[-1] * (1 + wacc)) / (wacc - g)
            vp_vt = vt / ((1 + wacc) ** 5)
            
            enterprise_value = sum(flujos) + vp_vt
            
            # Ajustar por deuda y acciones
            deuda_neta = (abs(self._get_balance_item(['Short Term Debt', 'Current Debt'], 0)) + 
                         abs(self._get_balance_item(['Long Term Debt'], 0)) - 
                         self._get_balance_item(['Cash And Cash Equivalents'], 0))
            
            equity_value = enterprise_value - deuda_neta
            shares = self.info.get('sharesOutstanding', 1e9)
            precio_justo = equity_value / shares
            
            margen = ((precio_justo / precio_actual) - 1) * 100
            return round(margen, 2)
            
        except Exception as e:
            logger.warning(f"No se pudo calcular margen de seguridad para {self.ticker}: {e}")
            return 0.0
    
    def _get_balance_item(self, keywords: list, default: float = 0.0) -> float:
        """Busca un item en el balance sheet usando palabras clave."""
        try:
            for kw in keywords:
                matches = [idx for idx in self.balance.index if kw in idx]
                if matches:
                    val = self.balance.loc[matches[0]].iloc[0]
                    return val if not pd.isna(val) else default
        except:
            pass
        return default
    
    def _get_cashflow_item(self, keywords: list, default: float = 0.0) -> float:
        """Busca un item en el cash flow usando palabras clave."""
        try:
            for kw in keywords:
                matches = [idx for idx in self.cashflow.index if kw in idx]
                if matches:
                    val = self.cashflow.loc[matches[0]].iloc[0]
                    return val if not pd.isna(val) else default
        except:
            pass
        return default


# Función auxiliar para uso rápido
def obtener_datos_financieros(ticker: str) -> Optional[Dict]:
    """
    Función rápida para obtener datos financieros de un ticker.
    
    Args:
        ticker: Símbolo bursátil
        
    Returns:
        Diccionario con métricas o None
    """
    fetcher = FinancialDataFetcher(ticker)
    return fetcher.fetch_data()
