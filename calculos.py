"""
Módulo de cálculos para distribuciones de probabilidad
Universidad Del Sinú - Simulador de Ventas
VERSIÓN CORREGIDA - Resultados exactos verificados
"""

from scipy import stats
import math

def binomial(n, p, k):
    """
    Calcula probabilidades y estadísticos para la distribución binomial.
    
    Parámetros:
        n: Número de ensayos (llamadas)
        p: Probabilidad de éxito (cierre)
        k: Número de éxitos (ventas)
    
    Retorna:
        dict con probabilidades y estadísticos
    """
    # Crear distribución binomial
    dist = stats.binom(n, p)
    
    # Calcular probabilidades
    p_exacta = dist.pmf(k)  # P(X = k)
    p_acumulada_menor = dist.cdf(k)  # P(X ≤ k)
    p_acumulada_mayor = 1 - dist.cdf(k - 1) if k > 0 else 1  # P(X ≥ k)
    
    # Calcular estadísticos
    media = n * p
    varianza = n * p * (1 - p)
    desviacion = math.sqrt(varianza)
    
    return {
        'p_exacta': float(p_exacta),
        'p_acumulada_menor': float(p_acumulada_menor),
        'p_acumulada_mayor': float(p_acumulada_mayor),
        'media': float(media),
        'varianza': float(varianza),
        'desviacion': float(desviacion)
    }


def poisson(lambda_val, k):
    """
    Calcula probabilidades y estadísticos para la distribución de Poisson.
    
    Parámetros:
        lambda_val: Tasa promedio de eventos
        k: Número de eventos a evaluar
    
    Retorna:
        dict con probabilidades y estadísticos
    """
    # Crear distribución de Poisson
    dist = stats.poisson(lambda_val)
    
    # Calcular probabilidades
    p_exacta = dist.pmf(k)  # P(X = k)
    p_acumulada_menor = dist.cdf(k)  # P(X ≤ k)
    p_acumulada_mayor = 1 - dist.cdf(k - 1) if k > 0 else 1  # P(X ≥ k)
    
    # Calcular estadísticos
    media = lambda_val
    varianza = lambda_val
    desviacion = math.sqrt(varianza)
    
    return {
        'p_exacta': float(p_exacta),
        'p_acumulada_menor': float(p_acumulada_menor),
        'p_acumulada_mayor': float(p_acumulada_mayor),
        'media': float(media),
        'varianza': float(varianza),
        'desviacion': float(desviacion)
    }


def geometrica(p, k):
    """
    Calcula probabilidades y estadísticos para la distribución geométrica.
    
    IMPORTANTE: Usa cálculo manual para evitar problemas de parametrización.
    k representa el intento en el que ocurre el PRIMER éxito.
    
    Parámetros:
        p: Probabilidad de éxito en cada intento
        k: Número del intento donde ocurre el primer éxito (k ≥ 1)
    
    Retorna:
        dict con probabilidades y estadísticos
    """
    # Cálculo MANUAL de probabilidades
    # P(X = k) = (1-p)^(k-1) * p
    p_exacta = ((1 - p) ** (k - 1)) * p
    
    # P(X ≤ k) = 1 - (1-p)^k
    p_acumulada_menor = 1 - ((1 - p) ** k)
    
    # P(X ≥ k) = (1-p)^(k-1)
    p_acumulada_mayor = (1 - p) ** (k - 1)
    
    # Calcular estadísticos
    media = 1 / p
    varianza = (1 - p) / (p ** 2)
    desviacion = math.sqrt(varianza)
    
    return {
        'p_exacta': float(p_exacta),
        'p_acumulada_menor': float(p_acumulada_menor),
        'p_acumulada_mayor': float(p_acumulada_mayor),
        'media': float(media),
        'varianza': float(varianza),
        'desviacion': float(desviacion)
    }


def hipergeometrica(N, K, n, k):
    """
    Calcula probabilidades y estadísticos para la distribución hipergeométrica.
    
    Parámetros:
        N: Tamaño de la población total
        K: Número de éxitos en la población
        n: Tamaño de la muestra
        k: Número de éxitos en la muestra
    
    Retorna:
        dict con probabilidades y estadísticos
    """
    # Crear distribución hipergeométrica
    dist = stats.hypergeom(N, K, n)
    
    # Calcular probabilidades
    p_exacta = dist.pmf(k)  # P(X = k)
    p_acumulada_menor = dist.cdf(k)  # P(X ≤ k)
    p_acumulada_mayor = 1 - dist.cdf(k - 1) if k > 0 else 1  # P(X ≥ k)
    
    # Calcular estadísticos
    media = n * (K / N)
    varianza = n * (K / N) * ((N - K) / N) * ((N - n) / (N - 1))
    desviacion = math.sqrt(varianza)
    
    return {
        'p_exacta': float(p_exacta),
        'p_acumulada_menor': float(p_acumulada_menor),
        'p_acumulada_mayor': float(p_acumulada_mayor),
        'media': float(media),
        'varianza': float(varianza),
        'desviacion': float(desviacion)
    }


# Función auxiliar para calcular rango de valores razonables
def rango_razonable(media, desviacion, min_val=0, max_val=None):
    """
    Calcula un rango razonable de valores para graficar (±3 desviaciones estándar)
    """
    inicio = max(min_val, int(media - 3 * desviacion))
    fin = int(media + 3 * desviacion)
    if max_val is not None:
        fin = min(fin, max_val)
    return inicio, fin