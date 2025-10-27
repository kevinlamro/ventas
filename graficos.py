"""
Módulo de generación de gráficos para distribuciones de probabilidad
Universidad Del Sinú - Simulador de Ventas
"""

import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')

def grafico_binomial(n, p, k, ruta_salida):
    """
    Genera gráfico de barras para la distribución binomial.
    
    Parámetros:
        n: Número de ensayos
        p: Probabilidad de éxito
        k: Valor a destacar
        ruta_salida: Ruta donde guardar la imagen
    """
    # Crear distribución
    dist = stats.binom(n, p)
    
    # Rango de valores a graficar
    media = n * p
    desv = np.sqrt(n * p * (1 - p))
    x_min = max(0, int(media - 3 * desv))
    x_max = min(n, int(media + 3 * desv) + 1)
    
    x = np.arange(x_min, x_max + 1)
    y = dist.pmf(x)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colores: azul para el valor k, gris para los demás
    colores = ['#3498db' if val == k else '#95a5a6' for val in x]
    
    # Gráfico de barras
    bars = ax.bar(x, y, color=colores, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Destacar la barra de k
    if k in x:
        idx = np.where(x == k)[0][0]
        bars[idx].set_color('#e74c3c')
        bars[idx].set_alpha(1.0)
        bars[idx].set_linewidth(2)
    
    # Configuración del gráfico
    ax.set_xlabel('Número de Ventas (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidad P(X = k)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribución Binomial (n={n}, p={p:.2f})\nP(X={k}) = {dist.pmf(k):.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Añadir línea vertical en la media
    ax.axvline(media, color='green', linestyle='--', linewidth=2, 
               label=f'Media = {media:.2f}', alpha=0.7)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


def grafico_poisson(lambda_val, k, ruta_salida):
    """
    Genera gráfico de barras para la distribución de Poisson.
    
    Parámetros:
        lambda_val: Tasa promedio
        k: Valor a destacar
        ruta_salida: Ruta donde guardar la imagen
    """
    # Crear distribución
    dist = stats.poisson(lambda_val)
    
    # Rango de valores a graficar
    desv = np.sqrt(lambda_val)
    x_min = max(0, int(lambda_val - 3 * desv))
    x_max = int(lambda_val + 3 * desv) + 1
    
    x = np.arange(x_min, x_max + 1)
    y = dist.pmf(x)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colores
    colores = ['#27ae60' if val == k else '#95a5a6' for val in x]
    
    # Gráfico de barras
    bars = ax.bar(x, y, color=colores, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Destacar la barra de k
    if k in x:
        idx = np.where(x == k)[0][0]
        bars[idx].set_color('#e74c3c')
        bars[idx].set_alpha(1.0)
        bars[idx].set_linewidth(2)
    
    # Configuración del gráfico
    ax.set_xlabel('Número de Ventas (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidad P(X = k)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribución de Poisson (λ={lambda_val:.2f})\nP(X={k}) = {dist.pmf(k):.4f}', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Línea vertical en la media
    ax.axvline(lambda_val, color='green', linestyle='--', linewidth=2, 
               label=f'Media = {lambda_val:.2f}', alpha=0.7)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


def grafico_geometrica(p, k, ruta_salida):
    """
    Genera gráfico de barras para la distribución geométrica.
    
    Parámetros:
        p: Probabilidad de éxito
        k: Valor a destacar
        ruta_salida: Ruta donde guardar la imagen
    """
    # Crear distribución
    dist = stats.geom(p)
    
    # Rango de valores a graficar (hasta 3 veces la media o 20, lo que sea menor)
    media = 1 / p
    x_max = min(int(media * 3), 20)
    
    x = np.arange(1, x_max + 1)
    y = dist.pmf(x)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colores
    colores = ['#f39c12' if val == k else '#95a5a6' for val in x]
    
    # Gráfico de barras
    bars = ax.bar(x, y, color=colores, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Destacar la barra de k
    if k in x:
        idx = np.where(x == k)[0][0]
        bars[idx].set_color('#e74c3c')
        bars[idx].set_alpha(1.0)
        bars[idx].set_linewidth(2)
    
    # Configuración del gráfico
    ax.set_xlabel('Número del Intento (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidad P(X = k)', fontsize=12, fontweight='bold')
    ax.set_title(f'Distribución Geométrica (p={p:.2f})\nP(X={k}) = {dist.pmf(k):.4f}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Línea vertical en la media
    ax.axvline(media, color='green', linestyle='--', linewidth=2, 
            label=f'Media = {media:.2f}', alpha=0.7)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()


def grafico_hipergeometrica(N, K, n, k, ruta_salida):
    """
    Genera gráfico de barras para la distribución hipergeométrica.
    
    Parámetros:
        N: Tamaño de la población
        K: Número de éxitos en la población
        n: Tamaño de la muestra
        k: Valor a destacar
        ruta_salida: Ruta donde guardar la imagen
    """
    # Crear distribución
    dist = stats.hypergeom(N, K, n)
    
    # Rango de valores posibles
    k_min = max(0, n + K - N)
    k_max = min(n, K)
    
    x = np.arange(k_min, k_max + 1)
    y = dist.pmf(x)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Colores
    colores = ['#e74c3c' if val == k else '#95a5a6' for val in x]
    
    # Gráfico de barras
    bars = ax.bar(x, y, color=colores, alpha=0.7, edgecolor='black', linewidth=1.2)
    
    # Destacar la barra de k
    if k in x:
        idx = np.where(x == k)[0][0]
        bars[idx].set_color('#c0392b')
        bars[idx].set_alpha(1.0)
        bars[idx].set_linewidth(2)
    
    # Configuración del gráfico
    ax.set_xlabel('Número de Prospects Buenos (k)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probabilidad P(X = k)', fontsize=12, fontweight='bold')
    
    media = n * (K / N)
    ax.set_title(f'Distribución Hipergeométrica (N={N}, K={K}, n={n})\nP(X={k}) = {dist.pmf(k):.4f}', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Línea vertical en la media
    ax.axvline(media, color='green', linestyle='--', linewidth=2, 
            label=f'Media = {media:.2f}', alpha=0.7)
    
    # Grid y leyenda
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Guardar
    plt.savefig(ruta_salida, dpi=150, bbox_inches='tight')
    plt.close()