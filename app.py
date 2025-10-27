from flask import Flask, request, render_template, send_from_directory
import calculos
import graficos
import os

app = Flask(__name__)

# Crear carpeta para imágenes si no existe
if not os.path.exists('static/imagenes'):
    os.makedirs('static/imagenes')

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Rutas para las páginas de cada distribución
@app.route('/binomial')
def binomial_page():
    return render_template('binomial.html')

@app.route('/poisson')
def poisson_page():
    return render_template('poisson.html')

@app.route('/geometrica')
def geometrica_page():
    return render_template('geometrica.html')

@app.route('/hipergeometrica')
def hipergeometrica_page():
    return render_template('hipergeometrica.html')

# Cálculo Binomial
@app.route('/calcular_binomial', methods=['POST'])
def calcular_binomial():
    try:
        n = int(request.form['n'])
        p = float(request.form['p'])
        k = int(request.form['k'])
        
        # Validaciones
        if n <= 0:
            raise ValueError("El número de llamadas (n) debe ser positivo")
        if p < 0 or p > 1:
            raise ValueError("La probabilidad (p) debe estar entre 0 y 1")
        if k < 0 or k > n:
            raise ValueError(f"El número de ventas (k) debe estar entre 0 y {n}")
        
        # Cálculos
        resultados = calculos.binomial(n, p, k)
        
        # Generar gráfico
        nombre_imagen = f'binomial_{n}_{p}_{k}.png'
        graficos.grafico_binomial(n, p, k, f'static/imagenes/{nombre_imagen}')
        
        # Interpretación
        interpretacion = f"""
        Al realizar {n} llamadas con una probabilidad de cierre del {p*100:.1f}%, 
        existe un {resultados['p_exacta']*100:.2f}% de probabilidad de lograr exactamente {k} ventas.
        
        En promedio, se esperan {resultados['media']:.2f} ventas con una desviación estándar de 
        {resultados['desviacion']:.2f} ventas. Esto significa que el resultado típico variará 
        entre {max(0, resultados['media'] - resultados['desviacion']):.1f} y 
        {min(n, resultados['media'] + resultados['desviacion']):.1f} ventas.
        
        La probabilidad de lograr al menos {k} ventas es del {resultados['p_acumulada_mayor']*100:.2f}%, 
        mientras que la probabilidad de lograr como máximo {k} ventas es del {resultados['p_acumulada_menor']*100:.2f}%.
        """
        
        parametros = {
            'Número de llamadas (n)': n,
            'Probabilidad de cierre (p)': f"{p} ({p*100}%)",
            'Ventas a evaluar (k)': k
        }
        
        return render_template('resultados.html',
            titulo="Distribución Binomial",
            icono="📞",
            url_volver="/binomial",
            parametros=parametros,
            resultados=resultados,
            imagen=nombre_imagen,
            interpretacion=interpretacion,
            error=None
        )
        
    except Exception as e:
        return render_template('resultados.html',
            titulo="Distribución Binomial",
            icono="📞",
            url_volver="/binomial",
            error=str(e),
            parametros={},
            resultados=None,
            imagen=None,
            interpretacion=None
        )

# Cálculo Poisson
@app.route('/calcular_poisson', methods=['POST'])
def calcular_poisson():
    try:
        lambda_val = float(request.form['lambda'])
        k = int(request.form['k'])
        periodo = request.form.get('periodo', 'día')
        
        if lambda_val <= 0:
            raise ValueError("El promedio (λ) debe ser mayor que 0")
        if k < 0:
            raise ValueError("El número de ventas (k) debe ser no negativo")
        
        resultados = calculos.poisson(lambda_val, k)
        nombre_imagen = f'poisson_{lambda_val}_{k}.png'
        graficos.grafico_poisson(lambda_val, k, f'static/imagenes/{nombre_imagen}')
        
        interpretacion = f"""
        Con un promedio de {lambda_val} ventas por {periodo}, hay un {resultados['p_exacta']*100:.2f}% 
        de probabilidad de lograr exactamente {k} ventas en un {periodo} específico.
        
        La media de la distribución es {resultados['media']:.2f} y la desviación estándar es 
        {resultados['desviacion']:.2f}. En la práctica, esto significa que en la mayoría de los {periodo}s, 
        las ventas oscilarán entre {max(0, resultados['media'] - resultados['desviacion']):.1f} y 
        {resultados['media'] + resultados['desviacion']:.1f}.
        """
        
        parametros = {
            'Promedio de ventas (λ)': lambda_val,
            'Período': periodo.capitalize(),
            'Ventas a evaluar (k)': k
        }
        
        return render_template('resultados.html',
            titulo="Distribución de Poisson",
            icono="📅",
            url_volver="/poisson",
            parametros=parametros,
            resultados=resultados,
            imagen=nombre_imagen,
            interpretacion=interpretacion,
            error=None
        )
        
    except Exception as e:
        return render_template('resultados.html',
            titulo="Distribución de Poisson",
            icono="📅",
            url_volver="/poisson",
            error=str(e),
            parametros={},
            resultados=None,
            imagen=None,
            interpretacion=None
        )

# Cálculo Geométrica
@app.route('/calcular_geometrica', methods=['POST'])
def calcular_geometrica():
    try:
        p = float(request.form['p'])
        k = int(request.form['k'])
        
        if p <= 0 or p > 1:
            raise ValueError("La probabilidad (p) debe estar entre 0 (exclusivo) y 1")
        if k < 1:
            raise ValueError("El número del intento (k) debe ser al menos 1")
        
        resultados = calculos.geometrica(p, k)
        nombre_imagen = f'geometrica_{p}_{k}.png'
        graficos.grafico_geometrica(p, k, f'static/imagenes/{nombre_imagen}')
        
        ordinal = {1: 'primer', 2: 'segundo', 3: 'tercer', 4: 'cuarto', 5: 'quinto'}
        k_ordinal = ordinal.get(k, f'{k}º')
        
        interpretacion = f"""
        Con una probabilidad de éxito del {p*100:.1f}% por intento, existe un {resultados['p_exacta']*100:.2f}% 
        de probabilidad de que la primera venta ocurra exactamente en el {k_ordinal} intento.
        
        En promedio, se necesitarán {resultados['media']:.2f} intentos para lograr la primera venta, 
        con una desviación estándar de {resultados['desviacion']:.2f} intentos.
        """
        
        parametros = {
            'Probabilidad de éxito (p)': f"{p} ({p*100}%)",
            'Número del intento (k)': f"{k}º intento"
        }
        
        return render_template('resultados.html',
            titulo="Distribución Geométrica",
            icono="🎯",
            url_volver="/geometrica",
            parametros=parametros,
            resultados=resultados,
            imagen=nombre_imagen,
            interpretacion=interpretacion,
            error=None
        )
        
    except Exception as e:
        return render_template('resultados.html',
            titulo="Distribución Geométrica",
            icono="🎯",
            url_volver="/geometrica",
            error=str(e),
            parametros={},
            resultados=None,
            imagen=None,
            interpretacion=None
        )

# Cálculo Hipergeométrica
@app.route('/calcular_hipergeometrica', methods=['POST'])
def calcular_hipergeometrica():
    try:
        N = int(request.form['N'])
        K = int(request.form['K'])
        n = int(request.form['n'])
        k = int(request.form['k'])
        
        if N <= 0:
            raise ValueError("El tamaño de la población (N) debe ser positivo")
        if K < 0 or K > N:
            raise ValueError(f"El número de prospects buenos (K) debe estar entre 0 y {N}")
        if n < 0 or n > N:
            raise ValueError(f"El tamaño de la muestra (n) debe estar entre 0 y {N}")
        
        k_min = max(0, n + K - N)
        k_max = min(n, K)
        if k < k_min or k > k_max:
            raise ValueError(f"Para los parámetros dados, k debe estar entre {k_min} y {k_max}")
        
        resultados = calculos.hipergeometrica(N, K, n, k)
        nombre_imagen = f'hipergeometrica_{N}_{K}_{n}_{k}.png'
        graficos.grafico_hipergeometrica(N, K, n, k, f'static/imagenes/{nombre_imagen}')
        
        porcentaje_prospects = (K/N)*100
        
        interpretacion = f"""
        De una base de {N} clientes con {K} prospects buenos ({porcentaje_prospects:.1f}% del total), 
        al contactar una muestra de {n} clientes, existe un {resultados['p_exacta']*100:.2f}% de 
        probabilidad de obtener exactamente {k} prospects buenos.
        
        En promedio, se esperan {resultados['media']:.2f} prospects buenos en la muestra.
        """
        
        parametros = {
            'Tamaño de la base (N)': N,
            'Prospects buenos (K)': f"{K} ({porcentaje_prospects:.1f}%)",
            'Tamaño de muestra (n)': n,
            'Prospects esperados (k)': k
        }
        
        return render_template('resultados.html',
            titulo="Distribución Hipergeométrica",
            icono="👥",
            url_volver="/hipergeometrica",
            parametros=parametros,
            resultados=resultados,
            imagen=nombre_imagen,
            interpretacion=interpretacion,
            error=None
        )
        
    except Exception as e:
        return render_template('resultados.html',
            titulo="Distribución Hipergeométrica",
            icono="👥",
            url_volver="/hipergeometrica",
            error=str(e),
            parametros={},
            resultados=None,
            imagen=None,
            interpretacion=None
        )

if __name__ == '__main__':
    # Railway proporciona el puerto a través de la variable de entorno PORT
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)