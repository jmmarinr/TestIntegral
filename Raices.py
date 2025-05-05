import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sympy import symbols, lambdify, diff, sympify
import time

st.set_page_config(
    page_title="Métodos Numéricos para Encontrar Raíces",
    page_icon="📊",
    layout="wide"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 1.5rem;
        font-weight: bold;
        color: #0D47A1;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .method-description {
        background-color: #E1F5FE;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .iteration-table {
        font-size: 0.9rem;
    }
    .convergence-plot {
        border: 1px solid #BDBDBD;
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown("<div class='main-header'>Métodos Numéricos para Encontrar Raíces de Funciones</div>", unsafe_allow_html=True)

st.markdown("""
Esta aplicación interactiva te permite explorar diferentes métodos numéricos para encontrar raíces de funciones.
Selecciona una función, un método y los parámetros para visualizar cómo el método converge hacia la raíz.
""")

# Sidebar para seleccionar función y método
with st.sidebar:
    st.markdown("### Configuración")
    
    # Selector de funciones predefinidas
    selected_function = st.selectbox(
        "Selecciona una función:",
        ["x^3 - 2*x - 5", "cos(x) - x", "e^x - 3*x", "x^2 - 4"]
    )
    
    # Opción para ingresar función personalizada
    custom_function = st.text_input("O ingresa tu propia función:", "")
    if custom_function:
        selected_function = custom_function
        
    # Mostrar la función seleccionada
    st.markdown(f"**Función seleccionada:** $f(x) = {selected_function}$")
    
    # Selector de método
    method = st.selectbox(
        "Selecciona un método:",
        ["Bisección", "Regula Falsi", "Punto Fijo", "Secante", "Newton-Raphson"]
    )
    
    # Mostrar entradas de parámetros según el método seleccionado
    if method == "Bisección" or method == "Regula Falsi":
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Límite inferior (a):", value=-2.0)
        with col2:
            b = st.number_input("Límite superior (b):", value=3.0)
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.10f")
        max_iter = st.number_input("Máximo de iteraciones:", value=100, min_value=1)
        
    elif method == "Punto Fijo":
        # Para punto fijo, necesitamos una función g(x) tal que x = g(x)
        g_x = st.text_input("Función g(x) para x = g(x):", "0.5*(x^2 + 4)/x")
        x0 = st.number_input("Valor inicial (x0):", value=1.0)
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.10f")
        max_iter = st.number_input("Máximo de iteraciones:", value=100, min_value=1)
        
    elif method == "Secante":
        col1, col2 = st.columns(2)
        with col1:
            x0 = st.number_input("Primer punto (x0):", value=1.0)
        with col2:
            x1 = st.number_input("Segundo punto (x1):", value=2.0)
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.10f")
        max_iter = st.number_input("Máximo de iteraciones:", value=100, min_value=1)
        
    elif method == "Newton-Raphson":
        x0 = st.number_input("Valor inicial (x0):", value=1.0)
        tol = st.number_input("Tolerancia:", value=1e-6, format="%.10f")
        max_iter = st.number_input("Máximo de iteraciones:", value=100, min_value=1)
    
    # Botón para iniciar el cálculo
    calculate = st.button("Calcular raíz")

# Función para crear lambda functions a partir de expresiones de texto
def create_function(expr):
    x = symbols('x')
    expr_sympy = sympify(expr)
    f = lambdify(x, expr_sympy, 'numpy')
    return f, expr_sympy

# Funciones para los diferentes métodos numéricos
def bisection(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        st.error("Error: La función debe tener signos opuestos en los extremos del intervalo.")
        return None, None
    
    iterations = []
    a_values = []
    b_values = []
    c_values = []
    error_values = []
    
    error = float('inf')
    i = 0
    c_old = a
    
    while error > tol and i < max_iter:
        c = (a + b) / 2
        iterations.append(i)
        a_values.append(a)
        b_values.append(b)
        c_values.append(c)
        
        if i > 0:
            error = abs(c - c_old)
        else:
            error = abs(b - a)
        
        error_values.append(error)
        
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        c_old = c
        i += 1
    
    results = pd.DataFrame({
        'Iteración': iterations,
        'a': a_values,
        'b': b_values,
        'c': c_values,
        'f(c)': [f(c) for c in c_values],
        'Error': error_values
    })
    
    return c, results

def regula_falsi(f, a, b, tol=1e-6, max_iter=100):
    if f(a) * f(b) > 0:
        st.error("Error: La función debe tener signos opuestos en los extremos del intervalo.")
        return None, None
    
    iterations = []
    a_values = []
    b_values = []
    c_values = []
    error_values = []
    
    error = float('inf')
    i = 0
    c_old = a
    
    while error > tol and i < max_iter:
        c = (a * f(b) - b * f(a)) / (f(b) - f(a))
        iterations.append(i)
        a_values.append(a)
        b_values.append(b)
        c_values.append(c)
        
        if i > 0:
            error = abs(c - c_old)
        else:
            error = abs(b - a)
        
        error_values.append(error)
        
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
        
        c_old = c
        i += 1
    
    results = pd.DataFrame({
        'Iteración': iterations,
        'a': a_values,
        'b': b_values,
        'c': c_values,
        'f(c)': [f(c) for c in c_values],
        'Error': error_values
    })
    
    return c, results

def fixed_point(f, g, x0, tol=1e-6, max_iter=100):
    iterations = []
    x_values = [x0]
    error_values = [float('inf')]
    
    x_old = x0
    
    for i in range(max_iter):
        x_new = g(x_old)
        x_values.append(x_new)
        error = abs(x_new - x_old)
        error_values.append(error)
        iterations.append(i)
        
        if error < tol:
            break
            
        x_old = x_new
    
    results = pd.DataFrame({
        'Iteración': iterations,
        'x': x_values[1:],  # Skip the initial value
        'g(x)': [g(x) for x in x_values[1:]],  # g(x) should equal x for fixed point
        'f(x)': [f(x) for x in x_values[1:]],
        'Error': error_values[1:]  # Skip the initial error
    })
    
    return x_values[-1], results

def secant(f, x0, x1, tol=1e-6, max_iter=100):
    iterations = []
    x_prev = x0
    x_curr = x1
    f_prev = f(x_prev)
    f_curr = f(x_curr)
    
    x_values_prev = []
    x_values_curr = []
    x_values_next = []
    f_values_next = []
    error_values = []
    
    for i in range(max_iter):
        # Verificar si la diferencia entre f_curr y f_prev es demasiado pequeña
        if abs(f_curr - f_prev) < 1e-10:
            st.warning("Método de la secante: División por un número cercano a cero.")
            break
        
        # Calcular el siguiente valor de x usando el método de la secante
        x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        f_next = f(x_next)
        
        # Calcular el error
        error = abs(x_next - x_curr)
        
        # Guardar valores para la visualización
        iterations.append(i)
        x_values_prev.append(x_prev)
        x_values_curr.append(x_curr)
        x_values_next.append(x_next)
        f_values_next.append(f_next)
        error_values.append(error)
        
        # Verificar convergencia
        if error < tol:
            break
        
        # Actualizar valores para la siguiente iteración
        x_prev = x_curr
        x_curr = x_next
        f_prev = f_curr
        f_curr = f_next
    
    results = pd.DataFrame({
        'Iteración': iterations,
        'x_n-1': x_values_prev,
        'x_n': x_values_curr,
        'x_n+1': x_values_next,
        'f(x_n+1)': f_values_next,
        'Error': error_values
    })
    
    return x_next, results

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100):
    iterations = []
    x_curr = x0
    
    x_values = []
    f_values = []
    df_values = []
    x_next_values = []
    error_values = []
    
    for i in range(max_iter):
        f_curr = f(x_curr)
        df_curr = df(x_curr)
        
        # Verificar si la derivada es cercana a cero
        if abs(df_curr) < 1e-10:
            st.warning("Método de Newton-Raphson: Derivada cercana a cero.")
            break
        
        # Calcular el siguiente valor usando el método de Newton-Raphson
        x_next = x_curr - f_curr / df_curr
        
        # Calcular el error
        error = abs(x_next - x_curr)
        
        # Guardar valores para la visualización
        iterations.append(i)
        x_values.append(x_curr)
        f_values.append(f_curr)
        df_values.append(df_curr)
        x_next_values.append(x_next)
        error_values.append(error)
        
        # Verificar convergencia
        if error < tol:
            break
        
        # Actualizar para la siguiente iteración
        x_curr = x_next
    
    results = pd.DataFrame({
        'Iteración': iterations,
        'x_n': x_values,
        'f(x_n)': f_values,
        'f\'(x_n)': df_values,
        'x_n+1': x_next_values,
        'Error': error_values
    })
    
    return x_curr, results

# Definir la función para graficar el método y la función
def plot_function_and_iterations(f, method_name, results, root, x_range=None):
    if x_range is None:
        # Determinar un rango adecuado para visualizar la función alrededor de la raíz
        x_range = [root - 3, root + 3]
    
    x = np.linspace(x_range[0], x_range[1], 1000)
    y = np.vectorize(f)(x)
    
    plt.figure(figsize=(10, 6))
    
    # Graficar la función
    plt.plot(x, y, 'b-', label=f'f(x)')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Graficar las iteraciones según el método
    if method_name == "Bisección" or method_name == "Regula Falsi":
        iterations = results['Iteración'].tolist()
        a_values = results['a'].tolist()
        b_values = results['b'].tolist()
        c_values = results['c'].tolist()
        
        for i, (a, b, c) in enumerate(zip(a_values, b_values, c_values)):
            if i == 0:  # Primera iteración
                plt.plot([a, b], [0, 0], 'ro-', label='Intervalo inicial')
            elif i == len(iterations) - 1:  # Última iteración
                plt.plot([a, b], [0, 0], 'go-', label='Intervalo final')
            else:
                plt.plot([a, b], [0, 0], 'yo-', alpha=0.3)
            
            plt.plot(c, 0, 'mo', markersize=8)
            plt.plot([c, c], [0, f(c)], 'm--', alpha=0.5)
    
    elif method_name == "Punto Fijo":
        x_values = results['x'].tolist()
        
        for i, x in enumerate(x_values):
            if i == 0:
                plt.plot(x, f(x), 'ro', label='Punto inicial')
            elif i == len(x_values) - 1:
                plt.plot(x, f(x), 'go', label='Punto final')
            else:
                plt.plot(x, f(x), 'yo', alpha=0.5)
            
            plt.plot([x, x], [0, f(x)], 'y--', alpha=0.3)
    
    elif method_name == "Secante":
        # Extraer valores directamente de las columnas para evitar problemas con listas
        try:
            for i in range(len(results)):
                x_n = results['x_n'].iloc[i]
                x_n_plus_1 = results['x_n+1'].iloc[i]
                
                if i == 0:
                    plt.plot([x_n, x_n_plus_1], [f(x_n), f(x_n_plus_1)], 'ro-', label='Puntos iniciales')
                elif i == len(results) - 1:
                    plt.plot([x_n, x_n_plus_1], [f(x_n), f(x_n_plus_1)], 'go-', label='Puntos finales')
                else:
                    plt.plot([x_n, x_n_plus_1], [f(x_n), f(x_n_plus_1)], 'yo-', alpha=0.3)
                
                # Marcar puntos de intersección con el eje x
                plt.plot([x_n_plus_1], [0], 'mo', markersize=6)
        except Exception as e:
            st.warning(f"No se pudieron graficar todas las iteraciones del método de la Secante: {e}")
    
    elif method_name == "Newton-Raphson":
        try:
            for i in range(len(results)):
                x_n = results['x_n'].iloc[i]
                x_n_plus_1 = results['x_n+1'].iloc[i]
                
                if i == 0:
                    plt.plot(x_n, f(x_n), 'ro', label='Punto inicial')
                elif i == len(results) - 1:
                    plt.plot(x_n, f(x_n), 'go', label='Punto final')
                else:
                    plt.plot(x_n, f(x_n), 'yo', alpha=0.5)
                
                # Graficar punto en el eje x
                plt.plot([x_n_plus_1], [0], 'mo', markersize=6)
                
                # Graficar la tangente si la derivada está disponible
                try:
                    df_x_n = results['f\'(x_n)'].iloc[i]
                    # Limitar el rango para que la tangente no domine la gráfica
                    x_range_local = min(2, abs(x_n_plus_1 - x_n) * 5)
                    x_tangent = np.linspace(x_n - x_range_local, x_n + x_range_local, 100)
                    y_tangent = f(x_n) + df_x_n * (x_tangent - x_n)
                    plt.plot(x_tangent, y_tangent, 'r--', alpha=0.3)
                    
                    # Línea vertical desde punto hasta intersección con eje x
                    plt.plot([x_n, x_n], [0, f(x_n)], 'm--', alpha=0.3)
                except Exception:
                    pass
        except Exception as e:
            st.warning(f"No se pudieron graficar todas las iteraciones del método de Newton-Raphson: {e}")
    
    # Graficar la raíz encontrada
    plt.plot(root, 0, 'g*', markersize=12, label=f'Raíz: x ≈ {root:.8f}')
    
    plt.title(f'Método de {method_name} para encontrar la raíz de f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    return plt

# Definir la función para mostrar la evolución del error
def plot_error_convergence(results):
    plt.figure(figsize=(10, 4))
    
    # Graficar el error en escala logarítmica
    plt.semilogy(results['Iteración'], results['Error'], 'bo-')
    
    plt.title('Convergencia del Error')
    plt.xlabel('Iteración')
    plt.ylabel('Error (escala logarítmica)')
    plt.grid(True, alpha=0.3)
    
    return plt

# Mostrar descripciones de los métodos
def show_method_description(method):
    if method == "Bisección":
        st.markdown("""
        <div class='method-description'>
            <h3>Método de Bisección</h3>
            <p>El método de bisección es un algoritmo de búsqueda de raíces que divide repetidamente un intervalo a la mitad y selecciona
            el subintervalo donde la raíz debe estar.</p>
            <p><strong>Ventajas:</strong> Simple, robusto y siempre converge si hay un cambio de signo en el intervalo inicial.</p>
            <p><strong>Desventajas:</strong> Converge lentamente comparado con otros métodos.</p>
            <p><strong>Orden de convergencia:</strong> Lineal (orden 1).</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif method == "Regula Falsi":
        st.markdown("""
        <div class='method-description'>
            <h3>Método de Regula Falsi (Falsa Posición)</h3>
            <p>El método de falsa posición es una combinación del método de bisección y el método de la secante.
            Usa una interpolación lineal entre los extremos del intervalo para estimar la raíz.</p>
            <p><strong>Ventajas:</strong> Más rápido que la bisección y siempre converge si hay un cambio de signo.</p>
            <p><strong>Desventajas:</strong> Puede ser lento si una de las fronteras del intervalo permanece fija.</p>
            <p><strong>Orden de convergencia:</strong> Superlineal (entre 1 y 1.618).</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif method == "Punto Fijo":
        st.markdown("""
        <div class='method-description'>
            <h3>Método de Punto Fijo</h3>
            <p>El método de punto fijo se basa en reescribir la ecuación f(x) = 0 como x = g(x), y luego iterar con x_new = g(x_old)
            hasta que converja a un punto fijo que es la raíz buscada.</p>
            <p><strong>Ventajas:</strong> Simple conceptualmente y fácil de implementar.</p>
            <p><strong>Desventajas:</strong> Necesita una función g(x) adecuada y puede diverger si |g'(x)| > 1 cerca de la raíz.</p>
            <p><strong>Orden de convergencia:</strong> Lineal (orden 1) si 0 < |g'(x)| < 1.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif method == "Secante":
        st.markdown("""
        <div class='method-description'>
            <h3>Método de la Secante</h3>
            <p>El método de la secante es similar al método de Newton-Raphson, pero usa una aproximación de la derivada
            utilizando dos puntos previos en lugar de calcular la derivada analíticamente.</p>
            <p><strong>Ventajas:</strong> No requiere el cálculo de la derivada y converge rápidamente.</p>
            <p><strong>Desventajas:</strong> Requiere dos valores iniciales y puede divergir en algunos casos.</p>
            <p><strong>Orden de convergencia:</strong> Superlineal (orden 1.618, número áureo).</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif method == "Newton-Raphson":
        st.markdown("""
        <div class='method-description'>
            <h3>Método de Newton-Raphson</h3>
            <p>El método de Newton-Raphson utiliza la derivada de la función para encontrar mejores aproximaciones
            a la raíz utilizando la fórmula: x_new = x_old - f(x_old)/f'(x_old).</p>
            <p><strong>Ventajas:</strong> Converge muy rápidamente cerca de la raíz.</p>
            <p><strong>Desventajas:</strong> Requiere el cálculo de la derivada y puede diverger si el punto inicial está lejos de la raíz.</p>
            <p><strong>Orden de convergencia:</strong> Cuadrático (orden 2).</p>
        </div>
        """, unsafe_allow_html=True)

# Lógica principal
if calculate:
    # Crear la función a partir de la expresión
    try:
        f, expr_sympy = create_function(selected_function)
        
        # Calcular la raíz según el método seleccionado
        if method == "Bisección":
            show_method_description(method)
            root, results = bisection(f, a, b, tol, max_iter)
            if root is not None:
                x_range = [min(a, b) - 0.5, max(a, b) + 0.5]
        
        elif method == "Regula Falsi":
            show_method_description(method)
            root, results = regula_falsi(f, a, b, tol, max_iter)
            if root is not None:
                x_range = [min(a, b) - 0.5, max(a, b) + 0.5]
        
        elif method == "Punto Fijo":
            show_method_description(method)
            g, _ = create_function(g_x)
            root, results = fixed_point(f, g, x0, tol, max_iter)
            x_range = [root - 2, root + 2]
        
        elif method == "Secante":
            show_method_description(method)
            root, results = secant(f, x0, x1, tol, max_iter)
            x_range = [min(x0, x1, root) - 1, max(x0, x1, root) + 1]
        
        elif method == "Newton-Raphson":
            show_method_description(method)
            # Calcular la derivada de la función
            x = symbols('x')
            df_expr = diff(expr_sympy, x)
            df = lambdify(x, df_expr, 'numpy')
            
            root, results = newton_raphson(f, df, x0, tol, max_iter)
            x_range = [root - 2, root + 2]
        
        # Mostrar resultados si se encontró una raíz
        if root is not None:
            # Mostrar la raíz encontrada
            st.markdown("<div class='subheader'>Resultados</div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"Raíz encontrada: x ≈ {root:.10f}")
                st.info(f"Valor de f(x) en la raíz: {f(root):.10e}")
                st.info(f"Número de iteraciones: {len(results)}")
            
            # Mostrar la gráfica de la función y las iteraciones
            st.markdown("<div class='subheader'>Visualización Gráfica</div>", unsafe_allow_html=True)
            plt_func = plot_function_and_iterations(f, method, results, root, x_range)
            st.pyplot(plt_func)
            
            # Mostrar la gráfica de convergencia del error
            st.markdown("<div class='subheader'>Convergencia del Error</div>", unsafe_allow_html=True)
            plt_error = plot_error_convergence(results)
            st.pyplot(plt_error)
            
            # Mostrar la tabla de iteraciones
            st.markdown("<div class='subheader'>Tabla de Iteraciones</div>", unsafe_allow_html=True)
            st.dataframe(results, hide_index=True, use_container_width=True, height=400)
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.error("Asegúrate de que la función esté correctamente definida y los parámetros sean válidos.")

else:
    # Mostrar descripciones de los métodos cuando no se ha calculado aún
    show_method_description(method)
    
    st.info("Configura los parámetros y presiona 'Calcular raíz' para ver los resultados.")
    
    # Mostrar ejemplo de la función
    try:
        f, _ = create_function(selected_function)
        x = np.linspace(-5, 5, 1000)
        y = np.vectorize(f)(x)
        
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'b-')
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.title(f'Gráfica de f(x) = {selected_function}')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True, alpha=0.3)
        st.pyplot(plt)
    except:
        pass

# Información adicional
with st.expander("Información sobre los métodos numéricos"):
    st.markdown("""
    ### Comparación de Métodos
    
    | Método | Orden de Convergencia | Requiere Derivada | Requiere Intervalo | Robustez |
    |--------|----------------------|-------------------|-------------------|----------|
    | Bisección | Lineal (1) | No | Sí | Alta |
    | Regula Falsi | Superlineal (1-1.618) | No | Sí | Alta |
    | Punto Fijo | Lineal (1) | No | No | Media |
    | Secante | Superlineal (1.618) | No | No (2 puntos) | Media |
    | Newton-Raphson | Cuadrática (2) | Sí | No | Baja |
    
    ### Cuándo usar cada método
    
    - **Bisección**: Cuando necesitas un método garantizado para encontrar una raíz pero no te importa la velocidad.
    - **Regula Falsi**: Cuando necesitas un método robusto pero más rápido que la bisección.
    - **Punto Fijo**: Cuando puedes encontrar fácilmente una función g(x) adecuada.
    - **Secante**: Cuando no puedes calcular fácilmente la derivada pero necesitas convergencia rápida.
    - **Newton-Raphson**: Cuando necesitas la convergencia más rápida y puedes calcular la derivada.
    """)
