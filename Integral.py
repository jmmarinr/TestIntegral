import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Funciones de ejemplo
def f1(x):
    return np.sin(x)

def f2(x):
    return x**2

def f3(x):
    return np.exp(-x)

# Diccionario de funciones
funciones = {
    "f(x) = sin(x)": f1,
    "f(x) = x^2": f2,
    "f(x) = exp(-x)": f3,
}

# Métodos de integración
def punto_medio(f, a, b, n):
    h = (b - a) / n
    puntos = np.linspace(a + h / 2, b - h / 2, n)
    aproximacion = h * np.sum(f(puntos))
    return aproximacion, puntos

def trapecio(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    aproximacion = h * (np.sum(y) - 0.5 * (y[0] + y[-1]))
    return aproximacion, x

def simpson(f, a, b, n):
    if n % 2 == 1:  # Ajustar para que n sea par
        n += 1
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    aproximacion = h / 3 * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]) + y[-1])
    return aproximacion, x

# Función para graficar
def graficar_metodos(funcion, metodo, a, b, n):
    f = funciones[funcion]
    x_f = np.linspace(a, b, 500)
    y_f = f(x_f)
    
    plt.plot(x_f, y_f, label=f"f(x) = {funcion.split('=')[1].strip()}", color="blue")
    
    # Selección del método
    if metodo == "Punto Medio":
        aproximacion, puntos = punto_medio(f, a, b, n)
        h = (b - a) / n
        for xi in puntos:
            plt.plot([xi - h / 2, xi + h / 2], [f(xi), f(xi)], color="red", label="Punto Medio" if xi == puntos[0] else "")
            plt.fill_between([xi - h / 2, xi + h / 2], [f(xi), f(xi)], color="red", alpha=0.3)
        plt.scatter(puntos, f(puntos), color="black", label="Nodos")
    elif metodo == "Trapecio":
        aproximacion, puntos = trapecio(f, a, b, n)
        for i in range(len(puntos) - 1):
            plt.plot([puntos[i], puntos[i + 1]], [f(puntos[i]), f(puntos[i + 1])], color="orange", label="Trapecio" if i == 0 else "")
            plt.fill_between([puntos[i], puntos[i + 1]], [f(puntos[i]), f(puntos[i + 1])], color="orange", alpha=0.3)
        plt.scatter(puntos, f(puntos), color="black", label="Nodos")
    elif metodo == "Simpson":
        aproximacion, puntos = simpson(f, a, b, n)
        plt.scatter(puntos, f(puntos), color="black", label="Nodos")
        for i in range(0, len(puntos) - 2, 2):
            px = np.linspace(puntos[i], puntos[i + 2], 100)
            py = np.polyval(np.polyfit(puntos[i:i + 3], f(puntos[i:i + 3]), 2), px)
            plt.plot(px, py, color="purple", label="Parábolas (Simpson)" if i == 0 else "")
            plt.fill_between(px, py, color="purple", alpha=0.3)
    
    # Gráfica principal
    plt.title(f"Método: {metodo} | Integral Aproximada: {aproximacion:.5f}")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)  # Usar st.pyplot para mostrar el gráfico en Streamlit

# Interfaz interactiva de Streamlit
def main():
    st.title("Métodos de Integración Numérica")
    
    funcion = st.selectbox("Selecciona la función:", list(funciones.keys()))
    metodo = st.selectbox("Selecciona el método de integración:", ["Punto Medio", "Trapecio", "Simpson"])
    
    a = st.slider("Límite Inferior", -10, 10, 0)
    b = st.slider("Límite Superior", -10, 10, 3)
    n = st.slider("Número de Nodos", 2, 50, 4, step=2)
    
    graficar_metodos(funcion, metodo, a, b, n)

if __name__ == "__main__":
    main()
