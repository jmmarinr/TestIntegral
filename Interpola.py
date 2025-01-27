import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange, interp1d
from scipy.interpolate import BarycentricInterpolator

# Funciones a utilizar
def f1(x):
    return np.sin(x)

def f2(x):
    return np.cos(x)

def f3(x):
    return np.tan(x)

def f4(x):
    return np.exp(x)

# Configuración inicial
st.title("Interpolación de Funciones")
st.write("Selecciona una función, el número de puntos y el método de interpolación.")

# Slider para seleccionar la función
funciones = ['Seno', 'Coseno', 'Tangente', 'Exponencial']
funcion_seleccionada = st.selectbox("Selecciona una función", funciones)

# Entrada para el número de puntos
n_puntos = st.slider("Selecciona el número de puntos", min_value=5, max_value=50, value=10)

# Definir la función seleccionada
if funcion_seleccionada == 'Seno':
    f = f1
elif funcion_seleccionada == 'Coseno':
    f = f2
elif funcion_seleccionada == 'Tangente':
    f = f3
else:
    f = f4

# Generar los puntos en x
x = np.linspace(0, 10, n_puntos)
y = f(x)

# Selección del método de interpolación
metodo = st.selectbox("Selecciona el método de interpolación", ["Lineal", "SplineCubic", "Lagrange", "Barycentric", "Hermite"])

# Crear los puntos de interpolación (más finos para graficar)
x_fino = np.linspace(0, 10, 1000)

# Interpolación dependiendo del método seleccionado
if metodo == "Lineal":
    interpolador = interp1d(x, y, kind='linear')
    y_interpolado = interpolador(x_fino)
    
elif metodo == "SplineCubic":
    interpolador = CubicSpline(x, y)
    y_interpolado = interpolador(x_fino)

elif metodo == "Lagrange":
    interpolador = lagrange(x, y)
    y_interpolado = interpolador(x_fino)

elif metodo == "Barycentric":
    interpolador = BarycentricInterpolator(x, y)
    y_interpolado = interpolador(x_fino)

elif metodo == "Hermite":
    # Para Hermite necesitamos derivadas, las calcularemos con las diferencias finitas
    def derivada(x, y):
        return np.gradient(y, x)
    
    dy = derivada(x, y)
    interpolador = CubicSpline(x, y, bc_type='natural', extrapolate=True)
    y_interpolado = interpolador(x_fino)

# Gráfico de los puntos originales y la interpolación
fig, ax = plt.subplots()
ax.plot(x, y, 'ro', label='Puntos Originales')
ax.plot(x_fino, y_interpolado, label=f'Interpolación {metodo}')
ax.set_title(f'Interpolación con {metodo}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Mostrar el gráfico
st.pyplot(fig)

# Calcular el error (si la función es continua y conocida, podemos usarla para calcular el error)
y_real = f(x_fino)
error = np.abs(y_real - y_interpolado)

# Mostrar el error
st.write(f"Error máximo de la interpolación: {np.max(error):.4f}")
