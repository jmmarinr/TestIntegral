import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, lagrange, interp1d
from scipy.interpolate import BarycentricInterpolator, CubicHermiteSpline

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
st.write("Selecciona una función, el rango, el número de puntos y el método de interpolación.")

# Slider para seleccionar la función
funciones = ['Seno', 'Coseno', 'Tangente', 'Exponencial']
funcion_seleccionada = st.selectbox("Selecciona una función", funciones)

# Slider para seleccionar el rango de la función
rango_min = st.slider("Selecciona el valor mínimo de x", min_value=0, max_value=20, value=0)
rango_max = st.slider("Selecciona el valor máximo de x", min_value=1, max_value=20, value=10)

# Slider para seleccionar el número de puntos
n_puntos = st.slider("Selecciona el número de puntos", min_value=3, max_value=20, value=10)

# Definir la función seleccionada
if funcion_seleccionada == 'Seno':
    f = f1
elif funcion_seleccionada == 'Coseno':
    f = f2
elif funcion_seleccionada == 'Tangente':
    f = f3
else:
    f = f4

# Generar los puntos en x según el rango y el número de puntos seleccionados
x = np.linspace(rango_min, rango_max, n_puntos)
y = f(x)

# Selección del método de interpolación
metodo = st.selectbox("Selecciona el método de interpolación", ["Lineal", "SplineCubic", "Lagrange", "Barycentric", "CubicHermite"])

# Crear los puntos de interpolación (más finos para graficar)
x_fino = np.linspace(rango_min, rango_max, 1000)

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

elif metodo == "CubicHermite":
    # Para CubicHermite necesitamos tanto los valores de la función como las derivadas
    def derivada(x, y):
        return np.gradient(y, x)
    
    dy = derivada(x, y)
    interpolador = CubicHermiteSpline(x, y, dy)
    y_interpolado = interpolador(x_fino)

# Gráfico de los puntos originales, la función original y la interpolación
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, y, 'ro', label='Puntos Originales')
ax.plot(x_fino, f(x_fino), label=f'Función Real: {funcion_seleccionada}')
ax.plot(x_fino, y_interpolado, label=f'Interpolación {metodo}')
ax.set_title(f'Interpolación con {metodo}')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend()

# Mostrar el gráfico
st.pyplot(fig)

# Calcular el error (comparando la función real y la interpolada)
y_real = f(x_fino)
error = np.abs(y_real - y_interpolado)

# Gráfico de error
fig_error, ax_error = plt.subplots(figsize=(8, 6))
ax_error.plot(x_fino, error, label='Error de la Interpolación', color='purple')
ax_error.set_title('Gráfico de Error de la Interpolación')
ax_error.set_xlabel('x')
ax_error.set_ylabel('Error')
ax_error.legend()

# Mostrar el gráfico de error
st.pyplot(fig_error)
