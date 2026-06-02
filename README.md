# Aplicación web: Métodos de Optimización con condiciones de Wolfe

Esta aplicación resuelve problemas de minimización sin restricciones usando:

- Método del gradiente
- Método del gradiente conjugado no lineal
- Método de Newton

Los tres métodos usan búsqueda de línea con la primera condición de Wolfe y la segunda condición de Wolfe en su versión fuerte, que también satisface la condición de curvatura de Wolfe clásica para direcciones de descenso.

## Cómo ejecutar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Cómo desplegar en Streamlit Community Cloud

1. Crear un repositorio en GitHub.
2. Subir `app.py`, `requirements.txt`, `README.md` y la carpeta `.streamlit`.
3. Entrar a https://streamlit.io/cloud.
4. Crear una nueva app apuntando al repositorio.
5. En `Main file path`, seleccionar `app.py`.
6. Copiar el enlace público generado y enviarlo al profesor.

## Ejemplos de funciones

Para 2 variables:

```text
(x1 - 1)**2 + 2*(x2 + 2)**2
```

```text
100*(x2 - x1**2)**2 + (1 - x1)**2
```

```text
sin(x1) + cos(x2) + x1**2 + x2**2
```

Para 3 variables:

```text
x1**2 + 2*x2**2 + 3*x3**2 - 4*x1 + x2
```

## Formato del punto inicial

El punto inicial puede escribirse separado por comas:

```text
0, 0
```

También se aceptan corchetes:

```text
[0, 0]
```
