import ast
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import sympy as sp


# -----------------------------
# Utilidades de lectura y parseo
# -----------------------------

ALLOWED_NAMES: Dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
}


class InputError(Exception):
    """Error controlado para entradas del usuario."""


def parse_starting_point(text: str, n: int) -> np.ndarray:
    """Convierte un texto tipo '1, 2' o '[1, 2]' en vector numpy."""
    cleaned = text.strip()
    if not cleaned:
        raise InputError("Debes ingresar un punto de partida.")

    if cleaned.startswith("[") or cleaned.startswith("("):
        try:
            values = ast.literal_eval(cleaned)
        except Exception as exc:
            raise InputError("El punto de partida no tiene un formato válido.") from exc
    else:
        cleaned = cleaned.replace(";", ",")
        if "," in cleaned:
            values = [item.strip() for item in cleaned.split(",") if item.strip()]
        else:
            values = [item.strip() for item in cleaned.split() if item.strip()]

    try:
        arr = np.array([float(v) for v in values], dtype=float)
    except Exception as exc:
        raise InputError("Todos los valores del punto inicial deben ser numéricos.") from exc

    if len(arr) != n:
        raise InputError(f"El punto inicial debe tener exactamente {n} valores.")
    if not np.all(np.isfinite(arr)):
        raise InputError("El punto inicial contiene valores no finitos.")
    return arr


def build_functions(expr_text: str, n: int) -> Tuple[Callable, Callable, Callable, sp.Expr, List[sp.Symbol]]:
    """Construye función objetivo, gradiente y Hessiano desde una expresión de SymPy."""
    if not expr_text.strip():
        raise InputError("Debes ingresar una función objetivo.")

    variables = sp.symbols(" ".join([f"x{i}" for i in range(1, n + 1)]))
    if n == 1:
        variables = [variables]
    else:
        variables = list(variables)

    local_dict = {f"x{i}": variables[i - 1] for i in range(1, n + 1)}
    local_dict.update(ALLOWED_NAMES)

    try:
        expr = sp.sympify(expr_text.replace("^", "**"), locals=local_dict)
    except Exception as exc:
        raise InputError("No se pudo interpretar la función objetivo. Revisa la sintaxis.") from exc

    extra_symbols = sorted([str(s) for s in expr.free_symbols if s not in set(variables)])
    if extra_symbols:
        raise InputError(
            "La función contiene variables no declaradas: " + ", ".join(extra_symbols)
        )

    grad_expr = [sp.diff(expr, var) for var in variables]
    hess_expr = sp.hessian(expr, variables)

    f_raw = sp.lambdify(variables, expr, modules="numpy")
    g_raw = sp.lambdify(variables, grad_expr, modules="numpy")
    h_raw = sp.lambdify(variables, hess_expr, modules="numpy")

    def f(x: np.ndarray) -> float:
        try:
            val = float(np.asarray(f_raw(*x), dtype=float))
        except Exception as exc:
            raise FloatingPointError("La función no pudo evaluarse en el punto actual.") from exc
        if not np.isfinite(val):
            raise FloatingPointError("La función produjo un valor no finito.")
        return val

    def grad(x: np.ndarray) -> np.ndarray:
        try:
            val = np.asarray(g_raw(*x), dtype=float).reshape(-1)
        except Exception as exc:
            raise FloatingPointError("El gradiente no pudo evaluarse en el punto actual.") from exc
        if val.size != n:
            val = np.resize(val, n)
        if not np.all(np.isfinite(val)):
            raise FloatingPointError("El gradiente produjo valores no finitos.")
        return val

    def hess(x: np.ndarray) -> np.ndarray:
        try:
            val = np.asarray(h_raw(*x), dtype=float).reshape(n, n)
        except Exception as exc:
            raise FloatingPointError("El Hessiano no pudo evaluarse en el punto actual.") from exc
        if not np.all(np.isfinite(val)):
            raise FloatingPointError("El Hessiano produjo valores no finitos.")
        return val

    return f, grad, hess, expr, variables


# -----------------------------
# Condiciones de Wolfe
# -----------------------------

@dataclass
class WolfeResult:
    alpha: float
    success: bool
    message: str
    armijo_ok: bool
    curvature_ok: bool
    tries: int


def check_wolfe(
    f: Callable,
    grad: Callable,
    x: np.ndarray,
    p: np.ndarray,
    alpha: float,
    c1: float,
    c2: float,
    strong: bool = True,
) -> Tuple[bool, bool]:
    """Retorna si se cumple la primera y segunda condición de Wolfe."""
    phi0 = f(x)
    g0p = float(np.dot(grad(x), p))
    x_new = x + alpha * p
    phi_a = f(x_new)
    gap = float(np.dot(grad(x_new), p))

    armijo_ok = phi_a <= phi0 + c1 * alpha * g0p
    if strong:
        curvature_ok = abs(gap) <= c2 * abs(g0p)
    else:
        curvature_ok = gap >= c2 * g0p
    return armijo_ok, curvature_ok


def line_search_wolfe(
    f: Callable,
    grad: Callable,
    x: np.ndarray,
    p: np.ndarray,
    c1: float,
    c2: float,
    alpha0: float = 1.0,
    alpha_max: float = 100.0,
    max_iter: int = 40,
    max_zoom: int = 40,
) -> WolfeResult:
    """Búsqueda de línea tipo zoom con condiciones fuertes de Wolfe."""
    phi0 = f(x)
    g0 = grad(x)
    dphi0 = float(np.dot(g0, p))

    if dphi0 >= 0:
        return WolfeResult(0.0, False, "La dirección no es de descenso.", False, False, 0)

    def phi(alpha: float) -> float:
        return f(x + alpha * p)

    def dphi(alpha: float) -> float:
        return float(np.dot(grad(x + alpha * p), p))

    def zoom(alpha_lo: float, alpha_hi: float, phi_lo: float, tries_start: int) -> WolfeResult:
        tries = tries_start
        last_alpha = alpha_lo
        last_armijo = False
        last_curvature = False

        for _ in range(max_zoom):
            tries += 1
            alpha_j = 0.5 * (alpha_lo + alpha_hi)
            last_alpha = alpha_j

            try:
                phi_j = phi(alpha_j)
            except FloatingPointError:
                alpha_hi = alpha_j
                continue

            armijo_j = phi_j <= phi0 + c1 * alpha_j * dphi0
            if (not armijo_j) or phi_j >= phi_lo:
                alpha_hi = alpha_j
            else:
                try:
                    dphi_j = dphi(alpha_j)
                except FloatingPointError:
                    alpha_hi = alpha_j
                    continue

                curvature_j = abs(dphi_j) <= c2 * abs(dphi0)
                last_armijo = armijo_j
                last_curvature = curvature_j
                if curvature_j:
                    return WolfeResult(
                        alpha_j,
                        True,
                        "Se cumplieron ambas condiciones de Wolfe.",
                        True,
                        True,
                        tries,
                    )

                if dphi_j * (alpha_hi - alpha_lo) >= 0:
                    alpha_hi = alpha_lo
                alpha_lo = alpha_j
                phi_lo = phi_j

            if abs(alpha_hi - alpha_lo) < 1e-14:
                break

        armijo_ok, curvature_ok = safe_wolfe_check(f, grad, x, p, last_alpha, c1, c2)
        return WolfeResult(
            last_alpha,
            armijo_ok and curvature_ok,
            "Zoom terminó por límite de iteraciones.",
            armijo_ok,
            curvature_ok,
            tries,
        )

    alpha_prev = 0.0
    phi_prev = phi0
    alpha = min(max(float(alpha0), 1e-12), alpha_max)
    tries = 0

    for i in range(max_iter):
        tries += 1
        try:
            phi_a = phi(alpha)
        except FloatingPointError:
            alpha *= 0.5
            continue

        armijo_a = phi_a <= phi0 + c1 * alpha * dphi0
        if (not armijo_a) or (i > 0 and phi_a >= phi_prev):
            return zoom(alpha_prev, alpha, phi_prev, tries)

        try:
            dphi_a = dphi(alpha)
        except FloatingPointError:
            alpha *= 0.5
            continue

        curvature_a = abs(dphi_a) <= c2 * abs(dphi0)
        if curvature_a:
            return WolfeResult(
                alpha,
                True,
                "Se cumplieron ambas condiciones de Wolfe.",
                True,
                True,
                tries,
            )

        if dphi_a >= 0:
            return zoom(alpha, alpha_prev, phi_a, tries)

        alpha_prev = alpha
        phi_prev = phi_a
        alpha = min(alpha * 2.0, alpha_max)

    # Respaldo: backtracking que intenta encontrar Armijo y luego reporta si Wolfe se cumple.
    alpha = min(max(float(alpha0), 1e-12), alpha_max)
    last_armijo, last_curvature = False, False
    for _ in range(60):
        tries += 1
        armijo_ok, curvature_ok = safe_wolfe_check(f, grad, x, p, alpha, c1, c2)
        last_armijo, last_curvature = armijo_ok, curvature_ok
        if armijo_ok and curvature_ok:
            return WolfeResult(
                alpha,
                True,
                "Se cumplieron ambas condiciones de Wolfe mediante respaldo.",
                True,
                True,
                tries,
            )
        if armijo_ok:
            return WolfeResult(
                alpha,
                False,
                "Solo se cumplió Armijo; se usa paso de respaldo para continuar.",
                True,
                curvature_ok,
                tries,
            )
        alpha *= 0.5
        if alpha < 1e-14:
            break

    return WolfeResult(
        max(alpha, 1e-14),
        False,
        "No se logró satisfacer Wolfe; se usa un paso muy pequeño.",
        last_armijo,
        last_curvature,
        tries,
    )


def safe_wolfe_check(
    f: Callable,
    grad: Callable,
    x: np.ndarray,
    p: np.ndarray,
    alpha: float,
    c1: float,
    c2: float,
) -> Tuple[bool, bool]:
    try:
        return check_wolfe(f, grad, x, p, alpha, c1, c2, strong=True)
    except Exception:
        return False, False


# -----------------------------
# Métodos de optimización
# -----------------------------

@dataclass
class OptimizationOutput:
    x_min: np.ndarray
    f_min: float
    iterations: int
    final_error: float
    stop_reason: str
    history: pd.DataFrame
    path: List[np.ndarray]


def optimize(
    method: str,
    f: Callable,
    grad: Callable,
    hess: Callable,
    x0: np.ndarray,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
    alpha_max: float,
) -> OptimizationOutput:
    x = x0.astype(float).copy()
    n = len(x)
    history_rows = []
    path = [x.copy()]
    g = grad(x)
    p_prev = None
    g_prev = None
    stop_reason = "Se alcanzó el número máximo de iteraciones."

    for k in range(max_iter + 1):
        f_val = f(x)
        g = grad(x)
        error = float(np.linalg.norm(g))

        if k == 0:
            history_rows.append(
                {
                    "iteración": k,
                    "f(x)": f_val,
                    "error ||∇f||": error,
                    "alpha": np.nan,
                    "Wolfe 1": np.nan,
                    "Wolfe 2": np.nan,
                    "criterio": "inicio",
                }
            )
        else:
            # La fila de la iteración se agrega después del paso.
            pass

        if error <= tol:
            stop_reason = "Convergencia: la norma del gradiente es menor o igual a la tolerancia."
            break

        if k == max_iter:
            break

        # Dirección de búsqueda
        if method == "Método del gradiente":
            p = -g
        elif method == "Gradiente conjugado":
            if k == 0 or p_prev is None or g_prev is None:
                p = -g
            else:
                denom = float(np.dot(g_prev, g_prev))
                beta_pr = 0.0 if denom <= 1e-30 else float(np.dot(g, g - g_prev) / denom)
                beta = max(0.0, beta_pr)  # Polak-Ribière+ para mayor estabilidad.
                p = -g + beta * p_prev
                if float(np.dot(g, p)) >= -1e-12 * max(1.0, np.linalg.norm(g) * np.linalg.norm(p)):
                    p = -g
        elif method == "Método de Newton":
            H = hess(x)
            try:
                # Regularización suave si el Hessiano no es definido positivo o está mal condicionado.
                eig_min = float(np.min(np.linalg.eigvalsh((H + H.T) / 2.0)))
                reg = max(0.0, 1e-8 - eig_min)
                p = -np.linalg.solve(H + reg * np.eye(n), g)
            except Exception:
                p = -g
            if float(np.dot(g, p)) >= 0:
                p = -g
        else:
            raise ValueError("Método no reconocido.")

        if np.linalg.norm(p) <= 1e-15:
            stop_reason = "Criterio de parada: la dirección de búsqueda es prácticamente cero."
            break

        wolfe = line_search_wolfe(
            f=f,
            grad=grad,
            x=x,
            p=p,
            c1=c1,
            c2=c2,
            alpha0=alpha0,
            alpha_max=alpha_max,
        )

        alpha = wolfe.alpha if wolfe.alpha > 0 else min(alpha0, 1e-6)
        x_new = x + alpha * p
        step_norm = float(np.linalg.norm(x_new - x))

        g_prev = g.copy()
        p_prev = p.copy()
        x = x_new
        path.append(x.copy())

        f_new = f(x)
        g_new = grad(x)
        error_new = float(np.linalg.norm(g_new))
        history_rows.append(
            {
                "iteración": k + 1,
                "f(x)": f_new,
                "error ||∇f||": error_new,
                "alpha": alpha,
                "Wolfe 1": "sí" if wolfe.armijo_ok else "no",
                "Wolfe 2": "sí" if wolfe.curvature_ok else "no",
                "criterio": wolfe.message,
            }
        )

        if step_norm <= tol * max(1.0, np.linalg.norm(x)):
            stop_reason = "Convergencia: el cambio entre iteraciones es menor o igual a la tolerancia."
            break
        if error_new <= tol:
            stop_reason = "Convergencia: la norma del gradiente es menor o igual a la tolerancia."
            break

    history = pd.DataFrame(history_rows)
    return OptimizationOutput(
        x_min=x,
        f_min=f(x),
        iterations=int(history["iteración"].iloc[-1]),
        final_error=float(np.linalg.norm(grad(x))),
        stop_reason=stop_reason,
        history=history,
        path=path,
    )


# -----------------------------
# Visualizaciones
# -----------------------------

def plot_convergence(history: pd.DataFrame):
    fig, ax = plt.subplots()
    ax.plot(history["iteración"], history["error ||∇f||"], marker="o")
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error ||∇f(x)||")
    ax.set_title("Gráfico de convergencia")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    return fig


def plot_1d_function(f: Callable, path: List[np.ndarray]):
    xs_path = np.array([p[0] for p in path])
    center = float(xs_path[-1])
    radius = max(1.0, float(np.max(np.abs(xs_path - center))) * 1.5)
    xs = np.linspace(center - radius, center + radius, 400)
    ys = []
    for val in xs:
        try:
            ys.append(f(np.array([val])))
        except Exception:
            ys.append(np.nan)
    fig, ax = plt.subplots()
    ax.plot(xs, ys)
    ax.scatter(xs_path, [f(np.array([v])) for v in xs_path], zorder=3)
    ax.set_xlabel("x1")
    ax.set_ylabel("f(x1)")
    ax.set_title("Función y puntos visitados")
    ax.grid(True, alpha=0.3)
    return fig


def plot_2d_contour(f: Callable, path: List[np.ndarray]):
    pts = np.array(path)
    center = pts[-1]
    span_x = max(1.0, float(np.ptp(pts[:, 0])) * 1.5)
    span_y = max(1.0, float(np.ptp(pts[:, 1])) * 1.5)
    x_vals = np.linspace(center[0] - span_x, center[0] + span_x, 120)
    y_vals = np.linspace(center[1] - span_y, center[1] + span_y, 120)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
            except Exception:
                Z[i, j] = np.nan

    fig, ax = plt.subplots()
    if np.isfinite(Z).any():
        finite_z = Z[np.isfinite(Z)]
        levels = np.linspace(np.nanmin(finite_z), np.nanpercentile(finite_z, 95), 25)
        ax.contour(X, Y, Z, levels=levels)
    ax.plot(pts[:, 0], pts[:, 1], marker="o")
    ax.scatter([pts[-1, 0]], [pts[-1, 1]], s=80, zorder=4)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Valor agregado: trayectoria sobre curvas de nivel")
    ax.grid(True, alpha=0.3)
    return fig




def plot_method_comparison(results_by_method: Dict[str, OptimizationOutput]):
    """Grafica la convergencia de varios métodos en un mismo gráfico."""
    fig, ax = plt.subplots()
    for method_name, result in results_by_method.items():
        hist = result.history.copy()
        ax.plot(hist["iteración"], hist["error ||∇f||"], marker="o", label=method_name)
    ax.set_xlabel("Iteración")
    ax.set_ylabel("Error ||∇f(x)||")
    ax.set_title("Valor agregado: comparación de convergencia entre métodos")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def build_comparison_table(results_by_method: Dict[str, OptimizationOutput]) -> pd.DataFrame:
    """Construye una tabla resumen para comparar los métodos."""
    rows = []
    for method_name, result in results_by_method.items():
        rows.append(
            {
                "método": method_name,
                "f(x*)": result.f_min,
                "iteraciones": result.iterations,
                "error final": result.final_error,
                "punto encontrado": np.array2string(result.x_min, precision=6, separator=", "),
                "criterio de parada": result.stop_reason,
            }
        )
    return pd.DataFrame(rows).sort_values(
        by=["error final", "iteraciones"], ascending=[True, True]
    )


def automatic_diagnosis(result: OptimizationOutput, tol: float, max_iter: int) -> str:
    """Genera una conclusión breve a partir del resultado numérico."""
    if result.final_error <= tol:
        quality = "El método convergió correctamente según la tolerancia solicitada."
    elif result.iterations >= max_iter:
        quality = "El método alcanzó el máximo de iteraciones antes de cumplir la tolerancia."
    else:
        quality = "El método se detuvo por un criterio alternativo de parada."

    if result.iterations <= max(5, 0.1 * max_iter):
        speed = "La convergencia fue rápida para la configuración usada."
    elif result.iterations <= max(20, 0.5 * max_iter):
        speed = "La convergencia fue moderada."
    else:
        speed = "La convergencia fue lenta o necesitó muchas iteraciones."

    return quality + " " + speed


def build_text_report(
    method: str,
    n_vars: int,
    function_text: str,
    x0_text: str,
    max_iter: int,
    tol: float,
    c1: float,
    c2: float,
    alpha0: float,
    alpha_max: float,
    result: OptimizationOutput,
) -> str:
    """Crea un reporte descargable en texto plano."""
    lines = [
        "REPORTE DE OPTIMIZACIÓN CON CONDICIONES DE WOLFE",
        "",
        "DATOS DE ENTRADA",
        f"Número de variables: {n_vars}",
        f"Método: {method}",
        f"Función objetivo: {function_text}",
        f"Punto de partida: {x0_text}",
        f"Máximo de iteraciones: {max_iter}",
        f"Tolerancia: {tol}",
        f"c1: {c1}",
        f"c2: {c2}",
        f"alpha inicial: {alpha0}",
        f"alpha máximo: {alpha_max}",
        "",
        "RESULTADOS",
        f"Punto mínimo encontrado: {np.array2string(result.x_min, precision=10, separator=', ')}",
        f"Valor de la función objetivo: {result.f_min:.12g}",
        f"Iteraciones realizadas: {result.iterations}",
        f"Error final: {result.final_error:.12e}",
        f"Criterio de parada: {result.stop_reason}",
        "",
        "DIAGNÓSTICO AUTOMÁTICO",
        automatic_diagnosis(result, tol, max_iter),
    ]
    return "\n".join(lines)


# -----------------------------
# Interfaz Streamlit
# -----------------------------

st.set_page_config(
    page_title="Optimización con Wolfe",
    page_icon="📉",
    layout="wide",
)

st.title("Aplicación web de Métodos de Optimización")
st.caption("Gradiente, Gradiente Conjugado y Newton con condiciones de Wolfe")

with st.sidebar:
    st.header("Datos de entrada")
    n_vars = st.number_input("Número de variables", min_value=1, max_value=10, value=2, step=1)
    method = st.selectbox(
        "Método de optimización",
        ["Método del gradiente", "Gradiente conjugado", "Método de Newton"],
    )

    default_function = "100*(x2 - x1**2)**2 + (1 - x1)**2" if n_vars == 2 else " + ".join([f"x{i}**2" for i in range(1, n_vars + 1)])
    function_text = st.text_area(
        "Función objetivo f(x)",
        value=default_function,
        help="Usa variables x1, x2, ..., xn. Puedes usar sin, cos, exp, log, sqrt, pi.",
    )

    default_x0 = "-1.2, 1" if n_vars == 2 else ", ".join(["1" for _ in range(n_vars)])
    x0_text = st.text_input("Punto de partida", value=default_x0)
    max_iter = st.number_input("Número máximo de iteraciones", min_value=1, max_value=5000, value=200, step=10)
    tol = st.number_input("Tolerancia de convergencia", min_value=1e-12, max_value=1e-1, value=1e-6, format="%.1e")

    st.subheader("Parámetros de Wolfe")
    c1 = st.number_input("c1, primera condición de Wolfe", min_value=1e-8, max_value=0.49, value=1e-4, format="%.1e")
    c2 = st.number_input("c2, segunda condición de Wolfe", min_value=0.50, max_value=0.999, value=0.90, format="%.3f")
    alpha0 = st.number_input("alpha inicial", min_value=1e-12, max_value=100.0, value=1.0, format="%.4f")
    alpha_max = st.number_input("alpha máximo", min_value=alpha0, max_value=1000.0, value=100.0, format="%.1f")

    st.subheader("Valores agregados")
    compare_methods = st.checkbox("Comparar automáticamente los 3 métodos", value=False)

    run_button = st.button("Ejecutar método", type="primary", use_container_width=True)

st.info(
    "Sintaxis: escribe la función usando x1, x2, ..., xn. Ejemplo: "
    "`100*(x2 - x1**2)**2 + (1 - x1)**2`."
)

if run_button:
    try:
        if not (0 < c1 < c2 < 1):
            raise InputError("Los parámetros deben cumplir 0 < c1 < c2 < 1.")

        x0 = parse_starting_point(x0_text, int(n_vars))
        f, grad, hess, expr, variables = build_functions(function_text, int(n_vars))

        with st.spinner("Ejecutando optimización..."):
            result = optimize(
                method=method,
                f=f,
                grad=grad,
                hess=hess,
                x0=x0,
                max_iter=int(max_iter),
                tol=float(tol),
                c1=float(c1),
                c2=float(c2),
                alpha0=float(alpha0),
                alpha_max=float(alpha_max),
            )

        st.success("Optimización finalizada")
        st.subheader("Resultados esperados")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valor de f(x*)", f"{result.f_min:.8g}")
        col2.metric("Iteraciones", result.iterations)
        col3.metric("Error final", f"{result.final_error:.3e}")
        col4.metric("Método", method)

        st.markdown("**Punto mínimo encontrado:**")
        point_df = pd.DataFrame(
            {"variable": [str(v) for v in variables], "valor": result.x_min}
        )
        st.dataframe(point_df, use_container_width=True, hide_index=True)

        st.markdown("**Criterio de parada alcanzado:**")
        st.write(result.stop_reason)

        st.markdown("**Función interpretada por el sistema:**")
        st.latex(sp.latex(expr))

        st.subheader("Gráfico de convergencia: error versus número de iteraciones")
        st.pyplot(plot_convergence(result.history), clear_figure=True)

        st.subheader("Tabla de iteraciones")
        st.dataframe(result.history, use_container_width=True, hide_index=True)

        csv = result.history.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar historial en CSV",
            data=csv,
            file_name="historial_optimizacion.csv",
            mime="text/csv",
        )

        st.subheader("Valor agregado: diagnóstico automático")
        st.write(automatic_diagnosis(result, float(tol), int(max_iter)))

        report_text = build_text_report(
            method=method,
            n_vars=int(n_vars),
            function_text=function_text,
            x0_text=x0_text,
            max_iter=int(max_iter),
            tol=float(tol),
            c1=float(c1),
            c2=float(c2),
            alpha0=float(alpha0),
            alpha_max=float(alpha_max),
            result=result,
        )
        st.download_button(
            "Descargar reporte completo en TXT",
            data=report_text.encode("utf-8"),
            file_name="reporte_optimizacion.txt",
            mime="text/plain",
        )

        if compare_methods:
            st.subheader("Valor agregado: comparación automática entre métodos")
            comparison_results = {}
            for method_to_compare in ["Método del gradiente", "Gradiente conjugado", "Método de Newton"]:
                try:
                    comparison_results[method_to_compare] = optimize(
                        method=method_to_compare,
                        f=f,
                        grad=grad,
                        hess=hess,
                        x0=x0,
                        max_iter=int(max_iter),
                        tol=float(tol),
                        c1=float(c1),
                        c2=float(c2),
                        alpha0=float(alpha0),
                        alpha_max=float(alpha_max),
                    )
                except Exception as exc:
                    st.warning(f"No se pudo ejecutar {method_to_compare}: {exc}")

            if comparison_results:
                comparison_table = build_comparison_table(comparison_results)
                st.dataframe(comparison_table, use_container_width=True, hide_index=True)
                st.pyplot(plot_method_comparison(comparison_results), clear_figure=True)
                best_row = comparison_table.iloc[0]
                st.info(
                    "Método recomendado para esta función: "
                    f"{best_row['método']}, porque obtuvo el menor error final "
                    "y, en caso de empate, menos iteraciones."
                )

        st.subheader("Valor agregado")
        if int(n_vars) == 1:
            st.write("Se muestra la función en una dimensión junto con los puntos visitados por el método.")
            st.pyplot(plot_1d_function(f, result.path), clear_figure=True)
        elif int(n_vars) == 2:
            st.write("Se muestra la trayectoria del algoritmo sobre curvas de nivel de la función objetivo.")
            st.pyplot(plot_2d_contour(f, result.path), clear_figure=True)
        else:
            st.write(
                "Para más de dos variables, el valor agregado consiste en la tabla completa de iteraciones, "
                "la verificación de Wolfe por paso y la descarga del historial en CSV."
            )

        with st.expander("Detalles de implementación"):
            st.write(
                "El error reportado corresponde a la norma euclidiana del gradiente, ||∇f(x)||. "
                "La búsqueda de línea verifica Armijo y curvatura fuerte de Wolfe en cada paso. "
                "Si Newton genera una dirección que no es de descenso, la aplicación usa descenso por gradiente como respaldo."
            )

    except InputError as exc:
        st.error(str(exc))
    except FloatingPointError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Ocurrió un error inesperado durante la ejecución.")
        st.exception(exc)
else:
    st.subheader("Qué hace esta aplicación")
    st.write(
        "La aplicación recibe el número de variables, el método, la función objetivo, "
        "el punto de partida, el máximo de iteraciones, la tolerancia y los parámetros de Wolfe. "
        "Luego muestra el punto mínimo encontrado, el valor objetivo, el número de iteraciones, "
        "el error final, el criterio de parada, el gráfico de convergencia y un valor agregado visual o exportable."
    )

    st.subheader("Prueba rápida sugerida")
    st.code(
        "Número de variables: 2\n"
        "Función: 100*(x2 - x1**2)**2 + (1 - x1)**2\n"
        "Punto de partida: -1.2, 1\n"
        "Método: Método de Newton\n"
        "Tolerancia: 1e-6",
        language="text",
    )


# Valores agregados nuevos implementados: comparación automática, diagnóstico y reporte TXT.
