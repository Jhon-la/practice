# -*- coding: utf-8 -*-
"""
EJEMPLO DIDACTICO: PERCEPTRON SIN SEPARABILIDAD
================================================
'Esto es un nuevo ejemplo' 'nuevo ejemplo'
Objetivo
hola..
--------
Comprender por que el perceptron de Rosenblatt puede seguir corrigiendo
sus pesos cuando NO existe una recta que separe todos los ejemplos.
Comparamos:
    1) Clasico: usa los ultimos pesos calculados.
    2) Pocket: conserva los pesos con menos errores de entrenamiento.
    3) Averaged: promedia los pesos visitados durante el aprendizaje.

Se generan SIETE FIGURAS:
  1. Datos no separables (incluye un par contradictorio).
  2. Frontera del perceptron clasico en cuatro epocas.
  3. Comparacion de tres fronteras finales.
  4. Actualizaciones por epoca y errores globales por epoca.
  5. Evolucion de w1, w2 y b en las tres estrategias.
  6. Errores globales despues de CADA correccion: clasico frente a Pocket.
  7. Ejemplo real en el que corregir un punto aumenta los errores globales.

Dependencias: numpy, matplotlib. Compatible con Spyder/Anaconda.

Para probar:
  - Ejecuta el archivo completo (F5 en Spyder).
  - Las siete graficas aparecen en el panel Graficos o en ventanas.
  - Si deseas archivos PNG, cambia GUARDAR_FIGURAS a True.
  - No se requieren scikit-learn ni archivos externos.

Importante: Pocket y Averaged NO cambian la regla basica de Rosenblatt.
Cambia la manera de obtener el modelo que usamos al terminar.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================================
# 0. PARAMETROS DEL EXPERIMENTO: CAMBIA ESTOS VALORES EN CLASE
# ============================================================================
SEMILLA = 14
PUNTOS_POR_CLASE = 45      # Se suman dos puntos contradictorios al final.
ETAPAS = 30         # Numero de epocas que vamos a ejecutar.
ETA = 0.20                 # Tasa de aprendizaje de Rosenblatt.
ETIQUETAS_INVERTIDAS = 10 # Etiquetas cambiadas para crear solapamiento.
BARAJAR_EN_CADA_EPOCA = True
GUARDAR_FIGURAS = False    # True: tambien escribe los PNG en disco.

CARPETA_SALIDA = Path(__file__).resolve().parent / "figuras_perceptron"
COLOR_POS = "#2A78D6"      # Clase +1: azul.
COLOR_NEG = "#EB6834"      # Clase -1: naranja.
COLOR_CLASICO = "#24384C"
COLOR_POCKET = "#0D947F"
COLOR_MEDIA = "#A053B6"


# ============================================================================
# 1. CONSTRUIR LOS DATOS
# ============================================================================
def crear_datos(semilla, puntos_por_clase, etiquetas_invertidas):
    """Crea un problema deliberadamente NO separable en dos dimensiones.

    Hay dos nubes de puntos: una azul (y = +1) y otra naranja (y = -1).
    Ademas, invertimos algunas etiquetas e insertamos dos puntos EXACTAMENTE
    iguales, uno de cada clase.

    La ultima decision es crucial: ¡no puede existir una recta que clasifique
    correctamente dos observaciones identicas con etiquetas opuestas!
    De esta forma la no separabilidad queda demostrada, no solo sugerida.
    """
    rng = np.random.default_rng(semilla)

    # Cada fila son dos coordenadas: (x1, x2).
    positivos = rng.normal(loc=(1.1, 0.7), scale=0.65,
                           size=(puntos_por_clase, 2))
    negativos = rng.normal(loc=(-1.1, -0.7), scale=0.65,
                           size=(puntos_por_clase, 2))

    X = np.vstack([positivos, negativos])
    y = np.r_[np.ones(puntos_por_clase), -np.ones(puntos_por_clase)]

    # Cambiamos algunas etiquetas para introducir ruido de clasificacion.
    indices = rng.choice(len(y), size=etiquetas_invertidas, replace=False)
    y[indices] *= -1

    # DOS EJEMPLOS CONTRADICTORIOS: misma X, dos etiquetas distintas.
    punto_repetido = np.array([0.0, 0.1])
    X = np.vstack([X, punto_repetido, punto_repetido])
    y = np.r_[y, 1, -1]

    # Mezclamos todo, para no presentar primero solo una de las clases.
    orden = rng.permutation(len(y))
    return X[orden], y[orden], punto_repetido


# ============================================================================
# 2. OPERACIONES BASICAS DEL PERCEPTRON
# ============================================================================
def puntuacion(X, w, b):
    """Calcula z = w1*x1 + w2*x2 + b para un punto o muchos puntos."""
    return X @ w + b


def numero_errores(X, y, w, b):
    """Cuenta los patrones que no tienen el signo correcto.

    Se considera que z = 0 tambien necesita correccion, exactamente como
    en la condicion original de Rosenblatt: y*z <= 0.
    """
    return int(np.count_nonzero(y * puntuacion(X, w, b) <= 0))


def entrenar(X, y, epocas, eta, semilla, barajar=True):
    """ENTRENAMIENTO ONLINE + POCKET + AVERAGED EN UN MISMO BUCLE.

    Se usa UNA SOLA secuencia de actualizaciones de Rosenblatt:
         si y_i*(w·x_i+b) <= 0:
             w = w + eta*y_i*x_i
             b = b + eta*y_i

    CLASICO: al final devolvemos (w,b), es decir, LOS ULTIMOS pesos.

    POCKET: despues de cada correccion contamos los errores SOBRE TODOS los
    puntos de entrenamiento. Si son menos que el mejor registro anterior,
    guardamos una COPIA de esos pesos. No se usan los pesos Pocket para
    continuar aprendiendo: es una 'foto' del mejor modelo visitado.

    AVERAGED: sumamos los pesos AL TERMINAR CADA PRESENTACION de un patron,
    haya habido correccion o no. Despues dividimos por el numero de pasos.
    Por eso el promedio pondera mas los estados que duran mas tiempo.

    Devuelve un diccionario con los modelos y las historias para las figuras.
    """
    rng = np.random.default_rng(semilla + 1000)

    w = np.zeros(2, dtype=float)  # Al principio: w1 = w2 = 0.
    b = 0.0                       # Al principio: sesgo igual a 0.

    mejor_w = w.copy()            # ¡.copy() evita guardar una referencia viva!
    mejor_b = b
    mejor_error = numero_errores(X, y, w, b)

    suma_w = np.zeros(2)
    suma_b = 0.0
    pasos = 0

    # Historiales para poder dibujar mas tarde SIN reentrenar el modelo.
    actualizaciones_epoca = []
    errores_clasico_epoca = []
    errores_pocket_epoca = []
    errores_media_epoca = []
    pesos_clasico_epoca = []
    pesos_pocket_epoca = []
    pesos_media_epoca = []

    # Cada entrada guarda el numero de errores DESPUES de una actualizacion.
    errores_por_correccion = [mejor_error]
    mejores_por_correccion = [mejor_error]

    ejemplo_que_empeora = None

    for epoca in range(epocas):
        num_correcciones = 0
        orden = rng.permutation(len(y)) if barajar else np.arange(len(y))

        for i in orden:
            xi = X[i]    # Punto actual: (x1, x2).
            yi = y[i]    # Etiqueta verdadera: +1 o -1.
            z = float(w @ xi + b)

            # Regla EXACTA de Rosenblatt: solo actualizar si y*z <= 0.
            if yi * z <= 0:
                errores_antes = numero_errores(X, y, w, b)
                w_antes = w.copy()
                b_antes = b

                w += eta * yi * xi    # Dos pesos: uno por variable.
                b += eta * yi         # El sesgo tambien se aprende.
                num_correcciones += 1

                errores_despues = numero_errores(X, y, w, b)
                errores_por_correccion.append(errores_despues)

                # POCKET: guardar solo si MEJORA la exactitud global.
                if errores_despues < mejor_error:
                    mejor_error = errores_despues
                    mejor_w = w.copy()
                    mejor_b = float(b)
                mejores_por_correccion.append(mejor_error)

                # Registrar una correccion que empeore la clasificacion global.
                # Buscamos un caso especialmente didactico: el punto que
                # provoco la correccion pasa al lado CORRECTO de la recta,
                # pero al hacerlo empeoran otros puntos del conjunto.
                # Preferimos un momento en que el modelo ya haya aprendido
                # parcialmente (menos del 40% de errores). Este suceso
                # procede del entrenamiento, NO es inventado.
                if (ejemplo_que_empeora is None
                        and errores_antes < 0.40 * len(y)
                        and errores_despues >= errores_antes + 3
                        and yi * float(w @ xi + b) > 0):
                    ejemplo_que_empeora = {
                        "x": xi.copy(), "y": float(yi),
                        "w_antes": w_antes, "b_antes": b_antes,
                        "w_despues": w.copy(), "b_despues": float(b),
                        "errores_antes": errores_antes,
                        "errores_despues": errores_despues,
                        "score_antes": z, "score_despues": float(w @ xi + b),
                    }

            # AVERAGED: sumamos los pesos tras procesar CADA observacion.
            # Se acumulan incluso cuando esta observacion no exigio correccion.
            suma_w += w
            suma_b += b
            pasos += 1

        # Al final de una epoca, guardamos el estado de cada estrategia.
        w_media = suma_w / pasos
        b_media = suma_b / pasos
        actualizaciones_epoca.append(num_correcciones)
        errores_clasico_epoca.append(numero_errores(X, y, w, b))
        errores_pocket_epoca.append(numero_errores(X, y, mejor_w, mejor_b))
        errores_media_epoca.append(numero_errores(X, y, w_media, b_media))
        pesos_clasico_epoca.append((*w.copy(), float(b)))
        pesos_pocket_epoca.append((*mejor_w.copy(), float(mejor_b)))
        pesos_media_epoca.append((*w_media.copy(), float(b_media)))

        # En este experimento no hay separacion perfecta (puntos duplicados).
        # Por eso NO se sale del bucle por convergencia: se completa el numero
        # de epocas fijado por el usuario para poder estudiar la oscilacion.

    return {
        "clasico": (w.copy(), float(b)),
        "pocket": (mejor_w.copy(), float(mejor_b)),
        "averaged": (suma_w / pasos, float(suma_b / pasos)),
        "actualizaciones_epoca": actualizaciones_epoca,
        "errores_epoca": {
            "Clasico": errores_clasico_epoca,
            "Pocket": errores_pocket_epoca,
            "Averaged": errores_media_epoca,
        },
        "pesos_epoca": {
            "Clasico": np.asarray(pesos_clasico_epoca),
            "Pocket": np.asarray(pesos_pocket_epoca),
            "Averaged": np.asarray(pesos_media_epoca),
        },
        "errores_por_correccion": errores_por_correccion,
        "mejor_por_correccion": mejores_por_correccion,
        "ejemplo_que_empeora": ejemplo_que_empeora,
    }


# ============================================================================
# 3. FUNCIONES DE DIBUJO: MISMO ESTILO EN TODAS LAS FIGURAS
# ============================================================================
def dibujar_puntos(ax, X, y, punto_repetido=None):
    """Coloca las dos clases con colores y marcas diferentes."""
    for clase, color, marca in [(+1, COLOR_POS, "o"), (-1, COLOR_NEG, "s")]:
        mask = y == clase
        ax.scatter(X[mask, 0], X[mask, 1], s=35, c=color,
                   marker=marca, edgecolors="white", linewidths=0.5,
                   alpha=0.9, label=f"Clase {clase:+d}")

    if punto_repetido is not None:
        ax.scatter([punto_repetido[0]], [punto_repetido[1]], s=240,
                   facecolors="none", edgecolors="black", linewidths=2,
                   zorder=5)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.grid(alpha=0.18)
    ax.set_aspect("equal", adjustable="box")


def ajustar_limites(ax, X):
    """Usar los mismos limites ayuda a comparar las rectas."""
    ax.set_xlim(float(X[:, 0].min() - 0.55), float(X[:, 0].max() + 0.55))
    ax.set_ylim(float(X[:, 1].min() - 0.55), float(X[:, 1].max() + 0.55))


def dibujar_frontera(ax, w, b, X, color="black", estilo="-", etiqueta=None):
    """Representa w1*x1 + w2*x2 + b = 0.

    Si w2 casi vale cero, la recta es vertical: x1 = -b/w1.
    Esto evita divisiones por cero en los primeros instantes de aprendizaje.
    """
    x_min, x_max = float(X[:, 0].min() - 0.6), float(X[:, 0].max() + 0.6)
    if abs(w[1]) > 1e-10:
        xx = np.array([x_min, x_max])
        yy = -(w[0] * xx + b) / w[1]
        ax.plot(xx, yy, linestyle=estilo, color=color, linewidth=2,
                label=etiqueta)
    elif abs(w[0]) > 1e-10:
        ax.axvline(-b / w[0], linestyle=estilo, color=color, linewidth=2,
                   label=etiqueta)
    # Si w = (0,0), la frontera todavia no esta definida: no se dibuja.


def presentar(fig, nombre):
    """Decide si solo mostrar la figura o tambien guardarla como PNG."""
    fig.tight_layout()
    if GUARDAR_FIGURAS:
        CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)
        ruta = CARPETA_SALIDA / nombre
        fig.savefig(ruta, dpi=180, bbox_inches="tight")
        print(f"Guardada: {ruta}")


# ============================================================================
# 4. CREAR LAS SIETE GRAFICAS
# ============================================================================
def crear_figuras(X, y, punto_repetido, r):
    """Construye todas las figuras y las deja listas para plt.show()."""
    estilos = {
        "Clasico": COLOR_CLASICO,
        "Pocket": COLOR_POCKET,
        "Averaged": COLOR_MEDIA,
    }
    n = len(y)

    # FIGURA 1: LOS DATOS (no pueden separarse perfectamente).
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    dibujar_puntos(ax, X, y, punto_repetido)
    ajustar_limites(ax, X)
    ax.annotate("Dos puntos identicos con clases opuestas",
                xy=punto_repetido, xytext=(-2.7, 1.7),
                arrowprops=dict(arrowstyle="->", color="black"), fontsize=10)
    ax.set_title("Figura 1. Datos NO separables: hay contradicciones")
    ax.legend(loc="lower right")
    presentar(fig, "01_datos_no_separables.png")

    # FIGURA 2: CUATRO FOTOS de la recta en diferentes epocas.
    indices = sorted(set([0, 2, min(9, ETAPAS - 1), ETAPAS - 1]))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.6))
    for ax, k in zip(axes.flat, indices):
        dibujar_puntos(ax, X, y)
        wk = r["pesos_epoca"]["Clasico"][k]
        dibujar_frontera(ax, wk[:2], wk[2], X, COLOR_CLASICO)
        ajustar_limites(ax, X)
        errores = r["errores_epoca"]["Clasico"][k]
        ax.set_title(f"Epoca {k+1}: {errores} errores globales")
    for ax in axes.flat[len(indices):]:
        ax.axis("off")
    fig.suptitle("Figura 2. La frontera clasica oscila entre epocas", fontsize=14)
    presentar(fig, "02_oscilacion_por_epocas.png")

    # FIGURA 3: LAS TRES FRONTERAS FINALES, EN LAS MISMAS CONDICIONES.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, nombre in zip(axes, ["Clasico", "Pocket", "Averaged"]):
        dibujar_puntos(ax, X, y)
        w, b = r[nombre.lower() if nombre != "Clasico" else "clasico"]
        dibujar_frontera(ax, w, b, X, estilos[nombre])
        ajustar_limites(ax, X)
        errores = numero_errores(X, y, w, b)
        ax.set_title(f"{nombre}: {errores}/{n} errores\n"
                     f"Exactitud = {100*(1-errores/n):.1f}%")
    fig.suptitle("Figura 3. El mismo entrenamiento, tres modelos finales",
                 fontsize=14)
    presentar(fig, "03_comparacion_fronteras.png")

    # FIGURA 4: ACTUALIZACIONES y ERRORES GLOBALES NO SON LO MISMO.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    epocas = np.arange(1, ETAPAS + 1)
    ax1.bar(epocas, r["actualizaciones_epoca"], color=COLOR_POS)
    ax1.set_title("Actualizaciones hechas en cada epoca")
    ax1.set_xlabel("Epoca")
    ax1.set_ylabel("Veces que se cambiaron w y b")
    ax2.set_title("Errores al evaluar TODOS los puntos")
    for nombre, serie in r["errores_epoca"].items():
        ax2.plot(epocas, serie, "-o", markersize=3.0,
                 color=estilos[nombre], label=nombre)
    ax2.set_xlabel("Epoca")
    ax2.set_ylabel("Numero de errores de entrenamiento")
    ax2.legend()
    fig.suptitle("Figura 4. El perceptron no termina en cero errores",
                 fontsize=14)
    presentar(fig, "04_errores_por_epoca.png")

    # FIGURA 5: EVOLUCION DE LOS PARAMETROS w1, w2 y b.
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.1))
    for j, (ax, parametro) in enumerate(zip(axes, ["w1", "w2", "b"])):
        for nombre, pesos in r["pesos_epoca"].items():
            ax.plot(epocas, pesos[:, j], color=estilos[nombre],
                    linewidth=1.8, label=nombre)
        ax.set_title(parametro)
        ax.set_xlabel("Epoca")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Valor del parametro")
    axes[-1].legend(loc="best")
    fig.suptitle("Figura 5. Los pesos clasicos siguen cambiando",
                 fontsize=14)
    presentar(fig, "05_evolucion_pesos.png")

    # FIGURA 6: LOS ERRORES PUEDEN SUBIR; EL MEJOR DE POCKET NUNCA SUBE.
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(r["errores_por_correccion"], color=COLOR_CLASICO, lw=1.1,
            alpha=0.9, label="Clasico: errores en cada momento")
    ax.plot(r["mejor_por_correccion"], color=COLOR_POCKET, lw=2.3,
            label="Pocket: minimo historico")
    ax.set_xlabel("Numero acumulado de correcciones")
    ax.set_ylabel("Errores de entrenamiento")
    ax.set_title("Figura 6. Una correccion puede empeorar el conjunto")
    ax.legend()
    ax.grid(alpha=0.2)
    presentar(fig, "06_errores_tras_cada_correccion.png")

    # FIGURA 7: SE ARREGLA UN PUNTO, PERO AUMENTAN LOS ERRORES GLOBALES.
    evento = r["ejemplo_que_empeora"]
    if evento is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
        for ax, situacion in zip(axes, ["antes", "despues"]):
            dibujar_puntos(ax, X, y)
            dibujar_frontera(ax, evento[f"w_{situacion}"],
                            evento[f"b_{situacion}"], X, COLOR_CLASICO)
            # Aro amarillo sobre el patron que desencadeno la correccion.
            ax.scatter([evento["x"][0]], [evento["x"][1]], s=260,
                       facecolors="none", edgecolors="gold", linewidths=3,
                       zorder=6)
            ajustar_limites(ax, X)
            ax.set_title(f"{situacion.upper()}: "
                         f"{evento[f'errores_{situacion}']} errores globales")
        fig.suptitle("Figura 7. Corregir UN punto puede empeorar el conjunto",
                     fontsize=13)
        presentar(fig, "07_una_correccion_empeora.png")
    else:
        print("No aparecio una correccion individual que arreglase un "
              "punto y aumentase al menos 3 errores globales; "
              "se omite la figura 7 con esta configuracion.")


# ============================================================================
# 5. EJECUCION PRINCIPAL
# ============================================================================
def main():
    X, y, punto_repetido = crear_datos(
        SEMILLA, PUNTOS_POR_CLASE, ETIQUETAS_INVERTIDAS)

    r = entrenar(X, y, ETAPAS, ETA, SEMILLA, BARAJAR_EN_CADA_EPOCA)

    print("=" * 66)
    print("PERCEPTRON SIN SEPARABILIDAD: CLASICO, POCKET Y AVERAGED")
    print("=" * 66)
    print(f"Observaciones: {len(y)}; epocas: {ETAPAS}; eta: {ETA}")
    print("Existen dos puntos con la misma posicion y clases opuestas.")
    print("Por tanto, ninguna recta puede acertar en todos los puntos.\n")

    for nombre in ["clasico", "pocket", "averaged"]:
        w, b = r[nombre]
        fallos = numero_errores(X, y, w, b)
        print(f"{nombre.upper():8s} | fallos = {fallos:3d} | "
              f"exactitud = {100*(1-fallos/len(y)):5.1f}% | "
              f"w = {np.round(w, 3)}, b = {b:.3f}")

    print(f"\nCorrecciones totales: {sum(r['actualizaciones_epoca'])}")
    print("Actualizaciones en las ultimas 5 epocas:",
          r["actualizaciones_epoca"][-5:])

    evento = r["ejemplo_que_empeora"]
    if evento is not None:
        print("\nEJEMPLO REAL DE EMPEORAMIENTO GLOBAL:")
        print(f"  Punto: {np.round(evento['x'], 3)}, "
              f"clase verdadera = {evento['y']:+.0f}")
        print(f"  Su puntuacion y*z mejora de "
              f"{evento['y']*evento['score_antes']:.3f} a "
              f"{evento['y']*evento['score_despues']:.3f}.")
        print(f"  Pero el total de errores sube: "
              f"{evento['errores_antes']} -> {evento['errores_despues']}.")

    crear_figuras(X, y, punto_repetido, r)

    # Se llama UNA sola vez al final. En Spyder puedes ver todas las figuras
    # en el panel Graficos; con un backend externo se abren ventanas.
    plt.show()
    return r


if __name__ == "__main__":
    resultados = main()
