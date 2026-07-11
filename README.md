# Proyecto de Entrenamiento de Agentes de IA en Overcooked 🍳🤖

Este repositorio implementa un sistema completo y modular para entrenar agentes de Inteligencia Artificial capaces de jugar eficientemente en **Overcooked AI**. El flujo de trabajo combina **Clonación de Comportamiento (*Behavioral Cloning* - BC)** a partir de grabaciones de demostraciones humanas y un posterior ajuste fino mediante **Optimización de Políticas Próximas (*Proximal Policy Optimization* - PPO)** usando auto-juego (*self-play*).

> [!TIP]
> **Repositorio Ligero y Optimizado:** Para evitar que el proyecto se sature y ocupe cientos de megabytes (`bloat`), el archivo `.gitignore` ha sido configurado específicamente para excluir archivos binarios pesados (`.npz`, `.pkl`, checkpoints `.pt`, archivos comprimidos y carpetas de grabaciones de los equipos). De esta manera, el código fuente y las configuraciones se mantienen ultraligeros en Git mientras todo el procesamiento de datos funciona a la perfección en local.

---

## 🗂️ Estructura del Proyecto

```text
deep_project/
├── overcooked/                    # Motor del juego y entorno de evaluación (NO modificar src/)
│   ├── src/                       # Código base de Overcooked (entorno, motor, evaluación, runner)
│   ├── policies/                  # Políticas base y agente entrenado (trained_agent.py)
│   ├── configs/                   # Configuraciones YAML para jugar, evaluar y recopilar demos
│   ├── layouts/                   # Directorio canónico donde se unifican todos los layouts de los grupos
│   ├── scripts/                   # Scripts del motor (consolidar layouts, filtrar grabaciones, métricas)
│   └── <carpetas_de_grupos>/      # ← AQUÍ se colocan localmente las carpetas de grabaciones de cada equipo
├── train/                         # Pipeline de procesamiento de datos y entrenamiento
│   ├── build_dataset.py           # Consolidación final del dataset filtrado
│   ├── train_bc.py                # Entrenador de Behavioral Cloning (BC)
│   ├── train_ppo.py               # Entrenador de PPO (Self-Play Fine-Tuning)
│   ├── training/                  # Módulos internos (modelos PyTorch, entorno PPO, buffers)
│   ├── data/                      # Datasets generados y estadísticas (ignorados por git)
│   └── models/                    # Checkpoints de los modelos entrenados .pt (ignorados por git)
├── configs/filter.yaml            # Configuración de umbrales y filtrado de calidad de grabaciones
├── requirements.txt               # Dependencias de Python necesarias
└── TUTORIAL.md                    # Documentación detallada e historial de arquitectura
```

---

## 📥 INSTRUCCIONES IMPORTANTES: ¿Dónde colocar las grabaciones de los grupos?

Dado que los archivos binarios de las demostraciones (`.npz`, `.pkl`, `.metadata.json`) no se suben al repositorio para evitar la saturación del almacenamiento, **cada usuario o equipo debe colocar las grabaciones localmente tras clonar el repositorio**.

> [!IMPORTANT]
> ### 📍 Ubicación Exacta para las Grabaciones de los Grupos
> 
> Para que todo el flujo de trabajo (`consolidación de layouts`, `filtrado de grabaciones` y `construcción de datasets`) funcione de manera idéntica y de forma 100% automática, **las carpetas de los grupos deben copiarse DIRECTAMENTE DENTRO del directorio `overcooked/`**.

### Ejemplo de cómo debe verse el directorio `overcooked/` tras añadir las grabaciones:

```text
deep_project/
└── overcooked/
    ├── src/                       # Directorio del motor del juego
    ├── policies/                  # Directorio de políticas
    ├── configs/                   # Directorio de configuraciones
    ├── layouts/                   # Directorio canónico de layouts
    ├── scripts/                   # Directorio de scripts del motor
    │
    ├── Flauta_s Company/          # ← Carpeta del grupo 1 con sus archivos .npz, .pkl y .layout
    ├── 001/                       # ← Carpeta del grupo 2 con sus archivos .npz, .pkl y .layout
    ├── Attention_t/               # ← Carpeta del grupo 3 con sus archivos .npz, .pkl y .layout
    ├── Grupo_Equipazo/            # ← Carpeta de cualquier otro grupo con sus demostraciones
    └── ...
```

### ¿Por qué colocarlas exactamente ahí y por qué funciona automáticamente?

1. **Detección Automática sin Cambiar Códigos ni Rutas:**
   Los scripts de preprocesamiento (`scripts/consolidate_layouts.py` y `scripts/filter_recordings.py`) están programados para escanear recursivamente todo el directorio `overcooked/` (`recordings_root: "."`). Al colocar la carpeta de un grupo dentro de `overcooked/`, el sistema:
   - Encuentra automáticamente todos los archivos `.layout` personalizados que haya creado el grupo y los unifica en `overcooked/layouts/` asignándoles un sufijo con el nombre del equipo (`<nombre_layout>_<equipo>.layout`).
   - Encuentra todos los archivos de grabación (`.npz` y `.pkl`), extrae las métricas de rendimiento humana y filtra las demostraciones de calidad para el entrenamiento.

2. **Protección Total contra Saturación en Git (`Bloat`):**
   El archivo `.gitignore` está configurado específicamente con la regla `overcooked/*` y excepciones solo para las carpetas del núcleo (`src/`, `policies/`, `configs/`, `layouts/`, `scripts/`). Por lo tanto, **puedes pegar las carpetas de los grupos dentro de `overcooked/` con total libertad**: Git las ignorará automáticamente, evitando que subas accidentalmente cientos de megabytes de grabaciones o archivos temporales de los alumnos/equipos.

---

## 🚀 Flujo de Trabajo Paso a Paso (Pipeline Completo)

Una vez que hayas colocado las carpetas de las grabaciones de los grupos dentro de `overcooked/`, sigue este flujo de comandos para procesar los datos, entrenar al agente y evaluarlo.

### 1. Consolidación de Layouts de los Grupos
Unifica los mapas (`.layout`) diseñados por cada grupo y los copia de forma segura en `overcooked/layouts/`:

```bash
cd overcooked
python scripts/consolidate_layouts.py
```
*Resultado:* Todos los layouts de los grupos estarán disponibles en `overcooked/layouts/` listos para ser utilizados por el motor del juego.

### 2. Filtrado de Calidad y Clasificación por Tiers (`filter_recordings.py`)
Inspecciona cada grabación para eliminar episodios con alta inactividad (*idle*), comportamientos robóticos o partidas abortadas prematuramente. Además, clasifica las partidas en **Oro**, **Plata** y **Bronce** según el percentil de su puntuación oficial:

```bash
python scripts/filter_recordings.py --config ../configs/filter.yaml
```
*Salidas generadas en `train/data/`:*
- `recording_quality.tsv`: Tabla exhaustiva con métricas y estado de retención de cada partida.
- `consolidated_filtered.npz`: Dataset filtrado con las grabaciones que superaron el control de calidad.

### 3. Construcción del Dataset de Entrenamiento (`build_dataset.py`)
Estandariza las dimensiones de las observaciones y empareja los tensores para el entrenamiento de PyTorch:

```bash
python ../train/build_dataset.py
```
*Salidas generadas en `train/data/`:*
- `consolidated.npz`: Dataset final normalizado y empaquetado.
- `dataset_stats.json`: Resumen estadístico global de distribuciones, acciones y pesos.

### 4. Entrenamiento del Agente por Behavioral Cloning (BC)
Entrena una red neuronal profunda (`MLP`) imitando las acciones del jugador humano ponderando la pérdida por el *Tier* de calidad (las partidas Oro influyen más que las Bronce):

```bash
python ../train/train_bc.py --epochs 50 --batch-size 256 --lr 1e-3
```
*Checkpoint generado:* `train/models/bc_agent.pt` (y su historial en `bc_agent.history.json`).

### 5. Fine-Tuning mediante PPO (Self-Play)
Ajusta la política entrenada con BC haciéndola competir y cooperar en auto-juego mediante aprendizaje por refuerzo:

```bash
python ../train/train_ppo.py
```
*Checkpoint generado:* `train/models/ppo_agent.pt`.

### 6. Evaluación del Agente Entrenado
El archivo `overcooked/policies/trained_agent.py` carga automáticamente los modelos de `train/models/` y actúa como un agente nativo dentro del entorno base de Overcooked:

```bash
python src/eval.py --config ../configs/eval/evaluate.yaml
```

---

## 🏆 Fórmula de la Puntuación Oficial del Torneo

El sistema de evaluación no se basa únicamente en la recompensa pura, sino en la **Puntuación Oficial de Competición**, que premia drásticamente la entrega de sopas y la velocidad:

\[
\text{Puntuación Oficial} = 10000 \times \text{num\_soups} + 10 \times (\text{horizon} - \text{last\_soup}) + (\text{horizon} - \text{first\_soup}) - \text{penalty}
\]

- **`num_soups`:** Número de sopas entregadas con éxito (ponderadas a `10,000` puntos cada una).
- **`horizon`:** Duración total del episodio (normalmente `250` pasos de tiempo).
- **`first_soup` & `last_soup`:** Paso de tiempo en el que se entregó la primera y la última sopa respectivamente (incentiva entregar lo antes posible).
- **`penalty`:** Penalizaciones acumuladas por comportamientos indebidos o tiempos expirados.

El script `scripts/official_score.py` se encarga de calcular exactamente este puntaje durante el filtrado y las evaluaciones del torneo.

---

## 🛠️ Requisitos e Instalación

Para instalar las dependencias necesarias de Python (requiere Python 3.9+):

```bash
pip install -r requirements.txt
```
