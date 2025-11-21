import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
# TensorFlow/Keras son opcionales para que la app pueda iniciar rápido
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except Exception:
    tf = None
    keras = None
    layers = None
    TF_AVAILABLE = False
import os
import joblib
import pickle

st.set_page_config(page_title='Clasificación de Cultivos - GUI', layout='wide')

PROJECT_SEED = 42
np.random.seed(PROJECT_SEED)
MODEL_DIR = 'models'
try:
    if TF_AVAILABLE:
        tf.keras.utils.set_random_seed(PROJECT_SEED)
except Exception:
    pass

st.title('🌾 Clasificación de cultivos')

st.markdown(
    'Cargue un CSV con las variables de suelo y clima y la columna objetivo (ej: `label` o `crop`). '
    'La app entrenará una red feedforward (FNN) y mostrará métricas y predicciones.'
)

# Explicación amigable para usuarios no técnicos
with st.expander('¿Qué significan N, P y K? ¿Cómo funciona esta app?'):
    st.markdown(
        """
        - N (Nitrógeno): nutriente esencial para el crecimiento vegetativo; influye en el color y desarrollo de las hojas.
        - P (Fósforo): importante para el desarrollo de raíces y la floración/fructificación.
        - K (Potasio): ayuda en la regulación del agua, la resistencia a enfermedades y la calidad del fruto.

        ¿Qué hace la app?
        1. Subís un archivo CSV con variables como N, P, K, pH, temperatura y humedad y una columna objetivo con el cultivo (ej. `label`).
        2. La app preprocesa los datos (escala variables numéricas y codifica categóricas) y entrena una red FNN para ofrecer recomendaciones.
        3. Podés seleccionar una fila del dataset o ingresar valores manuales para obtener la predicción del cultivo más adecuado y ver las probabilidades por clase.

        Nota: si querés entrenar la FNN (red neuronal) más avanzada, podemos añadir esa opción — puede tardar más tiempo en tu computador.
        """
    )

with st.sidebar:
    st.header('Acciones')
    uploaded_file = st.file_uploader('Cargar CSV (data/crops.csv)', type=['csv'])
    sample_data = st.checkbox('Usar ejemplo (si no hay CSV)')
    sample_choice = None
    if sample_data:
        sample_choice = st.selectbox(
            'Dataset de ejemplo',
            ['Fácil (limpio y separable)', 'Desafiante (clases solapadas)'],
            index=0
        )
    random_state = st.number_input('Seed (reproducible)', value=PROJECT_SEED, step=1)
    st.markdown('---')
    st.caption('Flujo: subir CSV → seleccionar target → Preprocess & Split → Entrenar FNN → Evaluar/Predecir')

DATA_PATH_EASY = 'data/crops_easy.csv'
DATA_PATH_HARD = 'data/crops_challenging.csv'

def load_sample(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    # pequeño ejemplo sintético
    rng = np.random.default_rng(PROJECT_SEED)
    df = pd.DataFrame({
        'N': rng.normal(50, 10, 200),
        'P': rng.normal(30, 5, 200),
        'K': rng.normal(20, 4, 200),
        'temperatura': rng.normal(25, 3, 200),
        'humedad': rng.normal(40, 8, 200),
        'ph': rng.normal(6.5, 0.6, 200),
        'label': rng.choice(['arroz','maíz','algodón','trigo'], size=200)
    })
    return df

@st.cache_data
def build_preprocessor(X, numeric_cols, categorical_cols):
    numeric_tf = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_tf = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_tf, numeric_cols),
            ('cat', categorical_tf, categorical_cols)
        ]
    )
    preprocessor.fit(X)
    return preprocessor

def build_fnn(input_dim, num_classes, hidden=[128,64], dropout=0.2):
    if not TF_AVAILABLE:
        raise RuntimeError('TensorFlow/Keras no está disponible en este entorno. Instálalo para usar la FNN.')
    # Evita problemas de name scopes y reconstrucciones cuando Streamlit re-ejecuta el script
    try:
        keras.backend.clear_session()
    except Exception:
        pass
    if input_dim is None or int(input_dim) <= 0:
        raise ValueError('El preprocesamiento produjo 0 features. Verifica columnas numéricas/categóricas y tipos de datos.')
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    for h in hidden:
        model.add(layers.Dense(h, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_confusion_matrix(cm, class_names):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='Real', xlabel='Predicho')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
    fig.tight_layout()
    return fig


def save_preprocessor(preprocessor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path):
    return joblib.load(path)


# --- Flujo simplificado: Subir CSV -> Entrenar FNN -> Predecir
df = None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success('CSV cargado correctamente')
elif sample_data:
    sample_path = DATA_PATH_EASY if (sample_choice is None or sample_choice.startswith('Fácil')) else DATA_PATH_HARD
    df = load_sample(sample_path)
    st.info(f'Usando dataset de ejemplo: {sample_choice or "Fácil (limpio y separable)"}')
else:
    st.info('Suba un CSV o marque "Usar ejemplo" en la barra lateral para probar la app')

if df is not None:
    st.subheader('Vista previa de datos')
    col_prev_a, col_prev_b = st.columns([1,1])
    with col_prev_a:
        show_all_preview = st.checkbox('Mostrar todos los registros', value=False, key='show_all_preview')
    with col_prev_b:
        n_default = min(100, len(df))
        n_preview = st.number_input('Filas a mostrar', min_value=5, max_value=len(df), value=int(n_default), step=5, key='n_preview')
    if show_all_preview:
        st.dataframe(df, use_container_width=True, height=500)
    else:
        st.dataframe(df.head(int(n_preview)), use_container_width=True, height=500)
    st.caption(f'Registros: {len(df)} | Columnas: {len(df.columns)}')
    st.markdown('**Columnas detectadas:** ' + ', '.join(df.columns))
    try:
        st.download_button('Descargar CSV cargado', df.to_csv(index=False).encode('utf-8'), file_name='dataset_actual.csv', mime='text/csv')
    except Exception:
        pass

    target_col = st.selectbox('Selecciona la columna objetivo (target)', options=list(df.columns), index=len(df.columns)-1, key='target_col')

    # Entrenamiento FNN (predeterminado)
    auto_train = st.checkbox('Entrenar FNN automáticamente (recomendado)', value=True, key='auto_train')

    if auto_train:
        with st.spinner('Preparando datos y entrenando FNN...'):
            X = df.drop(columns=[target_col])
            y = df[target_col].astype('category')
            class_names = y.cat.categories.tolist()
            numeric_cols = [c for c in X.columns if X[c].dtype != 'object']
            categorical_cols = [c for c in X.columns if X[c].dtype == 'object']

            # rápido split 80/20
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

            preprocessor = build_preprocessor(X_train, numeric_cols, categorical_cols)
            X_train_t = preprocessor.transform(X_train)
            X_test_t = preprocessor.transform(X_test)
            if hasattr(X_train_t, 'toarray'):
                X_train_t = X_train_t.toarray()
                X_test_t = X_test_t.toarray()

            y_train_idx = y_train.cat.codes.values
            y_test_idx = y_test.cat.codes.values

            # Entrenar FNN rápida por defecto
            try:
                # Fijar semilla de TF/Keras según el valor elegido en la UI
                try:
                    if TF_AVAILABLE:
                        tf.keras.utils.set_random_seed(int(random_state))
                except Exception:
                    pass
                hs = [128, 64]
                fnn = build_fnn(X_train_t.shape[1], len(class_names), hidden=hs, dropout=0.2)
            except RuntimeError as e:
                st.error(str(e))
                st.stop()
            cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                  keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)]
            history = fnn.fit(X_train_t, y_train_idx, validation_data=(X_test_t, y_test_idx),
                              epochs=60, batch_size=64, callbacks=cb, verbose=0)

            # evaluación rápida
            test_probs = fnn.predict(X_test_t)
            test_pred = test_probs.argmax(axis=1)
            test_acc = accuracy_score(y_test_idx, test_pred)
            test_f1 = f1_score(y_test_idx, test_pred, average='macro')
            # sugerir una fila "difícil" (menor confianza en test) y guardarla en sesión
            try:
                test_conf = test_probs.max(axis=1)
                mis_mask = (test_pred != y_test_idx)
                if mis_mask.any():
                    cand = np.where(mis_mask)[0]
                    worst_local = int(cand[np.argmin(test_conf[cand])])
                else:
                    worst_local = int(np.argmin(test_conf))
                suggested_idx = int(X_test.index[worst_local])
            except Exception:
                suggested_idx = int(X_test.index[0])
            st.session_state['suggested_idx'] = suggested_idx

            # guardar artefactos
            os.makedirs(MODEL_DIR, exist_ok=True)
            save_preprocessor(preprocessor, os.path.join(MODEL_DIR, 'preprocessor.joblib'))
            # Guardar en formato Keras 3 recomendado
            fnn.save(os.path.join(MODEL_DIR, 'fnn_saved.keras'))
            # guardar mapping de clases
            with open(os.path.join(MODEL_DIR, 'class_names.pkl'), 'wb') as f:
                pickle.dump(class_names, f)

        st.success(f'FNN entrenada. Test Accuracy: {test_acc:.4f} | F1 macro: {test_f1:.4f}')
        # mostrar curva de entrenamiento
        try:
            fig, ax = plt.subplots()
            ax.plot(history.history.get('loss', []), label='train loss')
            ax.plot(history.history.get('val_loss', []), label='val loss')
            ax.set_xlabel('Época')
            ax.set_ylabel('Loss')
            ax.legend()
            st.pyplot(fig)
        except Exception:
            pass

        with st.expander('¿Cómo interpretar estas métricas?'):
            st.markdown(
                f"""
- **Accuracy (exactitud)**: proporción de aciertos del modelo en test. En este entrenamiento fue de `{test_acc:.2%}`.
- **F1 macro**: promedio del F1 por clase, útil si las clases están desbalanceadas. Obtuviste `{test_f1:.2f}` (0–1).
- **Sugerencia**: si el F1 es bastante menor que el accuracy, probablemente haya clases difíciles o desbalanceadas.

Lectura de la matriz de confusión:
- **Filas = reales** y **columnas = predichas**; la **diagonal** indica aciertos.
- Una fila con valores altos fuera de la diagonal señala en qué clases el modelo suele confundirse.
- Si ves confusión entre dos clases (p. ej., arroz ↔ maíz), inspeccioná ejemplos de esas clases usando el selector de índice en la sección “Predecir” para revisar valores concretos de entrada.
- Usá la vista de “fila seleccionada” para comparar los atributos originales y entender por qué el modelo pudo confundirla.
                """
            )
            try:
                cm = confusion_matrix(y_test_idx, test_pred)
                st.pyplot(plot_confusion_matrix(cm, class_names))
                st.caption('Matriz de confusión: filas = reales, columnas = predichas.')
            except Exception:
                pass

        # Mostrar sección de predicción SOLO cuando el ejemplo es el desafiante
        show_prediction_section = bool(sample_data and sample_choice and sample_choice.startswith('Desafiante'))
        if show_prediction_section:
            st.subheader('Predecir')
            st.markdown('El sistema sugiere una fila con menor confianza en test para inspección y predicción.')
            suggested_idx = int(st.session_state.get('suggested_idx', 0))
            st.markdown('**Fila sugerida (caso difícil):**')
            st.write(f'Índice sugerido: {suggested_idx}')
            st.dataframe(df.iloc[[suggested_idx]])
            if st.button('Predecir fila sugerida', key='predict_suggested'):
                x_row = df.iloc[[suggested_idx]].drop(columns=[target_col])
                x_t = preprocessor.transform(x_row)
                if hasattr(x_t, 'toarray'):
                    x_t = x_t.toarray()
                try:
                    probs = fnn.predict(x_t)[0]
                except Exception:
                    from tensorflow import keras as tfkeras  # seguro en este punto
                    fnn_loaded = tfkeras.models.load_model(os.path.join(MODEL_DIR, 'fnn_saved.keras'))
                    probs = fnn_loaded.predict(x_t)[0]
                pred_idx = int(np.argmax(probs))
                with open(os.path.join(MODEL_DIR, 'class_names.pkl'), 'rb') as f:
                    class_names = pickle.load(f)
                st.write('Predicción (sugerida):', class_names[pred_idx])
                prob_df = pd.DataFrame({'class': class_names, 'prob': probs})
                st.dataframe(prob_df.sort_values('prob', ascending=False).reset_index(drop=True))

                # Explicación amigable de la respuesta
                confidence = float(np.max(probs))
                st.markdown(f'**Confianza estimada del modelo:** {confidence:.2%}')
                if confidence >= 0.8:
                    st.success('Alta confianza: el modelo está bastante seguro de esta predicción.')
                elif confidence >= 0.6:
                    st.info('Confianza media: revisa las clases cercanas en la tabla de probabilidades.')
                else:
                    st.warning('Confianza baja: podrían necesitarse más datos o features; revisa el preprocesamiento.')

                # Comparación con la etiqueta real si está disponible
                try:
                    true_value = df.iloc[suggested_idx][target_col]
                    st.markdown(f'**Etiqueta real en los datos:** `{true_value}`')
                    if str(true_value) == str(class_names[pred_idx]):
                        st.success('La predicción coincide con la etiqueta real para esta fila.')
                    else:
                        st.error('La predicción no coincide con la etiqueta real para esta fila. Revisá valores y posibles clases confusas.')
                except Exception:
                    pass

                with st.expander('Interpretación de la respuesta'):
                    st.markdown(
                        """
- **Predicción**: cultivo con mayor probabilidad según la FNN.
- **Probabilidad por clase**: cuán convencida está la red para cada cultivo; la suma es 1.
- **Confianza**: probabilidad de la clase elegida. Alta (≥80%), media (60–80%), baja (<60%).
- **Cómo usar la vista previa**: la fila mostrada es aquella donde el modelo tuvo menor confianza en test; revisá sus valores para entender posibles confusiones.
- **Si no coincide con la etiqueta real**: verificá la calidad de datos, outliers o clases muy parecidas. Consultá la matriz de confusión para ver si el error es frecuente entre esas dos clases.
- **Recomendación**: si la confianza es baja, considerá ajustar hiperparámetros o añadir datos; si la confusión es sistemática entre dos clases, añadí features que las distingan mejor.
                        """
                    )
        else:
            st.caption('Este dataset de ejemplo es fácil/separable: no se requiere sección de predicción interactiva.')

        # FIN sección de predicción condicionada

        # (La lógica de comparación e interpretación para la fila sugerida se maneja arriba)
        # Mantener resto del flujo de la app

        # placeholder para evitar variables no usadas si la sección no se muestra
        _ = MODEL_DIR
        # FIN

        st.markdown('---')
        st.caption('Modelo activo: FNN — Solo se utiliza red neuronal feedforward en esta app.')
