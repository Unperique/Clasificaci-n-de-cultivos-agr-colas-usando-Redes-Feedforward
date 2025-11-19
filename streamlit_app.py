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

st.title('🌾 Clasificación de cultivos — Interfaz amigable')

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
    random_state = st.number_input('Seed (reproducible)', value=PROJECT_SEED, step=1)
    st.markdown('---')
    st.caption('Flujo: subir CSV → seleccionar target → Preprocess & Split → Entrenar FNN → Evaluar/Predecir')

DATA_PATH = 'data/crops.csv'

def load_sample():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
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
    df = load_sample()
    st.info('Usando dataset de ejemplo')
else:
    st.info('Suba un CSV o marque "Usar ejemplo" en la barra lateral para probar la app')

if df is not None:
    st.subheader('Vista previa de datos')
    st.dataframe(df.head(10))
    st.markdown('**Columnas detectadas:** ' + ', '.join(df.columns))

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
            test_pred = fnn.predict(X_test_t).argmax(axis=1)
            test_acc = accuracy_score(y_test_idx, test_pred)
            test_f1 = f1_score(y_test_idx, test_pred, average='macro')

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
                """
            )
            try:
                cm = confusion_matrix(y_test_idx, test_pred)
                st.pyplot(plot_confusion_matrix(cm, class_names))
                st.caption('Matriz de confusión: filas = reales, columnas = predichas.')
            except Exception:
                pass

        st.subheader('Predecir')
        st.markdown('Seleccione una fila del dataset para predecir el mejor cultivo sugerido.')

        # elegir fila del dataset por índice (0 .. len(df)-1)
        max_index = max(0, len(df) - 1)
        selected_idx = st.slider('Elegir índice de fila para predecir', min_value=0, max_value=max_index, value=0, step=1, key='select_row')
        with st.expander('¿Qué hace esta sección?'):
            st.markdown(
                """
- Seleccioná el índice de una fila de la tabla superior. Los índices van de 0 a n-1.
- Tomamos las variables de esa fila (excepto la columna objetivo) y la FNN predice el cultivo.
- La tabla de probabilidades muestra alternativas y su confianza.
- Si querés validar visualmente la fila, buscala en la “Vista previa de datos”.
                """
            )
        if st.button('Predecir fila seleccionada', key='predict_row'):
            x_row = df.iloc[[selected_idx]].drop(columns=[target_col])
            x_t = preprocessor.transform(x_row)
            if hasattr(x_t, 'toarray'):
                x_t = x_t.toarray()
            try:
                probs = fnn.predict(x_t)[0]
            except Exception:
                # cargar modelo guardado si no está en memoria
                from tensorflow import keras as tfkeras  # seguro en este punto
                fnn_loaded = tfkeras.models.load_model(os.path.join(MODEL_DIR, 'fnn_saved.keras'))
                probs = fnn_loaded.predict(x_t)[0]
            pred_idx = int(np.argmax(probs))
            with open(os.path.join(MODEL_DIR, 'class_names.pkl'), 'rb') as f:
                class_names = pickle.load(f)
            st.write('Predicción:', class_names[pred_idx])
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

            with st.expander('¿Cómo interpretar estas respuestas?'):
                st.markdown(
                    """
- **Predicción**: cultivo con mayor probabilidad según la FNN.
- **Probabilidad por clase**: cuán convencida está la red para cada cultivo; la suma es 1.
- **Confianza**: probabilidad de la clase elegida. Alta (≥80%), media (60–80%), baja (<60%).
- **Recomendación**: si la confianza es baja, considerá ajustar hiperparámetros o añadir datos.
                    """
                )

        st.markdown('---')
        st.caption('Modelo activo: FNN — Solo se utiliza red neuronal feedforward en esta app.')