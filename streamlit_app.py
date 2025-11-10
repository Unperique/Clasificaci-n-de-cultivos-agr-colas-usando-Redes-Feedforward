import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import joblib
import pickle

st.set_page_config(page_title='Clasificación de Cultivos - GUI', layout='wide')

PROJECT_SEED = 42
np.random.seed(PROJECT_SEED)
tf.keras.utils.set_random_seed(PROJECT_SEED)

st.title('🌾 Clasificación de cultivos — Interfaz amigable')

st.markdown(
    'Cargue un CSV con las variables de suelo y clima y la columna objetivo (ej: `label` o `crop`). '
    'La app entrenará modelos simples y una red feedforward (FNN) y mostrará métricas y predicciones.'
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
        2. La app preprocesa los datos (escala variables numéricas y codifica categóricas) y entrena rápidamente un modelo confiable (RandomForest) en segundo plano para ofrecer recomendaciones.
        3. Podés seleccionar una fila del dataset o ingresar valores manuales para obtener la predicción del cultivo más adecuado y ver las probabilidades por clase.
        4. También se muestran las variables más importantes según el modelo (para ayudar a entender qué afecta la predicción).

        Nota: si querés entrenar la FNN (red neuronal) más avanzada, podemos añadir esa opción — puede tardar más tiempo en tu computador.
        """
    )

with st.sidebar:
    st.header('Acciones')
    uploaded_file = st.file_uploader('Cargar CSV (data/crops.csv)', type=['csv'])
    sample_data = st.checkbox('Usar ejemplo (si no hay CSV)')
    random_state = st.number_input('Seed (reproducible)', value=PROJECT_SEED, step=1)
    st.markdown('---')
    st.caption('Flujo: subir CSV → seleccionar target → Preprocess & Split → Train → Evaluar/Predecir')

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


# --- Flujo simplificado: Subir CSV -> Auto-entrenar modelo rápido (oculto) -> Predecir
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

    # Auto-train toggle (hidden guidance)
    auto_train = st.checkbox('Preparar modelo automáticamente (recomendado)', value=True, key='auto_train')

    if auto_train:
        with st.spinner('Preparando modelo automáticamente (rápido)...'):
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

            # Entrenar un RandomForest rápido y confiable para predicciones
            rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            rf.fit(X_train_t, y_train_idx)

            # guardar artefactos
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            save_preprocessor(preprocessor, os.path.join(model_dir, 'preprocessor.joblib'))
            joblib.dump(rf, os.path.join(model_dir, 'rf_model.joblib'))
            # guardar mapping de clases
            with open(os.path.join(model_dir, 'class_names.pkl'), 'wb') as f:
                pickle.dump(class_names, f)

        st.success('Modelo preparado y listo para predecir')

        st.markdown('**Por qué usamos RandomForest por defecto:** RandomForest es rápido de entrenar, robusto frente a datos ruidosos y ofrece importancias de variables interpretables; es una buena opción para obtener predicciones confiables sin esperar largos entrenamientos.')

        # Opción A: Entrenar FNN avanzada (opcional)
        with st.expander('Entrenar FNN avanzada (opcional) 🔬', expanded=False):
            st.warning('Entrenar la FNN puede tardar varios minutos u horas dependiendo del tamaño del dataset y de tu equipo. Activa solo si querés esperar.')
            fnn_epochs = st.slider('Épocas (FNN avanzada)', min_value=10, max_value=500, value=100, key='fnn_epochs')
            fnn_batch = st.selectbox('Batch size (FNN avanzada)', options=[32,64,128,256], index=1, key='fnn_batch')
            hidden_sizes = st.text_input('Tamaños de capas (coma separada)', value='128,64,32', key='fnn_hidden')
            train_fnn_confirm = st.checkbox('Confirmar: quiero entrenar la FNN avanzada', key='confirm_fnn')
            if st.button('Entrenar FNN ahora', key='train_fnn_btn'):
                if not train_fnn_confirm:
                    st.error('Primero marca la casilla de confirmación para evitar entrenamientos accidentales.')
                else:
                    with st.spinner('Entrenando FNN — esto puede tardar...'):
                        try:
                            hs = [int(x.strip()) for x in hidden_sizes.split(',') if x.strip()]
                        except Exception:
                            hs = [128,64,32]
                        fnn = build_fnn(X_train_t.shape[1], len(class_names), hidden=hs, dropout=0.2)
                        cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
                              keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)]
                        history = fnn.fit(X_train_t, y_train_idx, validation_data=(X_test_t, y_test_idx), epochs=fnn_epochs, batch_size=fnn_batch, callbacks=cb, verbose=0)
                        # evaluar
                        test_pred = fnn.predict(X_test_t).argmax(axis=1)
                        test_acc = accuracy_score(y_test_idx, test_pred)
                        test_f1 = f1_score(y_test_idx, test_pred, average='macro')
                        st.success(f'FNN entrenada. Test Accuracy: {test_acc:.4f} | F1 macro: {test_f1:.4f}')
                        # guardar
                        model_dir = 'models'
                        os.makedirs(model_dir, exist_ok=True)
                        fnn.save(os.path.join(model_dir, 'fnn_saved'))
                        st.success(f'Modelo FNN guardado en {os.path.join(model_dir, "fnn_saved")}')
                        # mostrar curva de entrenamiento
                        fig, ax = plt.subplots()
                        ax.plot(history.history.get('loss', []), label='train loss')
                        ax.plot(history.history.get('val_loss', []), label='val loss')
                        ax.set_xlabel('Época')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        st.pyplot(fig)

        st.subheader('Predecir')
        st.markdown('Seleccione una fila del dataset o ingrese valores manualmente para predecir el mejor cultivo sugerido.')

        # opción 1: elegir fila del dataset
        idx_options = df.index.tolist()
        selected_idx = st.selectbox('Elegir índice de fila para predecir', options=idx_options, key='select_row')
        if st.button('Predecir fila seleccionada', key='predict_row'):
            x_row = df.loc[[selected_idx]].drop(columns=[target_col])
            x_t = preprocessor.transform(x_row)
            if hasattr(x_t, 'toarray'):
                x_t = x_t.toarray()
            pred_idx = rf.predict(x_t)[0]
            probs = rf.predict_proba(x_t)[0]
            with open(os.path.join(model_dir, 'class_names.pkl'), 'rb') as f:
                class_names = pickle.load(f)
            st.write('Predicción:', class_names[pred_idx])
            prob_df = pd.DataFrame({'class': class_names, 'prob': probs})
            st.dataframe(prob_df.sort_values('prob', ascending=False).reset_index(drop=True))

        st.markdown('---')
        st.subheader('Entrada manual')
        manual_vals = {}
        for i, c in enumerate(X.columns):
            if c in numeric_cols:
                v = st.number_input(f'{c}', value=float(X[c].median()), key=f'man_{c}')
                manual_vals[c] = v
            else:
                opts = df[c].dropna().unique().tolist()
                v = st.selectbox(f'{c}', options=opts, key=f'sel_{c}')
                manual_vals[c] = v

        if st.button('Predecir con valores manuales', key='predict_manual'):
            x_manual = pd.DataFrame([manual_vals])
            x_t = preprocessor.transform(x_manual)
            if hasattr(x_t, 'toarray'):
                x_t = x_t.toarray()
            pred_idx = rf.predict(x_t)[0]
            probs = rf.predict_proba(x_t)[0]
            with open(os.path.join(model_dir, 'class_names.pkl'), 'rb') as f:
                class_names = pickle.load(f)
            st.write('Predicción:', class_names[pred_idx])
            prob_df = pd.DataFrame({'class': class_names, 'prob': probs})
            st.dataframe(prob_df.sort_values('prob', ascending=False).reset_index(drop=True))

        st.markdown('---')
        st.subheader('Importancia de variables (RandomForest)')
        try:
            importances = rf.feature_importances_
            # construir nombres de features si hay one-hot
            try:
                feat_names = preprocessor.get_feature_names_out()
            except Exception:
                feat_names = [f'f{i}' for i in range(len(importances))]
            fi_df = pd.DataFrame({'feature': feat_names, 'importance': importances}).sort_values('importance', ascending=False).head(20)
            fig, ax = plt.subplots(figsize=(6,4))
            ax.barh(fi_df['feature'][::-1], fi_df['importance'][::-1])
            st.pyplot(fig)
        except Exception as e:
            st.warning('No se pudo mostrar la importancia de variables: ' + str(e))

        st.info('Si querés, puedo añadir una opción para entrenar una FNN más avanzada detrás de escena (esto puede tardar).')
