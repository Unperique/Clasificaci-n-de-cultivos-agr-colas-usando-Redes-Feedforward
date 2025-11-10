# Clasificación de Cultivos — Interfaz amigable

Esta pequeña aplicación provee una interfaz sencilla para entrenar y probar modelos (baselines y una FNN) sobre un dataset de recomendación de cultivos.

Requisitos mínimos
- Windows (Probado localmente)
- Python 3.8+

Instalación
1. Crear un entorno virtual (recomendado) y activar:

```powershell
python -m venv venv; .\venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

Uso (ejecución)
1. Ejecutar la app Streamlit:

```powershell
streamlit run streamlit_app.py
```

2. En la barra lateral puede:
- Subir su CSV (por ejemplo `data/crops.csv`).
- O usar el dataset de ejemplo si no tiene uno.

Flujo en la app
1. Subir CSV y seleccionar la columna objetivo (target).
2. Hacer click en "Preprocess & Split" para preprocesar y dividir los datos.
3. Ajustar parámetros (épocas, batch size) y click en "Entrenar modelos ahora".
4. Ver resultados en validación/test, matriz de confusión y descargar (guardar) modelo FNN en `models/fnn_saved`.

Notas y recomendaciones
- Si su dataset es muy grande, reduzca `epochs` y el `batch_size` para entrenar más rápido.
- Para producción, exporte el `preprocessor` y el `SavedModel` y sirva mediante una API (FastAPI/Flask) o deploy en la nube.
- Nota: si querés entrenar la FNN (red neuronal) más avanzada, podemos añadir esa opción — puede tardar más tiempo en tu computador y consumir más recursos. Para pruebas rápidas, recomendamos usar el RandomForest que viene por defecto.

Contacto
Si quiere que adapte la interfaz (más formularios, ayuda contextuall, soporte para imágenes) dime cómo la imaginás y lo implemento.
