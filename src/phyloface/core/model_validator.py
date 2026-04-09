# =========================================
# ID: PHYLOFACE_MODEL_VALIDATE_001
# VERSION: v1.0
# =========================================
# Archivo sugerido:
# phyloface/core/model_validator.py
#
# Qué hace:
# - verifica que el modelo existe en ~/.insightface/models
# - chequea que contiene archivos .onnx
# - intenta inicializar FaceAnalysis de forma segura
# - detecta errores típicos (como el de antelopev2)
#
# Uso típico:
# from phyloface.core.model_validator import validate_insightface_model
# validate_insightface_model("buffalo_l")


from pathlib import Path


def validate_insightface_model(model_name: str) -> bool:
    """
    Valida que un modelo de InsightFace esté correctamente instalado y usable.
    """
    model_root = Path.home() / ".insightface" / "models"
    model_dir = model_root / model_name

    print(f"\n[CHECK] Modelo: {model_name}")
    print(f"Ruta esperada: {model_dir}")

    # -------------------------
    # 1) Existe carpeta
    # -------------------------
    if not model_dir.exists():
        print("❌ No existe la carpeta del modelo")
        return False

    # -------------------------
    # 2) Buscar archivos ONNX
    # -------------------------
    onnx_files = list(model_dir.rglob("*.onnx"))

    if len(onnx_files) == 0:
        print("❌ No se encontraron archivos .onnx")
        return False

    print(f"✔ ONNX encontrados: {len(onnx_files)}")

    # -------------------------
    # 3) Detectar estructura incorrecta
    # (ej: carpeta duplicada)
    # -------------------------
    nested_dir = model_dir / model_name
    if nested_dir.exists():
        print("⚠️ Posible estructura incorrecta: carpeta anidada detectada")
        print(f"   -> {nested_dir}")
        print("   Esto suele romper FaceAnalysis")

    # -------------------------
    # 4) Intentar cargar modelo
    # -------------------------
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name=model_name)
        app.prepare(ctx_id=-1, det_size=(640, 640))

        print("✔ Modelo cargado correctamente")
        print(f"   Submodelos disponibles: {list(app.models.keys())}")

        if "detection" not in app.models:
            print("❌ Falta submodelo de detección")
            return False

        return True

    except AssertionError as e:
        print("❌ AssertionError durante carga (típico de modelo incompleto)")
        print(f"   {e}")
        return False

    except Exception as e:
        print("❌ Error inesperado al cargar el modelo")
        print(f"   {type(e).__name__}: {e}")
        return False

####Ejecución:
# Ejecutar desde:
# .../mendelEmbeddings/src

# Validar buffalo (debería OK)
#python -c "from phyloface.core.model_validator import validate_insightface_model; validate_insightface_model('buffalo_l')"

# Validar antelope (el que falla)
#python -c "from phyloface.core.model_validator import validate_insightface_model; validate_insightface_model('antelopev2')"