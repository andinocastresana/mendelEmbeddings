# =========================================
# ID: GIT_NB_012
# VERSION: v1.0
# =========================================
# sync_notebooks.sh
#
# Exporta SIEMPRE de ipynb -> py
# - ignora notebooks vacíos
# - no modifica los .ipynb
# - añade al staging los .py generados
#
# Uso:
#   chmod +x sync_notebooks.sh
#   ./sync_notebooks.sh

set -e

echo ">> Buscando notebooks..."

find . -name "*.ipynb" -not -path "./.venv/*" -print0 | while IFS= read -r -d '' nb; do
    # Saltar archivos vacíos
    if [ ! -s "$nb" ]; then
        echo ">> Saltando notebook vacío: $nb"
        continue
    fi

    echo ">> Exportando: $nb"

    # Si falla la conversión, avisar y seguir con el siguiente
    if python -m jupytext --to py:percent "$nb"; then
        py_file="${nb%.ipynb}.py"
        if [ -f "$py_file" ]; then
            git add "$py_file"
        fi
    else
        echo ">> ERROR al convertir: $nb"
        echo ">> Lo salto y continúo."
    fi
done

echo ">> OK. Listo para commit."