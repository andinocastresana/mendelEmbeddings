# tests/smoke

Smoke tests **muy livianos** asociados a la migración Tarea #1 (ver `_meta/MIGRACION_TAREA1.md`). Cada archivo corresponde a un sub-paso de la migración y se ejecuta con:

```bash
python3 tests/smoke/<archivo>.py
```

## Propósito

Verificar, tras migrar un grupo de funciones, que:
1. Los imports del nuevo módulo funcionan.
2. Las funciones se ejecutan sin error con datos sintéticos.
3. Las invariantes clave se cumplen (shapes, dtypes, casos de borde básicos).

**NO son tests de calidad funcional** (eso es la Tarea #22 del proyecto: smoke test del pipeline completo en `tests/` con datos reales). Acá solo verificamos que la migración no rompió la cadena de imports ni la mecánica básica.

## Convención

- Un archivo por sub-paso de migración: `test_paso_<N>_<modulo>.py` (ej. `test_paso_7_viz.py`).
- Cada archivo es **autónomo**: agrega `src/` a `sys.path`, importa lo recién migrado, ejecuta los chequeos y termina con `print('[OK] Paso N verificado')` en éxito o levanta `AssertionError` en fallo.
- Sin dependencias de pytest/unittest — es ejecutable directamente con `python3`. Mantener simple.
- Sin red, sin escrituras fuera de `/tmp`, sin modelos pesados (usar mocks como en pasos 6b y 6d).

## Por qué archivos y no `python3 -c "..."`

El harness de Claude Code no puede auto-aprobar `python3 -c "..."` (es arbitrary code execution). Como archivos versionados bajo `tests/smoke/`, el patrón `Bash(python3 tests/smoke/*)` queda en el allowlist del proyecto (ver `.claude/settings.json`) y los smoke tests se ejecutan sin prompt — además de quedar **versionados y reproducibles**.

## Tras cerrar la Tarea #1

Estos archivos pueden quedar como referencia histórica, o eliminarse (siguiendo la regla de no-borrado del proyecto: mover a `_toReview/`). La Tarea #22 producirá smoke tests "de verdad" en `tests/` (sin `smoke/`), basados en el pipeline completo con datos reales.
