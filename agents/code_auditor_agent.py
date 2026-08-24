#!/usr/bin/env python3
"""Agente auditor unificado — código + datos de producción"""

import subprocess
from pathlib import Path
from datetime import datetime

REPORT_DIR = "/data/audits"

PROMPT = """Eres un auditor experto en sistemas de trading algorítmico con acceso completo al código fuente y datos de producción del sistema APP PREDICTIVA.

PASO 1 — LEE EL CÓDIGO COMPLETO del repo GitHub: cahuerta/spy-2w-price-prediction

NO uses una lista fija de archivos. Antes de leer nada, enumera el árbol REAL del
repositorio completo, incluyendo todas las subcarpetas, con:

    git ls-files '*.py'

o, si el repo no está en un working tree de git:

    find . -name "*.py" -not -path "*/node_modules/*"

Lee TODOS los archivos .py que aparezcan en esa enumeración, sin excepción —
incluyendo (pero no limitado a) los que ya sabes que existen:
- alpha_engine_v4.py
- trading_orchestrator.py
- darwin_engine/arena.py, executor_genome.py, predictor_arena.py, pnl_fitness.py
- pm_growth.py, pm_neutral.py, pm_defensive.py
- model.py, predictors_engine/master_orchestrator.py
- predictors_engine/predictor_h1.py ... predictor_h10.py (o donde sea que
  aparezcan según la enumeración — presta especial atención a CUALQUIER
  subcarpeta con lógica de entrenamiento de modelos, como predictors_engine/)

Si la enumeración revela archivos o carpetas que no estaban en esta lista,
LÉELOS TAMBIÉN — la lista de arriba es orientativa, no exhaustiva. El criterio
es: si el archivo termina en .py y está en el repo, se lee.

PASO 2 — LEE LOS DATOS DE PRODUCCIÓN en /data/:
- /data/darwin/trades/*.json → pnl_real_pct, pnl_teorico_pct, days_held, exit_reason
- /data/evaluations/*/*.json → hit_sign por horizonte H1-H10
- /data/equity_snapshots.json → evolución del capital
- /data/alpha_last.json → alpha scores actuales
- /data/darwin/champion.json y /data/darwin/evolution_history.json
- /data/darwin/fitness/*.json → fitness real de cada genoma
- /data/positions.json → posiciones abiertas
- /data/market_context.json → modo actual

PASO 3 — REPORTE EXHAUSTIVO:

## 🚨 ESTADO CRÍTICO
Capital actual vs máximo histórico. Pérdida total en $ y %.

## PROBLEMAS ENCONTRADOS
Para cada problema:
### Problema N: [nombre]
- **Impacto estimado**: cuánto dinero cuesta este problema
- **Causa raíz**: explicación con referencia al código
- **Archivo**: archivo.py — función exacta o línea aproximada
- **Código actual**:
```python
[código problemático]
```
- **Código propuesto**:
```python
[código corregido]
```
- **Por qué Darwin no lo corrigió**: si aplica

Si un problema aparece repetido con variaciones menores en varios archivos
independientes (por ejemplo, la misma falta de corrección en cada predictor
de horizonte H1 a H10), no lo reportes como N problemas separados: repórtalo
UNA vez como problema de calibración/arquitectura, indicando en qué archivos
se repite y si los parámetros (alpha, features, período de entrenamiento)
varían entre ellos.

## CALIBRACIÓN ENTRE MÓDULOS
Sección específica para inconsistencias que surgen de que el sistema está
compuesto por muchos archivos independientes (uno por horizonte, uno por
portfolio manager, etc.) sin una capa de coordinación compartida. Ejemplos a
buscar activamente:
- Mismo tipo de lógica (features, targets, hiperparámetros, umbrales)
  implementada de forma distinta o inconsistente en archivos hermanos.
- Falta de una función/módulo compartido donde debería haberlo.
- Un fix aplicado en un archivo pero no en sus equivalentes.

## DARWIN — POR QUÉ NO EVOLUCIONA
Con código y datos reales: por qué el campeón lleva meses sin ser reemplazado.
Bug específico en arena.py o pnl_fitness.py si existe.

## ACCIONES INMEDIATAS
Lista ordenada por impacto con archivo y función para cada acción.

Usa SOLO datos reales. El código propuesto debe ser funcional."""


def run_audit():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp   = int(datetime.utcnow().timestamp())
    report_file = f"{REPORT_DIR}/audit_{timestamp}.txt"
    fecha       = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"🔍 Iniciando auditoría unificada...")
    print(f"💾 Guardando en: {report_file}")

    with open(report_file, 'w') as f:
        result = subprocess.run(
            [
                "claude", "-p", PROMPT,
                "--permission-mode", "acceptEdits",
                "--max-turns", "100",
                "--allowedTools", "Read,Bash,WebSearch",
                "--output-format", "text"
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=1800
        )

    from notifier import send_audit_report, send_error_alert

    if result.returncode == 0:
        print(f"✅ Auditoría completada")
        with open(report_file) as f:
            content = f.read()
        send_audit_report(content, fecha)
    else:
        print(f"❌ Error código: {result.returncode}")
        send_error_alert("Code Auditor Agent", f"returncode={result.returncode}")

if __name__ == "__main__":
    run_audit()
