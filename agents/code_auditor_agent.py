#!/usr/bin/env python3
"""Agente auditor de código - Claude Code CLI"""

import subprocess
import os
from datetime import datetime
from pathlib import Path

REPORT_DIR = "/data/audits"

def run_audit():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp = int(datetime.utcnow().timestamp())
    report_file = f"{REPORT_DIR}/audit_{timestamp}.txt"

    print(f"🔍 Iniciando auditoría...")
    print(f"💾 Guardando en: {report_file}")

    with open(report_file, 'w') as f:
        result = subprocess.run(
            [
                "claude", "-p",
                "Lee el repo cahuerta/spy-2w-price-prediction de GitHub. Analiza TODO el código Python. Detecta: 1) Funciones definidas pero nunca usadas 2) Variables con mismo significado pero nombres distintos en archivos diferentes 3) Parámetros inconsistentes entre archivos 4) Estados que no sincronizan 5) APIs que esperan campos que no reciben 6) Código duplicado. Dame reporte detallado.",
                "--permission-mode", "acceptEdits",
                "--max-turns", "5",
                "--allowedTools", "Read,Bash,WebSearch",
                "--output-format", "text"
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=900
        )

    if result.returncode == 0:
        print(f"✅ Auditoría completada")
        with open(report_file) as f:
            print(f.read()[:500])
    else:
        print(f"❌ Error código: {result.returncode}")
        with open(report_file) as f:
            print(f.read()[:300])

if __name__ == "__main__":
    run_audit()
    
