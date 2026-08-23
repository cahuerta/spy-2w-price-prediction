#!/usr/bin/env python3
"""Agente auditor de código - Claude Code CLI"""

import subprocess
from pathlib import Path
from datetime import datetime

REPORT_DIR = "/data/audits"

def run_audit():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp   = int(datetime.utcnow().timestamp())
    report_file = f"{REPORT_DIR}/audit_{timestamp}.txt"
    fecha       = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"🔍 Iniciando auditoría...")
    print(f"💾 Guardando en: {report_file}")

    with open(report_file, 'w') as f:
        result = subprocess.run(
            [
                "claude", "-p",
                "Lee el repositorio de GitHub: cahuerta/spy-2w-price-prediction. Analiza TODO el código Python completo y detecta INCOHERENCIAS Y BUGS SERIOS. Busca: 1) Funciones definidas pero nunca usadas 2) Variables con mismo significado pero nombres distintos en archivos diferentes 3) Parámetros inconsistentes entre archivos 4) Estados que no sincronizan 5) APIs que esperan campos que no reciben 6) Código duplicado. Dame reporte detallado con hallazgos concretos.",
                "--permission-mode", "acceptEdits",
                "--max-turns", "50",
                "--allowedTools", "Read,Bash,WebSearch",
                "--output-format", "text"
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            timeout=900
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
