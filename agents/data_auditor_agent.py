#!/usr/bin/env python3
"""
data_auditor_agent.py — Auditor de producción APP PREDICTIVA
Analiza datos reales del sistema y propone mejoras si está perdiendo dinero.
Usa Claude Code CLI para análisis profundo con acceso a /data completo.
"""

import subprocess
from pathlib import Path
from datetime import datetime

REPORT_DIR = "/data/audits"

PROMPT = """Eres un auditor experto en sistemas de trading algorítmico.
Tu tarea es analizar el estado real de producción del sistema APP PREDICTIVA
que está corriendo en este servidor.

Lee y analiza los siguientes archivos en /data/:

1. /data/darwin/trades/ — todos los trades reales ejecutados
   - pnl_real_pct, pnl_teorico_pct, days_held, exit_reason, ticker
   - Calcula: win rate real, PnL promedio, drawdown máximo

2. /data/evaluations/ — precisión de predicciones por ticker
   - hit_sign, real_return_pct, predicted_direction
   - Calcula: accuracy del modelo por ticker y horizonte

3. /data/equity_snapshots.json — evolución del capital
   - ¿El capital está subiendo o bajando en el tiempo?
   - ¿Cuándo fueron las peores caídas?

4. /data/alpha_last.json — señales actuales
   - ¿Qué tickers están siendo seleccionados?
   - ¿Son los que mejor han performado históricamente?

5. /data/darwin/champion.json — estado del genoma campeón
   - ¿Cuándo fue la última evolución?
   - ¿El campeón está mejorando?

6. /data/positions.json — posiciones abiertas actuales
   - ¿Cuánto tiempo llevan abiertas?
   - ¿Están en ganancia o pérdida?

7. /data/market_context.json — modo actual (growth/neutral/defensive)
   - ¿El modo es apropiado para el contexto de mercado?

8. /data/positions_meta.json — metadata de posiciones
   - entry_date, confidence original al abrir

CON TODOS ESOS DATOS, genera un reporte ejecutivo que incluya:

## ESTADO GENERAL
- ¿El sistema está ganando o perdiendo dinero? ¿Cuánto?
- ¿Cuál es el win rate real vs teórico?
- ¿El modelo predice bien la dirección pero el sistema pierde por otros motivos?

## DIAGNÓSTICO DE PROBLEMAS
- Identifica las 3-5 causas principales de pérdidas si las hay
- Compara PnL real vs PnL teórico — ¿dónde se pierde la diferencia?
- ¿Hay tickers que siempre pierden? ¿Hay tickers que siempre ganan?
- ¿El sistema cierra posiciones ganadoras demasiado pronto?
- ¿El sistema mantiene posiciones perdedoras demasiado tiempo?

## MEJORAS CONCRETAS RECOMENDADAS
Para cada problema identificado, propone un cambio específico y accionable:
- ¿Ajustar umbrales de stop_loss o take_profit?
- ¿Cambiar el modo de mercado?
- ¿Excluir ciertos tickers?
- ¿Cambiar la frecuencia de trading?
- ¿Ajustar el confidence mínimo?

## ALERTA URGENTE
Si el sistema está perdiendo más del 5% del capital, indícalo claramente
al inicio del reporte con 🚨 ALERTA CRÍTICA.

Sé específico y usa números reales de los archivos. No inventes datos."""

def run_data_audit():
    Path(REPORT_DIR).mkdir(parents=True, exist_ok=True)
    timestamp   = int(datetime.utcnow().timestamp())
    report_file = f"{REPORT_DIR}/data_audit_{timestamp}.txt"
    fecha       = datetime.utcnow().strftime("%Y-%m-%d")

    print(f"📊 Iniciando auditoría de producción...")
    print(f"💾 Guardando en: {report_file}")

    with open(report_file, 'w') as f:
        result = subprocess.run(
            [
                "claude", "-p", PROMPT,
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

    from notifier import send_email

    if result.returncode == 0:
        print(f"✅ Auditoría de producción completada")
        with open(report_file) as f:
            content = f.read()

        # Detectar si hay alerta crítica para el subject
        alerta = "🚨 ALERTA CRÍTICA" if "ALERTA CRÍTICA" in content else "✅ OK"
        subject = f"📊 Auditoría Producción APP PREDICTIVA — {alerta} — {fecha}"

        body = f"""
<h2>Auditoría de Producción — {fecha}</h2>
<p>Análisis completo del sistema basado en datos reales de /data/</p>
<hr>
<pre style="background:#f4f4f4;padding:16px;border-radius:6px;font-size:13px;white-space:pre-wrap;">
{content[:10000]}
</pre>
"""
        if len(content) > 10000:
            body += "<p><em>⚠️ Reporte truncado — ver archivo completo en /data/audits/</em></p>"

        send_email(subject, body)
    else:
        print(f"❌ Error código: {result.returncode}")
        from notifier import send_error_alert
        send_error_alert("Data Auditor Agent", f"returncode={result.returncode}")

if __name__ == "__main__":
    run_data_audit()
