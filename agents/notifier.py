#!/usr/bin/env python3
"""
notifier.py — Módulo centralizado de notificaciones
Usado por agentes del sistema (code_auditor, etc.)
"""

import os
import httpx
from datetime import datetime

FROM_EMAIL = "info@icarticular.cl"
TO_EMAIL   = "cahuerta@gmail.com"

def send_email(subject: str, body: str, is_html: bool = True) -> bool:
    """
    Envía correo via Resend.
    Retorna True si fue exitoso, False si falló.
    """
    resend_key = os.getenv("RESEND_API_KEY", "")
    if not resend_key:
        print("⚠️ RESEND_API_KEY no definida — correo omitido")
        return False

    try:
        html_body = body.replace("\n", "<br>") if not is_html else body

        res = httpx.post(
            "https://api.resend.com/emails",
            headers={
                "Authorization":  f"Bearer {resend_key}",
                "Content-Type":   "application/json",
            },
            json={
                "from":    FROM_EMAIL,
                "to":      [TO_EMAIL],
                "subject": subject,
                "html":    html_body,
            },
            timeout=15,
        )

        if res.status_code == 200:
            print(f"✅ Correo enviado → {TO_EMAIL} | {subject}")
            return True
        else:
            print(f"❌ Resend error {res.status_code}: {res.text[:200]}")
            return False

    except Exception as e:
        print(f"❌ Error enviando correo: {e}")
        return False


def send_audit_report(report_text: str, fecha: str = None) -> bool:
    """
    Envía reporte de auditoría de código formateado.
    """
    if not fecha:
        fecha = datetime.utcnow().strftime("%Y-%m-%d")

    subject = f"🔍 Auditoría APP PREDICTIVA — {fecha}"

    body = f"""
<h2>Auditoría de Código — {fecha}</h2>
<p>El agente completó el análisis del repositorio <strong>cahuerta/spy-2w-price-prediction</strong>.</p>
<hr>
<pre style="background:#f4f4f4;padding:16px;border-radius:6px;font-size:13px;">
{report_text[:8000]}
</pre>
"""
    if len(report_text) > 8000:
        body += "<p><em>⚠️ Reporte truncado — ver archivo completo en /data/audits/</em></p>"

    return send_email(subject, body, is_html=True)


def send_error_alert(source: str, error: str) -> bool:
    """
    Envía alerta de error de cualquier agente.
    """
    fecha   = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    subject = f"❌ Error en {source} — {fecha}"
    body    = f"""
<h2>Error en {source}</h2>
<p><strong>Fecha:</strong> {fecha}</p>
<p><strong>Error:</strong></p>
<pre style="background:#fff0f0;padding:16px;border-radius:6px;">
{error[:2000]}
</pre>
<p>Revisar logs en Render para más detalles.</p>
"""
    return send_email(subject, body, is_html=True)


if __name__ == "__main__":
    # Test rápido
    ok = send_email(
        subject="✅ Test notifier APP PREDICTIVA",
        body="<p>Si recibes este correo, el notifier funciona correctamente.</p>",
    )
    print("Test OK" if ok else "Test FALLÓ")
