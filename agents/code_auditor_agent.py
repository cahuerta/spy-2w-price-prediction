#!/usr/bin/env python3
"""
Code Auditor Agent — Auditoría automática del repo usando Claude Code CLI
Ejecutable desde scheduler
"""

import subprocess
import json
import os
from pathlib import Path
from datetime import datetime

REPORT_DIR = "/data/audits"

class CodeAuditorAgent:
    def __init__(self, report_dir=REPORT_DIR):
        self.report_dir = report_dir
        self.timestamp = datetime.utcnow().isoformat()
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Ejecuta el agente auditor"""
        print(f"🔍 Iniciando auditoría de código...")
        print(f"⏰ {self.timestamp}")
        
        prompt = self._build_prompt()
        
        # Comando correcto para claude headless sin bloqueo
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--permission-mode", "acceptEdits",  # No espera confirmación
            "--max-turns", "5",  # Limita iteraciones
            "--allowedTools", "Read,Bash,WebSearch",  # Solo lectura + búsqueda
            "--cwd", "/opt/render/project/src"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                stdin=subprocess.DEVNULL  # No espera input
            )
            
            if result.returncode == 0:
                print(f"✅ Auditoría completada")
                return self._process_output(result.stdout)
            else:
                print(f"❌ Error (código {result.returncode})")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}")
                if result.stdout:
                    print(f"Output: {result.stdout[:500]}")
                return {"status": "error", "stderr": result.stderr[:500]}
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout (10 minutos)")
            return {"status": "error", "message": "Timeout"}
        except Exception as e:
            print(f"❌ Excepción: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _build_prompt(self):
        """Construye el prompt para Claude"""
        return """Lee el repositorio de GitHub: cahuerta/spy-2w-price-prediction

Analiza TODO el código Python completo y detecta INCOHERENCIAS Y BUGS SERIOS.

Busca:
1. Funciones definidas pero nunca usadas
2. Variables con el mismo significado pero nombres distintos en archivos diferentes
3. Parámetros que se pasan a funciones pero la función no los acepta
4. Estados que deberían sincronizarse pero no lo hacen
5. APIs/endpoints que esperan campos que no reciben
6. Código duplicado que podría reutilizarse
7. Valores hardcodeados que deberían ser configurables

RESPONDE SOLO EN JSON (válido):
{
  "findings": [
    {
      "severity": "critical|high|medium",
      "file": "ruta/archivo.py",
      "issue": "descripción breve del problema",
      "location": "línea/función aproximada"
    }
  ],
  "summary": "resumen de hallazgos principales",
  "total": <número>
}"""
    
    def _process_output(self, output):
        """Procesa y guarda la salida"""
        try:
            report = json.loads(output)
            
            # Guardar
            report_file = Path(self.report_dir) / f"audit_{self.timestamp.replace(':', '-')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            total = report.get('total', 0)
            summary = report.get('summary', 'Sin resumen')
            
            print(f"📊 Hallazgos: {total}")
            print(f"📋 {summary[:200]}")
            print(f"💾 Guardado en: {report_file}")
            
            return report
            
        except json.JSONDecodeError:
            print(f"⚠️ No se pudo parsear JSON")
            print(f"Raw output: {output[:300]}")
            return {"status": "parse_error", "raw": output[:500]}

def main():
    agent = CodeAuditorAgent()
    agent.run()

if __name__ == "__main__":
    main()
