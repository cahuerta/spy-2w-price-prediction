cat > agents/code_auditor_agent.py << 'EOF'
#!/usr/bin/env python3
"""Code Auditor Agent - Auditoría de código usando Claude Code CLI"""

import subprocess
import json
from pathlib import Path
from datetime import datetime

REPORT_DIR = "/data/audits"

class CodeAuditorAgent:
    def __init__(self):
        self.report_dir = REPORT_DIR
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.utcnow().isoformat()
    
    def run(self):
        """Ejecuta la auditoría"""
        print(f"🔍 Iniciando auditoría de código...")
        print(f"⏰ {self.timestamp}")
        
        prompt = """Lee el repositorio de GitHub: cahuerta/spy-2w-price-prediction

Analiza TODO el código Python completo y detecta INCOHERENCIAS Y BUGS SERIOS.

Busca:
1. Funciones definidas pero nunca usadas
2. Variables con el mismo significado pero nombres distintos en archivos diferentes
3. Parámetros que se pasan a funciones pero la función no los acepta
4. Estados que deberían sincronizarse pero no lo hacen
5. APIs/endpoints que esperan campos que no reciben
6. Código duplicado que podría reutilizarse
7. Valores hardcodeados que deberían ser configurables

RESPONDE SOLO EN JSON válido:
{
  "findings": [
    {
      "severity": "critical|high|medium",
      "file": "ruta/archivo.py",
      "issue": "descripción breve",
      "location": "línea o función"
    }
  ],
  "summary": "resumen principal",
  "total": 0
}"""
        
        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--permission-mode", "acceptEdits",
            "--max-turns", "5",
            "--allowedTools", "Read,Bash,WebSearch"
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,
                stdin=subprocess.DEVNULL
            )
            
            if result.returncode == 0:
                print(f"✅ Auditoría completada")
                self._save_report(result.stdout)
            else:
                print(f"❌ Error: {result.stderr[:300]}")
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout (15 minutos)")
        except Exception as e:
            print(f"❌ Excepción: {str(e)}")
    
    def _save_report(self, output):
        """Guarda el reporte"""
        try:
            report = json.loads(output)
            report_file = Path(self.report_dir) / f"audit_{self.timestamp.replace(':', '-')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            total = report.get('total', 0)
            summary = report.get('summary', 'Sin resumen')
            print(f"📊 Hallazgos: {total}")
            print(f"📋 {summary[:200]}")
            print(f"💾 Guardado en: {report_file}")
        except json.JSONDecodeError:
            print(f"⚠️ No se pudo parsear JSON")
            print(f"Output: {output[:500]}")

if __name__ == "__main__":
    CodeAuditorAgent().run()
EOF
