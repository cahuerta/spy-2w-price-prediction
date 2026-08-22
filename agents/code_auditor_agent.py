#!/usr/bin/env python3
"""
Code Auditor Agent — Detecta incoherencias en el repo usando Claude Code CLI
Ejecutable desde scheduler o manual
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from datetime import datetime

# Configuración
REPO_PATH = "/opt/render/project/src"  # Ajustar si es necesario
REPORT_DIR = "/data/audits"  # Donde guardamos reportes

class CodeAuditorAgent:
    def __init__(self, repo_path=REPO_PATH, report_dir=REPORT_DIR):
        self.repo_path = repo_path
        self.report_dir = report_dir
        self.timestamp = datetime.utcnow().isoformat()
        
        # Asegurar que el directorio de reportes existe
        Path(self.report_dir).mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Ejecuta el agente auditor"""
        print(f"🔍 Iniciando auditoría de código...")
        print(f"📁 Repo: {self.repo_path}")
        print(f"⏰ {self.timestamp}")
        
        # Prompt para Claude
        prompt = self._build_prompt()
        
        # Ejecutar claude via CLI
        try:
            result = subprocess.run(
                ["claude", "-p", prompt, "--output-format", "json"],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutos máximo
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                return self._process_output(result.stdout)
            else:
                print(f"❌ Error en ejecución:")
                print(result.stderr)
                return {"status": "error", "message": result.stderr}
                
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Auditoría excedió timeout (10 min)"}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def _build_prompt(self):
        """Construye el prompt para Claude"""
        return f"""Eres un auditor de código experto. Tu tarea es revisar el repositorio completo en {self.repo_path} 
y detectar incoherencias, bugs sutiles, y problemas de arquitectura.

Busca específicamente:
1. **Funciones no usadas** — definidas pero nunca llamadas
2. **Contratos rotos** — funciones que devuelven tipos distintos según rama, o parámetros inconsistentes
3. **Variables con nombres conflictivos** — la misma lógica con nombres distintos en archivos diferentes
4. **Imports circulares o no resueltos**
5. **Estados inconsistentes** — variables que debería sincronizarse pero no lo hacen
6. **APIs incompatibles** — endpoints que esperan ciertos campos pero no los reciben

Responde ÚNICAMENTE en JSON con este formato:
{{
  "timestamp": "{self.timestamp}",
  "repo_path": "{self.repo_path}",
  "findings": [
    {{
      "severity": "critical|high|medium|low",
      "type": "unused_function|contract_break|naming_conflict|circular_import|state_mismatch|api_incompatibility|other",
      "file": "ruta/al/archivo.py",
      "function_or_area": "nombre_función_o_zona",
      "description": "descripción concisa del problema",
      "code_snippet": "línea de código relevante si aplica",
      "suggested_fix": "cómo arreglarlo"
    }}
  ],
  "summary": "resumen ejecutivo de los hallazgos",
  "total_findings": <número>
}}"""
    
    def _process_output(self, output):
        """Procesa la salida de Claude y la guarda"""
        try:
            report = json.loads(output)
            
            # Guardar en disco
            report_file = Path(self.report_dir) / f"audit_{self.timestamp.replace(':', '-')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Auditoría completada")
            print(f"📊 Hallazgos: {report.get('total_findings', 0)}")
            print(f"💾 Guardado en: {report_file}")
            
            # Mostrar resumen en consola
            summary = report.get('summary', 'Sin resumen disponible')
            print(f"\n📋 Resumen:\n{summary}")
            
            # Si hay hallazgos críticos, log destacado
            findings = report.get('findings', [])
            critical = [f for f in findings if f.get('severity') == 'critical']
            if critical:
                print(f"\n🚨 CRÍTICOS ({len(critical)}):")
                for f in critical[:3]:  # Mostrar primeros 3
                    print(f"  - {f.get('type')}: {f.get('description')}")
            
            return report
            
        except json.JSONDecodeError as e:
            print(f"⚠️ No se pudo parsear JSON de Claude: {e}")
            print(f"Output: {output[:500]}")  # Primeros 500 chars
            return {"status": "error", "message": "Invalid JSON response", "raw_output": output}

def main():
    """Punto de entrada"""
    agent = CodeAuditorAgent()
    result = agent.run()
    
    # Retornar código de salida
    if result.get("status") == "error":
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
