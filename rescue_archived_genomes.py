"""
rescue_archived_genomes.py — [AUD-D2] Rescate único de genomas archivados
=========================================================================
Corregido en producción sobre el repo (2026-09-07).

Contexto: antes del fix de elitismo en arena.py (2026-09-05), la poda
_prune_shadow_genomes() archivaba de forma irreversible a cualquier
genoma fuera del top-8 del día, sin importar su calidad histórica.
executor_v2 (combined_fitness=+0.33, ~139 trades simulados) quedó
archivado el 2026-08-27 por ese bug, y el fix del 5-sep no es
retroactivo: _load_all_active_genomes() usa un glob que no entra a
subcarpetas, así que archived/ queda fuera del ciclo para siempre
a menos que alguien lo rescate a mano.

Este script:
  1. Lista todo lo que hay en darwin/genomes/archived/.
  2. Para cada uno, cuenta sus shadow trades reales en
     darwin/shadow_trades/{genome_id}/ (get_shadow_resolved_trades) —
     ese historial NUNCA se movió ni se borró, vive en una carpeta
     aparte del .json del genoma, así que el rescate no le hace
     empezar de cero.
  3. Reporta cuáles califican por MIN_TRADES_TO_COMPETE (el mismo
     umbral que usa el fix de elitismo del 5-sep).
  4. Sin --execute: solo reporta (dry-run), no mueve nada.
     Con --execute: mueve de vuelta a darwin/genomes/ únicamente a
     los que califican. No decide "quién es mejor" — solo les
     devuelve el derecho a competir; el próximo ciclo evolutivo
     (_select_champion) decide honestamente si alguno supera al
     campeón vigente.

Uso (correr desde /app en el shell del contenedor):
    python rescue_archived_genomes.py               # dry-run, solo reporta
    python rescue_archived_genomes.py --execute      # rescata a los que califican
"""

import sys
import argparse

from darwin_engine.executor_genome import GENOME_DIR
from darwin_engine.executor_shadow_evaluator import get_shadow_resolved_trades
from darwin_engine.arena import MIN_TRADES_TO_COMPETE


def main():
    parser = argparse.ArgumentParser(description="Rescate único de genomas archivados")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Si se pasa, mueve de vuelta los genomas que califican. Sin esto, solo reporta (dry-run).",
    )
    args = parser.parse_args()

    archive_dir = GENOME_DIR / "archived"

    if not archive_dir.exists():
        print(f"❌ No existe {archive_dir} — nada que rescatar.")
        sys.exit(1)

    archived_files = sorted(archive_dir.glob("executor_v*.json"))

    if not archived_files:
        print(f"✅ {archive_dir} está vacío — no hay genomas archivados.")
        sys.exit(0)

    print(f"📦 {len(archived_files)} genoma(s) en archived/ | umbral de rescate: "
          f"MIN_TRADES_TO_COMPETE={MIN_TRADES_TO_COMPETE}\n")

    candidatos = []
    for path in archived_files:
        genome_id = path.stem  # "executor_v2.json" -> "executor_v2"
        trades = get_shadow_resolved_trades(genome_id)
        n_trades = len(trades)
        califica = n_trades >= MIN_TRADES_TO_COMPETE

        estado = "✅ CALIFICA" if califica else "⛔ insuficiente"
        print(f"  {genome_id:<20} | shadow_trades={n_trades:<4} | {estado}")

        if califica:
            candidatos.append((genome_id, path, n_trades))

    print()

    if not candidatos:
        print("No hay ningún genoma archivado con evidencia suficiente para rescatar. Nada que hacer.")
        return

    print(f"🎯 {len(candidatos)} genoma(s) califican para rescate: "
          f"{[c[0] for c in candidatos]}")

    if not args.execute:
        print("\n(dry-run — no se movió nada. Corré con --execute para rescatarlos de verdad.)")
        return

    print("\n🚑 Ejecutando rescate...")
    for genome_id, src, n_trades in candidatos:
        dst = GENOME_DIR / f"{genome_id}.json"
        if dst.exists():
            print(f"  ⚠️ SKIP {genome_id}: ya existe un archivo activo con ese nombre en {GENOME_DIR}")
            continue
        src.rename(dst)
        print(f"  ✅ Rescatado: {genome_id} ({n_trades} shadow trades preservados) → {dst}")

    print("\nListo. En el próximo ciclo evolutivo (/internal/darwin/run), estos genomas "
          "vuelven a competir con su historial completo intacto.")


if __name__ == "__main__":
    main()
