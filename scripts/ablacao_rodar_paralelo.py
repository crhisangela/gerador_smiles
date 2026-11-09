"""
Roda as 5 execuções de ablação de peso_alerta em paralelo.

Uso (a partir da raiz do projeto):
    python scripts/ablacao_rodar_paralelo.py

Cada processo grava em data/execucoes/<data>_pa<valor>/
Os logs de cada processo são exibidos em tempo real com prefixo [pa=X].
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

VALORES = [0.0, 0.05, 0.1, 0.2, 0.5]
GERACOES = 20
SEMENTE = 42


def _rotulo(v: float) -> str:
    return f"pa{v}".replace(".", "_")


def _stream(proc, label, resultados, idx):
    """Lê stdout do processo linha a linha e imprime com prefixo."""
    for linha in proc.stdout:
        print(f"[{label}] {linha}", end="", flush=True)
    proc.wait()
    resultados[idx] = proc.returncode


def main():
    raiz = Path(__file__).resolve().parent.parent
    python = sys.executable

    processos = []
    threads = []
    resultados = [None] * len(VALORES)

    print(f"Iniciando {len(VALORES)} execuções em paralelo...\n")
    t0 = time.time()

    for i, v in enumerate(VALORES):
        rot = _rotulo(v)
        cmd = [
            python, "src/main_recombinacao.py",
            "--rotulo", rot,
            "--geracoes", str(GERACOES),
            "--peso-alerta", str(v),
            "--semente", str(SEMENTE),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(raiz),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        processos.append(proc)
        t = threading.Thread(target=_stream, args=(proc, f"pa={v}", resultados, i), daemon=True)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    elapsed = time.time() - t0
    print(f"\n{'='*50}")
    print(f"Concluído em {elapsed/60:.1f} min")
    for v, rc in zip(VALORES, resultados):
        status = "OK" if rc == 0 else f"ERRO (código {rc})"
        print(f"  peso_alerta={v}: {status}")

    print("\nAgora compare os resultados:")
    print("  python scripts/ablacao_peso_alerta.py --modo comparar")


if __name__ == "__main__":
    main()
