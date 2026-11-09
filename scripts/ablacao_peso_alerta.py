"""
Ablação do hiperparâmetro peso_alerta.

Uso:
    # 1. Imprimir os comandos para rodar manualmente:
    python scripts/ablacao_peso_alerta.py --modo comandos

    # 2. Comparar execuções já concluídas e gerar o relatório CSV:
    python scripts/ablacao_peso_alerta.py --modo comparar

Os rótulos seguem o padrão  pa<valor>  (ex.: pa0.1) para fácil identificação
na pasta data/execucoes/.
"""

import argparse
import os
import sys
import glob
import pandas as pd

VALORES = [0.0, 0.05, 0.1, 0.2, 0.5]
GERACOES = 20          # bastam para ver a convergência
SEMENTE = 42           # mesma semente para reprodutibilidade


def _rotulo(v: float) -> str:
    return f"pa{v}".replace(".", "_")


def modo_comandos():
    print("# Rode cada linha abaixo (em paralelo ou sequencialmente):\n")
    for v in VALORES:
        rot = _rotulo(v)
        cmd = (
            f"python src/main_recombinacao.py"
            f" --rotulo {rot}"
            f" --geracoes {GERACOES}"
            f" --peso-alerta {v}"
            f" --semente {SEMENTE}"
        )
        print(cmd)
    print(
        "\n# Após todas concluírem, compare com:\n"
        "python scripts/ablacao_peso_alerta.py --modo comparar"
    )


def modo_comparar():
    DIR_EXEC = "data/execucoes"
    if not os.path.isdir(DIR_EXEC):
        sys.exit(f"Pasta {DIR_EXEC!r} não encontrada. Rode a partir da raiz do projeto.")

    rows = []
    for v in VALORES:
        rot = _rotulo(v)
        # Encontra a pasta mais recente com esse rótulo
        padrao = os.path.join(DIR_EXEC, f"*_{rot}", "geracoes")
        candidatas = sorted(glob.glob(padrao))
        if not candidatas:
            print(f"[aviso] Nenhuma execução encontrada para peso_alerta={v} (rótulo={rot!r})")
            continue
        pasta_ger = candidatas[-1]  # mais recente

        for g in range(1, GERACOES + 1):
            arq = os.path.join(pasta_ger, f"g{g:04d}_aceitas.parquet")
            if not os.path.exists(arq):
                continue
            df = pd.read_parquet(arq)
            if df.empty:
                continue
            al_cols = [
                "alerta_carga_liquida", "alerta_S_hipervalente",
                "alerta_heteroatomo_halogenado", "alerta_radical",
            ]
            al_ok = [c for c in al_cols if c in df.columns]
            sem_alerta = (~df[al_ok].any(axis=1)).mean() * 100 if al_ok else float("nan")
            ps_mean = df["Peso_Semente"].mean() if "Peso_Semente" in df.columns else float("nan")
            rows.append({
                "peso_alerta": v,
                "geracao": g,
                "n_aceitas": len(df),
                "sem_alerta_pct": round(sem_alerta, 2),
                "peso_semente_mean": round(ps_mean, 4),
            })

    if not rows:
        sys.exit("Nenhum dado encontrado. Verifique se as execuções foram concluídas.")

    df_res = pd.DataFrame(rows)
    out = "data/ablacao_peso_alerta.csv"
    df_res.to_csv(out, index=False)
    print(f"Salvo em {out}\n")

    # Resumo: como cada valor convergiu
    pivot = df_res.pivot_table(
        index="geracao", columns="peso_alerta", values="sem_alerta_pct"
    )
    print("=== % sem alerta por geração ===")
    print(pivot.to_string(float_format=lambda x: f"{x:.1f}"))

    # Geração em que atinge 90%
    print("\n=== Geração de convergência (primeira vez ≥ 90% sem alerta) ===")
    for v in VALORES:
        sub = df_res[(df_res.peso_alerta == v) & (df_res.sem_alerta_pct >= 90)]
        g90 = sub["geracao"].min() if not sub.empty else ">20"
        print(f"  peso_alerta={v}: G{g90}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--modo", choices=["comandos", "comparar"], default="comandos")
    args = ap.parse_args()
    if args.modo == "comandos":
        modo_comandos()
    else:
        modo_comparar()
