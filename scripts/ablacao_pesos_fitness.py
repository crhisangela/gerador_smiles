"""
Ablacao dos pesos da funcao fitness UV-Vis.

Grade deterministica sobre o simplex unitario com passo=0.10.
Cada configuracao e avaliada sobre o pool ja gerado
(data/banco_smiles_geracao_fitness*.parquet) — nenhuma nova execucao e necessaria.

Metrica: Enrichment Factor a 10% (EF10) — Truchon & Bayly, J.Chem.Inf.Model. 47:488, 2007.
  EF10 = (hits no top-10%) / (hits esperados por acaso no top-10%)
       = (n_fortes_top10 / n_top10) / (n_fortes_total / n_total)

Criterio-alvo fixo (independente dos pesos testados):
    n_aneis_aromaticos >= 2  AND  n_doadores >= 1  AND  n_aceptores >= 1
    AND  mw em [200, 700]  AND  sa <= 5  (Ertl & Schuffenhauer, J.Cheminform. 1:8, 2009)

Uso:
    python scripts/ablacao_pesos_fitness.py
    python scripts/ablacao_pesos_fitness.py --passo 0.05 --pasta data
"""

import argparse
import glob
import sys
from itertools import product
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

COMPONENTES = ['mw_score', 'qed_score', 'sa_score', 'pi_score', 'd_a_score', 'planar_score']
# PAINS — conflito bibliografico; ablacao B.5 no notebook usa EF10 como desempate:
#   Baell & Holloway J.Med.Chem. 53:2719,2010 (Tab.9/Tab.1): dyes flagados por absorcao OPTICA -> 1.0
#   Dahlin et al. J.Med.Chem. 58:2091,2015 (DOI:10.1021/jm5019093): PAINS tiol-reativo -> penalizar
#   Brown et al. J.Chem.Inf.Model. 59:1096,2019 (GuacaMol): PAINS como filtro drug-like -> penalizar
# Valor abaixo e o default para ablacoes A/B; B.5 varre [0.3,1.0] via EF10.
PAINS_FATOR = 1.0
TOP_FRAC    = 0.10   # fracao para EF10 — Truchon & Bayly 2007

# Configuracoes de referencia — sem fonte bibliografica; incluidas para comparacao
CONFIGS_REF = {
    'base':         [0.10, 0.10, 0.10, 0.30, 0.25, 0.15],  # valores originais sem fonte
    'uvvis_puro':   [0.05, 0.05, 0.05, 0.40, 0.35, 0.10],
    'sintetizavel': [0.15, 0.20, 0.20, 0.20, 0.15, 0.10],
    'uniforme':     [1/6,  1/6,  1/6,  1/6,  1/6,  1/6 ],
    'planaridade':  [0.05, 0.05, 0.10, 0.30, 0.20, 0.30],
    'farmaco_like': [0.20, 0.25, 0.20, 0.15, 0.10, 0.10],
}


def grade_simplex(n_pesos: int, passo: float):
    """
    Gera TODOS os vetores de pesos deterministicos com soma=1 e passo dado.

    Stars-and-bars: solucoes inteiras nao-negativas de w1+...+wk = N (N=1/passo).
    Total de combinacoes para k=6, passo=0.1: C(15,5) = 3003.
    """
    n = round(1.0 / passo)

    def _enum(k, restante):
        if k == 1:
            yield (restante,)
            return
        for v in range(0, restante + 1):
            for resto in _enum(k - 1, restante - v):
                yield (v,) + resto

    for combo in _enum(n_pesos, n):
        yield [c / n for c in combo]


def carregar_pool(pasta: str = 'data') -> pd.DataFrame:
    arqs = sorted(glob.glob(str(Path(pasta) / 'banco_smiles_geracao_fitness*.parquet')))
    if not arqs:
        sys.exit(f'Nenhum parquet encontrado. Execute notebook_geracao_fitness_uvvis.ipynb primeiro.')
    df = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
    req = COMPONENTES + ['tem_pains', 'mw', 'sa', 'n_aneis_aromaticos', 'n_doadores', 'n_aceptores']
    faltando = [c for c in req if c not in df.columns]
    if faltando:
        sys.exit(f'Colunas ausentes: {faltando}')
    return df


def candidato_forte(df: pd.DataFrame) -> pd.Series:
    # Criterio fixo — sa <= 5: Ertl & Schuffenhauer J.Cheminform. 1:8, 2009
    return (
        (df['n_aneis_aromaticos'] >= 2) &
        (df['n_doadores'] >= 1) &
        (df['n_aceptores'] >= 1) &
        df['mw'].between(200, 700) &
        (df['sa'] <= 5)
    ).astype(int)


def calcular_fitness(df: pd.DataFrame, pesos: list,
                     pains_fator: float = PAINS_FATOR) -> pd.Series:
    """Calcula o fitness ponderado; ``pains_fator`` é o multiplicador para PAINS."""
    fator = df['tem_pains'].map({True: pains_fator, False: 1.0}).fillna(1.0)
    return sum(pesos[i] * df[c] for i, c in enumerate(COMPONENTES)) * fator


def avaliar(df: pd.DataFrame, fit: pd.Series, forte: pd.Series) -> dict:
    """
    EF10 = Enrichment Factor a 10% — Truchon & Bayly, J.Chem.Inf.Model. 47:488, 2007.
    rho_spearman: correlacao de Spearman entre fitness e rotulo binario.
    """
    n_total = len(df)
    n_top   = max(1, int(n_total * TOP_FRAC))
    top_i   = fit.nlargest(n_top).index
    n_fortes_total = forte.sum()
    n_fortes_top   = forte.loc[top_i].sum()
    taxa_base = n_fortes_total / n_total if n_total > 0 else 1e-9
    taxa_top  = n_fortes_top / n_top     if n_top  > 0 else 0.0
    ef10      = taxa_top / taxa_base if taxa_base > 0 else 0.0
    rho, _    = spearmanr(fit, forte)
    return {
        'ef10':          round(float(ef10), 4),   # EF10: Truchon & Bayly 2007
        'rho_spearman':  round(float(rho), 5),
        'pct_forte_top10': round(taxa_top * 100, 3),
    }


def avaliar_config(df: pd.DataFrame, pesos: list, forte: pd.Series, origem: str) -> dict:
    """Calcula fitness, avalia EF10/Spearman e devolve um dict com os pesos e métricas."""
    fit = calcular_fitness(df, pesos)
    m   = avaliar(df, fit, forte)
    return {
        **{f'w_{c.replace("_score","")}': round(float(w), 6)
           for c, w in zip(COMPONENTES, pesos)},
        **m, 'origem': origem,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--passo', type=float, default=0.10,
                    help='Passo da grade deterministica (default 0.10 → 3003 combinacoes)')
    ap.add_argument('--pasta', default='data')
    args = ap.parse_args()

    df    = carregar_pool(args.pasta)
    forte = candidato_forte(df)
    n_fortes = int(forte.sum())
    taxa_base = n_fortes / len(df) * 100
    print(f'Pool: {len(df):,} moleculas | candidatos fortes: {n_fortes} ({taxa_base:.1f}%)')
    print(f'EF10 aleatorio esperado: 1.00  (referencia: Truchon & Bayly 2007)\n')

    rows = []
    n_total_grade = 0
    for pesos in grade_simplex(len(COMPONENTES), args.passo):
        rows.append(avaliar_config(df, pesos, forte, 'grade'))
        n_total_grade += 1
        if n_total_grade % 500 == 0:
            print(f'  {n_total_grade} combinacoes avaliadas...')

    for nome, pesos in CONFIGS_REF.items():
        rows.append(avaliar_config(df, pesos, forte, nome))

    df_res = (pd.DataFrame(rows)
              .sort_values('ef10', ascending=False)
              .reset_index(drop=True))

    saida = Path(args.pasta) / 'ablacao_pesos_fitness.csv'
    df_res.to_csv(saida, index=False)
    print(f'\n{n_total_grade} configuracoes da grade + {len(CONFIGS_REF)} de referencia avaliadas.')
    print(f'Resultados salvos em {saida}')

    cols = ['origem', 'w_mw', 'w_qed', 'w_sa', 'w_pi', 'w_d_a', 'w_planar',
            'ef10', 'rho_spearman', 'pct_forte_top10']
    print(f'\n=== TOP-10 (por EF10 — Truchon & Bayly 2007) ===')
    print(df_res[cols].head(10).to_string(index=True))

    print('\n=== CONFIGS DE REFERENCIA (sem fonte) ===')
    print(df_res[df_res['origem'] != 'grade'][cols].to_string(index=False))

    melhor = df_res.iloc[0]
    print('\n=== MELHOR CONFIGURACAO ===')
    for c in COMPONENTES:
        k = f'w_{c.replace("_score","")}'
        print(f'  {k:<10}: {melhor[k]:.4f}')
    print(f'  ef10           : {melhor["ef10"]:.4f}  (EF10 aleatorio = 1.00)')
    print(f'  rho_spearman   : {melhor["rho_spearman"]:.4f}')

    top50 = df_res[df_res['origem'] == 'grade'].head(50)
    print('\n=== DISTRIBUICAO DOS PESOS NO TOP-50 DA GRADE ===')
    for c in COMPONENTES:
        k = f'w_{c.replace("_score","")}'
        print(f'  {k:<10}: media={top50[k].mean():.4f}  std={top50[k].std():.4f}')


if __name__ == '__main__':
    main()
