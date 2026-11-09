"""
Ablacao dos parametros de formula dos scores parciais da funcao fitness UV-Vis.

Varia, um por vez, os parametros de formula sem justificativa bibliografica:
  - Faixa de MW valida (mw_min, mw_max)
  - Numero de aneis aromaticos para pi_score maximo (pi_max_rings)
  - Numero de ligacoes rotaveis para planar_score zero (rot_max)
  - Fator de penalidade PAINS (pains_fator)

Cada variacao e avaliada sobre o pool ja gerado (data/banco_smiles_geracao_fitness*.parquet).
O criterio-alvo e fixo e independente do parametro testado:
    candidato forte = n_aneis_aromaticos >= 2  AND  D-pi-A  AND  mw in [200,700]  AND  sa <= 5
    (sa <= 5: Ertl & Schuffenhauer, J.Cheminform. 1:8, 2009)

Metrica primaria: EF10 (Enrichment Factor a 10%) — Truchon & Bayly, J.Chem.Inf.Model. 47:488, 2007.
  EF10 = (hits no top-10%) / (hits esperados por acaso) — EF10=1.0 equivale a selecao aleatoria.

Uso:
    python scripts/ablacao_formulas_scores.py
    python scripts/ablacao_formulas_scores.py --pasta data
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

COMPONENTES = ['mw_score', 'qed_score', 'sa_score', 'pi_score', 'd_a_score', 'planar_score']
PESOS_BASE  = [0.10, 0.10, 0.10, 0.30, 0.25, 0.15]  # sem fonte bibliografica
TOP_FRAC    = 0.10

# Grades de ablacao — todos os parametros sem fonte bibliografica
GRADES = {
    'mw_min':       [100, 150, 200, 250, 300],
    'mw_max':       [500, 600, 700, 800, 900],
    'pi_max_rings': [1, 2, 3, 4, 5],
    'rot_max':      [4, 6, 8, 10, 12],
    'pains_fator':  [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
}

# Valores de referencia sem justificativa bibliografica — exceto pains_fator (ver abaixo):
# PAINS — conflito bibliografico; a ablacao de pains_fator aqui usa EF10 como desempate:
#   Baell & Holloway J.Med.Chem. 53:2719,2010 (Tab.9/Tab.1): dyes flagados por absorcao OPTICA -> 1.0
#   Dahlin et al. J.Med.Chem. 58:2091,2015 (DOI:10.1021/jm5019093): PAINS tiol-reativo -> penalizar
#   Brown et al. J.Chem.Inf.Model. 59:1096,2019 (GuacaMol): PAINS como filtro drug-like -> penalizar
DEFAULTS = {
    'mw_min': 200, 'mw_max': 700,
    'pi_max_rings': 3, 'rot_max': 8,
    'pains_fator': 1.0,  # default p/ ablacoes B.1-B.4; B.5 varre [0.3,1.0] via EF10
}


def carregar_pool(pasta: str = 'data') -> pd.DataFrame:
    arqs = sorted(glob.glob(str(Path(pasta) / 'banco_smiles_geracao_fitness*.parquet')))
    if not arqs:
        sys.exit('Nenhum parquet encontrado. Execute o notebook primeiro.')
    df = pd.concat([pd.read_parquet(a) for a in arqs], ignore_index=True)
    colunas_req = COMPONENTES + ['tem_pains', 'mw', 'sa', 'rot',
                                  'n_aneis_aromaticos', 'n_doadores', 'n_aceptores']
    faltando = [c for c in colunas_req if c not in df.columns]
    if faltando:
        sys.exit(f'Colunas ausentes no pool: {faltando}')
    return df


def candidato_forte(df: pd.DataFrame) -> pd.Series:
    """Criterio fixo — independente dos parametros sendo ablateados."""
    return (
        (df['n_aneis_aromaticos'] >= 2) &
        (df['n_doadores'] >= 1) &
        (df['n_aceptores'] >= 1) &
        df['mw'].between(200, 700) &
        (df['sa'] <= 5)
    ).astype(int)


# Funcoes de score sem fonte bibliografica — parametros ablateados neste script

def mw_score_fn(mw: pd.Series, mw_min: float, mw_max: float) -> pd.Series:
    decaimento = 300.0  # sem fonte
    s = pd.Series(1.0, index=mw.index)
    s = s.where(mw >= mw_min, mw / mw_min)
    s = s.where(mw <= mw_max, (1.0 - (mw - mw_max) / decaimento).clip(lower=0.0))
    return s


def pi_score_fn(n_aromaticos: pd.Series, pi_max: int) -> pd.Series:
    return (n_aromaticos / pi_max).clip(upper=1.0)


def planar_score_fn(rot: pd.Series, rot_max: int) -> pd.Series:
    return (1.0 - rot / rot_max).clip(lower=0.0)


def fitness_completo(df: pd.DataFrame, scores_override: dict,
                     pains_fator: float) -> pd.Series:
    """Recalcula fitness com os scores sobrepostos e o pains_fator dado."""
    fator = df['tem_pains'].map({True: pains_fator, False: 1.0}).fillna(1.0)
    scores = {}
    for i, c in enumerate(COMPONENTES):
        scores[c] = scores_override.get(c, df[c])
    return sum(PESOS_BASE[i] * scores[c] for i, c in enumerate(COMPONENTES)) * fator


def avaliar(df: pd.DataFrame, fit: pd.Series, forte: pd.Series) -> dict:
    """
    EF10 = Enrichment Factor a 10% — Truchon & Bayly, J.Chem.Inf.Model. 47:488, 2007.
    EF10 = 1.0 equivale a selecao aleatoria; EF10 > 1 indica enriquecimento real.
    """
    n_total = len(df)
    n_top   = max(1, int(n_total * TOP_FRAC))
    top_i   = fit.nlargest(n_top).index
    taxa_base = forte.sum() / n_total if n_total > 0 else 1e-9
    taxa_top  = forte.loc[top_i].mean()
    ef10      = taxa_top / taxa_base if taxa_base > 0 else 0.0
    rho, _    = spearmanr(fit, forte)
    return {
        'ef10':            round(float(ef10), 4),   # EF10: Truchon & Bayly 2007
        'rho_spearman':    round(float(rho), 5),
        'pct_forte_top10': round(taxa_top * 100, 3),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pasta', default='data')
    args = ap.parse_args()

    df    = carregar_pool(args.pasta)
    forte = candidato_forte(df)
    n_forte = int(forte.sum())
    print(f'Pool: {len(df):,} moleculas | candidatos fortes: {n_forte} ({n_forte/len(df)*100:.1f}%)\n')

    rows = []

    # ── 1. mw_min (mw_max fixo = 700) ────────────────────────────────────
    print('--- Ablacao mw_min (mw_max fixo = 700) ---')
    for v in GRADES['mw_min']:
        novo_mw = mw_score_fn(df['mw'], mw_min=v, mw_max=DEFAULTS['mw_max'])
        fit = fitness_completo(df, {'mw_score': novo_mw}, DEFAULTS['pains_fator'])
        m   = avaliar(df, fit, forte)
        ref = ' <-- ref.s/fonte' if v == DEFAULTS['mw_min'] else ''
        rows.append({'parametro': 'mw_min', 'valor': v, **m})
        print(f'  mw_min={v:>4}  ef10={m["ef10"]:5.3f}  pct_forte={m["pct_forte_top10"]:6.2f}%{ref}')

    # ── 2. mw_max (mw_min fixo = 200) ────────────────────────────────────
    print('\n--- Ablacao mw_max (mw_min fixo = 200) ---')
    for v in GRADES['mw_max']:
        novo_mw = mw_score_fn(df['mw'], mw_min=DEFAULTS['mw_min'], mw_max=v)
        fit = fitness_completo(df, {'mw_score': novo_mw}, DEFAULTS['pains_fator'])
        m   = avaliar(df, fit, forte)
        ref = ' <-- ref.s/fonte' if v == DEFAULTS['mw_max'] else ''
        rows.append({'parametro': 'mw_max', 'valor': v, **m})
        print(f'  mw_max={v:>4}  ef10={m["ef10"]:5.3f}  pct_forte={m["pct_forte_top10"]:6.2f}%{ref}')

    # ── 3. pi_max_rings ───────────────────────────────────────────────────
    print('\n--- Ablacao pi_max_rings ---')
    for v in GRADES['pi_max_rings']:
        novo_pi = pi_score_fn(df['n_aneis_aromaticos'], pi_max=v)
        fit = fitness_completo(df, {'pi_score': novo_pi}, DEFAULTS['pains_fator'])
        m   = avaliar(df, fit, forte)
        ref = ' <-- ref.s/fonte' if v == DEFAULTS['pi_max_rings'] else ''
        rows.append({'parametro': 'pi_max_rings', 'valor': v, **m})
        print(f'  pi_max={v}  ef10={m["ef10"]:5.3f}  pct_forte={m["pct_forte_top10"]:6.2f}%{ref}')

    # ── 4. rot_max ────────────────────────────────────────────────────────
    print('\n--- Ablacao rot_max (planar_score) ---')
    for v in GRADES['rot_max']:
        novo_pl = planar_score_fn(df['rot'], rot_max=v)
        fit = fitness_completo(df, {'planar_score': novo_pl}, DEFAULTS['pains_fator'])
        m   = avaliar(df, fit, forte)
        ref = ' <-- ref.s/fonte' if v == DEFAULTS['rot_max'] else ''
        rows.append({'parametro': 'rot_max', 'valor': v, **m})
        print(f'  rot_max={v:>2}  ef10={m["ef10"]:5.3f}  pct_forte={m["pct_forte_top10"]:6.2f}%{ref}')

    # ── 5. pains_fator — PAINS: Baell & Holloway J.Med.Chem. 2010 (HTS, nao UV-Vis) ──
    print('\n--- Ablacao pains_fator (PAINS: Baell & Holloway J.Med.Chem. 53:2719, 2010) ---')
    for v in GRADES['pains_fator']:
        fit = fitness_completo(df, {}, pains_fator=v)
        m   = avaliar(df, fit, forte)
        ref = ' <-- ref.s/fonte' if abs(v - DEFAULTS['pains_fator']) < 1e-6 else ''
        rows.append({'parametro': 'pains_fator', 'valor': v, **m})
        print(f'  pains_fator={v:.2f}  ef10={m["ef10"]:5.3f}  pct_forte={m["pct_forte_top10"]:6.2f}%{ref}')

    df_res = pd.DataFrame(rows)
    saida  = Path(args.pasta) / 'ablacao_formulas_scores.csv'
    df_res.to_csv(saida, index=False)
    print(f'\nResultados salvos em {saida}')

    # Resumo: melhor valor por parametro
    print('\n=== MELHOR VALOR POR PARAMETRO (por EF10 — Truchon & Bayly 2007) ===')
    print('  EF10=1.00 = aleatorio; EF10>1.00 = enriquecimento real')
    for param in ['mw_min', 'mw_max', 'pi_max_rings', 'rot_max', 'pains_fator']:
        sub     = df_res[df_res['parametro'] == param]
        melhor  = sub.loc[sub['ef10'].idxmax()]
        default = DEFAULTS[param]
        flag    = '' if melhor['valor'] == default else f'  (ref.s/fonte era {default})'
        print(f'  {param:<15}: {melhor["valor"]}  ef10={melhor["ef10"]:.4f}{flag}')


if __name__ == '__main__':
    main()
