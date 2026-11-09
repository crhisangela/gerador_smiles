"""
Ablacao dos parametros do algoritmo genetico (k_torneio e n_sementes).

Roda um mini-loop evolutivo (10 geracoes, 300 sementes iniciais) para cada
combinacao de (k_torneio, n_sementes), usando a mesma semente aleatoria.

Metricas de comparacao:
  - total_candidatos_fortes: soma de candidatos fortes gerados em todas as geracoes
  - fitness_medio_final: fitness medio na ultima geracao com moleculas novas
  - n_geracoes: quantas geracoes produziram moleculas novas
  - total_novas: total de moleculas novas geradas

Criterio "candidato forte" (fixo):
    n_aneis_aromaticos >= 2  AND  D-pi-A  AND  mw in [200, 700]  AND  sa <= 5

Uso:
    python scripts/ablacao_ag_params.py
    python scripts/ablacao_ag_params.py --base data/banco_smiles_aromatizados.parquet
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
    from zss import Node, simple_distance
except ImportError as e:
    sys.exit(f'Dependencia ausente: {e}')

# SA Score
_SA_DISPONIVEL = False
_sascorer = None
try:
    from rdkit.Contrib.SA_Score import sascorer as _sascorer
    _SA_DISPONIVEL = True
except ImportError:
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
        import sascorer as _sascorer
        _SA_DISPONIVEL = True
    except ImportError:
        pass

# PAINS
_params_pains = FilterCatalogParams()
_params_pains.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
CATALOGO_PAINS = FilterCatalog(_params_pains)

# Padroes SMARTS UV-Vis
_DOADORES = {
    'amino_primario':   Chem.MolFromSmarts('[NX3;H2]'),
    'amino_secundario': Chem.MolFromSmarts('[NX3;H1;!$(NC=O)]'),
    'amino_terciario':  Chem.MolFromSmarts('[NX3;H0;!$(NC=O);!$(N=*)]'),
    'hidroxil':         Chem.MolFromSmarts('[OX2H]'),
    'alcoxil':          Chem.MolFromSmarts('[OX2;!$(OC=O)][#6]'),
    'tiol':             Chem.MolFromSmarts('[SX2H]'),
    'tioeter':          Chem.MolFromSmarts('[SX2;!$(SC=O)][#6]'),
}
_ACEPTORES = {
    'nitro':    Chem.MolFromSmarts('[N+](=O)[O-]'),
    'nitroso':  Chem.MolFromSmarts('[NX2]=O'),
    'ciano':    Chem.MolFromSmarts('[CX2]#[NX1]'),
    'carbonil': Chem.MolFromSmarts('[CX3](=O)[#6]'),
    'sulfonil': Chem.MolFromSmarts('[SX4](=O)(=O)'),
    'imina':    Chem.MolFromSmarts('[CX3]=[NX2]'),
}
_CROMOFOROS = {
    'azo':     Chem.MolFromSmarts('[NX2]=[NX2]'),
    'vinil':   Chem.MolFromSmarts('[#6]=[#6]'),
    'estiril': Chem.MolFromSmarts('c[CX3]=[CX3]'),
    'quinona': Chem.MolFromSmarts('O=c1cccc(=O)c1'),
}


def _n_matches(mol, padroes):
    return sum(1 for p in padroes.values() if p is not None and mol.HasSubstructMatch(p))


# ── Funcoes do loop (extraidas do notebook) ───────────────────────────────────

def extrair_catalogo_aneis(df: pd.DataFrame, sufixo: int) -> pd.DataFrame:
    colunas = {
        'base':              f'SMILES_Base_Anel{sufixo}',
        'smiles_modificado': f'SMILES_Modificado_Anel{sufixo}',
        'frag1': f'Frag_Estrutura_1_Anel{sufixo}', 'pos1': f'Posicao_1_Anel{sufixo}',
        'frag2': f'Frag_Estrutura_2_Anel{sufixo}', 'pos2': f'Posicao_2_Anel{sufixo}',
        'frag3': f'Frag_Estrutura_3_Anel{sufixo}', 'pos3': f'Posicao_3_Anel{sufixo}',
    }
    sub = df[[v for v in colunas.values() if v in df.columns]].copy()
    sub.columns = [k for k, v in colunas.items() if v in df.columns]
    sub['origem'] = f'Anel{sufixo}'
    return sub


def _valor_valido(v) -> bool:
    return pd.notna(v) and str(v).strip() not in {'', 'nan', 'None'}


def anel_para_arvore(row) -> Node:
    raiz = Node(f"ANEL:{row['base']}")
    for slot in (1, 2, 3):
        frag = row.get(f'frag{slot}')
        pos  = row.get(f'pos{slot}')
        if _valor_valido(frag):
            no_pos = Node(f'POS:{pos}')
            no_pos.addkid(Node(f'FRAG:{frag}'))
            raiz.addkid(no_pos)
    return raiz


def _tamanho_arvore(no: Node) -> int:
    return 1 + sum(_tamanho_arvore(f) for f in no.children)


def ted_normalizada(a: Node, b: Node) -> float:
    d     = simple_distance(a, b)
    denom = max(_tamanho_arvore(a), _tamanho_arvore(b), 1)
    return d / denom


def substituir_numeracao_anel(smiles: str, de: str = '1', para: str = '2') -> str:
    return smiles.replace(de, para)


def validar_e_canonizar(smiles: str):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, False
    return Chem.MolToSmiles(mol, canonical=True), True


def aromatizar_aneis_6_membros(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for anel in mol.GetRingInfo().AtomRings():
        if len(anel) != 6:
            continue
        for k in range(len(anel)):
            at1, at2 = anel[k], anel[(k + 1) % len(anel)]
            lig = mol.GetBondBetweenAtoms(at1, at2)
            if lig is None:
                return None
            lig.SetBondType(Chem.rdchem.BondType.AROMATIC)
        for idx in anel:
            mol.GetAtomWithIdx(idx).SetIsAromatic(True)
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol)


def calcular_fitness_uvvis(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw   = Descriptors.MolWt(mol)
    rot  = Descriptors.NumRotatableBonds(mol)
    qed  = float(QED.qed(mol))
    sa   = _sascorer.calculateScore(mol) if _SA_DISPONIVEL else 5.0
    aneis = mol.GetRingInfo().AtomRings()
    n_aromaticos = sum(
        1 for a in aneis
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in a)
    )
    n_doadores  = _n_matches(mol, _DOADORES)
    n_aceptores = _n_matches(mol, _ACEPTORES)
    n_cromoforos = _n_matches(mol, _CROMOFOROS)
    tem_pains    = CATALOGO_PAINS.HasMatch(mol)

    if 200 <= mw <= 700:
        mw_score = 1.0
    elif mw < 200:
        mw_score = mw / 200.0
    else:
        mw_score = max(0.0, 1.0 - (mw - 700.0) / 300.0)

    qed_score   = qed
    sa_score    = max(0.0, 1.0 - (sa - 1.0) / 9.0)
    pi_score    = min(1.0, n_aromaticos / 3.0)
    d_a_score   = (min(1.0, n_doadores) + min(1.0, n_aceptores) + min(1.0, n_cromoforos)) / 3.0
    planar_score = max(0.0, 1.0 - rot / 8.0)
    # Ablação B.5: pains_fator=1.0 → EF10=10.02 vs 7.24 (Baell & Holloway J.Med.Chem. 53:2719, 2010)
    pains_fator = 1.0

    fitness = _PESOS_FITNESS["mw"] * mw_score + _PESOS_FITNESS["qed"] * qed_score + \
              _PESOS_FITNESS["sa"] * sa_score + _PESOS_FITNESS["pi"] * pi_score + \
              _PESOS_FITNESS["d_a"] * d_a_score + _PESOS_FITNESS["planar"] * planar_score
    fitness *= pains_fator

    return {
        'mw': round(mw, 2), 'sa': round(sa, 2), 'rot': rot,
        'n_aneis_aromaticos': n_aromaticos,
        'n_doadores': n_doadores, 'n_aceptores': n_aceptores,
        'n_cromoforos': n_cromoforos, 'tem_pains': tem_pains,
        'fitness': round(fitness, 4),
    }


def _candidato_forte_linha(r: dict) -> bool:
    return (
        r.get('n_aneis_aromaticos', 0) >= 2 and
        r.get('n_doadores', 0) >= 1 and
        r.get('n_aceptores', 0) >= 1 and
        200 <= r.get('mw', 0) <= 700 and
        r.get('sa', 10) <= 5
    )


# Pesos da função fitness — valores sem fonte bibliográfica;
# estes são os defaults usados pelo notebook (ablação A confirmou degenerescência em EF10).
_PESOS_FITNESS = {"mw": 0.10, "qed": 0.10, "sa": 0.10, "pi": 0.30, "d_a": 0.25, "planar": 0.15}

MAPA_COLUNAS = {
    'base_Anel1': 'SMILES_Base_Anel1',
    'smiles_modificado_Anel1': 'SMILES_Modificado_Anel1',
    'frag1_Anel1': 'Frag_Estrutura_1_Anel1', 'pos1_Anel1': 'Posicao_1_Anel1',
    'frag2_Anel1': 'Frag_Estrutura_2_Anel1', 'pos2_Anel1': 'Posicao_2_Anel1',
    'frag3_Anel1': 'Frag_Estrutura_3_Anel1', 'pos3_Anel1': 'Posicao_3_Anel1',
    'base_Anel2': 'SMILES_Base_Anel2',
    'smiles_modificado_Anel2': 'SMILES_Modificado_Anel2',
    'frag1_Anel2': 'Frag_Estrutura_1_Anel2', 'pos1_Anel2': 'Posicao_1_Anel2',
    'frag2_Anel2': 'Frag_Estrutura_2_Anel2', 'pos2_Anel2': 'Posicao_2_Anel2',
    'frag3_Anel2': 'Frag_Estrutura_3_Anel2', 'pos3_Anel2': 'Posicao_3_Anel2',
}


def selecionar_sementes_torneio(df: pd.DataFrame, n_sementes: int,
                                 col_fitness: str = 'fitness',
                                 k_torneio: int = 3,
                                 rng: random.Random = None) -> pd.DataFrame:
    if rng is None:
        rng = random.Random(42)
    n = len(df)
    if n == 0:
        return df.copy()
    n_sementes = min(n_sementes, n)
    k = min(k_torneio, n)
    indices = []
    for _ in range(n_sementes):
        concorrentes = rng.sample(range(n), k)
        vencedor = max(concorrentes, key=lambda i: df.iloc[i][col_fitness])
        indices.append(vencedor)
    return df.iloc[indices].reset_index(drop=True)


def rodar_uma_geracao(df_sementes, df_historico, catalogo_pontes_lista,
                       canonicos_complexo, canonicos_aromatico,
                       numero_geracao, n_sondas=5, semente=44,
                       combinacoes_invalidas=None):
    if combinacoes_invalidas is None:
        combinacoes_invalidas = set()

    cat_bruto = pd.concat(
        [extrair_catalogo_aneis(df_historico, 1),
         extrair_catalogo_aneis(df_historico, 2)],
        ignore_index=True,
    )
    cat_aneis = cat_bruto.drop_duplicates(subset='smiles_modificado').reset_index(drop=True)
    arvores   = [anel_para_arvore(r) for _, r in cat_aneis.iterrows()]
    n_cat     = len(cat_aneis)

    base_anel1  = extrair_catalogo_aneis(df_sementes, 1)
    rng_pares   = random.Random(semente)
    rng_pontes  = random.Random(semente * 7 + 1)
    n_sondas_ef = min(n_sondas, n_cat)

    vistos = set()
    linhas_validas = []
    canonicos_validos = []
    tentativas = rem_invalidos = rem_dupla = rem_ja_existe = 0

    for pos in range(len(base_anel1)):
        row_i = base_anel1.iloc[pos]
        arv_i = anel_para_arvore(row_i)
        sondas = rng_pares.sample(range(n_cat), n_sondas_ef)
        dists  = [(idx, ted_normalizada(arv_i, arvores[idx])) for idx in sondas]
        dists.sort(key=lambda p: p[1], reverse=True)

        for melhor_idx, dist in dists:
            tentativas += 1
            row_j  = cat_aneis.iloc[melhor_idx]
            ponte  = rng_pontes.choice(catalogo_pontes_lista)
            smi_i  = row_i['smiles_modificado']
            smi_j  = substituir_numeracao_anel(row_j['smiles_modificado'])
            assin  = (smi_i, ponte['Ponte_Estrutura'], smi_j)

            if assin in combinacoes_invalidas:
                rem_invalidos += 1
                continue
            smiles_c = f"{smi_i}{ponte['Ponte_Estrutura']}{smi_j}"
            can, ok  = validar_e_canonizar(smiles_c)
            if not ok:
                combinacoes_invalidas.add(assin)
                rem_invalidos += 1
                continue
            if can in vistos:
                rem_dupla += 1
                continue
            if can in canonicos_complexo:
                rem_ja_existe += 1
                continue
            vistos.add(can)
            canonicos_validos.append(can)
            linhas_validas.append({
                'base_Anel1': row_i['base'], 'smiles_modificado_Anel1': row_i['smiles_modificado'],
                'frag1_Anel1': row_i.get('frag1'), 'pos1_Anel1': row_i.get('pos1'),
                'frag2_Anel1': row_i.get('frag2'), 'pos2_Anel1': row_i.get('pos2'),
                'frag3_Anel1': row_i.get('frag3'), 'pos3_Anel1': row_i.get('pos3'),
                'ID_Ponte': ponte['ID_Ponte'], 'Ponte_Estrutura': ponte['Ponte_Estrutura'],
                'base_Anel2': row_j['base'], 'smiles_modificado_Anel2': row_j['smiles_modificado'],
                'frag1_Anel2': row_j.get('frag1'), 'pos1_Anel2': row_j.get('pos1'),
                'frag2_Anel2': row_j.get('frag2'), 'pos2_Anel2': row_j.get('pos2'),
                'frag3_Anel2': row_j.get('frag3'), 'pos3_Anel2': row_j.get('pos3'),
                'distancia_ted_normalizada': dist,
                'SMILES_Complexo': smiles_c,
            })
            break

    df_fa = pd.DataFrame(linhas_validas)
    if not df_fa.empty:
        df_fa['SMILES_Canonico_Complexo_Candidato'] = canonicos_validos
        df_fa['SMILES_Aromatico_Candidato'] = df_fa['SMILES_Complexo'].apply(
            aromatizar_aneis_6_membros)
    else:
        df_fa = pd.DataFrame(columns=['SMILES_Complexo', 'SMILES_Canonico_Complexo_Candidato',
                                       'SMILES_Aromatico_Candidato', 'distancia_ted_normalizada'])

    df_fb = df_fa.dropna(subset=['SMILES_Aromatico_Candidato']).reset_index(drop=True)
    vistos_f = set()
    linhas_finais = []
    for _, row in df_fb.iterrows():
        mol = Chem.MolFromSmiles(row['SMILES_Aromatico_Candidato'])
        if mol is None:
            continue
        can_f = Chem.MolToSmiles(mol, canonical=True)
        if can_f in vistos_f or can_f in canonicos_aromatico:
            continue
        vistos_f.add(can_f)
        linhas_finais.append(row)

    df_ger = pd.DataFrame(linhas_finais).reset_index(drop=True)
    if not df_ger.empty:
        df_ger['smiles'] = df_ger['SMILES_Aromatico_Candidato']
        registros_fit = []
        for smi in df_ger['smiles']:
            f = calcular_fitness_uvvis(smi)
            registros_fit.append(f if f else {'fitness': 0.0})
        df_fit = pd.DataFrame(registros_fit)
        for col in df_fit.columns:
            df_ger[col] = df_fit[col].values
    else:
        df_ger['fitness'] = pd.Series(dtype=float)
    df_ger['Geracao'] = numero_geracao

    fitness_med = (df_ger['fitness'].mean()
                   if not df_ger.empty and 'fitness' in df_ger.columns else float('nan'))
    return df_ger, {
        'geracao': numero_geracao,
        'candidatos_gerados': tentativas,
        'moleculas_novas': len(df_ger),
        'fitness_medio': round(fitness_med, 4),
        'taxa_sobrevivencia': len(df_ger) / tentativas if tentativas else 0.0,
    }


# ── Configuração da grade de ablação ─────────────────────────────────────────

GRADES_K     = [2, 3, 4, 5]        # tamanhos de torneio a testar
GRADES_NSEM  = [50, 100, 150, 200, 300]  # número de sementes por geração
MAX_GERACOES = 10                   # orçamento de gerações por configuração
N_SONDAS     = 5                    # parceiros TED sorteados por semente
N_BASE       = 300                  # sub-amostra da base para o mini-loop
SEMENTE_BASE = 42                   # reprodutibilidade


def rodar_config(df_base, catalogo_pontes_lista, k_torneio, n_sementes, verboso=False):
    df_base_mini = df_base.sample(N_BASE, random_state=SEMENTE_BASE).reset_index(drop=True)
    can_c = set(df_base_mini['SMILES_Canonico_Complexo'].dropna())
    can_a = set()
    for smi in df_base_mini['smiles'].dropna():
        mol = Chem.MolFromSmiles(str(smi))
        if mol:
            can_a.add(Chem.MolToSmiles(mol))

    hist          = df_base_mini.copy()
    df_seeds      = df_base_mini.copy()
    comb_inv      = set()
    rng_ag        = random.Random(SEMENTE_BASE)
    total_fortes  = 0
    total_novas   = 0
    n_geracoes    = 0
    fitness_final = float('nan')

    for g in range(1, MAX_GERACOES + 1):
        df_nova, estat = rodar_uma_geracao(
            df_seeds, hist, catalogo_pontes_lista,
            can_c, can_a,
            numero_geracao=g,
            n_sondas=N_SONDAS,
            semente=SEMENTE_BASE + g,
            combinacoes_invalidas=comb_inv,
        )
        n_nova = estat['moleculas_novas']
        if verboso:
            print(f'    G{g:2d}: {n_nova:4d} novas | fitness={estat["fitness_medio"]:.4f}')

        if n_nova == 0:
            break

        total_novas   += n_nova
        n_geracoes    += 1
        fitness_final  = estat['fitness_medio']

        if 'n_aneis_aromaticos' in df_nova.columns:
            fortes = df_nova.apply(
                lambda r: _candidato_forte_linha(r.to_dict()), axis=1
            ).sum()
            total_fortes += int(fortes)

        can_c |= set(df_nova.get('SMILES_Canonico_Complexo_Candidato', pd.Series()).dropna())
        can_a |= set(df_nova.get('smiles', pd.Series()).dropna())
        df_nova_ren = df_nova.rename(columns=MAPA_COLUNAS)
        hist = pd.concat([hist, df_nova_ren], ignore_index=True)

        n_sel    = min(n_sementes, n_nova)
        df_seeds = selecionar_sementes_torneio(
            df_nova_ren, n_sel, col_fitness='fitness',
            k_torneio=k_torneio, rng=rng_ag,
        )

    return {
        'k_torneio':       k_torneio,
        'n_sementes':      n_sementes,
        'n_geracoes':      n_geracoes,
        'total_novas':     total_novas,
        'total_fortes':    total_fortes,
        'fitness_final':   round(fitness_final, 4) if not np.isnan(fitness_final) else None,
        'pct_fortes':      round(total_fortes / total_novas * 100, 2) if total_novas else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base',  default='data/banco_smiles_aromatizados.parquet')
    ap.add_argument('--pasta', default='data')
    ap.add_argument('--verboso', action='store_true')
    args = ap.parse_args()

    if not Path(args.base).exists():
        sys.exit(f'Base nao encontrada: {args.base}')
    df_base = pd.read_parquet(args.base)
    if len(df_base) < N_BASE:
        sys.exit(f'Base tem {len(df_base)} moleculas, minimo necessario: {N_BASE}')
    print(f'Base carregada: {len(df_base):,} moleculas')

    catalogo_pontes = (
        df_base[['ID_Ponte', 'Ponte_Estrutura']]
        .dropna().drop_duplicates().reset_index(drop=True)
        .to_dict('records')
    )
    print(f'Pontes disponiveis: {len(catalogo_pontes)}\n')

    total = len(GRADES_K) * len(GRADES_NSEM)
    print(f'Rodando {total} combinacoes '
          f'({MAX_GERACOES} ger. max, {N_SONDAS} sondas, {N_BASE} sementes iniciais)...\n')

    rows = []
    for i, k in enumerate(GRADES_K):
        for j, n in enumerate(GRADES_NSEM):
            idx = i * len(GRADES_NSEM) + j + 1
            print(f'[{idx:2d}/{total}] k_torneio={k}  n_sementes={n:3d}', flush=True)
            r = rodar_config(df_base, catalogo_pontes, k, n, verboso=args.verboso)
            rows.append(r)
            print(f'       -> {r["n_geracoes"]} ger. | '
                  f'{r["total_novas"]:,} novas | '
                  f'{r["total_fortes"]} fortes ({r["pct_fortes"]:.1f}%) | '
                  f'fit_final={r["fitness_final"]}')

    df_res = pd.DataFrame(rows).sort_values('pct_fortes', ascending=False).reset_index(drop=True)
    saida  = Path(args.pasta) / 'ablacao_ag_params.csv'
    df_res.to_csv(saida, index=False)
    print(f'\nResultados salvos em {saida}')

    print('\n=== RANKING POR % CANDIDATOS FORTES ===')
    print(df_res.to_string(index=True))

    melhor = df_res.iloc[0]
    print(f'\n=== MELHOR CONFIGURACAO AG ===')
    print(f'  k_torneio  : {int(melhor["k_torneio"])}')
    print(f'  n_sementes : {int(melhor["n_sementes"])}')
    print(f'  pct_fortes : {melhor["pct_fortes"]:.2f}%')
    print(f'  total_fortes: {int(melhor["total_fortes"])}')
    print(f'  n_geracoes  : {int(melhor["n_geracoes"])}')

    ref = df_res[(df_res['k_torneio'] == 3) & (df_res['n_sementes'] == 150)]
    if not ref.empty:
        r = ref.iloc[0]
        print(f'\n  Referencia (k=3, n=150): pct_fortes={r["pct_fortes"]:.2f}%')


if __name__ == '__main__':
    main()
