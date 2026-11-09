"""Alertas químicos aplicados depois da limpeza padrão do pipeline.

A limpeza padrão (``utils.validacao`` + ``utils.aromatizacao``) só garante que
o RDKit aceita a molécula. Algumas estruturas passam por ela mas são pouco
plausíveis para um cromóforo; cada uma vira uma coluna ``alerta_*`` e pode
ser promovida a filtro via ``filtros_estritos`` no pipeline.
"""

import pandas as pd
from rdkit import Chem

_HETERO_HALOGENADO = Chem.MolFromSmarts("[N,O,S]-[Cl,Br]")

ALERTAS = {
    # Cátion/ânion sem contra-íon — ex.: nitro escrito como [N+](=O)O.
    "alerta_carga_liquida": lambda m: Chem.GetFormalCharge(m) != 0,
    # S neutro com valência 4: ocorre quando um auxocromo é pendurado em S de anel.
    "alerta_S_hipervalente": lambda m: any(
        a.GetSymbol() == "S" and a.GetFormalCharge() == 0 and a.GetTotalValence() == 4
        for a in m.GetAtoms()
    ),
    # Halogênio ligado a heteroátomo (N-Cl, N-Br, O-Br...); instável em solução.
    "alerta_heteroatomo_halogenado": lambda m: m.HasSubstructMatch(_HETERO_HALOGENADO),
    "alerta_radical": lambda m: any(a.GetNumRadicalElectrons() for a in m.GetAtoms()),
}


def fracao_conjugada(mol: Chem.Mol) -> float:
    """Fração dos átomos pesados no maior sistema conjugado (0–1).

    Proxy de extensão do cromóforo para triagem UV-Vis. Não há limiar
    bibliográfico definido; a coluna é calculada mas não usada como filtro.
    """
    grupos: list[set[int]] = []
    for b in mol.GetBonds():
        if not b.GetIsConjugated():
            continue
        atomos = {b.GetBeginAtomIdx(), b.GetEndAtomIdx()}
        for g in [g for g in grupos if g & atomos]:
            atomos |= g
            grupos.remove(g)
        grupos.append(atomos)
    return max((len(g) for g in grupos), default=0) / max(mol.GetNumHeavyAtoms(), 1)


def anotar_alertas(
    df: pd.DataFrame,
    coluna_smiles: str = "smiles",
    filtros_estritos: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Anota alertas químicos e, opcionalmente, remove as linhas flagadas.

    Adiciona uma coluna ``alerta_*`` para cada entrada em ``ALERTAS`` e a
    coluna ``fracao_conjugada``. Quando ``filtros_estritos`` é informado,
    remove as linhas que disparam ao menos um dos alertas listados.
    """
    desconhecidos = set(filtros_estritos) - set(ALERTAS)
    if desconhecidos:
        raise ValueError(f"Filtros desconhecidos: {sorted(desconhecidos)}")

    df = df.copy()
    mols = [Chem.MolFromSmiles(s) for s in df[coluna_smiles]]
    for nome, regra in ALERTAS.items():
        df[nome] = [regra(m) for m in mols]
    df["fracao_conjugada"] = [fracao_conjugada(m) for m in mols]

    if not filtros_estritos:
        return df

    total = len(df)
    df = df[~df[list(filtros_estritos)].any(axis=1)].reset_index(drop=True)
    print(f"[filtros químicos] {len(df)} / {total} moléculas sem alertas estritos "
          f"({', '.join(filtros_estritos)}).")
    return df
