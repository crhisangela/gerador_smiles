"""Aromatização forçada de anéis de 6 membros formados por C e N.

O pipeline gera SMILES com ligações alternadas (Kekulé); a aromatização
força a notação aromática para facilitar comparações canônicas. Apenas anéis
de C e N são convertidos porque O, S e B em anel de 6 não formam aromático
neutro e quebram a kekulização do RDKit.
"""

import pandas as pd
from rdkit import Chem

# Restrição heurística: só anéis cujos átomos são todos C ou N são aromatizáveis
# sem quebrar a kekulização. O, S e B em anel de 6 não formam aromático neutro.
ELEMENTOS_AROMATIZAVEIS = {"C", "N"}


def _aromatizar_aneis_6_membros(smiles: str) -> str | None:
    """Aromatiza anéis de 6 membros formados só por C e N.

    Retorna o SMILES aromatizado ou None se a sanitização falhar.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    for anel in mol.GetRingInfo().AtomRings():
        if len(anel) != 6:
            continue
        if any(mol.GetAtomWithIdx(i).GetSymbol() not in ELEMENTOS_AROMATIZAVEIS
               for i in anel):
            continue
        for i in range(len(anel)):
            ligacao = mol.GetBondBetweenAtoms(anel[i], anel[(i + 1) % len(anel)])
            if ligacao is None:
                return None
            ligacao.SetBondType(Chem.rdchem.BondType.AROMATIC)
        for idx in anel:
            mol.GetAtomWithIdx(idx).SetIsAromatic(True)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(mol)


def aplicar_aromatizacao_e_filtrar(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Complexo",
) -> pd.DataFrame:
    """Aplica aromatização e descarta moléculas que falham na conversão.

    Adiciona a coluna ``SMILES_Aromatico`` e remove as linhas onde o valor
    é None (anel não aromatizável ou sanitização falhou).
    """
    df = df.copy()
    df["SMILES_Aromatico"] = df[coluna_smiles].apply(_aromatizar_aneis_6_membros)
    df_valido = df.dropna(subset=["SMILES_Aromatico"]).reset_index(drop=True)
    print(f"[aromatização] {len(df_valido)} / {len(df)} moléculas válidas após aromatização.")
    return df_valido

