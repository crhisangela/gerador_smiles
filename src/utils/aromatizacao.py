import pandas as pd
from rdkit import Chem


def _aromatizar_aneis_6_membros(smiles: str) -> str | None:
    """
    Tenta aromatizar todos os anéis de 6 membros de uma molécula.
    Retorna None se a aromatização falhar.
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    for anel in mol.GetRingInfo().AtomRings():
        if len(anel) != 6:
            continue

        for i in range(len(anel)):
            atomo1 = anel[i]
            atomo2 = anel[(i + 1) % len(anel)]

            ligacao = mol.GetBondBetweenAtoms(atomo1, atomo2)

            if ligacao is None:
                return None

            ligacao.SetBondType(Chem.rdchem.BondType.AROMATIC)

        for idx_atomo in anel:
            mol.GetAtomWithIdx(idx_atomo).SetIsAromatic(True)

    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    return Chem.MolToSmiles(mol)


def aplicar_aromatizacao_e_filtrar(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Complexo",
) -> pd.DataFrame:
    """
    Aplica aromatização e remove moléculas inválidas após aromatização.
    """
    df = df.copy()

    df["SMILES_Aromatico"] = df[coluna_smiles].apply(_aromatizar_aneis_6_membros)

    df_valido = df.dropna(subset=["SMILES_Aromatico"]).reset_index(drop=True)

    print(
        f"[aromatização] {len(df_valido)} / {len(df)} "
        "moléculas válidas após aromatização."
    )

    return df_valido

