import pandas as pd
from rdkit import Chem

def filtrar_smiles_validos(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Modificado",
) -> pd.DataFrame:
    """
    Mantém apenas SMILES válidos de acordo com o RDKit.
    """
    df = df.copy()

    def converter_para_mol(smiles: str):
        return Chem.MolFromSmiles(smiles)

    df["mol"] = df[coluna_smiles].apply(converter_para_mol)

    df_validos = df[df["mol"].notna()].copy()
    df_validos = df_validos.drop(columns=["mol"])

    return df_validos.reset_index(drop=True)


def remover_duplicatas_moleculares(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Modificado",
    nome_coluna_canonico: str = "SMILES_Canonico",
) -> pd.DataFrame:
    """
    Remove duplicatas moleculares usando SMILES canônico do RDKit.

    Isso é melhor que drop_duplicates, porque dois SMILES diferentes podem
    representar a mesma molécula.
    """
    vistos: set[str] = set()
    indices_unicos: list[int] = []
    smiles_canonicos: list[str] = []

    for idx, smiles in df[coluna_smiles].items():
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            continue

        smiles_canonico = Chem.MolToSmiles(mol, canonical=True)

        if smiles_canonico not in vistos:
            vistos.add(smiles_canonico)
            indices_unicos.append(idx)
            smiles_canonicos.append(smiles_canonico)

    df_unicos = df.loc[indices_unicos].copy()
    df_unicos[nome_coluna_canonico] = smiles_canonicos

    return df_unicos.reset_index(drop=True)


def limpar_fragmentos_gerados(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Modificado",
) -> pd.DataFrame:
    """
    Filtra SMILES inválidos e remove duplicatas moleculares dos fragmentos.
    """
    total_inicial = len(df)

    df_validos = filtrar_smiles_validos(df, coluna_smiles=coluna_smiles)
    total_validos = len(df_validos)

    df_unicos = remover_duplicatas_moleculares(
        df_validos,
        coluna_smiles=coluna_smiles,
        nome_coluna_canonico="SMILES_Canonico_Fragmento",
    )
    total_unicos = len(df_unicos)

    print("\n--- RELATÓRIO DE LIMPEZA DOS FRAGMENTOS ---")
    print(f"Total gerado inicialmente:       {total_inicial}")
    print(f"Removidos por SMILES inválido:   {total_inicial - total_validos}")
    print(f"Removidos por duplicata mol.:    {total_validos - total_unicos}")
    print(f"Total final de fragmentos:       {total_unicos}\n")

    return df_unicos


def _substituir_numeracao_anel(
    smiles: str,
    de: str = "1",
    para: str = "2",
) -> str:
    """
    Substitui a numeração de fechamento de anel para evitar conflitos
    ao juntar dois fragmentos no mesmo SMILES.
    """
    return smiles.replace(de, para)



def limpar_dataframe_complexos(
    df: pd.DataFrame,
    coluna_smiles: str = "SMILES_Complexo",
) -> pd.DataFrame:
    """
    Filtra complexos inválidos e remove duplicatas moleculares via RDKit.
    """
    total_inicial = len(df)

    df_validos = filtrar_smiles_validos(df, coluna_smiles=coluna_smiles)
    total_validos = len(df_validos)

    df_unicos = remover_duplicatas_moleculares(
        df_validos,
        coluna_smiles=coluna_smiles,
        nome_coluna_canonico="SMILES_Canonico_Complexo",
    )
    total_unicos = len(df_unicos)

    print("\n--- RELATÓRIO DE FUSÃO E LIMPEZA ---")
    print(f"Fusões tentadas:                 {total_inicial}")
    print(f"Removidas por SMILES inválido:   {total_inicial - total_validos}")
    print(f"Removidas por duplicata mol.:    {total_validos - total_unicos}")
    print(f"Total complexos válidos únicos:  {total_unicos}\n")

    return df_unicos