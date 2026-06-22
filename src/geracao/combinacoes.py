"""Combinação vetorizada de pares de fragmentos conectados por pontes."""

import pandas as pd
import random

from utils.validacao import filtrar_smiles_validos, remover_duplicatas_moleculares


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


def gerar_complexos(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    pontes_dict: dict[int, str],
    n_combinacoes: int,
    coluna_smiles: str = "SMILES_Modificado",
    ajustar_numeracao_anel2: bool = True,
) -> pd.DataFrame:
    """
    Combina aleatoriamente dois fragmentos por meio de pontes químicas.

    Mantém rastreamento completo:
    - colunas do Anel 1 recebem sufixo _Anel1;
    - dados da ponte são registrados;
    - colunas do Anel 2 recebem sufixo _Anel2;
    - SMILES_Complexo é criado.
    """
    dados_complexos = []
    chaves_pontes_disponiveis = list(pontes_dict.keys())

    if df1.empty or df2.empty:
        raise ValueError("df1 e df2 precisam conter fragmentos válidos.")

    if not chaves_pontes_disponiveis:
        raise ValueError("O dicionário de pontes está vazio.")

    for _ in range(n_combinacoes):
        row1 = df1.sample(1).iloc[0]
        row2 = df2.sample(1).iloc[0]

        chave_ponte = random.choice(chaves_pontes_disponiveis)
        ponte_str = pontes_dict[chave_ponte]

        smi_anel1 = row1[coluna_smiles]
        smi_anel2 = row2[coluna_smiles]

        if ajustar_numeracao_anel2:
            smi_anel2 = _substituir_numeracao_anel(smi_anel2)

        smi_complexo = f"{smi_anel1}{ponte_str}{smi_anel2}"

        linha = {}

        for col in df1.columns:
            linha[f"{col}_Anel1"] = row1[col]

        linha["ID_Ponte"] = str(chave_ponte)
        linha["Ponte_Estrutura"] = ponte_str

        for col in df2.columns:
            linha[f"{col}_Anel2"] = row2[col]

        linha["SMILES_Complexo"] = smi_complexo

        dados_complexos.append(linha)

    return pd.DataFrame(dados_complexos)


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
