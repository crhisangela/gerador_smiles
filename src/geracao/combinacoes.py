"""Combinação vetorizada de pares de fragmentos conectados por pontes."""

import pandas as pd
import random

from rdkit import Chem

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


def _ordem_ligacao_borda(ponte_str: str, borda: str) -> int:
    """
    Ordem da ligação (1, 2 ou 3) que a ponte forma no lado indicado.

    A concatenação é `smi_anel1 + ponte_str + smi_anel2`: o último átomo de
    smi_anel1 encosta no símbolo do INÍCIO de ponte_str (borda='fim', do
    ponto de vista do anel1), e o primeiro átomo de smi_anel2 encosta no
    símbolo do FIM de ponte_str (borda='inicio', do ponto de vista do
    anel2). Sem 1, assume-se ligação simples implícita.
    """
    simbolo = ponte_str[-1] if borda == "inicio" else ponte_str[0]
    return {"=": 2, "#": 3}.get(simbolo, 1)


def _tem_valencia_livre_para_ponte(smiles: str, borda: str, ponte_str: str) -> bool:
    """
    Confere se o átomo de encaixe (primeiro átomo de `smiles`, se
    borda='inicio', ou último átomo, se borda='fim') tem hidrogênios
    implícitos/explícitos suficientes para virar a ligação que a ponte
    exige nessa ponta, sem estourar a valência do átomo.

    Sem essa checagem, a ponte era colada direto na string e, se o átomo
    de encaixe já estivesse com a valência saturada (heteroátomo de anel
    tipo O/S, ou N já substituído por um auxocromo), o RDKit rejeitava o
    SMILES inteiro com "Explicit valence ... is greater than permitted".
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    idx_atomo = 0 if borda == "inicio" else mol.GetNumAtoms() - 1
    ordem_necessaria = _ordem_ligacao_borda(ponte_str, borda)
    return mol.GetAtomWithIdx(idx_atomo).GetTotalNumHs() >= ordem_necessaria


def gerar_complexos(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    pontes_dict: dict[int, str],
    n_combinacoes: int,
    coluna_smiles: str = "SMILES_Modificado",
    ajustar_numeracao_anel2: bool = True,
    max_tentativas_por_combinacao: int = 20,
) -> pd.DataFrame:
    """
    Combina aleatoriamente dois fragmentos por meio de pontes químicas.

    Para cada combinação, sorteia pares (anel1, anel2) até achar uma ponte
    do catálogo cuja ordem de ligação cabe na valência livre dos dois
    átomos de encaixe; combinações sem nenhuma ponte compatível em
    `max_tentativas_por_combinacao` tentativas são descartadas.

    Mantém rastreamento completo:
    - colunas do Anel 1 recebem sufixo _Anel1;
    - dados da ponte são registrados;
    - colunas do Anel 2 recebem sufixo _Anel2;
    - SMILES_Complexo é criado.
    """
    dados_complexos = []
    chaves_pontes_disponiveis = list(pontes_dict.keys())
    descartadas_sem_ponte_compativel = 0

    if df1.empty or df2.empty:
        raise ValueError("df1 e df2 precisam conter fragmentos válidos.")

    if not chaves_pontes_disponiveis:
        raise ValueError("O dicionário de pontes está vazio.")

    for _ in range(n_combinacoes):
        candidato = None

        for _tentativa in range(max_tentativas_por_combinacao):
            row1 = df1.sample(1).iloc[0]
            row2 = df2.sample(1).iloc[0]

            chave_ponte = random.choice(chaves_pontes_disponiveis)
            ponte_str = pontes_dict[chave_ponte]

            smi_anel1 = row1[coluna_smiles]
            smi_anel2_bruto = row2[coluna_smiles]

            if not _tem_valencia_livre_para_ponte(smi_anel1, "fim", ponte_str):
                continue
            if not _tem_valencia_livre_para_ponte(smi_anel2_bruto, "inicio", ponte_str):
                continue

            candidato = (row1, row2, chave_ponte, ponte_str, smi_anel2_bruto)
            break

        if candidato is None:
            descartadas_sem_ponte_compativel += 1
            continue

        row1, row2, chave_ponte, ponte_str, smi_anel2 = candidato
        smi_anel1 = row1[coluna_smiles]

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

    if descartadas_sem_ponte_compativel:
        print(
            f"[gerar_complexos] {descartadas_sem_ponte_compativel} combinações "
            f"descartadas: nenhuma ponte do catálogo coube na valência livre "
            f"dos átomos de encaixe em {max_tentativas_por_combinacao} tentativas."
        )

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
