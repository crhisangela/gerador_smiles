"""Orquestração do pipeline completo de geração de SMILES."""
import os
import pandas as pd

from fragmentos.aneis import ANEIS_6_MEMBROS
from fragmentos.auxocromos import  AUXOCROMOS
from fragmentos.pontes import PONTES

from geracao.combinacoes import gerar_complexos, limpar_dataframe_complexos
from geracao.fragmentos import gerar_fragmentos_com_auxocromos

from utils.aromatizacao import aplicar_aromatizacao_e_filtrar
from utils.validacao import     filtrar_smiles_validos, limpar_fragmentos_gerados, remover_duplicatas_moleculares


def executar_pipeline(
    caminho_saida: str = "data/banco_smiles_aromatizados.parquet",
    repeticoes_por_anel: int = 200,
    n_combinacoes: int = 10_000,
    quantidade_auxocromos: int = 3,
    posicoes_disponiveis: list[int] | None = None,
    aneis: dict[int, str] | None = None,
    auxocromos: dict[int, str] | None = None,
    pontes: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Executa o pipeline completo de geração de SMILES.

    Etapas
    ------
    1. Gera fragmentos com auxocromos aleatórios.
    2. Filtra SMILES inválidos.
    3. Remove duplicatas moleculares dos fragmentos.
    4. Combina pares de fragmentos via pontes, mantendo rastreio das colunas.
    5. Filtra complexos inválidos e remove duplicatas moleculares.
    6. Aromatiza anéis de 6 membros e filtra inválidos.
    7. Exporta para Parquet.

    Retorna
    -------
    pd.DataFrame
        DataFrame final com metadados dos fragmentos, ponte usada,
        SMILES complexo e SMILES aromático.
    """

    if aneis is None:
        aneis = ANEIS_6_MEMBROS

    if auxocromos is None:
        auxocromos = AUXOCROMOS

    if pontes is None:
        pontes = PONTES

    print("\n========== INÍCIO DO PIPELINE ==========\n")

    # ============================================================
    # Etapa 1 — Geração de fragmentos
    # ============================================================
    print("Etapa 1: Gerando fragmentos com auxocromos...")

    df_fragmentos_bruto = gerar_fragmentos_com_auxocromos(
        aneis=aneis,
        auxocromos=auxocromos,
        repeticoes=repeticoes_por_anel,
        posicoes_disponiveis=posicoes_disponiveis,
        quantidade_auxocromos=quantidade_auxocromos,
    )

    print(f"  → {len(df_fragmentos_bruto)} fragmentos brutos gerados.")

    if df_fragmentos_bruto.empty:
        raise ValueError("Nenhum fragmento foi gerado na Etapa 1.")

    # ============================================================
    # Etapas 2 e 3 — Validação e deduplicação dos fragmentos
    # ============================================================
    print("\nEtapa 2 e 3: Validando fragmentos e removendo duplicatas...")

    df_fragmentos = limpar_fragmentos_gerados(
        df_fragmentos_bruto,
        coluna_smiles="SMILES_Modificado",
    )

    if df_fragmentos.empty:
        raise ValueError("Nenhum fragmento válido restou após limpeza.")

    # ============================================================
    # Etapa 4 — Fusão via pontes
    # ============================================================
    print(f"\nEtapa 4: Combinando fragmentos via pontes...")
    print(f"  → Alvo de combinações: {n_combinacoes}")

    df_complexos_bruto = gerar_complexos(
        df1=df_fragmentos,
        df2=df_fragmentos,
        pontes_dict=pontes,
        n_combinacoes=n_combinacoes,
        coluna_smiles="SMILES_Modificado",
        ajustar_numeracao_anel2=True,
    )

    print(f"  → {len(df_complexos_bruto)} complexos brutos gerados.")

    if df_complexos_bruto.empty:
        raise ValueError("Nenhum complexo foi gerado na Etapa 4.")

    # ============================================================
    # Etapa 5 — Limpeza dos complexos
    # ============================================================
    print("\nEtapa 5: Validando complexos e removendo duplicatas...")

    df_complexos = limpar_dataframe_complexos(
        df_complexos_bruto,
        coluna_smiles="SMILES_Complexo",
    )

    if df_complexos.empty:
        raise ValueError("Nenhum complexo válido restou após limpeza.")

    # ============================================================
    # Etapa 6 — Aromatização
    # ============================================================
    print("\nEtapa 6: Aromatizando anéis de 6 membros e filtrando inválidos...")

    df_final = aplicar_aromatizacao_e_filtrar(
        df_complexos,
        coluna_smiles="SMILES_Complexo",
    )

    if df_final.empty:
        raise ValueError("Nenhuma molécula válida restou após aromatização.")

    # Para manter compatibilidade com o pipeline anterior,
    # cria também uma coluna final padronizada chamada 'smiles'.
    df_final["smiles"] = df_final["SMILES_Aromatico"]

    # ============================================================
    # Etapa 7 — Exportação
    # ============================================================
    print("\nEtapa 7: Exportando arquivo final...")

    pasta_saida = os.path.dirname(caminho_saida)

    if pasta_saida:
        os.makedirs(pasta_saida, exist_ok=True)

    df_final.to_parquet(caminho_saida, index=False)

    print(f"  → Arquivo exportado para: {caminho_saida}")
    print(f"  → Total final: {len(df_final)} moléculas.\n")

    print("========== PIPELINE FINALIZADO ==========\n")

    return df_final