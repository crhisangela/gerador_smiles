"""Geração v3: recombinação por operadores de árvore (Zhang–Shasha) a partir
da base já montada, com a mesma limpeza do pipeline principal."""
import itertools
import random
import time
import traceback

import pandas as pd
from rdkit import Chem

from geracao.combinacoes import limpar_dataframe_complexos
from geracao.recombinacao_ted import (
    OPERADORES,
    canonicos_complexo,
    moleculas_da_base,
    propor_candidatos,
    validar_base,
)
from geracao.selecao_sementes import PesosSementes
from utils.armazenamento_execucao import RAIZ_EXECUCOES, RegistroExecucao
from utils.aromatizacao import aplicar_aromatizacao_e_filtrar
from utils.filtros_quimicos import ALERTAS, anotar_alertas
from utils.validacao import filtrar_smiles_validos


def _canonico(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(m) if m is not None else None


def _rejeitadas(antes: pd.DataFrame, depois: pd.DataFrame, etapa: str, motivo) -> pd.DataFrame:
    """Linhas de ``antes`` que não chegaram a ``depois``. ``motivo`` pode ser
    texto fixo ou função linha -> texto."""
    rej = antes[~antes["ID_Candidato"].isin(depois["ID_Candidato"])].copy()
    rej["Etapa"] = etapa
    if callable(motivo) and not rej.empty:
        rej["Motivo"] = rej.apply(motivo, axis=1)
    else:
        rej["Motivo"] = None if callable(motivo) else motivo
    return rej


def _limpar_geracao(
    df_cand: pd.DataFrame,
    conhecidos_complexo: set,
    conhecidos_final: set,
    filtros_estritos: tuple[str, ...],
) -> tuple[pd.DataFrame, list[pd.DataFrame], dict]:
    """Mesma sequência do ``executar_pipeline`` (Etapas 5 e 6) + novidade
    frente ao histórico + reabertura do SMILES aromático + alertas.
    Retorna (aceitas, rejeitadas por etapa, funil)."""
    funil = {"candidatos": len(df_cand)}
    rejeitadas = []

    df = limpar_dataframe_complexos(df_cand, coluna_smiles="SMILES_Complexo")
    rejeitadas.append(_rejeitadas(
        df_cand, df, "limpeza_complexo",
        lambda r: "complexo_invalido" if Chem.MolFromSmiles(r["SMILES_Complexo"]) is None
        else "complexo_duplicado_na_geracao",
    ))
    funil["complexo_valido_unico"] = len(df)

    antes = df
    df = df[~df["SMILES_Canonico_Complexo"].isin(conhecidos_complexo)]
    rejeitadas.append(_rejeitadas(antes, df, "novidade_complexo", "complexo_ja_existe"))
    funil["complexo_novo"] = len(df)
    if df.empty:
        return df, rejeitadas, funil

    antes = df
    df = aplicar_aromatizacao_e_filtrar(df, coluna_smiles="SMILES_Complexo")
    rejeitadas.append(_rejeitadas(antes, df, "aromatizacao", "aromatizacao_falhou"))
    funil["aromatizado"] = len(df)

    # A aromatização forçada pode escrever SMILES que o RDKit não consegue
    # reler (kekulização impossível): aplica o mesmo critério de validade da
    # Etapa 2 sobre o SMILES final.
    antes = df
    df = filtrar_smiles_validos(df, coluna_smiles="SMILES_Aromatico")
    rejeitadas.append(_rejeitadas(antes, df, "reabertura_aromatico", "aromatico_nao_reabre"))
    funil["aromatico_reabre"] = len(df)

    df["smiles"] = df["SMILES_Aromatico"]
    df["SMILES_Canonico_Final"] = df["smiles"].map(_canonico)
    antes = df
    df = df[~df["SMILES_Canonico_Final"].isin(conhecidos_final)]
    df = df.drop_duplicates(subset="SMILES_Canonico_Final")
    rejeitadas.append(_rejeitadas(
        antes, df, "novidade_final",
        lambda r: "final_ja_existe" if r["SMILES_Canonico_Final"] in conhecidos_final
        else "final_duplicado_na_geracao",
    ))
    funil["final_novo_unico"] = len(df)

    df = anotar_alertas(df, coluna_smiles="smiles")
    if filtros_estritos:
        antes = df
        df = df[~df[list(filtros_estritos)].any(axis=1)]
        rejeitadas.append(_rejeitadas(
            antes, df, "filtros_quimicos",
            lambda r: "+".join(f for f in filtros_estritos if r[f]),
        ))
    funil["aceitas"] = len(df)
    return df.reset_index(drop=True), rejeitadas, funil


def executar_recombinacao_ted(
    caminho_entrada: str = "data/banco_smiles_aromatizados.parquet",
    pasta_execucoes: str = RAIZ_EXECUCOES,
    rotulo: str | None = None,
    n_geracoes: int | None = None,
    sementes_por_geracao: int | None = None,
    n_sondas: int = 10,
    criterio: str = "diverso",
    operadores: tuple[str, ...] = OPERADORES,
    filtros_estritos: tuple[str, ...] = (),
    semente: int = 42,
    aprender: bool = True,
    peso_alerta: float = 0.1,
    heranca: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Executa as gerações de recombinação por operadores de árvore (Zhang–Shasha).

    Etapas por geração
    ------------------
    1. Sementes: a base inteira (G1); depois as novas moléculas ou sorteio
       por peso aprendido (``aprender=True``).
    2. Parceiro Q de cada semente P escolhido por TED normalizada entre
       ``n_sondas`` sorteados do histórico (Zhang–Shasha, SIAM J. Comput. 18:1245, 1989).
    3. Operador de árvore (``troca_anel``, ``troca_ponte``, ``troca_ramo``,
       ``troca_auxocromo``) produz o filho; valência livre nas pontas.
    4. Limpeza: SMILES válido, deduplicação canônica, novidade frente ao histórico,
       aromatização, reabertura do aromático, novidade final, alertas químicos.

    Parâmetros de escala
    --------------------
    ``n_sondas`` é o principal parâmetro de qualidade: valores maiores (50–100)
    geram mais diversidade mas o custo da TED cresce linearmente. Para execuções
    em supercomputador, aumente também ``sementes_por_geracao`` (padrão: len(novas)).

    Aprendizado Beta
    ----------------
    Com ``aprender=True``, da G2 em diante as sementes são sorteadas do histórico
    inteiro com peso Beta(a, b) (ver ``geracao.selecao_sementes``): sucesso = filho
    aceito sem alerta; ``peso_alerta`` penaliza sementes com alertas; ``heranca``
    transmite fração do placar para o filho.

    Para quando: geração vazia, limite ``n_geracoes`` atingido ou Ctrl+C
    (gerações concluídas ficam salvas; a em andamento é descartada).

    Armazenamento
    -------------
    Pasta nova por execução em ``<pasta_execucoes>/<AAAAMMDD_HHMMSS>[_rotulo]``.
    Ver ``utils.armazenamento_execucao`` para a estrutura completa.

    Returns:
        (df_novas, df_rejeitadas, df_funil, pasta)
    """
    desconhecidos = set(filtros_estritos) - set(ALERTAS)
    if desconhecidos:
        raise ValueError(f"Filtros desconhecidos: {sorted(desconhecidos)}")

    config = {k: v for k, v in locals().items() if k != "desconhecidos"}
    reg = RegistroExecucao(config, raiz=pasta_execucoes, rotulo=rotulo)
    rng = random.Random(semente)

    print("\n========== RECOMBINAÇÃO v3 (Zhang–Shasha) ==========\n")
    df_base = pd.read_parquet(caminho_entrada)
    print(f"Base de entrada: {len(df_base)} moléculas.\n")
    df_validacao = validar_base(df_base)
    reg.salvar_tabela("validacao_base.csv", df_validacao)
    print(df_validacao.to_string(index=False), "\n")

    historico = moleculas_da_base(df_base)
    # ID de cada molécula do histórico, na mesma ordem de ``historico``.
    id_hist = list(df_base["SMILES_Canonico_Complexo"])
    id_de = dict(zip(historico, id_hist))
    conhecidos_complexo = (set(df_base["SMILES_Canonico_Complexo"])
                           | canonicos_complexo(historico))
    conhecidos_final = {c for c in df_base["smiles"].map(_canonico) if c}

    def _amostrar(moleculas):
        if sementes_por_geracao is None or sementes_por_geracao >= len(moleculas):
            return list(moleculas)
        return rng.sample(moleculas, sementes_por_geracao)

    colunas_alerta = list(ALERTAS)
    pesos = None
    if aprender:
        # Qualidade das moléculas da base: alertas calculados no complexo
        # (sempre abre) + penalidade se o SMILES aromático não reabre.
        df_q = anotar_alertas(df_base, coluna_smiles="SMILES_Canonico_Complexo")
        limpas = (~df_q[colunas_alerta].any(axis=1)
                  & df_base["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None))
        pesos = PesosSementes(peso_alerta=peso_alerta, heranca=heranca)
        for mol, limpa in zip(historico, limpas):
            pesos.registrar(mol, bool(limpa))
        print(f"Aprendizado ligado: {int(limpas.sum())} / {len(historico)} moléculas "
              f"da base limpas (as demais começam com peso × {peso_alerta}).\n")

    def _genealogia(df: pd.DataFrame) -> None:
        if not df.empty:
            df["Semente_ID"] = df["Semente_idx"].map(lambda k: id_de[sementes[int(k)]])
            df["Parceiro_ID"] = df["Parceiro_idx"].map(lambda j: id_hist[int(j)])

    # Geração 1: sem nada aprendido ainda, usa a base como está.
    sementes = _amostrar(historico)
    funis = []
    proximo_id = 1
    status, g_concluidas, total_aceitas = "concluida", 0, 0
    inicio = time.perf_counter()
    geracoes_alvo = itertools.count(1) if n_geracoes is None else range(1, n_geracoes + 1)
    try:
        for g in geracoes_alvo:
            t0 = time.perf_counter()
            print(f"--- Geração {g}: {len(sementes)} sementes, "
                  f"{len(historico)} parceiros possíveis ---", flush=True)

            df_cand, df_falhas, est = propor_candidatos(
                sementes, historico, n_sondas=n_sondas, criterio=criterio,
                operadores=operadores, rng=rng, progresso=1_000,
            )
            for df_ in (df_cand, df_falhas):
                df_["ID_Candidato"] = range(proximo_id, proximo_id + len(df_))
                proximo_id += len(df_)
                _genealogia(df_)
            rejeitadas_g = [df_falhas]

            if df_cand.empty:
                df_g, funil = df_cand, {"candidatos": 0}
            else:
                df_g, rej, funil = _limpar_geracao(df_cand, conhecidos_complexo,
                                                   conhecidos_final, filtros_estritos)
                rejeitadas_g += rej
            rej_validas = [r for r in rejeitadas_g if not r.empty]
            df_rej_g = (pd.concat(rej_validas, ignore_index=True)
                        if rej_validas else pd.DataFrame())
            if not df_rej_g.empty:
                df_rej_g["Geracao"] = g
            segundos = time.perf_counter() - t0

            # Registro das sementes desta geração (antes de o placar mudar).
            gerou = set(df_g["Semente_idx"].astype(int)) if not df_g.empty else set()
            limpas_g = (set(df_g.loc[~df_g[colunas_alerta].any(axis=1), "Semente_idx"].astype(int))
                        if not df_g.empty else set())
            df_sem = pd.DataFrame({"Geracao": g, "Semente_idx": range(len(sementes)),
                                   "Semente_ID": [id_de[s] for s in sementes]})
            df_sem["Gerou_aceita"] = df_sem["Semente_idx"].isin(gerou)
            df_sem["Gerou_sem_alerta"] = df_sem["Semente_idx"].isin(limpas_g)

            # Aprendizado: cada semente ganha sucesso (filho aceito e sem alerta)
            # ou falha no placar; o peso no sorteio da próxima geração muda.
            if pesos is not None:
                peso_usado = [pesos.peso(s) for s in sementes]
                df_sem["Peso"] = peso_usado
                df_sem["Placar_a"] = [pesos.placar[s][0] for s in sementes]
                df_sem["Placar_b"] = [pesos.placar[s][1] for s in sementes]
                if not df_g.empty:
                    df_g["Peso_Semente"] = df_g["Semente_idx"].map(lambda k: peso_usado[k])
                n_limpas = pesos.atualizar(sementes, df_g, colunas_alerta)
            else:
                n_limpas = len(limpas_g)
            taxa = n_limpas / max(len(sementes), 1)
            funis.append({"geracao": g, "sementes": len(sementes), **funil,
                          "aceitas_sem_alerta": n_limpas,
                          "taxa_sucesso": round(taxa, 3),
                          **dict(est), **(pesos.resumo() if pesos else {}),
                          "segundos": round(segundos, 1)})

            if not df_g.empty:
                df_g["Geracao"] = g
            reg.salvar_geracao(g, df_g, df_rej_g, df_sem, funis)
            g_concluidas, total_aceitas = g, total_aceitas + len(df_g)

            if df_g.empty:
                print("Nenhuma molécula nova nesta geração; encerrando.", flush=True)
                break

            conhecidos_complexo |= set(df_g["SMILES_Canonico_Complexo"])
            conhecidos_final |= set(df_g["SMILES_Canonico_Final"])

            novas = moleculas_da_base(df_g)
            ids_novas = list(df_g["SMILES_Canonico_Complexo"])
            historico = historico + novas
            id_hist += ids_novas
            id_de.update(zip(novas, ids_novas))
            if pesos is not None:
                limpas = ~df_g[colunas_alerta].any(axis=1)
                for mol, k, limpa in zip(novas, df_g["Semente_idx"], limpas):
                    pesos.registrar(mol, bool(limpa), pai=sementes[int(k)])
                # Mesmo orçamento que a geração anterior produziu, mas sorteado
                # por peso em TODO o histórico (com reposição).
                k_sementes = sementes_por_geracao or len(novas)
                sementes = pesos.sortear(historico, k_sementes, rng)
            else:
                sementes = _amostrar(novas)
            print(f"  → {len(df_g)} moléculas novas aceitas na geração {g}, "
                  f"{n_limpas} sem alerta (taxa de sucesso {taxa:.1%}) "
                  f"({segundos / 60:.1f} min; total {(time.perf_counter() - inicio) / 60:.1f} min; "
                  f"salvo).\n", flush=True)
    except KeyboardInterrupt:
        status = "interrompida"
        print(f"\nInterrompida pelo usuário; a geração {g_concluidas + 1} foi descartada.")
    except Exception:
        traceback.print_exc()
        reg.finalizar("erro", geracoes=g_concluidas, aceitas=total_aceitas)
        reg.fechar()
        raise

    if pesos is not None:
        reg.salvar_tabela("pesos_finais.parquet", pd.DataFrame(
            [{"ID": i, "Placar_a": pesos.placar[m][0], "Placar_b": pesos.placar[m][1],
              "Qualidade": pesos.qualidade[m], "Peso": pesos.peso(m)}
             for m, i in zip(historico, id_hist)]).drop_duplicates("ID"))

    df_funil = pd.DataFrame(funis)
    print("\n--- FUNIL POR GERAÇÃO ---")
    print(df_funil.T.to_string(header=False) if not df_funil.empty else "(vazio)")

    df_novas, df_rejeitadas, df_funil = reg.finalizar(
        status, geracoes=g_concluidas, aceitas=total_aceitas,
        minutos=round((time.perf_counter() - inicio) / 60, 1))
    if not df_rejeitadas.empty:
        print("\n--- REJEITADAS POR MOTIVO ---")
        print(df_rejeitadas.groupby(["Etapa", "Motivo"]).size().to_string())
    print("\n========== RECOMBINAÇÃO FINALIZADA ==========\n")
    reg.fechar()
    return df_novas, df_rejeitadas, df_funil, reg.pasta
