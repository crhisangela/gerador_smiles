"""Aplica o fix de valencia na celula `rodar_uma_geracao` do notebook.

Problema: a ponte era colada direto na string (f"{smi_i}{ponte}{smi_j}"),
sem checar se o atomo de encaixe (ultimo atomo de smi_i, primeiro atomo de
smi_j) tinha valencia livre pra receber a nova ligacao. Quando esse atomo
era um heteroatomo de anel ja saturado (O/S com as 2 ligacoes do anel, ou
um N ja substituido por um auxocromo), o RDKit rejeitava o SMILES inteiro
com "Explicit valence ... is greater than permitted".

Fix: antes de montar o candidato, sorteia entre as pontes do catalogo
embaralhadas e usa a primeira cuja ordem de ligacao (simples/dupla/tripla,
lida no simbolo na ponta da string da ponte) cabe na valencia livre dos
dois atomos de encaixe. Se nenhuma ponte do catalogo servir pra aquele par
de aneis, o par e descartado (contabilizado), em vez de gerar um SMILES
invalido que ia ser jogado fora la na Fase A mesmo assim.
"""

import nbformat

CAMINHO = "notebook_geracao2_recombinacao_zhang_shasha.ipynb"
CELL_ID = "8c2931af-d746-4245-9655-51c82c72d389"

NOVO_SOURCE = '''def ordem_ligacao_borda(ponte_str, borda):
    """Ordem da ligacao (1, 2 ou 3) que a ponte forma no lado indicado
    ('inicio' ou 'fim'), lida no simbolo de ligacao explicito na ponta da
    string da ponte (ou 1, ligacao simples implicita, se nao houver)."""
    simbolo = ponte_str[0] if borda == "inicio" else ponte_str[-1]
    return {"=": 2, "#": 3}.get(simbolo, 1)


def tem_valencia_livre_para_ponte(smiles, borda, ponte_str):
    """Confere se o atomo de encaixe (primeiro atomo de `smiles`, se
    borda='inicio', ou ultimo atomo, se borda='fim') tem hidrogenios
    implicitos/explicitos suficientes pra virar a ligacao que a ponte exige
    nessa ponta, sem estourar a valencia do atomo."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    idx_atomo = 0 if borda == "inicio" else mol.GetNumAtoms() - 1
    ordem_necessaria = ordem_ligacao_borda(ponte_str, borda)
    return mol.GetAtomWithIdx(idx_atomo).GetTotalNumHs() >= ordem_necessaria


def rodar_uma_geracao(df_sementes, df_historico_completo, catalogo_pontes_lista,
                       canonicos_complexo_conhecidos, canonicos_aromatico_conhecidos,
                       numero_geracao, n_sondas=10, semente=44):
    catalogo_bruto = pd.concat(
        [extrair_catalogo_aneis(df_historico_completo, 1),
         extrair_catalogo_aneis(df_historico_completo, 2)],
        ignore_index=True,
    )
    catalogo_aneis_geracao = catalogo_bruto.drop_duplicates(subset="smiles_modificado").reset_index(drop=True)
    arvores_catalogo_geracao = [anel_para_arvore(row) for _, row in catalogo_aneis_geracao.iterrows()]
    n_catalogo = len(catalogo_aneis_geracao)

    # So as moleculas SEMENTES (novas da geracao anterior) contribuem com um
    # candidato cada -- nao o historico inteiro. Isso evita que o numero de
    # candidatos cresca geometricamente a cada rodada (ver Secao 8).
    base_anel1_geracao = extrair_catalogo_aneis(df_sementes, 1)

    sorteador_pares = random.Random(semente)
    n_sondas_efetivo = min(n_sondas, n_catalogo)
    pares_registros = []
    for pos in range(len(base_anel1_geracao)):
        row_i = base_anel1_geracao.iloc[pos]
        arvore_i = anel_para_arvore(row_i)
        indices_sondados = sorteador_pares.sample(range(n_catalogo), n_sondas_efetivo)
        distancias_sondadas = [
            (idx, ted_normalizada(arvore_i, arvores_catalogo_geracao[idx]))
            for idx in indices_sondados
        ]
        melhor_idx, melhor_distancia = max(distancias_sondadas, key=lambda par: par[1])
        pares_registros.append({"pos_base": pos, "j": melhor_idx, "distancia": melhor_distancia})
    df_pares_geracao = pd.DataFrame(pares_registros)

    sorteador_pontes = random.Random(semente * 7 + 1)
    registros_novos = []
    pares_sem_ponte_compativel = 0

    for _, par in df_pares_geracao.iterrows():
        row_i = base_anel1_geracao.iloc[int(par["pos_base"])]
        row_j = catalogo_aneis_geracao.iloc[int(par["j"])]

        smi_i = row_i["smiles_modificado"]
        smi_j_bruto = row_j["smiles_modificado"]

        # Antes so sorteava 1 ponte e colava direto na string -- se o atomo
        # de encaixe nao tivesse valencia livre (heteroatomo de anel ja
        # saturado, ou N ja substituido), o SMILES nascia invalido e so
        # descobriamos isso na Fase A, via excecao do RDKit. Agora testa as
        # pontes do catalogo embaralhadas e fica com a primeira que cabe na
        # valencia livre dos dois lados.
        pontes_embaralhadas = list(catalogo_pontes_lista)
        sorteador_pontes.shuffle(pontes_embaralhadas)

        ponte_escolhida = None
        for ponte_candidata in pontes_embaralhadas:
            ponte_str = ponte_candidata["Ponte_Estrutura"]
            if not tem_valencia_livre_para_ponte(smi_i, "fim", ponte_str):
                continue
            if not tem_valencia_livre_para_ponte(smi_j_bruto, "inicio", ponte_str):
                continue
            ponte_escolhida = ponte_candidata
            break

        if ponte_escolhida is None:
            pares_sem_ponte_compativel += 1
            continue

        smi_j = substituir_numeracao_anel(smi_j_bruto)
        smiles_complexo = f"{smi_i}{ponte_escolhida['Ponte_Estrutura']}{smi_j}"

        registros_novos.append({
            "base_Anel1": row_i["base"], "smiles_modificado_Anel1": row_i["smiles_modificado"],
            "frag1_Anel1": row_i["frag1"], "pos1_Anel1": row_i["pos1"],
            "frag2_Anel1": row_i["frag2"], "pos2_Anel1": row_i["pos2"],
            "frag3_Anel1": row_i["frag3"], "pos3_Anel1": row_i["pos3"],
            "ID_Ponte": ponte_escolhida["ID_Ponte"], "Ponte_Estrutura": ponte_escolhida["Ponte_Estrutura"],
            "base_Anel2": row_j["base"], "smiles_modificado_Anel2": row_j["smiles_modificado"],
            "frag1_Anel2": row_j["frag1"], "pos1_Anel2": row_j["pos1"],
            "frag2_Anel2": row_j["frag2"], "pos2_Anel2": row_j["pos2"],
            "frag3_Anel2": row_j["frag3"], "pos3_Anel2": row_j["pos3"],
            "distancia_ted_normalizada": par["distancia"],
            "SMILES_Complexo": smiles_complexo,
        })
    df_candidatos_geracao = pd.DataFrame(registros_novos)

    # Fase A -- SMILES valido / unico na leva / realmente novo
    tentativas = len(df_candidatos_geracao)
    removidos_invalidos = 0
    removidos_duplicata_interna = 0
    removidos_ja_existentes = 0
    vistos_internos = set()
    linhas_validas = []
    canonicos_validos = []

    for _, row in df_candidatos_geracao.iterrows():
        canonico, valido = validar_e_canonizar(row["SMILES_Complexo"])
        if not valido:
            removidos_invalidos += 1
            continue
        if canonico in vistos_internos:
            removidos_duplicata_interna += 1
            continue
        if canonico in canonicos_complexo_conhecidos:
            removidos_ja_existentes += 1
            continue
        vistos_internos.add(canonico)
        linhas_validas.append(row)
        canonicos_validos.append(canonico)

    df_fase_a = pd.DataFrame(linhas_validas).reset_index(drop=True)
    df_fase_a["SMILES_Canonico_Complexo_Candidato"] = canonicos_validos

    # Fase B -- aromatizacao + revalidacao + duplicata/novidade final
    if len(df_fase_a) > 0:
        df_fase_a["SMILES_Aromatico_Candidato"] = df_fase_a["SMILES_Complexo"].apply(aromatizar_aneis_6_membros)
    else:
        df_fase_a["SMILES_Aromatico_Candidato"] = pd.Series(dtype=object)

    antes_aromatizacao = len(df_fase_a)
    df_fase_b = df_fase_a.dropna(subset=["SMILES_Aromatico_Candidato"]).reset_index(drop=True)
    removidos_aromatizacao = antes_aromatizacao - len(df_fase_b)

    vistos_finais = set()
    removidos_nao_revalidados = 0
    removidos_duplicata_pos_aromatizacao = 0
    removidos_ja_existentes_pos_aromatizacao = 0
    linhas_finais = []

    for _, row in df_fase_b.iterrows():
        mol = Chem.MolFromSmiles(row["SMILES_Aromatico_Candidato"])
        if mol is None:
            removidos_nao_revalidados += 1
            continue
        canonico_final = Chem.MolToSmiles(mol, canonical=True)
        if canonico_final in vistos_finais:
            removidos_duplicata_pos_aromatizacao += 1
            continue
        if canonico_final in canonicos_aromatico_conhecidos:
            removidos_ja_existentes_pos_aromatizacao += 1
            continue
        vistos_finais.add(canonico_final)
        linhas_finais.append(row)

    df_geracao = pd.DataFrame(linhas_finais).reset_index(drop=True)
    if len(df_geracao) > 0:
        df_geracao["smiles"] = df_geracao["SMILES_Aromatico_Candidato"]
    df_geracao["Geracao"] = numero_geracao

    estatisticas = {
        "geracao": numero_geracao,
        "sementes_entrada": len(df_sementes),
        "historico_completo_no_momento": len(df_historico_completo),
        "catalogo_aneis_disponivel": n_catalogo,
        "pares_descartados_sem_ponte_compativel": pares_sem_ponte_compativel,
        "candidatos_gerados": tentativas,
        "distancia_media_pares": df_pares_geracao["distancia"].mean() if tentativas else float("nan"),
        "removidos_invalidos": removidos_invalidos,
        "removidos_duplicata_interna": removidos_duplicata_interna,
        "removidos_ja_existentes": removidos_ja_existentes,
        "validos_apos_fase_a": len(df_fase_a),
        "removidos_falha_aromatizacao": removidos_aromatizacao,
        "validos_apos_aromatizacao": len(df_fase_b),
        "removidos_nao_revalidados": removidos_nao_revalidados,
        "removidos_duplicata_pos_aromatizacao": removidos_duplicata_pos_aromatizacao,
        "removidos_ja_existentes_pos_aromatizacao": removidos_ja_existentes_pos_aromatizacao,
        "moleculas_novas": len(df_geracao),
        "taxa_sobrevivencia": (len(df_geracao) / tentativas) if tentativas else 0.0,
    }

    return df_geracao, estatisticas
'''

nb = nbformat.read(CAMINHO, as_version=4)

alvo = None
for cell in nb.cells:
    if cell.get("id") == CELL_ID:
        alvo = cell
        break

if alvo is None:
    raise SystemExit(f"Celula {CELL_ID} nao encontrada.")

alvo["source"] = NOVO_SOURCE
alvo["outputs"] = []
alvo["execution_count"] = None

nbformat.write(nb, CAMINHO)
print("Celula atualizada com sucesso.")
