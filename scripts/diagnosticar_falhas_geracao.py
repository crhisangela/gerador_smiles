"""Diagnóstico: por que tantos candidatos falham na Geração 2?

Reproduz exatamente a lógica de `rodar_uma_geracao` (mesma semente,
mesmos n_sondas) usando `base_daniel.csv` como semente + histórico, mas em
vez de só descartar os SMILES inválidos, categoriza CADA falha por causa-raiz
usando as exceções tipadas do RDKit (AtomValenceException, KekulizeException,
etc.) em vez de tentar interpretar texto de stderr.

Rodar: .venv/Scripts/python.exe scripts/diagnosticar_falhas_geracao.py
"""

import random
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from zss import Node, simple_distance

RDLogger.DisableLog("rdApp.*")  # silencia stderr do RDKit; classificamos via exceções tipadas


def extrair_catalogo_aneis(df, sufixo):
    colunas = {
        "base": f"SMILES_Base_Anel{sufixo}",
        "smiles_modificado": f"SMILES_Modificado_Anel{sufixo}",
        "frag1": f"Frag_Estrutura_1_Anel{sufixo}", "pos1": f"Posicao_1_Anel{sufixo}",
        "frag2": f"Frag_Estrutura_2_Anel{sufixo}", "pos2": f"Posicao_2_Anel{sufixo}",
        "frag3": f"Frag_Estrutura_3_Anel{sufixo}", "pos3": f"Posicao_3_Anel{sufixo}",
    }
    sub = df[list(colunas.values())].copy()
    sub.columns = list(colunas.keys())
    sub["origem"] = f"Anel{sufixo}"
    return sub


def valor_valido(v):
    return pd.notna(v) and str(v).strip() not in {"", "nan", "None"}


def anel_para_arvore(row):
    raiz = Node(f"ANEL:{row['base']}")
    itens = []
    for slot in (1, 2, 3):
        frag = row[f"frag{slot}"]
        pos = row[f"pos{slot}"]
        if valor_valido(frag):
            itens.append((str(pos), str(frag)))
    for pos, frag in sorted(itens, key=lambda x: x[0]):
        no_pos = Node(f"POS:{pos}")
        no_pos.addkid(Node(f"FRAG:{frag}"))
        raiz.addkid(no_pos)
    return raiz


def tamanho_arvore(no):
    return 1 + sum(tamanho_arvore(f) for f in no.children)


def ted_normalizada(a, b):
    d = simple_distance(a, b)
    denom = max(tamanho_arvore(a), tamanho_arvore(b), 1)
    return d / denom


def substituir_numeracao_anel(smiles, de="1", para="2"):
    return smiles.replace(de, para)


def classificar_sanitize_exception(exc):
    nome = type(exc).__name__
    if nome in ("AtomValenceException", "AtomKekulizeException"):
        return "valencia" if nome == "AtomValenceException" else "kekulizacao"
    if nome == "KekulizeException":
        return "kekulizacao"
    if nome == "AtomSanitizeException":
        return str(exc)
    return f"sanitize_outro:{nome}"


def diagnosticar_smiles(smiles):
    """Retorna (ok, categoria). Reclassifica com exceções tipadas do RDKit,
    em vez de só olhar se MolFromSmiles(default) retornou None."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False, "erro_parse_sintaxe"
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.MolSanitizeException as exc:
        return False, classificar_sanitize_exception(exc)
    except Exception as exc:  # pragma: no cover - defensivo
        return False, f"erro_inesperado:{type(exc).__name__}"
    return True, "ok"


def aromatizar_aneis_6_membros_diagnostico(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "falhou_antes_de_aromatizar"

    for anel in mol.GetRingInfo().AtomRings():
        if len(anel) != 6:
            continue
        for k in range(len(anel)):
            a1, a2 = anel[k], anel[(k + 1) % len(anel)]
            ligacao = mol.GetBondBetweenAtoms(a1, a2)
            if ligacao is None:
                return None, "ligacao_ausente_no_anel"
            ligacao.SetBondType(Chem.rdchem.BondType.AROMATIC)
        for idx_atomo in anel:
            mol.GetAtomWithIdx(idx_atomo).SetIsAromatic(True)

    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.MolSanitizeException as exc:
        return None, classificar_sanitize_exception(exc)
    except Exception as exc:  # pragma: no cover
        return None, f"erro_inesperado:{type(exc).__name__}"

    return Chem.MolToSmiles(mol), "ok"


def main():
    df_base = pd.read_csv("base_daniel.csv", index_col=0)
    print(f"df_base: {len(df_base):,} moleculas (Geracao 1)")

    catalogo_bruto = pd.concat(
        [extrair_catalogo_aneis(df_base, 1), extrair_catalogo_aneis(df_base, 2)],
        ignore_index=True,
    )
    catalogo_aneis = catalogo_bruto.drop_duplicates(subset="smiles_modificado").reset_index(drop=True)
    arvores_catalogo = [anel_para_arvore(row) for _, row in catalogo_aneis.iterrows()]
    n_catalogo = len(catalogo_aneis)
    print(f"catalogo_aneis: {n_catalogo:,} aneis unicos")

    catalogo_pontes_lista = (
        df_base[["ID_Ponte", "Ponte_Estrutura"]].drop_duplicates().to_dict("records")
    )

    # Reproduz Geracao 2: sementes = df_base inteiro (como no notebook)
    base_anel1 = extrair_catalogo_aneis(df_base, 1)

    N_SONDAS = 10
    SEMENTE = 44 + 2  # SEMENTE_BASE + proxima_geracao, igual ao notebook p/ geracao 2

    sorteador_pares = random.Random(SEMENTE)
    n_sondas_efetivo = min(N_SONDAS, n_catalogo)
    pares = []
    for pos in range(len(base_anel1)):
        arvore_i = anel_para_arvore(base_anel1.iloc[pos])
        indices = sorteador_pares.sample(range(n_catalogo), n_sondas_efetivo)
        dists = [(idx, ted_normalizada(arvore_i, arvores_catalogo[idx])) for idx in indices]
        melhor_idx, _ = max(dists, key=lambda p: p[1])
        pares.append((pos, melhor_idx))

    sorteador_pontes = random.Random(SEMENTE * 7 + 1)
    candidatos = []
    for pos, j in pares:
        row_i = base_anel1.iloc[pos]
        row_j = catalogo_aneis.iloc[j]
        ponte = sorteador_pontes.choice(catalogo_pontes_lista)
        smi_i = row_i["smiles_modificado"]
        smi_j = substituir_numeracao_anel(row_j["smiles_modificado"])
        smiles_complexo = f"{smi_i}{ponte['Ponte_Estrutura']}{smi_j}"
        candidatos.append(smiles_complexo)

    print(f"\ncandidatos gerados: {len(candidatos):,}\n")

    # Fase A: SMILES bruto (antes de aromatizar) -- capturar causa raiz tipada
    contagem_fase_a = Counter()
    exemplos_fase_a = {}
    validos_fase_a = []
    for smi in candidatos:
        ok, categoria = diagnosticar_smiles(smi)
        contagem_fase_a[categoria] += 1
        if not ok and categoria not in exemplos_fase_a:
            exemplos_fase_a[categoria] = smi
        if ok:
            validos_fase_a.append(smi)

    print("=" * 70)
    print("FASE A -- validar_e_canonizar (SMILES bruto, antes de aromatizar)")
    print("=" * 70)
    for categoria, n in contagem_fase_a.most_common():
        pct = 100 * n / len(candidatos)
        print(f"  {categoria:35s} {n:5d}  ({pct:5.1f}%)")
    print(f"\n  validos apos Fase A: {len(validos_fase_a):,} / {len(candidatos):,}")

    print("\nExemplos por categoria de falha (Fase A):")
    for categoria, smi in exemplos_fase_a.items():
        if categoria == "ok":
            continue
        print(f"  [{categoria}] {smi}")

    # Fase B: aromatizacao dos que passaram na Fase A
    contagem_fase_b = Counter()
    exemplos_fase_b = {}
    for smi in validos_fase_a:
        _, categoria = aromatizar_aneis_6_membros_diagnostico(smi)
        contagem_fase_b[categoria] += 1
        if categoria != "ok" and categoria not in exemplos_fase_b:
            exemplos_fase_b[categoria] = smi

    print("\n" + "=" * 70)
    print("FASE B -- aromatizar_aneis_6_membros (so quem passou na Fase A)")
    print("=" * 70)
    for categoria, n in contagem_fase_b.most_common():
        pct = 100 * n / max(len(validos_fase_a), 1)
        print(f"  {categoria:35s} {n:5d}  ({pct:5.1f}%)")

    print("\nExemplos por categoria de falha (Fase B):")
    for categoria, smi in exemplos_fase_b.items():
        print(f"  [{categoria}] {smi}")

    total_ok_final = contagem_fase_b.get("ok", 0)
    print(f"\nSobrevivencia total desta geracao (antes de checar duplicata/novidade): "
          f"{total_ok_final:,} / {len(candidatos):,} ({100*total_ok_final/len(candidatos):.1f}%)")


if __name__ == "__main__":
    main()
