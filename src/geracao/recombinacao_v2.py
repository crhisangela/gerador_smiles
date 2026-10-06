"""Recombinação v2 (Zhang–Shasha) — árvores uniformes, encaixe por valência,
Kekulé tolerante, rebote com escada de fallback, memória e guia por interesse.

Diferenças em relação ao notebook da Geração 2:

* Toda árvore de anel tem o MESMO esquema: ANEL -> k x (POS -> FRAG), com
  k fixo (`N_SUBSTITUINTES`), preenchendo com "-" se faltar. Começa com k=1.
* A ponte NÃO é colada na string. O átomo de encaixe de cada anel é escolhido
  pela valência livre (H suficientes para a ordem de ligação que a ponta da
  ponte exige). Pontes terminadas em "=" deixaram de estourar valência.
* Kekulé não inviabiliza: se a kekulização falha, a molécula é mantida com
  `kekule_ok=False` (sanitização sem KEKULIZE/SETAROMATICITY).
* O rebote cobre o pipeline inteiro (junção -> sanitização -> aromatização ->
  novidade), não só a validade do SMILES. Se as sondas esgotam, a escada de
  fallback troca a ordem dos anéis, testa outras pontes e amplia as sondas.
* `Memoria` aprende a cada candidato (sucesso por ponte/base/fragmento e
  recompensa de interesse) e enviesa a escolha das pontes na geração seguinte.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

import pandas as pd
from rdkit import Chem, RDLogger
from zss import Node, simple_distance

RDLogger.DisableLog("rdApp.*")

N_SUBSTITUINTES = 1
_VAZIO = {"", "nan", "None"}
ELEMENTOS_AROMATIZAVEIS = {"C", "N"}  # só anéis C/N são aromatizados


def valor_valido(v) -> bool:
    return pd.notna(v) and str(v).strip() not in _VAZIO


# ---------------------------------------------------------------- anéis/árvores
def montar_anel(base: str, pos, frag: str) -> str:
    """Mesma regra de `inserir_fragmento`: ramificação no offset da string."""
    pos = max(0, min(int(pos), len(base)))
    return base[:pos] + f"({frag})" + base[pos:]


@dataclass
class Anel:
    base: str
    subs: tuple  # ((pos, frag), ...) com exatamente k itens
    smiles: str
    mol: Chem.Mol | None
    arvore: Node
    chave: str = field(init=False)

    def __post_init__(self):
        self.chave = self.smiles


def arvore_anel(base: str, subs: tuple, k: int = N_SUBSTITUINTES) -> Node:
    """Árvore de forma fixa: raiz + k pares POS/FRAG (preenche com '-')."""
    itens = sorted(subs, key=lambda x: str(x[0]))[:k]
    itens += [("-", "-")] * (k - len(itens))
    raiz = Node(f"ANEL:{base}")
    for pos, frag in itens:
        no = Node(f"POS:{pos}")
        no.addkid(Node(f"FRAG:{frag}"))
        raiz.addkid(no)
    return raiz


def tamanho_arvore(no: Node) -> int:
    return 1 + sum(tamanho_arvore(f) for f in no.children)


def ted_normalizada(a: Node, b: Node) -> float:
    return simple_distance(a, b) / max(tamanho_arvore(a), tamanho_arvore(b), 1)


def validar_arvores_uniformes(aneis: list[Anel]) -> bool:
    return len({tamanho_arvore(a.arvore) for a in aneis}) == 1


def catalogo_aneis(df: pd.DataFrame, k: int = N_SUBSTITUINTES) -> tuple[list[Anel], Counter]:
    """Anéis únicos (base + k primeiros substituintes) de Anel1 e Anel2."""
    vistos, aneis, descartes = set(), [], Counter()
    for suf in (1, 2):
        for _, r in df.iterrows():
            base = str(r[f"SMILES_Base_Anel{suf}"])
            subs = tuple(
                (str(r[f"Posicao_{i}_Anel{suf}"]), str(r[f"Frag_Estrutura_{i}_Anel{suf}"]))
                for i in range(1, k + 1)
                if valor_valido(r.get(f"Frag_Estrutura_{i}_Anel{suf}"))
            )
            if (base, subs) in vistos:
                continue
            vistos.add((base, subs))
            smi = base
            for pos, frag in sorted(subs, key=lambda x: -int(float(x[0]))):
                smi = montar_anel(smi, int(float(pos)), frag)
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            mol, status = sanitizar_tolerante(mol) if mol is not None else (None, "parse")
            if mol is None or status not in ("ok", "sem_kekule"):
                descartes[status] += 1
                continue
            if status == "sem_kekule":
                descartes["anel_sem_kekule_mantido"] += 1
            aneis.append(Anel(base, subs, smi, mol, arvore_anel(base, subs, k)))
    return aneis, descartes


# ------------------------------------------------------------ sanitização/junção
def sanitizar_tolerante(mol: Chem.Mol):
    """(mol, status): 'ok' | 'sem_kekule' | 'valencia' | 'outro'.
    Kekulé NÃO descarta; valência sim (é erro químico real)."""
    base = Chem.Mol(mol)
    try:
        Chem.SanitizeMol(mol)
        return mol, "ok"
    except Chem.rdchem.KekulizeException:
        pass
    except Chem.rdchem.AtomValenceException:
        return None, "valencia"
    except Exception:
        return None, "outro"
    tolerante = Chem.Mol(base)
    ops = Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE ^ Chem.SANITIZE_SETAROMATICITY
    try:
        Chem.SanitizeMol(tolerante, sanitizeOps=ops)
        return tolerante, "sem_kekule"
    except Chem.rdchem.AtomValenceException:
        return None, "valencia"
    except Exception:
        return None, "outro"


def _ordens_ponte(ponte: str) -> tuple[Chem.Mol, int, int, int, int]:
    """Parseia a ponte entre dois curingas '*' e devolve as ordens/vizinhos
    de cada ponta (ordem de ligação que cada anel precisa aceitar)."""
    br = Chem.MolFromSmiles(f"[*]{ponte}[*]", sanitize=False)
    d1, d2 = [a.GetIdx() for a in br.GetAtoms() if a.GetAtomicNum() == 0]
    b1 = br.GetAtomWithIdx(d1).GetBonds()[0]
    b2 = br.GetAtomWithIdx(d2).GetBonds()[0]
    return br, int(b1.GetBondTypeAsDouble()), int(b2.GetBondTypeAsDouble()), d1, d2


def _candidatos_encaixe(mol: Chem.Mol, ordem: int, rng: random.Random) -> list[int]:
    ok = [a.GetIdx() for a in mol.GetAtoms() if a.GetTotalNumHs() >= ordem]
    rng.shuffle(ok)
    return ok


def ligar(m1: Chem.Mol, ponte: str, m2: Chem.Mol, rng: random.Random):
    """Une anel1 + ponte + anel2 escolhendo átomos de encaixe com valência livre.
    Retorna (mol_nao_sanitizada, (a1, a2)) ou (None, motivo)."""
    br, o1, o2, d1, d2 = _ordens_ponte(ponte)
    c1 = _candidatos_encaixe(m1, o1, rng)
    c2 = _candidatos_encaixe(m2, o2, rng)
    if not c1 or not c2:
        return None, "sem_encaixe"
    a1, a2 = c1[0], c2[0]

    off_b = m1.GetNumAtoms()
    off_2 = off_b + br.GetNumAtoms()
    rw = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(m1, br), m2))
    tipo = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    n1 = br.GetAtomWithIdx(d1).GetBonds()[0].GetOtherAtomIdx(d1)
    n2 = br.GetAtomWithIdx(d2).GetBonds()[0].GetOtherAtomIdx(d2)
    if n1 == d2:  # ponte é só uma ligação ("-", "=")
        rw.AddBond(a1, off_2 + a2, tipo[o1])
    else:
        rw.AddBond(a1, off_b + n1, tipo[o1])
        rw.AddBond(a2 + off_2, off_b + n2, tipo[o2])
    for idx in sorted((off_b + d1, off_b + d2), reverse=True):
        rw.RemoveAtom(idx)
    return rw.GetMol(), (a1, a2)


def aromatizar_aneis_6(mol: Chem.Mol) -> Chem.Mol | None:
    """Aromatiza só anéis de 6 membros formados apenas por C e N (O, S, B
    nunca viram aromáticos aqui). None se a aromatização falhar."""
    rw = Chem.RWMol(Chem.MolFromSmiles(Chem.MolToSmiles(mol)))
    for anel in rw.GetRingInfo().AtomRings():
        if len(anel) != 6:
            continue
        if any(rw.GetAtomWithIdx(i).GetSymbol() not in ELEMENTOS_AROMATIZAVEIS for i in anel):
            continue
        for k in range(6):
            lig = rw.GetBondBetweenAtoms(anel[k], anel[(k + 1) % 6])
            if lig is None:
                return None
            lig.SetBondType(Chem.BondType.AROMATIC)
        for i in anel:
            rw.GetAtomWithIdx(i).SetIsAromatic(True)
    try:
        m = rw.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


# ---------------------------------------------------------------------- interesse
def interesse_conjugacao(mol: Chem.Mol) -> float:
    """Fração dos átomos pesados no maior sistema conjugado (0–1).
    Substituível: qualquer f(mol) -> float em [0, 1] serve como guia."""
    grupos: list[set[int]] = []
    for b in mol.GetBonds():
        if not b.GetIsConjugated():
            continue
        ab = {b.GetBeginAtomIdx(), b.GetEndAtomIdx()}
        unidos = [g for g in grupos if g & ab]
        for g in unidos:
            ab |= g
            grupos.remove(g)
        grupos.append(ab)
    return max((len(g) for g in grupos), default=0) / max(mol.GetNumHeavyAtoms(), 1)


# ------------------------------------------------------------------------ memória
class Memoria:
    """Aprende a cada candidato: (tentativas, sucessos, soma de interesse) por
    característica. Os pesos enviesam a escolha da ponte e podem ser
    inspecionados em `tabela()` entre gerações."""

    def __init__(self):
        self.stat = defaultdict(lambda: [0, 0, 0.0])
        self.falhas: set[tuple] = set()

    def registrar(self, chaves, sucesso: bool, interesse: float = 0.0):
        for c in chaves:
            s = self.stat[c]
            s[0] += 1
            s[1] += int(sucesso)
            s[2] += interesse if sucesso else 0.0

    def peso(self, chave, peso_interesse: float = 0.0) -> float:
        t, s, i = self.stat[chave]
        p_ok = (s + 1) / (t + 2)  # Laplace
        media_int = (i / s) if s else 0.5
        return p_ok * (1 + peso_interesse * media_int)

    def tabela(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(k[0], k[1], t, s, s / t if t else 0.0, i / s if s else 0.0)
             for k, (t, s, i) in self.stat.items()],
            columns=["tipo", "chave", "tentativas", "sucessos", "taxa_ok", "interesse_medio"],
        ).sort_values(["tipo", "taxa_ok"], ascending=[True, False])


# ------------------------------------------------------------------- uma geração
def _tentar(a: Anel, b: Anel, ponte: dict, rng, mem: Memoria, est: Counter,
            conhecidos_complexo: set, conhecidos_final: set, vistos: set,
            interesse: Callable[[Chem.Mol], float]):
    """Pipeline completo de um candidato. Retorna dict ou None (motivo em est)."""
    assin = (a.chave, ponte["Ponte_Estrutura"], b.chave)
    chaves = [("ponte", ponte["ID_Ponte"]), ("base", a.base), ("base", b.base)]
    if assin in mem.falhas:
        est["memoria_evitou"] += 1
        return None
    mol, enc = ligar(a.mol, ponte["Ponte_Estrutura"], b.mol, rng)
    if mol is None:
        est["sem_encaixe"] += 1
        mem.falhas.add(assin)
        mem.registrar(chaves, False)
        return None
    mol, status = sanitizar_tolerante(mol)
    if mol is None:
        est[f"invalido_{status}"] += 1
        mem.falhas.add(assin)
        mem.registrar(chaves, False)
        return None
    canon_complexo = Chem.MolToSmiles(mol)
    if canon_complexo in vistos:
        est["duplicata_interna"] += 1
        return None
    if canon_complexo in conhecidos_complexo:
        est["ja_existente"] += 1
        return None

    kekule_ok = status == "ok"
    final = canon_complexo
    aromatizada = False
    if kekule_ok:
        arom = aromatizar_aneis_6(mol)
        smi_arom = Chem.MolToSmiles(arom) if arom is not None else None
        # Ida-e-volta: só vale se o RDKit relê a string (~55% das aromatizações
        # forçadas geram SMILES que não reabre). Senão mantém a forma não
        # aromatizada, sem descartar a molécula.
        if smi_arom is not None and Chem.MolFromSmiles(smi_arom) is not None:
            final, aromatizada = smi_arom, True
        else:
            est["aromatizacao_nao_reabre_mantida"] += 1
    else:
        est["kekule_tolerado"] += 1
    if final in conhecidos_final or final in vistos:
        est["ja_existente_final"] += 1
        return None

    gostou = interesse(mol)
    mem.registrar(chaves, True, gostou)
    vistos.update({canon_complexo, final})
    return {"mol": mol, "complexo": canon_complexo, "final": final, "kekule_ok": kekule_ok,
            "aromatizada": aromatizada, "interesse": gostou, "encaixe": enc}


def _linha(a: Anel, b: Anel, ponte: dict, r: dict, dist: float, nivel: int, ger: int) -> dict:
    lin = {}
    for suf, an in ((1, a), (2, b)):
        lin[f"SMILES_Base_Anel{suf}"] = an.base
        lin[f"SMILES_Modificado_Anel{suf}"] = an.smiles
        for i in range(1, 4):
            sub = an.subs[i - 1] if i <= len(an.subs) else (None, None)
            lin[f"Posicao_{i}_Anel{suf}"], lin[f"Frag_Estrutura_{i}_Anel{suf}"] = sub
    lin.update({
        "ID_Ponte": ponte["ID_Ponte"], "Ponte_Estrutura": ponte["Ponte_Estrutura"],
        "SMILES_Canonico_Complexo": r["complexo"], "smiles": r["final"],
        "kekule_ok": r["kekule_ok"], "aromatizada": r["aromatizada"],
        "interesse": r["interesse"], "distancia_ted": dist,
        "nivel_fallback": nivel, "Geracao": ger,
    })
    return lin


def rodar_uma_geracao_v2(sementes: list[Anel], catalogo: list[Anel], pontes: list[dict],
                         conhecidos_complexo: set, conhecidos_final: set, memoria: Memoria,
                         numero_geracao: int, n_sondas: int = 10, semente: int = 44,
                         interesse: Callable[[Chem.Mol], float] = interesse_conjugacao,
                         peso_interesse: float = 0.0, rodadas_extra: int = 3):
    """Para cada semente aplica o rebote em 4 níveis:
    1. sondas ordenadas por distância (empates embaralhados), 1ª aceitável vence;
    2. mesma lista com a ordem dos anéis trocada (partner + semente);
    3. todas as pontes (ordenadas pela memória) para cada sonda, nas 2 ordens;
    4. `rodadas_extra` novos lotes de sondas até esgotar o catálogo.
    Com `peso_interesse` > 0, entre os aceitáveis do nível 1 escolhe-se o de
    maior distância + peso * interesse em vez do primeiro."""
    rng = random.Random(semente)
    est, vistos, linhas, pares = Counter(), set(), [], []
    n = len(catalogo)
    pesos_pontes = lambda: [memoria.peso(("ponte", p["ID_Ponte"]), peso_interesse) for p in pontes]

    def sortear_ponte():
        return rng.choices(pontes, weights=pesos_pontes(), k=1)[0]

    def sondar(sem, usados):
        livres = [i for i in range(n) if i not in usados and catalogo[i] is not sem]
        idx = rng.sample(livres, min(n_sondas, len(livres)))
        usados.update(idx)
        lote = [(i, ted_normalizada(sem.arvore, catalogo[i].arvore)) for i in idx]
        rng.shuffle(lote)  # empates deixam de favorecer índice baixo
        lote.sort(key=lambda p: p[1], reverse=True)
        return lote

    for sem in sementes:
        usados: set[int] = set()
        lote = sondar(sem, usados)
        achou = None
        for nivel in (1, 2, 3, 4):
            if nivel == 4:
                for _ in range(rodadas_extra):
                    if len(usados) >= n - 1:
                        break
                    lote = sondar(sem, usados)
                    achou = _varrer(sem, lote, catalogo, sortear_ponte, False, est, rng, memoria,
                                    conhecidos_complexo, conhecidos_final, vistos, interesse,
                                    peso_interesse)
                    if achou:
                        break
            else:
                ordem_trocada = nivel == 2
                todas = nivel == 3
                achou = _varrer(sem, lote, catalogo, sortear_ponte, ordem_trocada, est, rng, memoria,
                                conhecidos_complexo, conhecidos_final, vistos, interesse,
                                peso_interesse, todas_pontes=todas, pontes=pontes,
                                pesos=pesos_pontes() if todas else None)
            if achou:
                a, b, ponte, r, dist = achou
                linhas.append(_linha(a, b, ponte, r, dist, nivel, numero_geracao))
                pares.append((a, b))
                est[f"resolvido_nivel_{nivel}"] += 1
                break
        else:
            est["semente_sem_parceiro"] += 1

    df = pd.DataFrame(linhas)
    est_df = {"geracao": numero_geracao, "sementes": len(sementes), "novas": len(df),
              "taxa_sementes_resolvidas": len(df) / max(len(sementes), 1),
              "interesse_medio": float(df["interesse"].mean()) if len(df) else float("nan"),
              **dict(est)}
    return df, est_df, pares


def _varrer(sem, lote, catalogo, sortear_ponte, ordem_trocada, est, rng, mem,
            conh_c, conh_f, vistos, interesse, peso_interesse,
            todas_pontes=False, pontes=None, pesos=None):
    aceitos = []
    for idx, dist in lote:
        par = catalogo[idx]
        a, b = (par, sem) if ordem_trocada else (sem, par)
        if todas_pontes:
            ordem = sorted(zip(pesos, range(len(pontes))), key=lambda x: -x[0])
            opcoes = [pontes[i] for _, i in ordem]
            opcoes += opcoes  # nível 3 também tenta a ordem trocada abaixo
        else:
            opcoes = [sortear_ponte()]
        for k, ponte in enumerate(opcoes):
            aa, bb = (a, b) if k < len(opcoes) // (2 if todas_pontes else 1) else (b, a)
            est["tentativas"] += 1
            r = _tentar(aa, bb, ponte, rng, mem, est, conh_c, conh_f, vistos, interesse)
            if r:
                aceitos.append((aa, bb, ponte, r, dist))
                break
        if aceitos and peso_interesse <= 0:
            break  # modo "primeiro aceitável" (comportamento do rebote original)
    if not aceitos:
        return None
    if peso_interesse > 0:  # modo guiado: melhor distância + interesse
        aceitos.sort(key=lambda x: x[4] + peso_interesse * x[3]["interesse"], reverse=True)
        for rej in aceitos[1:]:  # libera os não escolhidos para outras sementes
            vistos.discard(rej[3]["complexo"])
            vistos.discard(rej[3]["final"])
    return aceitos[0]


# ---------------------------------------------------------------- geração inicial
def geracao_inicial(df: pd.DataFrame, aneis: list[Anel], memoria: Memoria,
                    interesse: Callable[[Chem.Mol], float] = interesse_conjugacao,
                    semente: int = 1):
    """Geração 1 com k substituintes: reaproveita os pares anel+ponte+anel da
    base, mas com cada anel reduzido aos k primeiros substituintes.
    Retorna (df_g1, conhecidos_complexo, conhecidos_final, aneis_semente, est)."""
    k = len(aneis[0].subs)
    por_chave = {(a.base, a.subs): a for a in aneis}
    rng, est, vistos = random.Random(semente), Counter(), set()
    conh_c, conh_f, linhas, sementes = set(), set(), [], []

    def anel_de(r, suf):
        subs = tuple((str(r[f"Posicao_{i}_Anel{suf}"]), str(r[f"Frag_Estrutura_{i}_Anel{suf}"]))
                     for i in range(1, k + 1) if valor_valido(r.get(f"Frag_Estrutura_{i}_Anel{suf}")))
        return por_chave.get((str(r[f"SMILES_Base_Anel{suf}"]), subs))

    for _, r in df.iterrows():
        a, b = anel_de(r, 1), anel_de(r, 2)
        if a is None or b is None:
            est["anel_fora_do_catalogo"] += 1
            continue
        ponte = {"ID_Ponte": r["ID_Ponte"], "Ponte_Estrutura": r["Ponte_Estrutura"]}
        res = _tentar(a, b, ponte, rng, memoria, est, conh_c, conh_f, vistos, interesse)
        if res:
            conh_c.add(res["complexo"])
            conh_f.add(res["final"])
            linhas.append(_linha(a, b, ponte, res, 0.0, 0, 1))
            sementes.append(b)  # parceiro vira semente: o Anel1 nunca mudaria entre gerações
    return pd.DataFrame(linhas), conh_c, conh_f, sementes, dict(est)
