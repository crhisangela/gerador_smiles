"""Recombinação v3 — operadores de árvore guiados por Zhang–Shasha (TED).

Cada molécula da base (anel + ponte + anel) vira uma árvore ordenada:

    MOL
    ├── ANEL:<base1>
    │   ├── POS:2 ── FRAG:N
    │   ├── POS:3 ── FRAG:Br
    │   └── POS:6 ── FRAG:N(C)C
    ├── PONTE:C=C
    └── ANEL:<base2>
        └── ... (mesmo esquema)

A recombinação é feita por troca de subárvores entre dois pais P e Q:

* ``troca_anel``        — um ANEL inteiro de P é substituído por um de Q;
* ``troca_ponte``       — o nó PONTE de P recebe o rótulo da ponte de Q;
* ``troca_ramo``        — um ramo POS→FRAG de um anel de P vem de Q;
* ``troca_auxocromo``   — só a folha FRAG é trocada (posição mantida).

O parceiro Q de cada semente P é escolhido pela distância de edição de
árvore (Zhang–Shasha), normalizada pelo tamanho da maior árvore.

A montagem do SMILES é a MESMA do pipeline principal (inserção do auxocromo
como ramificação no offset da string, ``anel1 + ponte + anel2`` com
renumeração do anel 2, checagem de valência livre nas pontas) e a limpeza
reaproveita as funções de ``utils.validacao`` e ``utils.aromatizacao``.
Por cima dela entram os filtros químicos de ``filtros_quimicos``.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from rdkit import Chem, RDLogger
from zss import Node, simple_distance

from fragmentos.auxocromos import AUXOCROMOS
from fragmentos.pontes import PONTES
from geracao.combinacoes import (
    _substituir_numeracao_anel,
    _tem_valencia_livre_para_ponte,
)
from geracao.fragmentos import inserir_fragmento

RDLogger.DisableLog("rdApp.*")

# Número máximo de posições de auxocromo por anel na representação interna.
# Limite de representação: bases da v3 têm no máximo 3 posições por anel.
N_SLOTS = 3
OPERADORES = ("troca_anel", "troca_ponte", "troca_ramo", "troca_auxocromo")
_ID_AUXOCROMO = {v: str(k) for k, v in AUXOCROMOS.items()}
_ID_PONTE = {v: str(k) for k, v in PONTES.items()}

# Grafias antigas de auxocromos presentes em bases já geradas -> grafia atual.
# O nitro era "[N+](=O)O" (cátion sem contra-íon, carga líquida +1).
CORRECOES_AUXOCROMO = {"[N+](=O)O": "[N+](=O)[O-]"}


# ------------------------------------------------------------------ estruturas
@dataclass(frozen=True)
class Anel:
    id_anel: int
    base: str
    subs: tuple  # ((posicao:int, frag:str), ...) na ordem dos slots 1..3

    @property
    def smiles(self) -> str:
        """Mesma regra de ``gerar_smiles_modificado``: insere da maior
        posição para a menor para não deslocar os offsets."""
        smi = self.base
        for pos, frag in sorted(self.subs, key=lambda x: -x[0]):
            smi = inserir_fragmento(smi, pos, frag)
        return smi


@dataclass(frozen=True)
class Molecula:
    anel1: Anel
    ponte: str
    id_ponte: str
    anel2: Anel


def _anel_da_linha(r, suf: int, corrigir: bool = True) -> Anel:
    def frag(i):
        f = str(r[f"Frag_Estrutura_{i}_Anel{suf}"])
        return CORRECOES_AUXOCROMO.get(f, f) if corrigir else f

    subs = tuple(
        (int(r[f"Posicao_{i}_Anel{suf}"]), frag(i))
        for i in range(1, N_SLOTS + 1)
        if pd.notna(r.get(f"Frag_Estrutura_{i}_Anel{suf}"))
    )
    return Anel(int(r[f"ID_Anel_Anel{suf}"]), str(r[f"SMILES_Base_Anel{suf}"]), subs)


def moleculas_da_base(df: pd.DataFrame) -> list[Molecula]:
    """Moléculas do histórico já com as grafias de auxocromo corrigidas."""
    return [
        Molecula(_anel_da_linha(r, 1), str(r["Ponte_Estrutura"]), str(r["ID_Ponte"]),
                 _anel_da_linha(r, 2))
        for _, r in df.iterrows()
    ]


def canonicos_complexo(moleculas: list[Molecula]) -> set[str]:
    """Canônicos dos complexos remontados (com as correções aplicadas), para
    que uma molécula da base com o nitro corrigido não conte como nova."""
    canon = set()
    for mol in moleculas:
        smi, _ = montar_smiles(mol)
        c = _canonico(smi) if smi else None
        if c:
            canon.add(c)
    return canon


# ------------------------------------------------------------------- validação
def validar_base(df: pd.DataFrame) -> pd.DataFrame:
    """Confere se o histórico da base é coerente antes de recombinar.

    Retorna uma tabela (verificação, ok, total) com:
    - reconstrução: base + (posição, auxocromo) reproduz SMILES_Modificado;
    - SMILES dos anéis, do complexo e final abrem no RDKit;
    - canônicos sem duplicata.
    """
    n = len(df)
    linhas = []
    for suf in (1, 2):
        ok = sum(
            _anel_da_linha(r, suf, corrigir=False).smiles == r[f"SMILES_Modificado_Anel{suf}"]
            for _, r in df.iterrows()
        )
        linhas.append((f"Anel{suf}: base + auxocromos reproduz SMILES_Modificado", ok, n))
    for col in ("SMILES_Modificado_Anel1", "SMILES_Modificado_Anel2",
                "SMILES_Complexo", "SMILES_Canonico_Complexo", "smiles"):
        ok = sum(Chem.MolFromSmiles(str(s)) is not None for s in df[col])
        linhas.append((f"{col} abre no RDKit", ok, n))
    linhas.append(("SMILES_Canonico_Complexo sem duplicata",
                   n - int(df["SMILES_Canonico_Complexo"].duplicated().sum()), n))
    return pd.DataFrame(linhas, columns=["verificacao", "ok", "total"])


# ---------------------------------------------------------------------- árvores
def arvore_anel(anel: Anel) -> Node:
    raiz = Node(f"ANEL:{anel.base}")
    for pos, frag in sorted(anel.subs):
        raiz.addkid(Node(f"POS:{pos}").addkid(Node(f"FRAG:{frag}")))
    return raiz


def arvore_molecula(mol: Molecula) -> Node:
    return (Node("MOL")
            .addkid(arvore_anel(mol.anel1))
            .addkid(Node(f"PONTE:{mol.ponte}"))
            .addkid(arvore_anel(mol.anel2)))


def _tamanho(no: Node) -> int:
    return 1 + sum(_tamanho(f) for f in no.children)


def ted_normalizada(a: Node, b: Node) -> float:
    return simple_distance(a, b) / max(_tamanho(a), _tamanho(b), 1)


# -------------------------------------------------------------------- operadores
def _trocar_slot(anel: Anel, i: int, novo: tuple) -> Anel | None:
    """Substitui o slot i; recusa se a posição colidir com outro slot."""
    outras = {p for k, (p, _) in enumerate(anel.subs) if k != i}
    if novo[0] in outras:
        return None
    subs = list(anel.subs)
    subs[i] = novo
    return Anel(anel.id_anel, anel.base, tuple(subs))


def aplicar_operador(op: str, p: Molecula, q: Molecula, rng: random.Random) -> Molecula | None:
    lado_p = rng.choice((1, 2))
    anel_p = p.anel1 if lado_p == 1 else p.anel2

    def com_anel(novo: Anel) -> Molecula:
        if lado_p == 1:
            return Molecula(novo, p.ponte, p.id_ponte, p.anel2)
        return Molecula(p.anel1, p.ponte, p.id_ponte, novo)

    if op == "troca_anel":
        return com_anel(rng.choice((q.anel1, q.anel2)))
    if op == "troca_ponte":
        if q.ponte == p.ponte:
            return None
        return Molecula(p.anel1, q.ponte, q.id_ponte, p.anel2)

    anel_q = rng.choice((q.anel1, q.anel2))
    if not anel_p.subs or not anel_q.subs:
        return None
    i = rng.randrange(len(anel_p.subs))
    pos_q, frag_q = rng.choice(anel_q.subs)
    if op == "troca_ramo":
        novo = _trocar_slot(anel_p, i, (pos_q, frag_q))
    elif op == "troca_auxocromo":
        novo = _trocar_slot(anel_p, i, (anel_p.subs[i][0], frag_q))
    else:
        raise ValueError(f"Operador desconhecido: {op}")
    if novo is None or novo == anel_p:
        return None
    return com_anel(novo)


# ----------------------------------------------------------------- montagem
def montar_smiles(mol: Molecula) -> tuple[str | None, str]:
    """Montagem idêntica a ``gerar_complexos``: valida cada anel (como
    ``limpar_fragmentos_gerados``), confere a valência livre nas duas pontas
    e concatena com o anel 2 renumerado. Retorna (smiles, "ok") ou
    (None, motivo)."""
    smi1, smi2 = mol.anel1.smiles, mol.anel2.smiles
    if Chem.MolFromSmiles(smi1) is None or Chem.MolFromSmiles(smi2) is None:
        return None, "anel_invalido"
    if not _tem_valencia_livre_para_ponte(smi1, "fim", mol.ponte):
        return None, "sem_valencia_para_ponte"
    if not _tem_valencia_livre_para_ponte(smi2, "inicio", mol.ponte):
        return None, "sem_valencia_para_ponte"
    return f"{smi1}{mol.ponte}{_substituir_numeracao_anel(smi2)}", "ok"


def _canonico(smiles: str) -> str | None:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def linha_historico(mol: Molecula, smiles_complexo: str) -> dict:
    """Linha no MESMO formato de colunas da base de entrada, para que a
    saída possa alimentar a geração seguinte."""
    linha = {}
    for suf, anel in ((1, mol.anel1), (2, mol.anel2)):
        smi = anel.smiles
        linha[f"ID_Anel_Anel{suf}"] = anel.id_anel
        linha[f"SMILES_Base_Anel{suf}"] = anel.base
        linha[f"SMILES_Modificado_Anel{suf}"] = smi
        for i in range(1, N_SLOTS + 1):
            pos, frag = anel.subs[i - 1] if i <= len(anel.subs) else (None, None)
            linha[f"ID_Frag_{i}_Anel{suf}"] = _ID_AUXOCROMO.get(frag)
            linha[f"Frag_Estrutura_{i}_Anel{suf}"] = frag
            linha[f"Posicao_{i}_Anel{suf}"] = pos
        linha[f"SMILES_Canonico_Fragmento_Anel{suf}"] = _canonico(smi)
    linha["ID_Ponte"] = _ID_PONTE.get(mol.ponte, mol.id_ponte)
    linha["Ponte_Estrutura"] = mol.ponte
    linha["SMILES_Complexo"] = smiles_complexo
    return linha


# ------------------------------------------------------------ propor candidatos
def propor_candidatos(
    sementes: list[Molecula],
    parceiros: list[Molecula],
    n_sondas: int = 10,
    criterio: str = "diverso",
    operadores: tuple = OPERADORES,
    rng: random.Random | None = None,
    progresso: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter]:
    """Gera um candidato por semente via recombinação TED-guiada.

    Para cada semente P, sorteia ``n_sondas`` parceiros Q do histórico,
    ordena-os pela TED normalizada (``diverso``: maior distância primeiro;
    ``similar``: menor distância primeiro) e tenta cada operador em ordem
    aleatória até produzir um filho montável. Uma semente gera no máximo
    um candidato por chamada.

    ``n_sondas`` é o principal parâmetro de qualidade vs. custo: valores
    maiores (50–100) aumentam a diversidade mas escalam linearmente o custo
    da TED (Zhang–Shasha, SIAM J. Comput. 18:1245, 1989).

    Retorna:
        (candidatos montados, tentativas que falharam na montagem, estatísticas).
        Nas falhas, ``SMILES_Complexo`` guarda a concatenação tentada (só inspeção).
        ``progresso`` > 0 imprime o andamento a cada N sementes.
    """
    rng = rng or random.Random(42)
    est = Counter()
    arv_parceiros: dict[int, Node] = {}
    linhas, falhas = [], []

    for k, p in enumerate(sementes):
        arv_p = arvore_molecula(p)
        idx = rng.sample(range(len(parceiros)), min(n_sondas, len(parceiros)))
        sondas = []
        for j in idx:
            if j not in arv_parceiros:
                arv_parceiros[j] = arvore_molecula(parceiros[j])
            sondas.append((ted_normalizada(arv_p, arv_parceiros[j]), j))
        rng.shuffle(sondas)  # empates não favorecem índice baixo
        sondas.sort(key=lambda x: x[0], reverse=(criterio == "diverso"))

        achou = False
        for dist_pq, j in sondas:
            q = parceiros[j]
            if q == p:
                continue
            for op in rng.sample(list(operadores), len(operadores)):
                est["tentativas"] += 1
                filho = aplicar_operador(op, p, q, rng)
                if filho is None or filho in (p, q):
                    est["operador_sem_efeito"] += 1
                    continue
                smi, motivo = montar_smiles(filho)
                tentado = (f"{filho.anel1.smiles}{filho.ponte}"
                           f"{_substituir_numeracao_anel(filho.anel2.smiles)}")
                linha = linha_historico(filho, smi or tentado)
                linha.update({"Operador": op, "TED_pais": dist_pq, "Semente_idx": k,
                              "Parceiro_idx": j})
                if smi is None:
                    est[motivo] += 1
                    falhas.append({**linha, "Etapa": "montagem", "Motivo": motivo})
                    continue
                # TED do filho só para os montados: é o passo mais caro (zss
                # em Python puro) e nas falhas o valor não é usado.
                arv_f = arvore_molecula(filho)
                linha["TED_filho_P"] = ted_normalizada(arv_f, arv_p)
                linha["TED_filho_Q"] = ted_normalizada(arv_f, arv_parceiros[j])
                linhas.append(linha)
                est[f"proposto_{op}"] += 1
                achou = True
                break
            if achou:
                break
        if not achou:
            est["semente_sem_candidato"] += 1
        if progresso and (k + 1) % progresso == 0:
            print(f"    {k + 1}/{len(sementes)} sementes processadas", flush=True)

    return pd.DataFrame(linhas), pd.DataFrame(falhas), est
