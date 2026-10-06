"""Monta e executa um notebook novo, autocontido, documentando o diagnostico
e o fix do erro 'Explicit valence ... is greater than permitted' na etapa
de montagem de complexos (anel + ponte + anel).

Gera: diagnostico_fix_valencia_ponte.ipynb (na raiz do repo), ja executado,
com todos os numeros de antes/depois reais.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(texto):
    cells.append(nbf.v4.new_markdown_cell(texto))

def code(texto):
    cells.append(nbf.v4.new_code_cell(texto))


md("""\
# Diagnostico e fix: erro de valencia na montagem anel + ponte + anel

Este notebook documenta, passo a passo, a investigacao do erro que aparecia
repetidas vezes ao rodar as geracoes (ex.: `[18:48:44] Explicit valence for
atom # 24 N, 4, is greater than permitted`) e o fix aplicado.

**Onde o fix foi aplicado de verdade (fora deste notebook):**
- `src/geracao/combinacoes.py` -- funcao `gerar_complexos` (pipeline em `src/`).
- `notebook_geracao2_recombinacao_zhang_shasha.ipynb` -- funcao
  `rodar_uma_geracao` (celula do loop de geracoes).

Este notebook aqui e só o registro do diagnostico: reproduz a Geracao 2
(mesma semente) com o codigo ANTES do fix, mede o motivo de cada falha via
excecoes tipadas do RDKit, aplica o fix e mede de novo.""")

md("""\
## 1. Causa raiz

`gerar_complexos` / `rodar_uma_geracao` montavam o SMILES do complexo assim:

```python
smiles_complexo = f"{smi_anel1}{ponte['Ponte_Estrutura']}{smi_anel2}"
```

Ou seja, colavam a ponte direto na string, ligando-a ao **ultimo atomo
escrito** de `smi_anel1` e ao **primeiro atomo escrito** de `smi_anel2` --
sem checar se esse atomo tinha valencia livre pra receber mais uma ligacao.

Isso funciona bem quando esse atomo e um carbono de anel comum (CH2 -> CH,
sobra valencia). Mas varios aneis da base sao heterociclos (`O1CCCCC1`,
`S1CCCC1`, `N1CCCCC1`, ...), e quando o atomo de encaixe e um O/S de anel
(valencia 2, sem H sobrando) ou um N que ja recebeu um auxocromo (3 ligacoes
usadas, sem H sobrando), colar mais uma ligacao estoura a valencia.""")

code("""\
import random
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger
from zss import Node, simple_distance

RDLogger.DisableLog("rdApp.*")  # vamos classificar via excecoes tipadas, nao via stderr
""")

code('''\
# Exemplo concreto (candidato real gerado pela Geracao 2 com o codigo antigo):
# a ponte "N=N" e colada direto no primeiro atomo de smi_anel2, que e um O de anel
# ja com as 2 ligacoes do anel usadas -- nao sobra valencia pra essa 3a ligacao.
exemplo = "C1(N)C(Br)CCC(N(C)C)C1N=NO2CC(O)C(=O)C(N)2"

mol = Chem.MolFromSmiles(exemplo, sanitize=False)
try:
    Chem.SanitizeMol(mol)
except Exception as exc:
    print(type(exc).__name__, "->", exc)
''')

md("## 2. Funcoes auxiliares (reproduzem exatamente `rodar_uma_geracao` do notebook principal)")

code('''\
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
    return sub


def valor_valido(v):
    return pd.notna(v) and str(v).strip() not in {"", "nan", "None"}


def anel_para_arvore(row):
    raiz = Node(f"ANEL:{row['base']}")
    itens = []
    for slot in (1, 2, 3):
        frag, pos = row[f"frag{slot}"], row[f"pos{slot}"]
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
    return d / max(tamanho_arvore(a), tamanho_arvore(b), 1)


def substituir_numeracao_anel(smiles, de="1", para="2"):
    return smiles.replace(de, para)


def diagnosticar_smiles(smiles):
    """Classifica a falha usando as excecoes tipadas do RDKit, em vez de
    so olhar se MolFromSmiles(default) retornou None."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return False, "erro_parse_sintaxe"
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.AtomValenceException:
        return False, "valencia"
    except Chem.rdchem.MolSanitizeException as exc:
        return False, type(exc).__name__
    return True, "ok"
''')

md("""\
## 3. Reproduzir a Geracao 2

Carrega `base_daniel.csv` (Geracao 1, 5.974 moleculas -- a mesma base usada
pelo notebook principal), reconstroi o catalogo de aneis e de pontes, e
reproduz o pareamento por distancia Zhang-Shasha com a mesma semente
(`SEMENTE_BASE=44`, `numero_geracao=2` -> `semente=46`) usada em
`rodar_uma_geracao`.""")

code('''\
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

catalogo_pontes_lista = df_base[["ID_Ponte", "Ponte_Estrutura"]].drop_duplicates().to_dict("records")
base_anel1 = extrair_catalogo_aneis(df_base, 1)

N_SONDAS = 10
SEMENTE = 44 + 2  # SEMENTE_BASE + numero_geracao, igual ao notebook principal

sorteador_pares = random.Random(SEMENTE)
n_sondas_efetivo = min(N_SONDAS, n_catalogo)
pares = []
for pos in range(len(base_anel1)):
    arvore_i = anel_para_arvore(base_anel1.iloc[pos])
    indices = sorteador_pares.sample(range(n_catalogo), n_sondas_efetivo)
    dists = [(idx, ted_normalizada(arvore_i, arvores_catalogo[idx])) for idx in indices]
    melhor_idx, _ = max(dists, key=lambda p: p[1])
    pares.append((pos, melhor_idx))

print(f"pares formados: {len(pares):,}")
''')

md("## 4. ANTES do fix -- ponte colada direto na string")

code('''\
sorteador_pontes = random.Random(SEMENTE * 7 + 1)
candidatos_antes = []
for pos, j in pares:
    row_i = base_anel1.iloc[pos]
    row_j = catalogo_aneis.iloc[j]
    ponte = sorteador_pontes.choice(catalogo_pontes_lista)
    smi_i = row_i["smiles_modificado"]
    smi_j = substituir_numeracao_anel(row_j["smiles_modificado"])
    candidatos_antes.append(f"{smi_i}{ponte['Ponte_Estrutura']}{smi_j}")

contagem_antes = Counter()
for smi in candidatos_antes:
    _, categoria = diagnosticar_smiles(smi)
    contagem_antes[categoria] += 1

print(f"candidatos tentados: {len(candidatos_antes):,}\\n")
for categoria, n in contagem_antes.most_common():
    print(f"  {categoria:25s} {n:5d}  ({100*n/len(candidatos_antes):5.1f}%)")
''')

md("""\
## 5. O fix

Em vez de sortear 1 ponte e colar direto, escolhe -- entre as pontes do
catalogo, embaralhadas -- a primeira cuja ordem de ligacao (simples/dupla/
tripla, lida no simbolo na ponta da string da ponte) cabe na valencia livre
dos dois atomos de encaixe (ultimo atomo de `smi_anel1`, primeiro atomo de
`smi_anel2`). Se nenhuma ponte do catalogo servir para aquele par de aneis
especifico, o par e descartado (contabilizado) em vez de gerar um SMILES
que ia falhar de qualquer forma.

Atencao ao detalhe que gerou uma correcao secundaria durante este mesmo
diagnostico: a concatenacao e `smi_anel1 + ponte + smi_anel2`, entao o
**inicio** da string da ponte encosta no anel1 (`borda="fim"`, do ponto de
vista do anel1) e o **fim** da string da ponte encosta no anel2
(`borda="inicio"`, do ponto de vista do anel2) -- e facil trocar essa ponta
sem querer (foi o que aconteceu na primeira versao deste fix, corrigido
abaixo).""")

code('''\
def ordem_ligacao_borda(ponte_str, borda):
    """Ordem da ligacao (1, 2 ou 3) que a ponte forma no lado indicado.

    smi_anel1 + ponte_str + smi_anel2: o ultimo atomo de smi_anel1 encosta
    no simbolo do INICIO de ponte_str (borda="fim"), e o primeiro atomo de
    smi_anel2 encosta no simbolo do FIM de ponte_str (borda="inicio")."""
    simbolo = ponte_str[-1] if borda == "inicio" else ponte_str[0]
    return {"=": 2, "#": 3}.get(simbolo, 1)


def tem_valencia_livre_para_ponte(smiles, borda, ponte_str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    idx_atomo = 0 if borda == "inicio" else mol.GetNumAtoms() - 1
    ordem_necessaria = ordem_ligacao_borda(ponte_str, borda)
    return mol.GetAtomWithIdx(idx_atomo).GetTotalNumHs() >= ordem_necessaria
''')

md("## 6. DEPOIS do fix -- mesmos pares, ponte escolhida por compatibilidade de valencia")

code('''\
sorteador_pontes = random.Random(SEMENTE * 7 + 1)
candidatos_depois = []
pares_descartados = 0

for pos, j in pares:
    row_i = base_anel1.iloc[pos]
    row_j = catalogo_aneis.iloc[j]
    smi_i = row_i["smiles_modificado"]
    smi_j_bruto = row_j["smiles_modificado"]

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
        pares_descartados += 1
        continue

    smi_j = substituir_numeracao_anel(smi_j_bruto)
    candidatos_depois.append(f"{smi_i}{ponte_escolhida['Ponte_Estrutura']}{smi_j}")

contagem_depois = Counter()
for smi in candidatos_depois:
    _, categoria = diagnosticar_smiles(smi)
    contagem_depois[categoria] += 1

print(f"pares descartados (nenhuma ponte do catalogo cabia): {pares_descartados:,}")
print(f"candidatos tentados: {len(candidatos_depois):,}\\n")
for categoria, n in contagem_depois.most_common():
    print(f"  {categoria:25s} {n:5d}  ({100*n/len(candidatos_depois):5.1f}%)")
''')

md("## 7. Comparacao final")

code('''\
validos_antes = contagem_antes["ok"]
validos_depois = contagem_depois["ok"]

resumo = pd.DataFrame([
    {
        "etapa": "ANTES do fix",
        "candidatos_tentados": len(candidatos_antes),
        "validos": validos_antes,
        "taxa_validos_sobre_tentados": validos_antes / len(candidatos_antes),
    },
    {
        "etapa": "DEPOIS do fix",
        "candidatos_tentados": len(candidatos_depois),
        "validos": validos_depois,
        "taxa_validos_sobre_tentados": validos_depois / len(candidatos_depois),
    },
])
resumo
''')

md("""\
## 8. Leitura dos resultados

- A taxa de validade **entre os candidatos que chegam a ser tentados** sobe
  bastante com o fix (o pre-filtro elimina quase todo erro de valencia
  nessa etapa).
- O numero de **pares descartados antes de tentar** existe porque, para
  alguns pares (anel1, anel2) sorteados pelo pareamento Zhang-Shasha,
  **nenhuma das 20 pontes do catalogo** tem ordem de ligacao compativel com
  a valencia livre dos atomos de encaixe -- isso nao e um bug, e uma
  restricao quimica real (ex.: o atomo de encaixe e um O/S de anel que ja
  usa as duas ligacoes que pode ter). Antes do fix, esses pares eram
  tentados do mesmo jeito e contribuiam pros ~30% de falhas de valencia.
- Efeito colateral a observar rodando o loop completo de geracoes: como
  menos pares "impossiveis" chegam a virar candidato, o numero total de
  `candidatos_gerados` por geracao cai um pouco, mas a fracao que sobrevive
  sobe muito -- o proximo passo natural e comparar o total de moleculas
  novas por geracao (nao so a taxa) rodando o loop completo com o fix.""")

nb["cells"] = cells

import nbclient

client = nbclient.NotebookClient(nb, timeout=600, kernel_name="python3")
client.execute()

nbf.write(nb, "diagnostico_fix_valencia_ponte.ipynb")
print("Notebook gerado e executado: diagnostico_fix_valencia_ponte.ipynb")
