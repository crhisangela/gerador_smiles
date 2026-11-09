"""Relatório HTML do que foi e não foi aceito na recombinação v3.

Uso (a partir da raiz do projeto, depois de rodar main_recombinacao.py):
    python src/relatorio_recombinacao.py                       # execução mais recente
    python src/relatorio_recombinacao.py data/execucoes/<pasta>

O HTML é salvo dentro da pasta da execução como ``relatorio.html``.
Moléculas rejeitadas são agrupadas por etapa + motivo com exemplos sorteados.
"""

import html
import os
import random
import sys

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

from utils.armazenamento_execucao import carregar_execucao, ultima_execucao

RDLogger.DisableLog("rdApp.*")

EXEMPLOS_POR_GRUPO = 24

EXPLICACAO = {
    "anel_invalido": "Depois da troca de ramo/auxocromo, o anel modificado não abre no RDKit "
                     "(ex.: dois substituintes com ligação dupla no mesmo átomo).",
    "sem_valencia_para_ponte": "O último átomo do anel 1 ou o primeiro do anel 2 não tem "
                               "hidrogênios livres para a ordem de ligação que a ponte exige. "
                               "O desenho mostra a concatenação que teria sido feita.",
    "complexo_invalido": "A molécula montada (anel + ponte + anel) não abre no RDKit.",
    "complexo_duplicado_na_geracao": "Outra semente já gerou a mesma molécula nesta geração.",
    "complexo_ja_existe": "A molécula montada já existe na base ou numa geração anterior.",
    "aromatizacao_falhou": "A aromatização dos anéis de 6 membros de C/N quebrou a sanitização.",
    "aromatico_nao_reabre": "A aromatização gerou um SMILES que o RDKit não consegue reler "
                            "(kekulização impossível). O desenho é o complexo antes da aromatização.",
    "final_ja_existe": "Depois da aromatização, coincide com uma molécula já conhecida.",
    "final_duplicado_na_geracao": "Depois da aromatização, coincide com outra aceita nesta geração.",
}


def _svg(smiles: str, tamanho=(260, 190)) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:  # inválidas: desenha sem sanitizar para mostrar o problema
        mol = Chem.MolFromSmiles(str(smiles), sanitize=False)
        if mol is None:
            return "<div class='sem-desenho'>não desenhável</div>"
        mol.UpdatePropertyCache(strict=False)
    try:
        mol = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=False)
        d = rdMolDraw2D.MolDraw2DSVG(*tamanho)
        d.drawOptions().clearBackground = False
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    except Exception:
        return "<div class='sem-desenho'>não desenhável</div>"


def _smiles_para_desenho(r) -> str:
    if r.get("Motivo") == "aromatico_nao_reabre":
        return r["SMILES_Complexo"]
    s = r.get("smiles")
    return s if isinstance(s, str) else r["SMILES_Complexo"]


def _cartao(r) -> str:
    alertas = [c.removeprefix("alerta_") for c in r.index
               if c.startswith("alerta_") and r[c] == True]  # noqa: E712 (NaN/np.bool_)
    etiquetas = "".join(f"<span class='alerta'>{html.escape(a)}</span>" for a in alertas)
    linhas_smiles = f"<code>{html.escape(str(r['SMILES_Complexo']))}</code>"
    if isinstance(r.get("SMILES_Aromatico"), str):
        linhas_smiles += f"<code class='arom'>arom.: {html.escape(r['SMILES_Aromatico'])}</code>"
    return f"""
    <div class="cartao">
      {_svg(_smiles_para_desenho(r))}
      <div class="meta">G{int(r['Geracao'])} · {html.escape(str(r['Operador']))}
        · ponte <b>{html.escape(str(r['Ponte_Estrutura']))}</b>
        · TED pais {r['TED_pais']:.2f}</div>
      {etiquetas}
      {linhas_smiles}
    </div>"""


def _secao(titulo: str, df: pd.DataFrame, explicacao: str, aberta: bool, rng) -> str:
    idx = rng.sample(range(len(df)), min(EXEMPLOS_POR_GRUPO, len(df)))
    cartoes = "".join(_cartao(df.iloc[i]) for i in idx)
    return f"""
    <details {'open' if aberta else ''}>
      <summary><span class="titulo">{html.escape(titulo)}</span>
        <span class="n">{len(df):,}</span></summary>
      <p class="explica">{html.escape(explicacao)}
        Mostrando {len(idx)} exemplos sorteados.</p>
      <div class="grade">{cartoes}</div>
    </details>"""


def _tabela_funil(ac: pd.DataFrame, rej: pd.DataFrame) -> str:
    """Uma linha por geração (cabe qualquer número de gerações); colunas =
    motivo de rejeição + aceitas."""
    t = rej.groupby(["Geracao", "Motivo"]).size().unstack("Motivo", fill_value=0)
    t["aceitas"] = ac.groupby("Geracao").size()
    alertas = [c for c in ac.columns if c.startswith("alerta_")]
    t["aceitas sem alerta"] = (~ac[alertas].any(axis=1)).groupby(ac["Geracao"]).sum()
    t = t.fillna(0).astype(int)
    t.loc["total"] = t.sum()
    cab = "".join(f"<th class='{'ok' if m.startswith('aceitas') else ''}'>{html.escape(m)}</th>"
                  for m in t.columns)
    corpo = "".join(
        f"<tr><td>{'G' + str(g) if g != 'total' else '<b>total</b>'}</td>"
        + "".join(f"<td class='{'ok' if m.startswith('aceitas') else ''}'>{v:,}</td>"
                  for m, v in linha.items()) + "</tr>"
        for g, linha in t.iterrows()
    )
    return (f"<div class='rolagem'><table><tr><th>Geração</th>{cab}</tr>{corpo}"
            f"</table></div>")


def gerar_relatorio(pasta: str | None = None, semente=7) -> str:
    """Gera ``relatorio.html`` na pasta da execução (a mais recente se
    ``pasta`` for None)."""
    pasta = pasta or ultima_execucao()
    caminho_html = os.path.join(pasta, "relatorio.html")
    rng = random.Random(semente)
    ac, rej, _ = carregar_execucao(pasta)

    secoes = [_secao("Aceitas", ac, "Passaram por toda a limpeza. Etiquetas laranja são "
                     "alertas químicos (não removem a molécula).", True, rng)]
    for (etapa, motivo), grupo in rej.groupby(["Etapa", "Motivo"], sort=False):
        secoes.append(_secao(f"Rejeitadas · {etapa} · {motivo}", grupo,
                             EXPLICACAO.get(motivo, f"Alerta químico estrito: {motivo}."),
                             False, rng))

    pagina = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recombinação v3 — aceitas e rejeitadas</title>
<style>
:root {{ --fundo:#fafaf9; --cartao:#fff; --texto:#1c1917; --suave:#78716c; --borda:#e7e5e4;
        --ok:#dcfce7; --alerta:#fed7aa; --alerta-txt:#9a3412; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --fundo:#1c1917; --cartao:#292524; --texto:#f5f5f4; --suave:#a8a29e; --borda:#44403c;
          --ok:#14532d; --alerta:#7c2d12; --alerta-txt:#fed7aa; }}
  svg {{ filter: invert(1) hue-rotate(180deg); }} }}
body {{ font-family: system-ui, sans-serif; background:var(--fundo); color:var(--texto);
       margin:0 auto; padding:24px 16px; max-width:1300px; }}
h1 {{ font-size:1.4rem; margin:0 0 4px; }} .sub {{ color:var(--suave); margin:0 0 20px; }}
table {{ border-collapse:collapse; margin-bottom:24px; font-size:.85rem; }}
td, th {{ border:1px solid var(--borda); padding:4px 10px; text-align:right; }}
td:first-child, th:first-child {{ text-align:left; }}
td.ok, th.ok {{ background:var(--ok); font-weight:600; }}
.rolagem {{ overflow-x:auto; max-height:60vh; overflow-y:auto; margin-bottom:24px; }}
.rolagem table {{ margin-bottom:0; }}
details {{ border:1px solid var(--borda); border-radius:8px; margin-bottom:10px;
          background:var(--cartao); }}
summary {{ cursor:pointer; padding:10px 14px; display:flex; justify-content:space-between; }}
.titulo {{ font-weight:600; }} .n {{ color:var(--suave); font-variant-numeric:tabular-nums; }}
.explica {{ margin:0 14px 10px; color:var(--suave); font-size:.9rem; }}
.grade {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(270px, 1fr));
         gap:10px; padding:0 14px 14px; }}
.cartao {{ border:1px solid var(--borda); border-radius:6px; padding:6px; font-size:.75rem;
          overflow:hidden; }}
.cartao svg {{ width:100%; height:auto; }}
.meta {{ color:var(--suave); margin:2px 0 4px; }}
code {{ display:block; word-break:break-all; font-size:.7rem; margin-top:3px; }}
code.arom {{ color:var(--suave); }}
.alerta {{ display:inline-block; background:var(--alerta); color:var(--alerta-txt);
          border-radius:4px; padding:0 5px; margin:0 3px 3px 0; font-size:.7rem; }}
.sem-desenho {{ height:120px; display:grid; place-items:center; color:var(--suave); }}
</style></head><body>
<h1>Recombinação v3 (Zhang–Shasha) — o que foi e não foi aceito</h1>
<p class="sub">{len(ac):,} aceitas · {len(rej):,} rejeitadas · gerações 1–{int(ac['Geracao'].max())}</p>
{_tabela_funil(ac, rej)}
{''.join(secoes)}
</body></html>"""

    with open(caminho_html, "w", encoding="utf-8") as f:
        f.write(pagina)
    print(f"Relatório salvo em: {caminho_html}")
    return caminho_html


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")  # console do Windows (cp1252)
    gerar_relatorio(sys.argv[1] if len(sys.argv) > 1 else None)
