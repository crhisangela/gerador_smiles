"""Armazenamento de uma execução da recombinação, sem nunca sobrescrever.

Cada execução ganha a própria pasta em ``data/execucoes/``::

    data/execucoes/20261008_143000[_rotulo]/
    ├── config.json            parâmetros, entrada, commit, início/fim, status
    ├── execucao.log           tudo o que foi impresso no console
    ├── validacao_base.csv     conferência da base de entrada
    ├── codigo/                cópia dos módulos usados (o código muda entre execuções)
    ├── geracoes/
    │   ├── g0001_aceitas.parquet
    │   ├── g0001_rejeitadas.parquet
    │   ├── g0001_sementes.parquet
    │   └── ...                (um trio por geração, gravado ao fim dela)
    ├── funil.csv              uma linha por geração (regravado a cada geração)
    ├── aceitas.parquet        consolidado, gerado ao finalizar
    ├── rejeitadas.parquet     consolidado, gerado ao finalizar
    └── pesos_finais.parquet   placar Beta de cada molécula (se houve aprendizado)

Os arquivos por geração são escritos uma única vez; se a execução for
interrompida (Ctrl+C, queda de energia), ``consolidar(pasta)`` remonta os
consolidados a partir do que já está em ``geracoes/``.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import pandas as pd

RAIZ_EXECUCOES = "data/execucoes"

# Módulos copiados para ``codigo/`` (caminhos relativos a src/).
MODULOS_CODIGO = (
    "main_recombinacao.py",
    "pipelines/pipeline_recombinacao_ted.py",
    "geracao/recombinacao_ted.py",
    "geracao/selecao_sementes.py",
    "geracao/combinacoes.py",
    "geracao/fragmentos.py",
    "fragmentos/auxocromos.py",
    "fragmentos/pontes.py",
    "fragmentos/aneis.py",
    "utils/aromatizacao.py",
    "utils/validacao.py",
    "utils/filtros_quimicos.py",
    "utils/armazenamento_execucao.py",
)


class _Tee:
    """Repete a saída do console num arquivo de log."""

    def __init__(self, console, arquivo):
        self.console, self.arquivo = console, arquivo

    def write(self, texto):
        self.console.write(texto)
        self.arquivo.write(texto)
        return len(texto)

    def flush(self):
        self.console.flush()
        self.arquivo.flush()

    def __getattr__(self, nome):
        return getattr(self.console, nome)


def _git(*args) -> str | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def _gravar_json(caminho: str, dados: dict) -> None:
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2, default=str)


class RegistroExecucao:
    def __init__(self, config: dict, raiz: str = RAIZ_EXECUCOES, rotulo: str | None = None):
        nome = datetime.now().strftime("%Y%m%d_%H%M%S") + (f"_{rotulo}" if rotulo else "")
        pasta, n = os.path.join(raiz, nome), 2
        while os.path.exists(pasta):  # duas execuções no mesmo segundo
            pasta, n = os.path.join(raiz, f"{nome}_{n}"), n + 1
        self.pasta = pasta
        self.pasta_geracoes = os.path.join(pasta, "geracoes")
        os.makedirs(self.pasta_geracoes)

        self.config = {
            **config,
            "inicio": datetime.now().isoformat(timespec="seconds"),
            "status": "em_andamento",
            "git_commit": _git("rev-parse", "HEAD"),
            "git_alteracoes_locais": bool(_git("status", "--porcelain")),
            "python": sys.version.split()[0],
        }
        _gravar_json(self._caminho("config.json"), self.config)
        self._copiar_codigo()

        self._log = open(self._caminho("execucao.log"), "w", encoding="utf-8")
        self._stdout = sys.stdout
        sys.stdout = _Tee(sys.stdout, self._log)
        print(f"Execução armazenada em: {self.pasta}\n")

    def _caminho(self, *partes) -> str:
        return os.path.join(self.pasta, *partes)

    def _copiar_codigo(self) -> None:
        src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for rel in MODULOS_CODIGO:
            origem = os.path.join(src, rel)
            if os.path.exists(origem):
                destino = self._caminho("codigo", rel)
                os.makedirs(os.path.dirname(destino), exist_ok=True)
                shutil.copy2(origem, destino)

    # ------------------------------------------------------------- gravação
    def salvar_tabela(self, nome: str, df: pd.DataFrame) -> None:
        if nome.endswith(".csv"):
            df.to_csv(self._caminho(nome), index=False)
        else:
            df.to_parquet(self._caminho(nome), index=False)

    def salvar_geracao(self, g: int, aceitas: pd.DataFrame, rejeitadas: pd.DataFrame,
                       sementes: pd.DataFrame, funis: list[dict]) -> None:
        """Grava os arquivos da geração ``g`` (uma única vez) e o funil."""
        for tipo, df in (("aceitas", aceitas), ("rejeitadas", rejeitadas),
                         ("sementes", sementes)):
            df.to_parquet(os.path.join(self.pasta_geracoes, f"g{g:04d}_{tipo}.parquet"),
                          index=False)
        pd.DataFrame(funis).to_csv(self._caminho("funil.csv"), index=False)

    def finalizar(self, status: str, **extra) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Consolida as gerações e registra o status final
        (``concluida``, ``interrompida`` ou ``erro``)."""
        resultado = consolidar(self.pasta, verboso=True)
        self.config.update(status=status, fim=datetime.now().isoformat(timespec="seconds"),
                           **extra)
        _gravar_json(self._caminho("config.json"), self.config)
        print(f"\nStatus: {status}. Tudo em: {self.pasta}")
        return resultado

    def fechar(self) -> None:
        """Devolve o console e fecha o log."""
        if not self._log.closed:
            sys.stdout = self._stdout
            self._log.close()


# ------------------------------------------------------------------ leitura
def _concatenar(arquivos: list[str]) -> pd.DataFrame:
    dfs = [pd.read_parquet(a) for a in arquivos]
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def consolidar(pasta: str, verboso: bool = False,
               gravar: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Junta os arquivos de ``geracoes/`` em ``aceitas.parquet`` e
    ``rejeitadas.parquet``. Serve também para remontar uma execução
    interrompida. ``gravar=False`` só lê (execução ainda em andamento)."""
    saida = []
    for tipo in ("aceitas", "rejeitadas"):
        df = _concatenar(sorted(glob.glob(os.path.join(pasta, "geracoes", f"g*_{tipo}.parquet"))))
        if gravar and not df.empty:
            caminho = os.path.join(pasta, f"{tipo}.parquet")
            df.to_parquet(caminho, index=False)
            if verboso:
                print(f"  → {len(df)} linhas em: {caminho}")
        saida.append(df)
    caminho_funil = os.path.join(pasta, "funil.csv")
    funil = pd.read_csv(caminho_funil) if os.path.exists(caminho_funil) else pd.DataFrame()
    return saida[0], saida[1], funil


def ultima_execucao(raiz: str = RAIZ_EXECUCOES) -> str:
    pastas = sorted(p for p in glob.glob(os.path.join(raiz, "*")) if os.path.isdir(p))
    if not pastas:
        raise FileNotFoundError(f"Nenhuma execução em {raiz}")
    return pastas[-1]


def carregar_execucao(pasta: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(aceitas, rejeitadas, funil) de uma execução; a mais recente se
    ``pasta`` for None. Lê os consolidados ou, se a execução ainda está em
    andamento (ou foi derrubada antes de finalizar), os arquivos por geração."""
    pasta = pasta or ultima_execucao()
    with open(os.path.join(pasta, "config.json"), encoding="utf-8") as f:
        status = json.load(f).get("status")
    if status != "em_andamento" and os.path.exists(os.path.join(pasta, "aceitas.parquet")):
        ac = pd.read_parquet(os.path.join(pasta, "aceitas.parquet"))
        caminho_rej = os.path.join(pasta, "rejeitadas.parquet")
        rej = pd.read_parquet(caminho_rej) if os.path.exists(caminho_rej) else pd.DataFrame()
        funil = pd.read_csv(os.path.join(pasta, "funil.csv"))
        return ac, rej, funil
    return consolidar(pasta, gravar=False)
