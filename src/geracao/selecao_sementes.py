"""Seleção de sementes com pesos aprendidos ao longo das gerações.

Cada molécula do histórico guarda um placar Beta(a, b) do que aconteceu
quando ela foi usada como semente:

* sucesso (a += 1): o filho passou por toda a limpeza e não dispara nenhum
  alerta químico;
* falha (b += 1): a semente não gerou filho montável, o filho foi rejeitado
  em alguma etapa da limpeza ou foi aceito mas com alerta.

O peso de sorteio é

    peso = qualidade × a / (a + b)

onde ``a / (a + b)`` é a taxa de sucesso estimada (começa em 0,5 com
a = b = 1, prior de Laplace) e ``qualidade`` penaliza a própria molécula:
1 se ela é limpa, ``peso_alerta`` se dispara algum alerta ou se o SMILES
aromático dela não reabre (só acontece em moléculas da base).

Herança: o filho aceito começa com uma fração (``heranca``) do placar da
semente que o gerou. Assim uma linhagem que vem dando certo transmite o
"crédito" para os descendentes, e uma que vem falhando, o descrédito.
"""

from __future__ import annotations

import random

import pandas as pd

from geracao.recombinacao_ted import Molecula


class PesosSementes:
    """Gerencia os pesos de sorteio das sementes com aprendizado Beta.

    Args:
        peso_alerta: Fator multiplicador para moléculas com alerta químico
            (padrão 0.1); penaliza sementes com estruturas problemáticas.
        heranca: Fração [0,1] do placar Beta da semente transmitida ao filho
            aceito (padrão 0.5); propaga crédito/descrédito pela linhagem.
    """

    def __init__(self, peso_alerta: float = 0.1, heranca: float = 0.5):
        self.peso_alerta = peso_alerta
        self.heranca = heranca
        self.placar: dict[Molecula, list[float]] = {}
        self.qualidade: dict[Molecula, float] = {}

    # ------------------------------------------------------------- cadastro
    def registrar(self, mol: Molecula, limpa: bool, pai: Molecula | None = None) -> None:
        """Inclui uma molécula no histórico de sementes possíveis."""
        if mol in self.placar:
            return
        a, b = 1.0, 1.0
        if pai is not None and pai in self.placar:
            a_p, b_p = self.placar[pai]
            a += self.heranca * (a_p - 1)
            b += self.heranca * (b_p - 1)
        self.placar[mol] = [a, b]
        self.qualidade[mol] = 1.0 if limpa else self.peso_alerta

    # --------------------------------------------------------------- pesos
    def peso(self, mol: Molecula) -> float:
        a, b = self.placar[mol]
        return self.qualidade[mol] * a / (a + b)

    def sortear(self, candidatas: list[Molecula], k: int, rng: random.Random) -> list[Molecula]:
        """Sorteia ``k`` sementes COM reposição, proporcional ao peso: uma
        molécula boa pode ser semente várias vezes na mesma geração (cada
        vez com um parceiro e operador sorteados de novo)."""
        pesos = [self.peso(m) for m in candidatas]
        return rng.choices(candidatas, weights=pesos, k=k)

    # ---------------------------------------------------------- aprendizado
    def atualizar(self, sementes: list[Molecula], df_aceitas: pd.DataFrame,
                  colunas_alerta: list[str]) -> int:
        """Atualiza o placar de cada semente da geração. ``df_aceitas`` deve
        ter a coluna ``Semente_idx`` (posição em ``sementes``). Retorna o
        número de sucessos."""
        sucesso_idx = set()
        if not df_aceitas.empty:
            limpas = ~df_aceitas[colunas_alerta].any(axis=1)
            sucesso_idx = set(df_aceitas.loc[limpas, "Semente_idx"].astype(int))
        for k, mol in enumerate(sementes):
            self.placar[mol][0 if k in sucesso_idx else 1] += 1
        return len(sucesso_idx)

    def resumo(self) -> dict:
        pesos = pd.Series([self.peso(m) for m in self.placar])
        return {"peso_medio": round(pesos.mean(), 3),
                "peso_max": round(pesos.max(), 3),
                "moleculas_peso_baixo": int((pesos < 0.1).sum())}
