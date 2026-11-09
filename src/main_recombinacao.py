"""Ponto de entrada da recombinação v3 (Zhang–Shasha) sobre a base montada.

Uso (a partir da raiz do projeto):
    python src/main_recombinacao.py
    python src/main_recombinacao.py --rotulo teste --geracoes 5 --sementes 500 --sondas 50

Cada execução grava tudo em uma pasta nova e nunca sobrescreve outra:
    data/execucoes/<AAAAMMDD_HHMMSS>[_rotulo]/
        config.json, execucao.log, validacao_base.csv, codigo/,
        geracoes/gNNNN_{aceitas,rejeitadas,sementes}.parquet, funil.csv,
        aceitas.parquet, rejeitadas.parquet, pesos_finais.parquet

Ctrl+C interrompe com segurança: gerações concluídas ficam salvas.
Para remontar consolidados de uma execução derrubada sem Ctrl+C:
    python -c "import sys; sys.path.insert(0, 'src'); \
from utils.armazenamento_execucao import consolidar; consolidar('data/execucoes/<pasta>')"

Escala para supercomputador: use --sondas 50 (ou mais) e --sementes N.
"""

import argparse
import sys

from pipelines.pipeline_recombinacao_ted import executar_recombinacao_ted

# Defaults; argumentos de linha de comando têm prioridade quando informados.
CONFIG = dict(
    caminho_entrada="data/banco_smiles_aromatizados.parquet",
    pasta_execucoes="data/execucoes",
    rotulo=None,                # sufixo do nome da pasta, ex.: "aprendizado"
    n_geracoes=None,            # None = até uma geração não produzir nada novo
    sementes_por_geracao=None,  # None = base inteira na G1, depois nº de novas
    n_sondas=10,                # parceiros sorteados por semente
    criterio="diverso",         # "diverso" (maior TED primeiro) ou "similar"
    filtros_estritos=(),        # ex.: ("alerta_S_hipervalente", "alerta_heteroatomo_halogenado")
    aprender=True,              # pesos aprendidos na escolha das sementes
    peso_alerta=0.1,
    heranca=0.5,
    semente=42,                 # semente do gerador aleatório (reprodutibilidade)
)


def _argumentos() -> dict:
    ap = argparse.ArgumentParser(description="Recombinação v3 (Zhang–Shasha)")
    ap.add_argument("--rotulo")
    ap.add_argument("--geracoes", type=int, dest="n_geracoes")
    ap.add_argument("--sementes", type=int, dest="sementes_por_geracao")
    ap.add_argument("--sondas", type=int, dest="n_sondas")
    ap.add_argument("--criterio", choices=("diverso", "similar"))
    ap.add_argument("--semente", type=int)
    ap.add_argument("--sem-aprendizado", action="store_false", dest="aprender", default=None)
    ap.add_argument("--peso-alerta", type=float, dest="peso_alerta")
    ap.add_argument("--heranca", type=float)
    ap.add_argument("--filtros-estritos", nargs="+", dest="filtros_estritos")
    return {k: (tuple(v) if k == "filtros_estritos" else v)
            for k, v in vars(ap.parse_args()).items() if v is not None}


if __name__ == "__main__":
    # O console do Windows usa cp1252 por padrão e não imprime "→" nem "–".
    sys.stdout.reconfigure(encoding="utf-8")
    executar_recombinacao_ted(**{**CONFIG, **_argumentos()})
