import random
import pandas as pd

def inserir_fragmento(smiles: str, pos: int, frag: str) -> str:
    """
    Insere um fragmento em uma posição específica de uma string SMILES.

    O fragmento é inserido como ramificação, entre parênteses.
    """
    pos = max(0, min(pos, len(smiles)))
    return smiles[:pos] + f"({frag})" + smiles[pos:]


def gerar_smiles_modificado(
    smiles_base: str,
    frags_dict: dict[int, str],
    posicoes_possiveis: list[int],
    n_fragmentos: int,
) -> tuple[str, list[int], list[str], list[str]]:
    """
    Gera uma variante de um SMILES base inserindo fragmentos aleatórios.

    As posições são sorteadas sem repetição.
    Os fragmentos podem se repetir.
    As inserções são feitas da maior posição para a menor para não deslocar índices.
    """
    smiles_mod = smiles_base

    n_fragmentos = min(n_fragmentos, len(posicoes_possiveis))

    posicoes_sorteadas = random.sample(posicoes_possiveis, n_fragmentos)

    chaves_frags_disponiveis = list(frags_dict.keys())
    chaves_sorteadas = random.choices(chaves_frags_disponiveis, k=n_fragmentos)
    fragmentos_sorteados = [frags_dict[chave] for chave in chaves_sorteadas]

    insercoes = list(zip(posicoes_sorteadas, fragmentos_sorteados, chaves_sorteadas))
    insercoes.sort(key=lambda x: x[0], reverse=True)

    posicoes_usadas = []
    frags_usados = []
    chaves_usadas = []

    for pos, frag, chave in insercoes:
        smiles_mod = inserir_fragmento(smiles_mod, pos, frag)

        posicoes_usadas.append(pos)
        frags_usados.append(frag)
        chaves_usadas.append(str(chave))

    posicoes_usadas.reverse()
    frags_usados.reverse()
    chaves_usadas.reverse()

    return smiles_mod, posicoes_usadas, frags_usados, chaves_usadas


def gerar_fragmentos_com_auxocromos(
    aneis: dict[int, str],
    auxocromos: dict[int, str],
    repeticoes: int = 200,
    posicoes_disponiveis: list[int] | None = None,
    quantidade_auxocromos: int = 3,
) -> pd.DataFrame:
    """
    Gera variantes de cada anel com auxocromos inseridos aleatoriamente.

    Retorna um DataFrame com:
    - id do fragmento gerado;
    - ID do anel;
    - SMILES base;
    - auxocromos usados;
    - posições usadas;
    - SMILES modificado.
    """
    if posicoes_disponiveis is None:
        posicoes_disponiveis = [2, 3, 4, 5, 6]

    dados = []
    contador = 1

    for id_anel, smiles_base in aneis.items():
        for _ in range(repeticoes):
            smiles_mod, posicoes_usadas, frags_usados, chaves_usadas = gerar_smiles_modificado(
                smiles_base=smiles_base,
                frags_dict=auxocromos,
                posicoes_possiveis=posicoes_disponiveis,
                n_fragmentos=quantidade_auxocromos,
            )

            linha = {
                "id": contador,
                "ID_Anel": id_anel,
                "SMILES_Base": smiles_base,
                "SMILES_Modificado": smiles_mod,
            }

            for i in range(len(posicoes_usadas)):
                linha[f"ID_Frag_{i + 1}"] = chaves_usadas[i]
                linha[f"Frag_Estrutura_{i + 1}"] = frags_usados[i]
                linha[f"Posicao_{i + 1}"] = posicoes_usadas[i]

            dados.append(linha)
            contador += 1

    return pd.DataFrame(dados)
