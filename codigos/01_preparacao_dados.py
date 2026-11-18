"""
01_preparacao_dados.py
----------------------
Gera a base analítica `dados/atlas_municipios_2010.csv` a partir da base original
do Atlas do Desenvolvimento Humano no Brasil (PNUD / IPEA / FJP), Censo 2010.

Fonte original : Atlas do Desenvolvimento Humano no Brasil - http://www.atlasbrasil.org.br/acervo/biblioteca
Arquivo usado  : redistribuição em formato R (.rda) do pacote abjData
                 https://github.com/abjur/abjData  (arquivo data/pnud_muni.rda)

Etapas (todas reproduzíveis):
  1. Baixa o arquivo .rda (se ainda não existir em dados/raw/).
  2. Lê o .rda com pyreadr.
  3. Filtra apenas o ano de 2010 (Censo mais recente disponível no Atlas municipal).
  4. Seleciona as variáveis de interesse e cria a taxa de urbanização (t_urb).
  5. Remove registros com valores ausentes (não há nenhum em 2010) e salva o .csv.

Uso:  python codigos/01_preparacao_dados.py   (executar a partir da raiz do repositório)
"""
from pathlib import Path
import urllib.request

import pandas as pd
import pyreadr

RAIZ = Path(__file__).resolve().parents[1]
RAW = RAIZ / "dados" / "raw" / "pnud_muni.rda"
SAIDA = RAIZ / "dados" / "atlas_municipios_2010.csv"
URL = "https://raw.githubusercontent.com/abjur/abjData/master/data/pnud_muni.rda"

# variável original -> nome amigável (dicionário de dados)
COLUNAS = {
    "codmun7": "cod_ibge",
    "municipio": "municipio",
    "uf": "cod_uf",
    "ufn": "uf",
    "mort1": "mort_infantil",        # óbitos de menores de 1 ano por mil nascidos vivos
    "agua_esgoto": "pct_agua_esgoto_inadeq",  # % pop. em domicílios com água e esgoto inadequados
    "t_lixo": "pct_coleta_lixo",     # % pop. em domicílios urbanos com coleta de lixo
    "t_agua": "pct_agua_encanada",   # % pop. em domicílios com água encanada
    "rdpc": "renda_per_capita",      # renda domiciliar per capita média (R$ ago/2010)
    "t_analf18m": "pct_analf_18mais", # taxa de analfabetismo - 18 anos ou mais (%)
    "e_anosestudo": "anos_estudo_esperados",
    "gini": "gini",
    "pind": "pct_extremamente_pobres",
    "idhm": "idhm",
    "pesotot": "populacao",
    "pesourb": "populacao_urbana",
}


def main() -> None:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        print(f"Baixando {URL} ...")
        urllib.request.urlretrieve(URL, RAW)

    bruto = pyreadr.read_r(str(RAW))["pnud_muni"]
    print("Base original:", bruto.shape, "| anos:", sorted(bruto["ano"].unique()))

    df = bruto.loc[bruto["ano"] == 2010, list(COLUNAS)].rename(columns=COLUNAS)
    df["pct_urbanizacao"] = (100 * df["populacao_urbana"] / df["populacao"]).round(2)
    df["cod_ibge"] = df["cod_ibge"].astype(int)
    df["populacao"] = df["populacao"].astype(int)
    df["populacao_urbana"] = df["populacao_urbana"].astype(int)

    n0 = len(df)
    df = df.dropna()
    print(f"Registros 2010: {n0} | após remover ausentes: {len(df)}")

    df = df.sort_values("cod_ibge").reset_index(drop=True)
    df.to_csv(SAIDA, index=False, encoding="utf-8")
    print("Base analítica salva em:", SAIDA, df.shape)


if __name__ == "__main__":
    main()
