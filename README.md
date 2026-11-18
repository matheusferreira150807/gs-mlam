# Saneamento básico e mortalidade infantil nos municípios brasileiros
**Global Solution – CitizenTech · Machine Learning and Advanced Mathematics (MLAM)**

## Integrantes
Repositório: https://github.com/matheusferreira150807/gs-mlam

| Nome completo | RM |
|---|---|
| Gustavo Henrique | 569921 |
| Matheus Caviglia Ferreira | 569638 |

## Problema e objetivo
A mortalidade infantil é um dos indicadores mais sensíveis às condições de vida nas cidades e comunidades, e a falta de
saneamento básico (água tratada, esgoto e coleta de lixo) é um de seus determinantes conhecidos. O objetivo é
**quantificar, com dados reais dos 5.565 municípios brasileiros, a associação entre saneamento básico e mortalidade infantil**,
controlando por renda, educação e urbanização, e construir modelos capazes de prever a mortalidade e identificar municípios
críticos para priorização de investimentos (ODS 3 e ODS 6).

## Base de dados
- **Fonte:** Atlas do Desenvolvimento Humano no Brasil – PNUD / IPEA / FJP, indicadores municipais do Censo 2010 (IBGE) – http://www.atlasbrasil.org.br
- **Arquivo original:** `dados/raw/pnud_muni.rda` – redistribuição pública da tabela municipal do Atlas no pacote abjData
  (https://github.com/abjur/abjData, `data/pnud_muni.rda`).
- **Base analítica:** `dados/atlas_municipios_2010.csv` (5.565 municípios × 17 colunas), gerada de forma reproduzível por `codigos/01_preparacao_dados.py`.
- Dicionário de dados e detalhes da fonte: [`dados/FONTE.md`](dados/FONTE.md).

## Estrutura do repositório
```
├── README.md
├── requirements.txt
├── integrantes.txt                  # nomes, RMs e link do repositório
├── preencher_integrantes.py          # preenche README, entrega_portal.txt e capa do PDF
├── dados/
│   ├── FONTE.md                      # fonte e dicionário de dados
│   ├── atlas_municipios_2010.csv     # base usada nas análises
│   └── raw/pnud_muni.rda             # base original (Atlas Brasil, 1991/2000/2010)
├── codigos/
│   ├── 01_preparacao_dados.py        # limpeza/seleção/transformação -> CSV
│   ├── 02_analise_estatistica.ipynb  # itens 02 a 06, COM as saídas preservadas
│   └── 02_analise_estatistica.py     # mesmo código do notebook em formato script
├── figuras/                          # gráficos gerados pelo notebook
└── relatorio/
    ├── Relatorio_Estatistico_MLAM.pdf  # relatório estatístico (item 07)
    └── resultados.json                 # números gerados pelo notebook
```

| Item do enunciado | Onde está |
|---|---|
| 02 Distribuição normal | notebook, seção 02 · relatório 4.1 |
| 03 Testes de hipóteses | notebook, seção 03 · relatório 4.2 |
| 04 Análise bivariada | notebook, seção 04 · relatório 4.3 |
| 05 Regressão linear múltipla | notebook, seção 05 · relatório 4.4 |
| 06 Regressão logística | notebook, seção 06 · relatório 4.5 |
| 07 Relatório | `relatorio/Relatorio_Estatistico_MLAM.pdf` |

## Como executar
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python codigos/01_preparacao_dados.py                 # (re)gera dados/atlas_municipios_2010.csv
jupyter notebook codigos/02_analise_estatistica.ipynb # ou: python codigos/02_analise_estatistica.py
```
O notebook salva os gráficos em `figuras/` e os resultados em `relatorio/resultados.json`. Sementes aleatórias fixas garantem os mesmos resultados.

## Bibliotecas
Python 3.10+ · pandas · NumPy · SciPy · statsmodels · scikit-learn · matplotlib · seaborn · pyreadr (leitura do .rda) · Jupyter.

## Síntese dos resultados
- **Normal (mortalidade infantil, μ = 19,25, σ = 7,14):** P(X < média) = 0,50 (equiprovável); P(X > mediana 16,9) = 0,6289 (provável).
- **Testes Z:** (a) α = 6%, n = 20, Z = -1,016 > Z crít. -1,555 → não rejeita H0; (b) α = 9%, n = 25, Z = -0,131 < Z crít. 1,341 → não rejeita H0.
- **Bivariada:** saneamento inadequado × mortalidade infantil: r = 0,611 (moderada positiva); ŷ = 16,122 + 0,340x; R² = 0,373; teste F significativo.
- **Regressão múltipla** (5 explicativas): R² = 0,723; F significativo; todas as variáveis significativas. Universalizar água/esgoto e lixo num município pobre típico → mortalidade prevista cai de 25,4 para 21,8 ‰.
- **Regressão logística** (alta mortalidade = acima da mediana): acurácia de 89,7% na base de teste (treino 88,3%), sensibilidade 87,4%, especificidade 92,0%.

## Limitações
- Dados observacionais e transversais: associações, não causalidade.
- Censo 2010 é o dado municipal mais completo do Atlas; a realidade atual pode ser diferente.
- A mortalidade infantil é assimétrica à direita; a normalidade dos itens 02 e 03 é uma suposição do enunciado.
- Análise ecológica (unidade = município); variáveis importantes (pré-natal, atenção básica) não foram incluídas; renda e analfabetismo são correlacionados.
