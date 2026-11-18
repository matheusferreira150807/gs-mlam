# Fonte dos dados

- **Base original:** Atlas do Desenvolvimento Humano no Brasil – PNUD / IPEA / Fundação João Pinheiro.
  Indicadores municipais calculados a partir do Censo Demográfico 2010 (IBGE).
  Endereço: http://www.atlasbrasil.org.br (seção Acervo/Biblioteca → base de dados municipal).
- **Arquivo efetivamente baixado:** `raw/pnud_muni.rda`, redistribuição pública da tabela municipal do Atlas
  (anos 1991, 2000 e 2010; 5.565 municípios; 125 indicadores) no pacote abjData:
  https://github.com/abjur/abjData/blob/master/data/pnud_muni.rda
  (download direto: https://raw.githubusercontent.com/abjur/abjData/master/data/pnud_muni.rda)
- **Base analítica:** `atlas_municipios_2010.csv` – gerada por `codigos/01_preparacao_dados.py`
  (filtro do ano 2010, seleção de variáveis, criação de `pct_urbanizacao`, remoção de ausentes – nenhum em 2010).

## Dicionário de dados (`atlas_municipios_2010.csv`)

| Coluna | Variável original no Atlas | Descrição |
|---|---|---|
| cod_ibge | codmun7 | Código IBGE do município (7 dígitos) |
| municipio | municipio | Nome do município |
| cod_uf / uf | uf / ufn | Código e nome da Unidade da Federação |
| mort_infantil | mort1 | Mortalidade até 1 ano de idade (por mil nascidos vivos) |
| pct_agua_esgoto_inadeq | agua_esgoto | % da população em domicílios com abastecimento de água e esgotamento sanitário inadequados |
| pct_coleta_lixo | t_lixo | % da população em domicílios urbanos com serviço de coleta de lixo |
| pct_agua_encanada | t_agua | % da população em domicílios com água encanada |
| renda_per_capita | rdpc | Renda per capita média (R$ de agosto de 2010) |
| pct_analf_18mais | t_analf18m | Taxa de analfabetismo – 18 anos ou mais (%) |
| anos_estudo_esperados | e_anosestudo | Expectativa de anos de estudo |
| gini | gini | Índice de Gini |
| pct_extremamente_pobres | pind | % de extremamente pobres |
| idhm | idhm | Índice de Desenvolvimento Humano Municipal |
| populacao / populacao_urbana | pesotot / pesourb | População total e urbana |
| pct_urbanizacao | (calculada) | 100 × populacao_urbana / populacao |
