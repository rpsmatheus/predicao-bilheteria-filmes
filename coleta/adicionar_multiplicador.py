import numpy as np
import pandas as pd

FICHEIRO_ENTRADA = "dados/dataset_filmes.csv"
FICHEIRO_SAIDA = "dados/dataset_filmes_multiplicador.csv"

# 1. Carregar os dados
df = pd.read_csv(FICHEIRO_ENTRADA)

# 2. Assegurar formato numérico
budget_num = pd.to_numeric(df["budget"], errors="coerce")
revenue_num = pd.to_numeric(df["revenue"], errors="coerce")

# 3. Calcular multiplicador onde houver dados financeiros válidos
condicao = (budget_num >= 10000) & (revenue_num >= 10000)

df["multiplicador"] = np.where(
    condicao, (revenue_num / budget_num).round(4), np.nan
)

# 4. Salvar o arquivo
df.to_csv(FICHEIRO_SAIDA, index=False, encoding="utf-8")

print(f"Arquivo gerado: {FICHEIRO_SAIDA}")
print(f"Total de filmes: {len(df)}")
print(f"Filmes com multiplicador calculado: {condicao.sum()}")