import pandas as pd

ARQUIVO_DATASET = "dados/dataset_filmes.csv"

# 1. Carregar os dados
try:
  df = pd.read_csv(ARQUIVO_DATASET)
except FileNotFoundError:
  print(f"Erro: Arquivo '{ARQUIVO_DATASET}' não encontrado no diretório.")
  exit()

# 2. Filtrar apenas registros válidos com budget e revenue
df_valido = df[(df["budget"] >= 10000) & (df["revenue"] >= 10000)].copy()

# 3. Calcular a métrica de retorno e as 3 classes do Vogel
df_valido["multiplicador"] = df_valido["revenue"] / df_valido["budget"]


def classificar_sucesso(mult):
  if mult < 2.0:
    return "Sem Sucesso"
  elif mult < 4.0:
    return "Sucesso Mediano"
  else:
    return "Muito Sucesso"


df_valido["classe_sucesso"] = df_valido["multiplicador"].apply(
    classificar_sucesso
)

# 4. Definição do Gabarito Canônico para validação
# Dicionário com: (termo_busca, ano_aproximado): classe_esperada
gabarito = {
    ("Avatar", 2009): "Muito Sucesso",
    ("Get Out", 2017): "Muito Sucesso",
    ("The Avengers", 2012): "Muito Sucesso",
    ("Edge of Tomorrow", 2014): "Sucesso Mediano",
    ("Dune", 2021): "Sucesso Mediano",
    ("The Bourne Legacy", 2012): "Sucesso Mediano",
    ("John Carter", 2012): "Sem Sucesso",
    ("The Flash", 2023): "Sem Sucesso",
    ("Battleship", 2012): "Sem Sucesso",
}

print("=" * 80)
print("TESTE DE CONFORMIDADE DO GABARITO (Ground Truth Validation)")
print("Critério: < 2.0x (Sem Sucesso) | 2.0x a 4.0x (Mediano) | >= 4.0x (Muito Sucesso)")
print("=" * 80)

resultados = []
acertos = 0
total_testados = 0

for (titulo_busca, ano), classe_esperada in gabarito.items():
  # Busca pelo título aproximado e ano para não confundir com remakes
  filtro = (
      df_valido["title"].str.contains(titulo_busca, case=False, na=False)
      & df_valido["release_date"].str.startswith(str(ano), na=False)
  )
  correspondencias = df_valido[filtro]

  if correspondencias.empty:
    # Se não achar por ano estrito, tenta apenas pelo título
    correspondencias = df_valido[
        df_valido["title"].str.contains(titulo_busca, case=False, na=False)
    ]

  if not correspondencias.empty:
    filme = correspondencias.iloc[0]
    total_testados += 1
    titulo_real = filme["title"]
    orcamento = filme["budget"]
    receita = filme["revenue"]
    mult = filme["multiplicador"]
    classe_obtida = filme["classe_sucesso"]

    passou = classe_obtida == classe_esperada
    if passou:
      acertos += 1
      status = "✓ APROVADO"
    else:
      status = "✗ DIVERGENTE"

    resultados.append({
        "Status": status,
        "Filme": f"{titulo_real} ({str(filme['release_date'])[:4]})",
        "Orçamento": f"${orcamento:,.0f}",
        "Bilheteria": f"${receita:,.0f}",
        "Multiplicador": f"{mult:.2f}x",
        "Esperado": classe_esperada,
        "Obtido": classe_obtida,
    })
  else:
    resultados.append({
        "Status": "⚠ NÃO ENCONTRADO",
        "Filme": f"{titulo_busca} ({ano})",
        "Orçamento": "-",
        "Bilheteria": "-",
        "Multiplicador": "-",
        "Esperado": classe_esperada,
        "Obtido": "Fora da base",
    })

# Formatação tabular dos resultados
tabela_df = pd.DataFrame(resultados)
print(
    tabela_df[[
        "Status",
        "Filme",
        "Orçamento",
        "Bilheteria",
        "Multiplicador",
        "Esperado",
        "Obtido",
    ]].to_string(index=False)
)

print("-" * 80)
if total_testados > 0:
  taxa_conformidade = (acertos / total_testados) * 100
  print(
      f"Taxa de Conformidade do Gabarito: {taxa_conformidade:.1f}% ({acertos}/{total_testados} filmes validados)"
  )
else:
  print(
      "Nenhum filme do gabarito foi localizado. Verifique se o dataset_filmes.csv já contém os filmes coletados."
  )
print("=" * 80)