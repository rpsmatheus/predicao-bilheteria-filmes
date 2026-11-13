"""
Análise de atributos significativos para o multiplicador (revenue / budget).

Como rodar:
    pip install pandas numpy scipy scikit-learn matplotlib
    python analise_completa.py

O CSV deve estar na mesma pasta. Leva alguns minutos (a seção 7 é a mais lenta).
Saídas:
    resultados/*.csv   tabelas para o relatório
    figuras/*.png      gráficos
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.decomposition import FactorAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==============================================================================
# PARÂMETROS (decisões metodológicas: declare todas no relatório)
# ==============================================================================
ARQUIVO = "dados/dataset_filmes_multiplicador.csv"
ORCAMENTO_MINIMO = 100_000  # abaixo disso o budget do TMDB costuma ser ruído
MESES_EM_CARTAZ = 6  # exclui lançamentos recentes (bilheteria incompleta)
ANO_CORTE_TEMPORAL = 2019  # treina até esse ano, testa nos seguintes
SEMENTE = 42

os.makedirs("resultados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)


def titulo(texto):
  print("\n" + "=" * 80 + f"\n{texto}\n" + "=" * 80)


# ==============================================================================
# 1. CARGA E LIMPEZA
# ==============================================================================
titulo("1. CARGA E LIMPEZA")
df = pd.read_csv(ARQUIVO)
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
n0 = len(df)

df = df.dropna(subset=["multiplicador", "release_date"])
print(f"Com multiplicador calculado: {len(df)} de {n0}")

data_limite = df["release_date"].max() - pd.DateOffset(months=MESES_EM_CARTAZ)
df = df[df["release_date"] <= data_limite]
print(f"Após excluir lançamentos depois de {data_limite.date()}: {len(df)}")

df = df[df["budget"] >= ORCAMENTO_MINIMO]
print(f"Após excluir orçamento < {ORCAMENTO_MINIMO:,}: {len(df)}")

df = df.sort_values("release_date").reset_index(drop=True)

# Alvos: log10 para a regressão (cauda longa), 3 classes de Vogel para a
# classificação
df["log_mult"] = np.log10(df["multiplicador"])
df["classe"] = pd.cut(
    df["multiplicador"],
    bins=[-np.inf, 2, 4, np.inf],
    labels=["Fracasso", "Mediano", "Sucesso"],
    right=False,
).astype(str)
print("\nDistribuição das classes:")
print(df["classe"].value_counts(normalize=True).round(3).to_string())

# ==============================================================================
# 2. ENGENHARIA DE ATRIBUTOS
# ==============================================================================
df["log_budget"] = np.log10(df["budget"])
df["ano"] = df["release_date"].dt.year
mes = df["release_date"].dt.month
df["is_summer"] = mes.isin([5, 6, 7]).astype(int)
df["is_holiday"] = mes.isin([11, 12]).astype(int)
df["eh_franquia"] = df["collection"].notna().astype(int)
df["idioma_ingles"] = (df["original_language"] == "en").astype(int)
df["rt_score"] = pd.to_numeric(
    df["rotten_tomatoes"].astype(str).str.rstrip("%"), errors="coerce"
)
df["log_tmdb_votes"] = np.log10(df["tmdb_vote_count"] + 1)
df["log_imdb_votes"] = np.log10(df["imdb_votes"] + 1)
df["log_popularity"] = np.log10(df["tmdb_popularity"] + 1)

# Gêneros: uma coluna binária por gênero (um filme pode ter vários)
generos = df["genres"].fillna("").str.get_dummies("|")
generos = generos.loc[:, generos.sum() >= 50].add_prefix("gen_")
df = pd.concat([df, generos], axis=1)

# Classificação etária
cols_mpaa = []
for r in ["G", "PG", "PG-13", "R"]:
  df[f"mpaa_{r}"] = (df["mpaa_rating"] == r).astype(int)
  cols_mpaa.append(f"mpaa_{r}")


def historico(df, coluna, prefixo, top=None):
  """Desempenho ANTERIOR das pessoas/empresas de um filme.

  Para cada filme, usa apenas filmes lançados antes dele (sem vazamento):
    - {prefixo}_hist_mult: média do log-multiplicador dos trabalhos anteriores
    - {prefixo}_n_previos: quantidade de trabalhos anteriores na base
  Quem não tem histórico fica com NaN na média (imputado depois).
  """
  nomes = df[coluna].fillna("").str.split("|")
  if top:
    nomes = nomes.apply(lambda x: x[:top])
  longo = pd.DataFrame({
      "filme": df.index.repeat(nomes.str.len()),
      "nome": np.concatenate(nomes.values),
  })
  longo = longo[longo["nome"] != ""]
  longo["data"] = df["release_date"].values[longo["filme"]]
  longo["y"] = df["log_mult"].values[longo["filme"]]
  longo = longo.sort_values("data")
  g = longo.groupby("nome")["y"]
  longo["n_prev"] = g.cumcount()
  longo["media_prev"] = (g.cumsum() - longo["y"]) / longo["n_prev"].replace(0, np.nan)
  agg = longo.groupby("filme").agg(
      media=("media_prev", "mean"), n=("n_prev", "mean")
  )
  df[f"{prefixo}_hist_mult"] = agg["media"].reindex(df.index)
  df[f"{prefixo}_n_previos"] = agg["n"].reindex(df.index).fillna(0)
  return [f"{prefixo}_hist_mult", f"{prefixo}_n_previos"]


cols_diretor = historico(df, "director", "diretor")
cols_elenco = historico(df, "actors", "elenco", top=3)
cols_estudio = historico(df, "production_companies", "estudio", top=2)

# ==============================================================================
# 3. GRUPOS CONCEITUAIS DE ATRIBUTOS
# ==============================================================================
GRUPOS = {
    "Produção": ["log_budget", "runtime", "eh_franquia", "idioma_ingles"]
    + cols_mpaa,
    "Gênero": list(generos.columns),
    "Equipe (histórico)": cols_diretor + cols_elenco + cols_estudio,
    "Calendário": ["ano", "is_summer", "is_holiday"],
    "Crítica": ["metascore", "rt_score"],
    "Nota do público": ["imdb_rating", "tmdb_vote_average"],
    "Engajamento": ["log_tmdb_votes", "log_imdb_votes", "log_popularity"],
}
PRE = ["Produção", "Gênero", "Equipe (histórico)", "Calendário"]
feats_pre = sum((GRUPOS[g] for g in PRE), [])
feats_todos = sum(GRUPOS.values(), [])
CENARIOS = {"Pré-lançamento": feats_pre, "Todos os atributos": feats_todos}

X = df[feats_todos]
y_reg = df["log_mult"]
y_cls = df["classe"]


def rf_reg():
  return make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestRegressor(
          n_estimators=300, min_samples_leaf=3, random_state=SEMENTE, n_jobs=-1
      ),
  )


def rf_cls():
  return make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestClassifier(
          n_estimators=300, min_samples_leaf=2, random_state=SEMENTE, n_jobs=-1
      ),
  )


kf = KFold(5, shuffle=True, random_state=SEMENTE)
skf = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)

# ==============================================================================
# 4. AGRUPAMENTO DE ATRIBUTOS POR CORRELAÇÃO (dendrograma + análise fatorial)
# ==============================================================================
titulo("4. AGRUPAMENTO DE ATRIBUTOS")
# Só atributos contínuos/ordinais: binários raros distorcem a correlação
cols_cluster = [
    c for c in feats_todos
    if not c.startswith(("gen_", "mpaa_")) and c != "idioma_ingles"
]
Xc = df[cols_cluster].fillna(df[cols_cluster].median())
corr = Xc.corr(method="spearman").abs()
dist = squareform(1 - corr.values, checks=False)
Z = hierarchy.linkage(dist, method="average")

fig, ax = plt.subplots(figsize=(10, 6))
hierarchy.dendrogram(Z, labels=cols_cluster, orientation="right", ax=ax)
ax.set_xlabel("Distância (1 − |correlação de Spearman|)")
ax.set_title("Agrupamento hierárquico dos atributos")
plt.tight_layout()
plt.savefig("figuras/dendrograma_atributos.png", dpi=150)
plt.close()

# Corte em distância 0.6 (= atributos com |correlação| média acima de 0.4)
rotulos = hierarchy.fcluster(Z, t=0.6, criterion="distance")
for c in sorted(set(rotulos)):
  membros = [f for f, r in zip(cols_cluster, rotulos) if r == c]
  print(f"Cluster {c}: {', '.join(membros)}")

# Análise fatorial: permite que um atributo pertença a mais de um grupo
N_FATORES = 5
fa = FactorAnalysis(n_components=N_FATORES, rotation="varimax", random_state=SEMENTE)
fa.fit(StandardScaler().fit_transform(Xc))
cargas = pd.DataFrame(
    fa.components_.T,
    index=cols_cluster,
    columns=[f"Fator {i + 1}" for i in range(N_FATORES)],
).round(2)
cargas.to_csv("resultados/cargas_fatoriais.csv")
print("\nCargas fatoriais (|carga| >= 0.4 indica pertencimento ao fator):")
print(cargas.to_string())
multi = cargas.index[(cargas.abs() >= 0.4).sum(axis=1) > 1].tolist()
print(f"\nAtributos em mais de um fator: {multi or 'nenhum'}")

# ==============================================================================
# 5. REGRESSÃO: comparação de modelos nos dois cenários
# ==============================================================================
titulo("5. REGRESSÃO (alvo: log10 do multiplicador) — R² em validação cruzada")


def linear(modelo):
  return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), modelo)


MODELOS = {
    "Ridge": linear(Ridge(alpha=1.0)),
    "Huber": linear(HuberRegressor(max_iter=1000)),
    "KNN": linear(KNeighborsRegressor(n_neighbors=15)),
    "Random Forest": rf_reg(),
}
linhas = []
for nome_m, modelo in MODELOS.items():
  linha = {"Modelo": nome_m}
  for nome_c, cols in CENARIOS.items():
    r2 = cross_val_score(modelo, X[cols], y_reg, cv=kf, scoring="r2")
    linha[nome_c] = f"{r2.mean():.3f} ± {r2.std():.3f}"
  linhas.append(linha)
tab_modelos = pd.DataFrame(linhas)
tab_modelos.to_csv("resultados/comparacao_modelos.csv", index=False)
print(tab_modelos.to_string(index=False))

# ==============================================================================
# 6. IMPORTÂNCIA POR PERMUTAÇÃO (com coluna de ruído como referência)
# ==============================================================================
titulo("6. IMPORTÂNCIA POR PERMUTAÇÃO (queda de R² no conjunto de teste)")
Xr = X.copy()
Xr["RUIDO_ALEATORIO"] = np.random.RandomState(SEMENTE).rand(len(Xr))

fig, eixos = plt.subplots(1, 2, figsize=(14, 7))
for ax, (nome_c, cols) in zip(eixos, CENARIOS.items()):
  cols_r = cols + ["RUIDO_ALEATORIO"]
  Xtr, Xte, ytr, yte = train_test_split(
      Xr[cols_r], y_reg, test_size=0.25, random_state=SEMENTE
  )
  m = rf_reg().fit(Xtr, ytr)
  pi = permutation_importance(
      m, Xte, yte, n_repeats=30, random_state=SEMENTE, n_jobs=-1
  )
  tab = pd.DataFrame({
      "Atributo": cols_r,
      "Permutação (média)": pi.importances_mean.round(4),
      "Desvio": pi.importances_std.round(4),
      "RF nativa (%)": (m[-1].feature_importances_ * 100).round(2),
  }).sort_values("Permutação (média)", ascending=False)
  arq = nome_c.lower().replace(" ", "_").replace("é", "e").replace("ç", "c")
  tab.to_csv(f"resultados/importancia_{arq}.csv", index=False)
  print(f"\n--- {nome_c} (R² no teste = {m.score(Xte, yte):.3f}) ---")
  print(tab.head(15).to_string(index=False))

  top = tab.head(15).iloc[::-1]
  cores = ["gray" if a == "RUIDO_ALEATORIO" else "C0" for a in top["Atributo"]]
  ax.barh(top["Atributo"], top["Permutação (média)"], xerr=top["Desvio"], color=cores)
  ax.set_title(nome_c)
  ax.set_xlabel("Queda de R² ao permutar o atributo")
plt.tight_layout()
plt.savefig("figuras/importancia_permutacao.png", dpi=150)
plt.close()
print("\nAtributos abaixo de RUIDO_ALEATORIO não se distinguem do acaso.")

# ==============================================================================
# 7. IMPORTÂNCIA POR GRUPO (ablação)
# ==============================================================================
titulo("7. IMPORTÂNCIA POR GRUPO DE ATRIBUTOS (R² em validação cruzada)")


def r2_cv(cols):
  # Floresta menor aqui: são dezenas de treinos e o resultado muda pouco
  modelo = make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestRegressor(
          n_estimators=100, min_samples_leaf=3, random_state=SEMENTE, n_jobs=-1
      ),
  )
  return cross_val_score(modelo, X[cols], y_reg, cv=kf, scoring="r2").mean()


for nome_c, grupos_c in [("Pré-lançamento", PRE), ("Todos os atributos", list(GRUPOS))]:
  cols_c = sum((GRUPOS[g] for g in grupos_c), [])
  completo = r2_cv(cols_c)
  linhas = []
  for g in grupos_c:
    sem = [c for c in cols_c if c not in GRUPOS[g]]
    r2_sem = r2_cv(sem)
    linhas.append({
        "Grupo": g,
        "Nº atributos": len(GRUPOS[g]),
        "R² só o grupo": round(r2_cv(GRUPOS[g]), 3),
        "R² sem o grupo": round(r2_sem, 3),
        "Perda ao remover": round(completo - r2_sem, 3),
    })
  tab = pd.DataFrame(linhas).sort_values("Perda ao remover", ascending=False)
  arq = nome_c.lower().replace(" ", "_").replace("é", "e").replace("ç", "c")
  tab.to_csv(f"resultados/grupos_{arq}.csv", index=False)
  print(f"\n--- {nome_c} (R² completo = {completo:.3f}) ---")
  print(tab.to_string(index=False))

# ==============================================================================
# 8. CLASSIFICAÇÃO EM 3 CLASSES (Vogel)
# ==============================================================================
titulo("8. CLASSIFICAÇÃO (Fracasso < 2x <= Mediano < 4x <= Sucesso)")
ordem = ["Fracasso", "Mediano", "Sucesso"]
treino = df["ano"] <= ANO_CORTE_TEMPORAL
teste = ~treino

linhas = []
fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
for ax, (nome_c, cols) in zip(eixos, CENARIOS.items()):
  # (a) validação cruzada estratificada
  pred = cross_val_predict(rf_cls(), X[cols], y_cls, cv=skf)
  # (b) divisão temporal: treina no passado, testa no futuro
  m = rf_cls().fit(X.loc[treino, cols], y_cls[treino])
  pred_t = m.predict(X.loc[teste, cols])
  linhas.append({
      "Cenário": nome_c,
      "Acurácia (CV)": round(accuracy_score(y_cls, pred), 3),
      "F1 macro (CV)": round(f1_score(y_cls, pred, average="macro"), 3),
      "Acurácia (temporal)": round(accuracy_score(y_cls[teste], pred_t), 3),
      "F1 macro (temporal)": round(f1_score(y_cls[teste], pred_t, average="macro"), 3),
  })
  print(f"\n--- {nome_c} (validação cruzada) ---")
  print(classification_report(y_cls, pred, labels=ordem))
  ConfusionMatrixDisplay.from_predictions(
      y_cls, pred, labels=ordem, normalize="true", cmap="Blues", ax=ax,
      colorbar=False, values_format=".2f",
  )
  ax.set_title(nome_c)
plt.tight_layout()
plt.savefig("figuras/matriz_confusao.png", dpi=150)
plt.close()

# Linha de base: sempre prever a classe mais frequente do treino
base_cv = cross_val_predict(DummyClassifier(), X, y_cls, cv=skf)
base_t = DummyClassifier().fit(X[treino], y_cls[treino]).predict(X[teste])
linhas.append({
    "Cenário": "Linha de base (classe majoritária)",
    "Acurácia (CV)": round(accuracy_score(y_cls, base_cv), 3),
    "F1 macro (CV)": round(f1_score(y_cls, base_cv, average="macro"), 3),
    "Acurácia (temporal)": round(accuracy_score(y_cls[teste], base_t), 3),
    "F1 macro (temporal)": round(f1_score(y_cls[teste], base_t, average="macro"), 3),
})
tab_cls = pd.DataFrame(linhas)
tab_cls.to_csv("resultados/classificacao.csv", index=False)
print(f"\nDivisão temporal: treino até {ANO_CORTE_TEMPORAL} "
      f"({treino.sum()} filmes), teste depois ({teste.sum()} filmes)")
print(tab_cls.to_string(index=False))

# ==============================================================================
# 9. DIREÇÃO DOS EFEITOS (dependência parcial, cenário pré-lançamento)
# ==============================================================================
titulo("9. DIREÇÃO DOS EFEITOS")
Xp = pd.DataFrame(
    SimpleImputer(strategy="median").fit_transform(X[feats_pre]), columns=feats_pre
)
rf = RandomForestRegressor(
    n_estimators=300, min_samples_leaf=3, random_state=SEMENTE, n_jobs=-1
).fit(Xp, y_reg)
alvo_pdp = ["log_budget", "ano", "runtime", "diretor_hist_mult",
            "elenco_hist_mult", "estudio_hist_mult"]
fig, ax = plt.subplots(2, 3, figsize=(14, 8))
PartialDependenceDisplay.from_estimator(rf, Xp, alvo_pdp, ax=ax)
fig.suptitle("Dependência parcial: efeito médio sobre o log10 do multiplicador")
plt.tight_layout()
plt.savefig("figuras/dependencia_parcial.png", dpi=150)
plt.close()

# Medianas do multiplicador por categoria (leitura direta)
linhas = []
binarios = ["eh_franquia", "idioma_ingles", "is_summer", "is_holiday"] + cols_mpaa + list(generos.columns)
for c in binarios:
  linhas.append({
      "Atributo": c,
      "N (=1)": int(df[c].sum()),
      "Mediana mult. (=1)": round(df.loc[df[c] == 1, "multiplicador"].median(), 2),
      "Mediana mult. (=0)": round(df.loc[df[c] == 0, "multiplicador"].median(), 2),
  })
tab_bin = pd.DataFrame(linhas)
tab_bin["Diferença"] = (tab_bin["Mediana mult. (=1)"] - tab_bin["Mediana mult. (=0)"]).round(2)
tab_bin = tab_bin.sort_values("Diferença", ascending=False)
tab_bin.to_csv("resultados/medianas_binarios.csv", index=False)
print(tab_bin.to_string(index=False))

decadas = df.groupby((df["ano"] // 10) * 10).agg(
    n=("ano", "size"), mediana_mult=("multiplicador", "median")
).round(2)
decadas.to_csv("resultados/multiplicador_por_decada.csv")
print("\nMultiplicador por década (atenção ao viés de seleção):")
print(decadas.to_string())

print("\nConcluído. Tabelas em resultados/ e gráficos em figuras/.")
