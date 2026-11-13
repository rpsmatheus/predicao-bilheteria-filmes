"""
Sprint 5 classes: refaz a parte de classificação com 5 classes e compara
com as 3 classes de antes.

Como rodar (mesma pasta do CSV, com o ambiente ativado):
    python analise_5_classes.py
Leva de 20 a 40 minutos (as seções D e E são as mais lentas).

A preparação dos dados (seções 1 a 3) é idêntica à de analise_completa.py.
O que NÃO depende das classes (regressão, clusters, análise fatorial) não
é refeito aqui: os resultados continuam os de analise_completa.py.

Saídas: resultados_sprint5/*.csv e figuras_sprint5/*.png
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
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
)
from sklearn.pipeline import make_pipeline

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


OUT = "resultados_sprint5"
FIG = "figuras_sprint5"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# ==============================================================================
# A. OS DOIS ESQUEMAS DE CLASSES
# ==============================================================================
# 5 classes: cortes em 1x, 2x, 4x e 8x. Cada faixa dobra a anterior, e os
# cortes de 2x e 4x são os mesmos das 3 classes (Vogel): as 5 classes são as
# 3 antigas com Fracasso e Sucesso divididos em dois.
ESQUEMAS = {
    "3 classes": ([2, 4], ["Fracasso", "Mediano", "Sucesso"]),
    "5 classes": ([1, 2, 4, 8],
                  ["Prejuízo", "Fracasso", "Mediano", "Sucesso", "Grande sucesso"]),
}


def rotular(mult, cortes, nomes):
  """Devolve o código ordinal da classe (0 = pior) e o nome."""
  cod = np.searchsorted(cortes, mult, side="right")
  return pd.Series(cod, index=mult.index), [nomes[c] for c in cod]


Y = {}
titulo("A. DISTRIBUIÇÃO DAS CLASSES")
linhas = []
for nome, (cortes, nomes) in ESQUEMAS.items():
  cod, _ = rotular(df["multiplicador"], cortes, nomes)
  Y[nome] = cod
  limites = [-np.inf] + cortes + [np.inf]
  for i, n in enumerate(nomes):
    faixa = (f"< {cortes[0]}x" if i == 0 else
             f">= {cortes[-1]}x" if i == len(nomes) - 1 else
             f"{limites[i]}x a {limites[i + 1]}x")
    linhas.append({"Esquema": nome, "Classe": n, "Faixa": faixa,
                   "Filmes": int((cod == i).sum()),
                   "%": round(100 * (cod == i).mean(), 1)})
tab = pd.DataFrame(linhas)
tab.to_csv(f"{OUT}/distribuicao_classes.csv", index=False)
print(tab.to_string(index=False))


def rf_clf(n=300):
  return make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestClassifier(n_estimators=n, min_samples_leaf=2,
                             random_state=SEMENTE, n_jobs=-1))


def rf_reg_n(n=300):
  return make_pipeline(
      SimpleImputer(strategy="median"),
      RandomForestRegressor(n_estimators=n, min_samples_leaf=3,
                            random_state=SEMENTE, n_jobs=-1))


def metricas(real, prev):
  real, prev = np.asarray(real), np.asarray(prev)
  return {
      "Acurácia": round(accuracy_score(real, prev), 3),
      "F1 macro": round(f1_score(real, prev, average="macro"), 3),
      # classes são ordenadas: errar por uma classe é menos grave
      "Acerto ±1 classe": round(np.mean(np.abs(real - prev) <= 1), 3),
      "Erro médio (nº de classes)": round(np.mean(np.abs(real - prev)), 3),
  }


# ==============================================================================
# B. DESEMPENHO: 3 CONTRA 5 CLASSES
# ==============================================================================
titulo("B. DESEMPENHO DOS MODELOS")
treino = df["ano"] <= ANO_CORTE_TEMPORAL
teste = ~treino
linhas = []
# A regressão não depende das classes: previsões calculadas uma vez
pred_reg = {c: cross_val_predict(rf_reg_n(), X[cols], y_reg, cv=kf)
            for c, cols in CENARIOS.items()}
for esquema, (cortes, nomes) in ESQUEMAS.items():
  y = Y[esquema]
  lim_log = np.log10(cortes)
  for cen, cols in CENARIOS.items():
    print(f"  {esquema} / {cen}...", flush=True)
    # (1) classificador direto
    pred = cross_val_predict(rf_clf(), X[cols], y, cv=skf)
    linhas.append({"Esquema": esquema, "Cenário": cen,
                   "Método": "Random Forest (classificação)",
                   "Validação": "Cruzada", **metricas(y, pred)})
    if cen == "Pré-lançamento":
      print(f"\n{esquema}, pré-lançamento, relatório por classe:")
      print(classification_report(y, pred, target_names=nomes))
      fig, ax = plt.subplots(figsize=(7, 6))
      ConfusionMatrixDisplay.from_predictions(
          y, pred, display_labels=nomes, normalize="true", cmap="Blues",
          ax=ax, colorbar=False, values_format=".2f")
      ax.set_title(f"Matriz de confusão ({esquema}, pré-lançamento)")
      plt.xticks(rotation=30, ha="right")
      plt.tight_layout()
      plt.savefig(f"{FIG}/matriz_confusao_{esquema[0]}c.png", dpi=150)
      plt.close()
      cm = pd.crosstab(pd.Series(y, name="Real").map(dict(enumerate(nomes))),
                       pd.Series(pred, name="Previsto").map(dict(enumerate(nomes))))
      cm = cm.reindex(index=nomes, columns=nomes, fill_value=0)
      cm.to_csv(f"{OUT}/matriz_confusao_{esquema[0]}c_pre.csv")
      print(cm.to_string())
    # (2) regressão do log-multiplicador, cortada nas mesmas faixas
    pr = np.searchsorted(lim_log, pred_reg[cen], side="right")
    linhas.append({"Esquema": esquema, "Cenário": cen,
                   "Método": "Regressão + cortes",
                   "Validação": "Cruzada", **metricas(y, pr)})
    # (3) divisão temporal: treina até 2019, testa depois
    m = rf_clf().fit(X.loc[treino, cols], y[treino])
    linhas.append({"Esquema": esquema, "Cenário": cen,
                   "Método": "Random Forest (classificação)",
                   "Validação": "Temporal",
                   **metricas(y[teste], m.predict(X.loc[teste, cols]))})
  # linhas de base
  base = cross_val_predict(DummyClassifier(), X, y, cv=skf)
  linhas.append({"Esquema": esquema, "Cenário": "—",
                 "Método": "Chute: classe mais comum", "Validação": "Cruzada",
                 **metricas(y, base)})
  base = cross_val_predict(DummyClassifier(strategy="stratified",
                                           random_state=SEMENTE), X, y, cv=skf)
  linhas.append({"Esquema": esquema, "Cenário": "—",
                 "Método": "Chute: sorteio proporcional", "Validação": "Cruzada",
                 **metricas(y, base)})
  b = DummyClassifier().fit(X[treino], y[treino]).predict(X[teste])
  linhas.append({"Esquema": esquema, "Cenário": "—",
                 "Método": "Chute: classe mais comum", "Validação": "Temporal",
                 **metricas(y[teste], b)})
tab_des = pd.DataFrame(linhas)
tab_des.to_csv(f"{OUT}/desempenho.csv", index=False)
print("\n" + tab_des.to_string(index=False))

# Robustez: 5 classes de tamanho igual (quintis), só pré-lançamento
q = df["multiplicador"].quantile([.2, .4, .6, .8]).values
yq = pd.Series(np.searchsorted(q, df["multiplicador"], side="right"), index=df.index)
pq = cross_val_predict(rf_clf(), X[feats_pre], yq, cv=skf)
print(f"\nRobustez, 5 classes por quintis (cortes {np.round(q, 2).tolist()}):")
print("  modelo:", metricas(yq, pq))
print("  chute: ", metricas(yq, cross_val_predict(DummyClassifier(), X, yq, cv=skf)))

# ==============================================================================
# C. IMPORTÂNCIA POR PERMUTAÇÃO (queda de F1 macro), 3 contra 5 classes
# ==============================================================================
titulo("C. IMPORTÂNCIA POR PERMUTAÇÃO (queda de F1 macro, validação cruzada)")


def permutacao_cv(cols, y, n_arvores=150, n_rep=5):
  """Permutação nas 5 partes; devolve (média, desvio entre rodadas, rodadas>0)."""
  rodadas = []
  for tr, te in skf.split(X, y):
    m = rf_clf(n_arvores).fit(X.iloc[tr][cols], y.iloc[tr])
    pi = permutation_importance(m, X.iloc[te][cols], y.iloc[te],
                                scoring="f1_macro", n_repeats=n_rep,
                                random_state=SEMENTE, n_jobs=-1)
    rodadas.append(pi.importances_mean)
  r = np.array(rodadas)
  return r.mean(0), r.std(0), (r > 0).sum(0)


linhas = []
for cen, cols in CENARIOS.items():
  for esquema in ESQUEMAS:
    print(f"  {esquema} / {cen}...", flush=True)
    med, dp, pos = permutacao_cv(cols, Y[esquema])
    for c, a, b, p in zip(cols, med, dp, pos):
      linhas.append({"Cenário": cen, "Esquema": esquema, "Atributo": c,
                     "Queda F1 (média)": round(a, 4), "Desvio": round(b, 4),
                     "Rodadas > 0": int(p)})
tab_perm = pd.DataFrame(linhas)
tab_perm.to_csv(f"{OUT}/importancia_permutacao.csv", index=False)
for cen in CENARIOS:
  t = (tab_perm[tab_perm["Cenário"] == cen]
       .pivot(index="Atributo", columns="Esquema", values="Queda F1 (média)"))
  t["Posição 3c"] = t["3 classes"].rank(ascending=False).astype(int)
  t["Posição 5c"] = t["5 classes"].rank(ascending=False).astype(int)
  print(f"\n--- {cen} ---")
  print(t.sort_values("3 classes", ascending=False).head(12).to_string())

# ==============================================================================
# D. IMPORTÂNCIA POR GRUPO (F1 macro), 3 contra 5 classes
# ==============================================================================
titulo("D. IMPORTÂNCIA POR GRUPO (F1 macro em validação cruzada)")


def f1_cv(cols, y):
  return cross_val_score(rf_clf(150), X[cols], y, cv=skf,
                         scoring="f1_macro").mean()


linhas = []
for esquema in ESQUEMAS:
  y = Y[esquema]
  sozinho = {}
  for g, cols in GRUPOS.items():
    sozinho[g] = f1_cv(cols, y)
  for cen, grupos_c in [("Pré-lançamento", PRE), ("Todos os atributos", list(GRUPOS))]:
    print(f"  {esquema} / {cen}...", flush=True)
    cols_c = sum((GRUPOS[g] for g in grupos_c), [])
    completo = f1_cv(cols_c, y)
    for g in grupos_c:
      sem = f1_cv([c for c in cols_c if c not in GRUPOS[g]], y)
      linhas.append({"Esquema": esquema, "Cenário": cen, "Grupo": g,
                     "F1 completo": round(completo, 3),
                     "F1 só o grupo": round(sozinho[g], 3),
                     "F1 sem o grupo": round(sem, 3),
                     "Perda ao remover": round(completo - sem, 3)})
tab_grp = pd.DataFrame(linhas)
tab_grp.to_csv(f"{OUT}/importancia_grupos.csv", index=False)
for cen in CENARIOS:
  t = tab_grp[tab_grp["Cenário"] == cen]
  print(f"\n--- {cen} ---")
  print(t.pivot(index="Grupo", columns="Esquema",
                values=["F1 só o grupo", "Perda ao remover"]).to_string())

# ==============================================================================
# E. ATRIBUTO MAIS IMPORTANTE DENTRO DE CADA GRUPO, 3 contra 5 classes
# ==============================================================================
titulo("E. IMPORTÂNCIA DENTRO DE CADA GRUPO (modelo só com o grupo)")
linhas = []
for esquema in ESQUEMAS:
  for g, cols in GRUPOS.items():
    print(f"  {esquema} / {g}...", flush=True)
    med, dp, pos = permutacao_cv(cols, Y[esquema], n_arvores=100, n_rep=10)
    for c, a, b, p in zip(cols, med, dp, pos):
      linhas.append({"Esquema": esquema, "Grupo": g, "Atributo": c,
                     "Queda F1 (média)": round(a, 4), "Desvio": round(b, 4),
                     "Rodadas > 0": int(p)})
tab_dg = pd.DataFrame(linhas)
tab_dg.to_csv(f"{OUT}/importancia_dentro_grupos.csv", index=False)
for g in GRUPOS:
  t = tab_dg[tab_dg["Grupo"] == g].pivot(
      index="Atributo", columns="Esquema", values=["Queda F1 (média)", "Rodadas > 0"])
  t = t.sort_values(("Queda F1 (média)", "3 classes"), ascending=False)
  print(f"\n--- {g} ---")
  print(t.head(6).to_string())

# ==============================================================================
# F. PERFIL DE CADA UMA DAS 5 CLASSES
# ==============================================================================
titulo("F. PERFIL DAS 5 CLASSES (medianas e proporções)")
cortes, nomes = ESQUEMAS["5 classes"]
d = df.assign(classe5=Y["5 classes"].map(dict(enumerate(nomes))))
perfil = d.groupby("classe5").agg(
    Filmes=("ano", "size"),
    Mult_mediano=("multiplicador", "median"),
    Orcamento_mediano_US_mi=("budget", lambda s: s.median() / 1e6),
    Pct_franquia=("eh_franquia", lambda s: 100 * s.mean()),
    Ano_mediano=("ano", "median"),
    Pct_terror=("gen_Horror", lambda s: 100 * s.mean()),
    Pct_acao=("gen_Action", lambda s: 100 * s.mean()),
    RT_mediano=("rt_score", "median"),
    Votos_TMDB_medianos=("tmdb_vote_count", "median"),
    Hist_estudio_mediano=("estudio_hist_mult", "median"),
).reindex(nomes).round(2)
perfil.to_csv(f"{OUT}/perfil_5_classes.csv")
print(perfil.to_string())
print("\nConcluído.")
