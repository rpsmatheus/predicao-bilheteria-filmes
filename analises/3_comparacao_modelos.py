"""
Comparação de 7 modelos (5 classes, F1 macro): Regressão Logística, SVM, KNN,
Random Forest, Gradient Boosting, XGBoost e CatBoost.

Como rodar (mesma pasta do CSV, com o ambiente ativado):
    pip install xgboost catboost
    python comparacao_modelos.py
Leva cerca de 1 a 2 horas. Cada etapa salva um CSV em resultados_modelos/;
se o script for interrompido, ao rodar de novo ele pula as etapas prontas.

A preparação dos dados (seções 1 a 3) é idêntica à de analise_completa.py e
as 5 classes são as mesmas de analise_5_classes.py.
"""

import json
import os
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.stats import loguniform, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

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


OUT = "resultados_modelos"
FIG = "figuras_modelos"
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

# 5 classes: cortes em 1x, 2x, 4x e 8x (mesmas de analise_5_classes.py)
CORTES = [1, 2, 4, 8]
NOMES = ["Prejuízo", "Fracasso", "Mediano", "Sucesso", "Grande sucesso"]
y = pd.Series(np.searchsorted(CORTES, df["multiplicador"], side="right"),
              index=df.index)

# ==============================================================================
# OS 7 MODELOS E O ESPAÇO DE BUSCA DE CADA UM
# ==============================================================================
# Todos recebem os mesmos atributos, com valores faltantes preenchidos pela
# mediana. Regressão Logística, SVM e KNN dependem de distâncias ou de pesos
# por atributo, então também padronizam os atributos (média 0, desvio 1).
# Os modelos de árvore não precisam disso.


def imp():
  return SimpleImputer(strategy="median")


MODELOS = {
    "Regressão Logística": (
        lambda: make_pipeline(imp(), StandardScaler(),
                              LogisticRegression(max_iter=3000)),
        {"logisticregression__C": loguniform(1e-3, 10)},
    ),
    "SVM": (
        lambda: make_pipeline(imp(), StandardScaler(), SVC()),
        {"svc__C": [0.1, 0.3, 1, 3, 10, 30],
         "svc__gamma": ["scale", 0.003, 0.01, 0.03, 0.1]},
    ),
    "KNN": (
        lambda: make_pipeline(imp(), StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": [5, 10, 15, 25, 40, 60, 90],
         "kneighborsclassifier__weights": ["uniform", "distance"]},
    ),
    "Random Forest": (
        lambda: make_pipeline(imp(), RandomForestClassifier(
            n_estimators=300, random_state=SEMENTE, n_jobs=-1)),
        {"randomforestclassifier__max_depth": [None, 8, 12, 20],
         "randomforestclassifier__min_samples_leaf": [1, 2, 5, 10],
         "randomforestclassifier__max_features": ["sqrt", 0.3, 0.5]},
    ),
    "Gradient Boosting": (
        lambda: make_pipeline(imp(), GradientBoostingClassifier(
            random_state=SEMENTE)),
        {"gradientboostingclassifier__n_estimators": [100, 200, 300],
         "gradientboostingclassifier__learning_rate": [0.03, 0.05, 0.1],
         "gradientboostingclassifier__max_depth": [2, 3, 4],
         "gradientboostingclassifier__subsample": [0.8, 1.0]},
    ),
    "XGBoost": (
        lambda: make_pipeline(imp(), XGBClassifier(
            random_state=SEMENTE, n_jobs=2, tree_method="hist")),
        {"xgbclassifier__n_estimators": [200, 400, 600],
         "xgbclassifier__learning_rate": [0.03, 0.05, 0.1],
         "xgbclassifier__max_depth": [3, 4, 6],
         "xgbclassifier__subsample": [0.8, 1.0],
         "xgbclassifier__colsample_bytree": [0.6, 0.8, 1.0],
         "xgbclassifier__min_child_weight": [1, 5]},
    ),
    "CatBoost": (
        lambda: make_pipeline(imp(), CatBoostClassifier(
            random_seed=SEMENTE, verbose=0, thread_count=2,
            allow_writing_files=False)),
        {"catboostclassifier__iterations": [300, 600],
         "catboostclassifier__depth": [4, 6, 8],
         "catboostclassifier__learning_rate": [0.03, 0.05, 0.1],
         "catboostclassifier__l2_leaf_reg": [1, 3, 10]},
    ),
}
N_BUSCA = 12  # configurações sorteadas por modelo e cenário

# A busca usa uma divisão em 3 partes com outra semente; a avaliação usa as
# 5 partes de sempre (skf). Assim a nota final não vem das mesmas divisões
# usadas para escolher os parâmetros.
cv_busca = StratifiedKFold(3, shuffle=True, random_state=7)


def etapa(arquivo):
  caminho = f"{OUT}/{arquivo}"
  return caminho, os.path.exists(caminho)


# ==============================================================================
# A. BUSCA DE PARÂMETROS
# ==============================================================================
titulo("A. BUSCA DE PARÂMETROS (F1 macro, 3 partes)")
caminho, pronto = etapa("melhores_parametros.json")
if pronto:
  PARAMS = json.load(open(caminho))
else:
  PARAMS = {}
  for nome, (fabrica, espaco) in MODELOS.items():
    PARAMS[nome] = {}
    for cen, cols in CENARIOS.items():
      t0 = time.time()
      busca = RandomizedSearchCV(fabrica(), espaco, n_iter=N_BUSCA,
                                 scoring="f1_macro", cv=cv_busca,
                                 random_state=SEMENTE, n_jobs=1)
      busca.fit(X[cols], y)
      melhores = {k: (v.item() if hasattr(v, "item") else v)
                  for k, v in busca.best_params_.items()}
      PARAMS[nome][cen] = melhores
      print(f"{nome} / {cen}: F1 na busca = {busca.best_score_:.3f} "
            f"({time.time() - t0:.0f}s) {melhores}", flush=True)
  json.dump(PARAMS, open(caminho, "w"), indent=2, ensure_ascii=False)
for nome in PARAMS:
  for cen in PARAMS[nome]:
    print(f"{nome} / {cen}: {PARAMS[nome][cen]}")


def modelo(nome, cen):
  """Modelo novo, já com os parâmetros escolhidos na busca."""
  return MODELOS[nome][0]().set_params(**PARAMS[nome][cen])


def metricas(real, prev):
  # ravel: o CatBoost devolve as previsões como coluna (n, 1)
  real, prev = np.asarray(real).ravel(), np.asarray(prev).ravel()
  return {
      "Acurácia": round(accuracy_score(real, prev), 3),
      "F1 macro": round(f1_score(real, prev, average="macro"), 3),
      "Acerto ±1 classe": round(float(np.mean(np.abs(real - prev) <= 1)), 3),
      "Erro médio (classes)": round(float(np.mean(np.abs(real - prev))), 3),
  }


# ==============================================================================
# B. DESEMPENHO DOS 7 MODELOS
# ==============================================================================
titulo("B. DESEMPENHO (validação cruzada em 5 partes e divisão temporal)")
treino = df["ano"] <= ANO_CORTE_TEMPORAL
teste = ~treino
caminho, pronto = etapa("desempenho.csv")
if pronto:
  tab_des = pd.read_csv(caminho)
  tab_cls = pd.read_csv(f"{OUT}/f1_por_classe.csv")
else:
  linhas, linhas_cls = [], []
  for nome in MODELOS:
    for cen, cols in CENARIOS.items():
      t0 = time.time()
      pred = np.asarray(
          cross_val_predict(modelo(nome, cen), X[cols], y, cv=skf)).ravel()
      seg = (time.time() - t0) / 5
      # F1 macro em cada uma das 5 partes, para saber se as diferenças
      # entre modelos são maiores que a variação natural entre partes
      f1_partes = [f1_score(y.iloc[te], pred[te], average="macro")
                   for _, te in skf.split(X, y)]
      m = modelo(nome, cen).fit(X.loc[treino, cols], y[treino])
      temp = metricas(y[teste], m.predict(X.loc[teste, cols]))
      linhas.append({"Modelo": nome, "Cenário": cen, **metricas(y, pred),
                     "Desvio F1 entre partes": round(np.std(f1_partes), 3),
                     "F1 macro (temporal)": temp["F1 macro"],
                     "Acurácia (temporal)": temp["Acurácia"],
                     "Segundos por treino": round(seg, 1)})
      f1c = f1_score(y, pred, average=None)
      linhas_cls.append({"Modelo": nome, "Cenário": cen,
                         **{n: round(v, 3) for n, v in zip(NOMES, f1c)}})
      print(f"{nome} / {cen}: F1 macro {linhas[-1]['F1 macro']}", flush=True)
  for nome_b, dummy in [("Chute: classe mais comum", DummyClassifier()),
                        ("Chute: sorteio proporcional",
                         DummyClassifier(strategy="stratified",
                                         random_state=SEMENTE))]:
    pred = cross_val_predict(dummy, X, y, cv=skf)
    f1_partes = [f1_score(y.iloc[te], pred[te], average="macro")
                 for _, te in skf.split(X, y)]
    b = clone(dummy).fit(X[treino], y[treino]).predict(X[teste])
    linhas.append({"Modelo": nome_b, "Cenário": "—", **metricas(y, pred),
                   "Desvio F1 entre partes": round(np.std(f1_partes), 3),
                   "F1 macro (temporal)": metricas(y[teste], b)["F1 macro"],
                   "Acurácia (temporal)": metricas(y[teste], b)["Acurácia"],
                   "Segundos por treino": 0})
  tab_des = pd.DataFrame(linhas)
  tab_cls = pd.DataFrame(linhas_cls)
  tab_des.to_csv(caminho, index=False)
  tab_cls.to_csv(f"{OUT}/f1_por_classe.csv", index=False)
print(tab_des.to_string(index=False))
print("\nF1 por classe:")
print(tab_cls.to_string(index=False))

# Gráfico: F1 macro por modelo nos dois cenários
fig, ax = plt.subplots(figsize=(10, 5))
t = tab_des[tab_des["Cenário"] != "—"].pivot(index="Modelo", columns="Cenário",
                                              values="F1 macro")
t = t.sort_values("Pré-lançamento")
t.plot.barh(ax=ax, color=["#9aa5b1", "#2b6cb0"])
sorteio = tab_des.loc[tab_des["Modelo"] == "Chute: sorteio proporcional",
                      "F1 macro"].iloc[0]
ax.axvline(sorteio, color="gray", ls="--", lw=1)
ax.text(sorteio, -0.7, " sorteio", color="gray", va="top")
ax.set_xlabel("F1 macro (validação cruzada)")
ax.set_title("F1 macro dos 7 modelos (5 classes)")
plt.tight_layout()
plt.savefig(f"{FIG}/f1_modelos.png", dpi=150)
plt.close()


def permutacao_cv(make, cols, n_rep):
  """Permutação nas 5 partes. Devolve média, desvio e rodadas > 0."""
  rodadas = []
  for tr, te in skf.split(X, y):
    m = make().fit(X.iloc[tr][cols], y.iloc[tr])
    pi = permutation_importance(m, X.iloc[te][cols], y.iloc[te],
                                scoring="f1_macro", n_repeats=n_rep,
                                random_state=SEMENTE, n_jobs=1)
    rodadas.append(pi.importances_mean)
  r = np.array(rodadas)
  return r.mean(0), r.std(0), (r > 0).sum(0)


# ==============================================================================
# C. RANKING DE ATRIBUTOS POR MODELO (permutação, queda de F1 macro)
# ==============================================================================
titulo("C. IMPORTÂNCIA POR PERMUTAÇÃO, MODELO A MODELO")
caminho, pronto = etapa("importancia_atributos.csv")
if pronto:
  tab_perm = pd.read_csv(caminho)
else:
  linhas = []
  for nome in MODELOS:
    for cen, cols in CENARIOS.items():
      print(f"  {nome} / {cen}...", flush=True)
      med, dp, pos = permutacao_cv(lambda: modelo(nome, cen), cols, n_rep=3)
      for c, a, b, p in zip(cols, med, dp, pos):
        linhas.append({"Modelo": nome, "Cenário": cen, "Atributo": c,
                       "Queda F1": round(a, 4), "Desvio": round(b, 4),
                       "Rodadas > 0": int(p)})
  tab_perm = pd.DataFrame(linhas)
  tab_perm.to_csv(caminho, index=False)
tab_perm["Posição"] = (tab_perm.groupby(["Modelo", "Cenário"])["Queda F1"]
                       .rank(ascending=False, method="min").astype(int))
for cen in CENARIOS:
  t = tab_perm[tab_perm["Cenário"] == cen].pivot(
      index="Atributo", columns="Modelo", values="Posição")
  t["Posição média"] = t.mean(axis=1).round(1)
  t = t.sort_values("Posição média")
  t.to_csv(f"{OUT}/posicoes_{'pre' if cen.startswith('Pré') else 'todos'}.csv")
  print(f"\n--- {cen}: posição de cada atributo em cada modelo (top 12) ---")
  print(t.head(12).to_string())

# ==============================================================================
# D. CONCORDÂNCIA ENTRE OS MODELOS
# ==============================================================================
# Correlação de Spearman entre as importâncias de cada par de modelos:
# perto de 1 = os dois modelos ordenam os atributos do mesmo jeito.
titulo("D. CONCORDÂNCIA ENTRE MODELOS (Spearman entre rankings)")
for cen in CENARIOS:
  t = tab_perm[tab_perm["Cenário"] == cen].pivot(
      index="Atributo", columns="Modelo", values="Queda F1")
  t = t[list(MODELOS)]
  conc = t.corr(method="spearman").round(2)
  sufixo = "pre" if cen.startswith("Pré") else "todos"
  conc.to_csv(f"{OUT}/concordancia_{sufixo}.csv")
  print(f"\n--- {cen} ---")
  print(conc.to_string())
  fig, ax = plt.subplots(figsize=(8, 6.5))
  im = ax.imshow(conc.values, vmin=0, vmax=1, cmap="Blues")
  ax.set_xticks(range(len(conc)), conc.columns, rotation=35, ha="right")
  ax.set_yticks(range(len(conc)), conc.index)
  for i in range(len(conc)):
    for j in range(len(conc)):
      ax.text(j, i, f"{conc.values[i, j]:.2f}", ha="center", va="center",
              color="white" if conc.values[i, j] > 0.6 else "black", fontsize=9)
  ax.set_title(f"Concordância entre rankings de atributos ({cen})")
  plt.colorbar(im, ax=ax, fraction=0.04)
  plt.tight_layout()
  plt.savefig(f"{FIG}/concordancia_{sufixo}.png", dpi=150)
  plt.close()

# ==============================================================================
# E. IMPORTÂNCIA POR GRUPO, MODELO A MODELO
# ==============================================================================
titulo("E. IMPORTÂNCIA POR GRUPO (F1 macro em validação cruzada)")
# Grupos pré-lançamento usam os parâmetros do cenário pré-lançamento; os
# grupos pós-lançamento, os do cenário com todos os atributos.
POS = [g for g in GRUPOS if g not in PRE]


def cen_do_grupo(g):
  return "Pré-lançamento" if g in PRE else "Todos os atributos"


def f1_cv(nome, cen, cols):
  return cross_val_score(modelo(nome, cen), X[cols], y, cv=skf,
                         scoring="f1_macro").mean()


caminho, pronto = etapa("importancia_grupos.csv")
if pronto:
  tab_grp = pd.read_csv(caminho)
else:
  linhas = []
  for nome in MODELOS:
    print(f"  {nome}...", flush=True)
    sozinho = {g: f1_cv(nome, cen_do_grupo(g), GRUPOS[g]) for g in GRUPOS}
    for cen, grupos_c in [("Pré-lançamento", PRE),
                          ("Todos os atributos", list(GRUPOS))]:
      cols_c = sum((GRUPOS[g] for g in grupos_c), [])
      completo = f1_cv(nome, cen, cols_c)
      for g in grupos_c:
        sem = f1_cv(nome, cen, [c for c in cols_c if c not in GRUPOS[g]])
        linhas.append({"Modelo": nome, "Cenário": cen, "Grupo": g,
                       "F1 completo": round(completo, 3),
                       "F1 só o grupo": round(sozinho[g], 3),
                       "F1 sem o grupo": round(sem, 3),
                       "Perda ao remover": round(completo - sem, 3)})
  tab_grp = pd.DataFrame(linhas)
  tab_grp.to_csv(caminho, index=False)
for cen in CENARIOS:
  t = tab_grp[tab_grp["Cenário"] == cen].pivot(
      index="Grupo", columns="Modelo", values="Perda ao remover")[list(MODELOS)]
  t["Média"] = t.mean(axis=1).round(3)
  print(f"\n--- {cen}: perda de F1 ao remover o grupo ---")
  print(t.sort_values("Média", ascending=False).to_string())

# ==============================================================================
# F. ATRIBUTO LÍDER DENTRO DE CADA GRUPO, MODELO A MODELO
# ==============================================================================
titulo("F. IMPORTÂNCIA DENTRO DE CADA GRUPO (modelo só com o grupo)")
caminho, pronto = etapa("importancia_dentro_grupos.csv")
if pronto:
  tab_dg = pd.read_csv(caminho)
else:
  linhas = []
  for nome in MODELOS:
    for g, cols in GRUPOS.items():
      print(f"  {nome} / {g}...", flush=True)
      cen = cen_do_grupo(g)
      med, dp, pos = permutacao_cv(lambda: modelo(nome, cen), cols, n_rep=5)
      for c, a, b, p in zip(cols, med, dp, pos):
        linhas.append({"Modelo": nome, "Grupo": g, "Atributo": c,
                       "Queda F1": round(a, 4), "Desvio": round(b, 4),
                       "Rodadas > 0": int(p)})
  tab_dg = pd.DataFrame(linhas)
  tab_dg.to_csv(caminho, index=False)
lider = (tab_dg.sort_values("Queda F1", ascending=False)
         .groupby(["Grupo", "Modelo"]).head(1)
         .pivot(index="Grupo", columns="Modelo", values="Atributo")[list(MODELOS)])
lider.to_csv(f"{OUT}/lider_por_grupo.csv")
print(lider.to_string())

print("\nConcluído.")
