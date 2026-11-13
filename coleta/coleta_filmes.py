import csv
from datetime import datetime
import os
import time
import requests
from tqdm import tqdm

try:
  from dotenv import load_dotenv

  load_dotenv()
except ImportError:
  pass

# Chaves lidas do ambiente (arquivo .env, fora do git). Veja .env.example.
TMDB_API_KEY = os.environ["TMDB_API_KEY"]
OMDB_API_KEY = os.environ["OMDB_API_KEY"]
ARQUIVO_SAIDA = "dados/dataset_filmes.csv"

# Data limite para descartar estreias futuras
HOJE = datetime.now().strftime("%Y-%m-%d")


def requisicao_com_retry(url, tentativas=3, espera=2):
  for tentativa in range(tentativas):
    try:
      resp = requests.get(url, timeout=10)
      if resp.status_code == 200:
        return resp
    except (
        requests.exceptions.RequestException,
        requests.exceptions.ConnectionError,
    ):
      if tentativa < tentativas - 1:
        time.sleep(espera)
  return None


def obter_ids_tmdb(paginas_totais=200):
  filmes_ids = []
  print(f"A recolher IDs ao longo de {paginas_totais} páginas...")
  for pagina in range(1, paginas_totais + 1):
    url = (
        f"https://api.themoviedb.org/3/discover/movie?"
        f"api_key={TMDB_API_KEY}&sort_by=popularity.desc&"
        f"primary_release_date.gte=1980-01-01&"
        f"primary_release_date.lte={HOJE}&"
        f"vote_count.gte=100&page={pagina}"
    )
    resp = requisicao_com_retry(url)
    if resp:
      dados = resp.json()
      for item in dados.get("results", []):
        filmes_ids.append(item["id"])
    time.sleep(0.05)
  return filmes_ids


def coletar_detalhes(tmdb_id):
  url_tmdb = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
  r_tmdb = requisicao_com_retry(url_tmdb)
  if not r_tmdb:
    return None
  d_tmdb = r_tmdb.json()

  imdb_id = d_tmdb.get("imdb_id")
  release_date = d_tmdb.get("release_date", "")

  revenue = d_tmdb.get("revenue") or 0
  budget = d_tmdb.get("budget") or 0
  revenue_val = revenue if revenue >= 10000 else None
  budget_val = budget if budget >= 10000 else None

  # Descarte de filmes sem data válida, sem imdb_id ou futuros
  if not imdb_id or not release_date or release_date > HOJE:
    return None

  # OMDb
  url_omdb = f"https://www.omdbapi.com/?i={imdb_id}&apikey={OMDB_API_KEY}"
  r_omdb = requisicao_com_retry(url_omdb)
  d_omdb = r_omdb.json() if r_omdb else {}

  rt_score = None
  for rating in d_omdb.get("Ratings", []):
    if rating.get("Source") == "Rotten Tomatoes":
      rt_score = rating.get("Value")

  # Tratamento do número de votos do IMDb para evitar separador de milhar com vírgula
  imdb_votes_raw = d_omdb.get("imdbVotes")
  imdb_votes = None
  if (
      imdb_votes_raw
      and imdb_votes_raw != "N/A"
      and isinstance(imdb_votes_raw, str)
  ):
    imdb_votes = imdb_votes_raw.replace(",", "").strip()

  # Tratamento do BoxOffice da OMDb
  box_office_raw = d_omdb.get("BoxOffice")
  box_office_domestic = None
  if box_office_raw and box_office_raw != "N/A":
    try:
      box_office_domestic = int(
          box_office_raw.replace("$", "").replace(",", "")
      )
    except ValueError:
      box_office_domestic = None

  # Limpeza do texto da sinopse (remove quebras de linha que quebram CSV)
  plot_clean = d_omdb.get("Plot")
  if plot_clean and plot_clean != "N/A":
    plot_clean = plot_clean.replace("\n", " ").replace("\r", " ").strip()

  elenco = [
      ator["name"] for ator in d_tmdb.get("credits", {}).get("cast", [])[:10]
  ]
  crew = d_tmdb.get("credits", {}).get("crew", [])
  diretores = [m["name"] for m in crew if m.get("job") == "Director"]
  roteiristas = [
      m["name"]
      for m in crew
      if m.get("job") in ("Screenplay", "Writer", "Story")
  ]
  produtoras = [p["name"] for p in d_tmdb.get("production_companies", [])[:5]]
  paises = [p["name"] for p in d_tmdb.get("production_countries", [])]
  idiomas = [l["english_name"] for l in d_tmdb.get("spoken_languages", [])]
  franquia = (d_tmdb.get("belongs_to_collection") or {}).get("name")

  return {
      "tmdb_id": tmdb_id,
      "imdb_id": imdb_id,
      "title": d_tmdb.get("title"),
      "original_title": d_tmdb.get("original_title"),
      "release_date": release_date,
      "budget": budget_val,
      "tem_budget": budget_val is not None,
      "revenue": revenue_val,
      "tem_revenue": revenue_val is not None,
      "box_office_domestic_omdb": box_office_domestic,
      "runtime": d_tmdb.get("runtime"),
      "genres": "|".join([g["name"] for g in d_tmdb.get("genres", [])]),
      "director": diretores[0] if diretores else None,
      "all_directors": "|".join(diretores),
      "writers": "|".join(roteiristas),
      "actors": "|".join(elenco),
      "production_companies": "|".join(produtoras),
      "production_countries": "|".join(paises),
      "spoken_languages": "|".join(idiomas),
      "original_language": d_tmdb.get("original_language"),
      "collection": franquia,
      "tagline": (d_tmdb.get("tagline") or "").replace("\n", " ").strip(),
      "tmdb_popularity": d_tmdb.get("popularity"),
      "tmdb_vote_average": d_tmdb.get("vote_average"),
      "tmdb_vote_count": d_tmdb.get("vote_count"),
      "imdb_rating": d_omdb.get("imdbRating"),
      "imdb_votes": imdb_votes,
      "metascore": d_omdb.get("Metascore"),
      "rotten_tomatoes": rt_score,
      "mpaa_rating": d_omdb.get("Rated"),
      "country_omdb": d_omdb.get("Country"),
      "language_omdb": d_omdb.get("Language"),
      "plot": plot_clean,
      "awards": d_omdb.get("Awards"),
  }


if __name__ == "__main__":
  print("A iniciar o script de recolha...")

  lista_ids = obter_ids_tmdb(paginas_totais=200)
  print(f"Total de IDs candidatos recolhidos: {len(lista_ids)}")

  colunas = [
      "tmdb_id",
      "imdb_id",
      "title",
      "original_title",
      "release_date",
      "budget",
      "tem_budget",
      "revenue",
      "tem_revenue",
      "box_office_domestic_omdb",
      "runtime",
      "genres",
      "director",
      "all_directors",
      "writers",
      "actors",
      "production_companies",
      "production_countries",
      "spoken_languages",
      "original_language",
      "collection",
      "tagline",
      "tmdb_popularity",
      "tmdb_vote_average",
      "tmdb_vote_count",
      "imdb_rating",
      "imdb_votes",
      "metascore",
      "rotten_tomatoes",
      "mpaa_rating",
      "country_omdb",
      "language_omdb",
      "plot",
      "awards",
  ]

  if not os.path.exists(ARQUIVO_SAIDA):
    with open(ARQUIVO_SAIDA, mode="w", newline="", encoding="utf-8") as f:
      # QUOTE_MINIMAL garante aspas em campos com delimitador ou texto
      writer = csv.DictWriter(
          f, fieldnames=colunas, quoting=csv.QUOTE_MINIMAL
      )
      writer.writeheader()

  sucessos = 0

  for tmdb_id in tqdm(lista_ids, desc="A processar filmes"):
    try:
      registro = coletar_detalhes(tmdb_id)
      if registro:
        with open(ARQUIVO_SAIDA, mode="a", newline="", encoding="utf-8") as f:
          writer = csv.DictWriter(
              f, fieldnames=colunas, quoting=csv.QUOTE_MINIMAL
          )
          writer.writerow(registro)
        sucessos += 1
      time.sleep(0.1)
    except Exception as e:
      print(f"\nErro no ID {tmdb_id}: {e}")

  print(f"\nRecolha concluída! Total de filmes guardados: {sucessos}")