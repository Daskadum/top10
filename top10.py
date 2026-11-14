import csv
import datetime as dt
import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "docs"
STATE = ROOT / "state"
TZ = ZoneInfo("America/New_York")
NOW = dt.datetime.now(TZ)
MODELS = [m for m in os.environ.get("TOP10_MODELS", "claude-haiku-5-5,claude-sonnet-5-5").split(",") if m]
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")

SOURCES = [
    dict(key="netflix_movies", service="netflix", chart="weekly", region="GLOBAL", official=True, want="movies",
         url="https://www.netflix.com/tudum/top10/",
         hint="Global Top 10 English movies for the most recent week shown."),
    dict(key="netflix_series", service="netflix", chart="weekly", region="GLOBAL", official=True, want="series",
         url="https://www.netflix.com/tudum/top10/tv",
         hint="Global Top 10 English TV shows for the most recent week shown."),
    dict(key="prime_video", service="prime_video", chart="weekly", region="GLOBAL", official=True, want="both",
         url="https://www.aboutamazon.com/news/entertainment/prime-video-top-10-movies-shows",
         hint="The Top 10 Prime Original films and Top 10 Prime Original series lists that combine English and "
              "non-English titles (not the non-English-only lists)."),
    dict(key="hulu_movies", service="hulu", chart="daily", region="US", official=True, want="movies",
         url="https://www.hulu.com/hub/movies/collections/10022",
         hint="The 'Top 15 Movies Today' list. Return only the first 10."),
    dict(key="hulu_series", service="hulu", chart="daily", region="US", official=True, want="series",
         url="https://www.hulu.com/hub/tv/collections/9979",
         hint="The 'Top 15 Series Today' list. Return only the first 10."),
    dict(key="disney_plus", service="disney_plus", chart="daily", region="US", official=False, want="both",
         url="https://streamscope.app/disney-plus",
         hint="Today's Disney+ Top 10 movies and Top 10 series."),
    dict(key="hbo_max", service="hbo_max", chart="weekly", region="US", official=False, want="combined",
         url="https://www.showlabs.tv/en/hbo-top-10",
         hint="One combined weekly Top 10 list. Put each title under movies or series using its Movie/Series "
              "label, and use its position in the combined list as its rank."),
]

SYSTEM = (
    "You extract ranked lists from the text of a web page. Copy titles exactly as written, in rank order. "
    "Use only the page text; never add titles from memory. If a list is not on the page, return it empty. "
    "Set period_start and period_end to the dates the list covers (YYYY-MM-DD) if the page states them, "
    "otherwise empty strings."
)
ITEMS = {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["rank", "title"],
                                    "properties": {"rank": {"type": "integer"}, "title": {"type": "string"}}}}
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["period_start", "period_end", "movies", "series"],
          "properties": {"period_start": {"type": "string"}, "period_end": {"type": "string"},
                         "movies": ITEMS, "series": ITEMS}}


def fetch_pages():
    from playwright.sync_api import sync_playwright
    texts, errors = {}, {}
    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(user_agent=UA, locale="en-US", viewport={"width": 1280, "height": 2200})
        for s in SOURCES:
            page = ctx.new_page()
            try:
                page.goto(s["url"], wait_until="domcontentloaded", timeout=60000)
                try:
                    page.wait_for_load_state("networkidle", timeout=15000)
                except Exception:
                    pass
                text = re.sub(r"\n\s*\n+", "\n", page.inner_text("body"))
                if len(text) < 200:
                    raise RuntimeError(f"page text too short ({len(text)} chars)")
                texts[s["key"]] = text[:60000]
            except Exception as e:
                errors[s["key"]] = f"fetch failed: {e}".splitlines()[0][:200]
            finally:
                page.close()
        browser.close()
    return texts, errors


def extract(client, s, text):
    import anthropic
    user = (f"Page: {s['url']}\nToday: {NOW:%Y-%m-%d}\nTask: {s['hint']}\n\n<page>\n{text}\n</page>")
    last = None
    for model in MODELS:
        try:
            r = client.messages.create(
                model=model, max_tokens=3000, system=SYSTEM,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}},
            )
        except (anthropic.NotFoundError, anthropic.BadRequestError) as e:
            last = e
            continue
        if r.stop_reason != "end_turn":
            last = RuntimeError(f"{model} stopped: {r.stop_reason}")
            continue
        out = json.loads(next(b.text for b in r.content if b.type == "text"))
        print(f"  {s['key']}: {model} in={r.usage.input_tokens} out={r.usage.output_tokens}")
        return out
    raise last


def problems(s, data):
    found = []
    want = s["want"]
    if want in ("movies", "both") and not data["movies"]:
        found.append("no movies")
    if want in ("series", "both") and not data["series"]:
        found.append("no series")
    if want == "combined" and not (data["movies"] or data["series"]):
        found.append("empty list")
    end = parse_date(data.get("period_end"))
    if end:
        age = (NOW.date() - end).days
        if (s["chart"] == "weekly" and age > 16) or (s["chart"] == "daily" and age > 3):
            found.append(f"old data (ends {end})")
    return found


def parse_date(v):
    try:
        return dt.date.fromisoformat((v or "")[:10])
    except ValueError:
        return None


def clean_title(raw):
    t = raw.split(" / ")[0].strip()
    m = re.match(r"^(.*?):\s*Season\s+(\d+)$", t, re.I) or re.match(r"^(.*?):\s*S(\d+)(?::.*)?$", t)
    if m:
        return m.group(1).strip(), int(m.group(2))
    m = (re.match(r"^(.*?):\s*(?:Limited Series|Miniseries)$", t, re.I)
         or re.match(r"^(.*?):\s*\d{4}\s*-\s*\w+ \d{1,2}, \d{4}$", t))
    return (m.group(1).strip() if m else t), None


def norm(s):
    s = re.sub(r"[^\w\s]", "", (s or "").lower().replace("&", "and"))
    return re.sub(r"\s+", " ", re.sub(r"^the\s+", "", s)).strip()


def tmdb_get(path, **params):
    token, key = os.environ.get("TMDB_READ_TOKEN"), os.environ.get("TMDB_API_KEY")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif key:
        params["api_key"] = key
    url = f"https://api.themoviedb.org/3{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=20) as r:
        return json.load(r)


def lookup_ids(title, kind, cache):
    blank = {"tmdb_id": None, "imdb_id": None, "year": None, "id_match": "none"}
    if not (os.environ.get("TMDB_READ_TOKEN") or os.environ.get("TMDB_API_KEY")):
        return blank
    ck = f"{kind}|{norm(title)}"
    if ck in cache:
        return cache[ck]
    try:
        results = tmdb_get(f"/search/{kind}", query=title, include_adult="false").get("results", [])
        q = norm(title)
        exact = [r for r in results if q in (norm(r.get("title") or r.get("name")),
                                             norm(r.get("original_title") or r.get("original_name")))]
        pool = exact or results[:1]
        if not pool:
            return blank
        pool.sort(key=lambda r: r.get("popularity") or 0, reverse=True)
        best = pool[0]
        match = "fuzzy"
        if exact:
            p = [r.get("popularity") or 0 for r in exact]
            match = "exact" if len(p) == 1 or p[0] >= 3 * p[1] else "ambiguous"
        imdb = tmdb_get(f"/{kind}/{best['id']}/external_ids").get("imdb_id") or None
        date = best.get("release_date") or best.get("first_air_date") or ""
        entry = {"tmdb_id": best["id"], "imdb_id": imdb,
                 "year": int(date[:4]) if date[:4].isdigit() else None, "id_match": match}
        if match == "exact" and imdb:
            cache[ck] = entry
        return entry
    except Exception as e:
        print(f"  id lookup failed for {title!r}: {e}")
        return blank


def build_rows(s, data, fetched_at, cache):
    rows = []
    lists = {"movies": data["movies"], "series": data["series"]}
    if s["want"] == "movies":
        lists["series"] = []
    if s["want"] == "series":
        lists["movies"] = []
    for name, items in lists.items():
        kind = "movie" if name == "movies" else "tv"
        for it in sorted(items, key=lambda i: i["rank"])[:10]:
            title, season = clean_title(it["title"])
            ids = lookup_ids(title, kind, cache)
            rows.append({
                "source": s["key"], "service": s["service"], "chart": s["chart"], "region": s["region"],
                "official": s["official"], "period_start": data["period_start"] or None,
                "period_end": data["period_end"] or None, "type": kind, "rank": it["rank"], "title": title,
                "season": season, "year": ids["year"], "imdb_id": ids["imdb_id"], "tmdb_id": ids["tmdb_id"],
                "id_match": ids["id_match"], "raw_title": it["title"], "source_url": s["url"],
                "fetched_at": fetched_at, "stale": False,
            })
    return rows


def main():
    import anthropic
    OUT.mkdir(exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    STATE.mkdir(exist_ok=True)
    cache_file = STATE / "ids.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    prev_file = OUT / "latest.json"
    prev = json.loads(prev_file.read_text()) if prev_file.exists() else {"rows": []}

    key, tok = os.environ.get("ANTHROPIC_API_KEY", ""), os.environ.get("TMDB_READ_TOKEN", "")
    if not key.startswith("sk-ant-"):
        print(f"warning: ANTHROPIC_API_KEY looks wrong (starts {key[:3]!r}, {len(key)} chars); expected 'sk-ant-...'")
    if tok and not tok.startswith("eyJ"):
        print(f"warning: TMDB_READ_TOKEN looks wrong (starts {tok[:3]!r}, {len(tok)} chars); expected 'eyJ...'")

    texts, status = fetch_pages()
    if "--fetch-only" in sys.argv:
        (STATE / "pages").mkdir(exist_ok=True)
        for k, t in texts.items():
            (STATE / "pages" / f"{k}.txt").write_text(t)
        print(json.dumps({k: len(t) for k, t in texts.items()}, indent=1), status)
        return 0

    client = anthropic.Anthropic()
    fetched_at = NOW.isoformat(timespec="seconds")
    rows = []
    for s in SOURCES:
        new = None
        if s["key"] in texts:
            try:
                data = extract(client, s, texts[s["key"]])
                issues = problems(s, data)
                if issues:
                    status[s["key"]] = "; ".join(issues)
                else:
                    new = build_rows(s, data, fetched_at, cache)
            except Exception as e:
                status[s["key"]] = f"extract failed: {e}".splitlines()[0][:200]
        if new:
            status[s["key"]] = "ok"
            rows += new
        else:
            old = [dict(r, stale=True) for r in prev["rows"] if r.get("source") == s["key"]]
            rows += old
            print(f"  {s['key']}: {status[s['key']]} (kept {len(old)} old rows)")

    doc = {"generated_at": fetched_at, "status": status, "rows": rows}
    payload = json.dumps(doc, indent=1, ensure_ascii=False)
    prev_file.write_text(payload)
    (OUT / "data" / f"{NOW:%Y-%m-%d}.json").write_text(payload)
    with open(OUT / "latest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["source"])
        w.writeheader()
        w.writerows(rows)
    cache_file.write_text(json.dumps(cache, indent=1, sort_keys=True, ensure_ascii=False))
    bad = {k: v for k, v in status.items() if v != "ok"}
    print("status:", json.dumps(status, indent=1))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
