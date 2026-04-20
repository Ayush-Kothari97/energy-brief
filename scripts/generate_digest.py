"""
generate_digest.py — The Energy Intelligence Brief
----------------------------------------------------
Runs daily at 7am IST via GitHub Actions.
Uses OpenAI GPT-4o with web_search tool to find real articles,
then writes structured cards to data/content.json.

SECURITY: OPENAI_API_KEY read from environment only.
Never written to disk, never logged, never in any file.
"""

import os, json, datetime, sys, time
from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError

# ── Key from environment only ──────────────────────────────────────────────
_key = os.environ.get("OPENAI_API_KEY", "")
if not _key:
    print("ERROR: OPENAI_API_KEY not found.", file=sys.stderr)
    sys.exit(1)
client = OpenAI(api_key=_key, timeout=90.0)
del _key

# ── Date (IST = UTC+5:30) ──────────────────────────────────────────────────
today = datetime.datetime.utcnow() + datetime.timedelta(hours=5, minutes=30)
date_str  = today.strftime("%A, %d %B %Y")
date_iso  = today.strftime("%Y-%m-%d")

# ── Tools: web search ──────────────────────────────────────────────────────
TOOLS = [{"type": "web_search_preview"}]

# ── Section configs ────────────────────────────────────────────────────────
# card_count: how many cards per section
# long_read: True = 8-10 min analytical article style
# sources: publication names for search targeting

SECTIONS = {
    "market-pulse": {
        "card_count": 5,
        "long_read": False,
        "sources": "OGJ, OilPrice.com, EIA, Platts, Argus Media",
        "prompt": """Search today's energy news ({date}) and write {n} market intelligence cards.
Cover: Brent crude price, WTI price, Henry Hub gas, a key OPEC+ development, one supply disruption.
For each card find the ACTUAL article URL from {sources}."""
    },
    "geopolitical": {
        "card_count": 4,
        "long_read": False,
        "sources": "CSIS, Columbia CGEP, OIES, Doomberg, OilPrice.com",
        "prompt": """Search today's geopolitical energy news ({date}) and write {n} analysis cards.
Cover: sanctions on energy flows, pipeline diplomacy, OPEC+ politics, one conflict affecting infrastructure.
Find ACTUAL article URLs from {sources}."""
    },
    "india": {
        "card_count": 4,
        "long_read": False,
        "sources": "India Energy Week, TERI, OilPrice.com, Argus, MoPNG",
        "prompt": """Search today's India energy news ({date}) and write {n} intelligence cards.
Cover: India crude imports, a PSU development (ONGC/BPCL/Reliance), LNG/gas news, one policy update.
Find ACTUAL article URLs from {sources}."""
    },
    "upstream": {
        "card_count": 1,
        "long_read": True,
        "sources": "SPE/JPT, Wood Mackenzie, Rystad Energy, Hart Energy, OGJ",
        "prompt": """Search for the single most important upstream oil & gas story published today or this week ({date}).
Write a detailed 8-10 minute read analytical article about it. Cover the discovery/FID/development in depth:
background context, technical details, company positions, market implications, what analysts say.
Find the ACTUAL source article URL. Write at least 600 words."""
    },
    "midstream": {
        "card_count": 1,
        "long_read": True,
        "sources": "LNG Industry, Gas Processing & LNG, Pipeline & Gas Journal, Offshore Energy",
        "prompt": """Search for the single most important midstream energy story ({date}) — LNG, pipeline, tanker markets.
Write a detailed 8-10 minute read analytical article. Cover the story in depth:
context, technical/commercial details, market implications, key players.
Find the ACTUAL source article URL. Write at least 600 words."""
    },
    "downstream": {
        "card_count": 1,
        "long_read": True,
        "sources": "Hydrocarbon Processing, OGJ Downstream, Platts, Argus",
        "prompt": """Search for the most important downstream/refining story ({date}) — margins, crack spreads, product markets.
Write a detailed 8-10 minute read analytical article covering refining margins, key drivers, regional dynamics.
Find the ACTUAL source article URL. Write at least 600 words."""
    },
    "petrochems": {
        "card_count": 1,
        "long_read": True,
        "sources": "ICIS, Hydrocarbon Processing, GlobalData",
        "prompt": """Search for the most important petrochemicals story ({date}) — ethylene, propylene, naphtha, polymers.
Write a detailed 8-10 minute read analytical article covering feedstock dynamics, margin trends, new projects.
Find the ACTUAL source article URL. Write at least 600 words."""
    },
    "og-projects": {
        "card_count": 3,
        "long_read": False,
        "sources": "Hart Energy, Upstream Online, Rystad Energy, Wood Mackenzie",
        "prompt": """Search for the top 3 oil & gas project developments from this week ({date}).
Cover: FIDs, first oil milestones, offshore awards, LNG sanctions.
Find ACTUAL article URLs from {sources}."""
    },
    "re-projects": {
        "card_count": 3,
        "long_read": False,
        "sources": "Renewables Now, reNews, Renewable Energy World",
        "prompt": """Search for the top 3 renewable energy project developments this week ({date}).
Cover: solar/wind commissionings, storage awards, offshore wind milestones.
Find ACTUAL article URLs from {sources}."""
    },
    "supply-demand": {
        "card_count": 3,
        "long_read": False,
        "sources": "IEA Oil Market Report, EIA Short-Term Energy Outlook, OPEC Monthly Oil Market Report",
        "prompt": """Search for the latest IEA, EIA, and OPEC forecast data ({date}).
Write 3 cards — one per agency — with their most recent demand/supply figures.
Find ACTUAL report URLs from {sources}."""
    },
    "frameworks": {
        "card_count": 1,
        "long_read": True,
        "sources": "McKinsey Energy, BCG, Deloitte, HBR",
        "prompt": """Choose the single most relevant strategic framework for today's energy market context ({date}).
Write a detailed 8-10 minute read explaining the framework and applying it analytically to current energy sector conditions.
Reference specific companies, numbers, and developments from this week.
Structure the article with clear sections. Find a real source URL from {sources}.
Add a note: [AI-generated analysis]"""
    },
    "narratives": {
        "card_count": 3,
        "long_read": False,
        "sources": "Canary Media, Carbon Brief, BloombergNEF, Gerard Reid",
        "prompt": """Search for the top 3 energy transition narrative stories this week ({date}).
Cover: the dominant transition story, a policy/net-zero development, one key analysis piece.
Find ACTUAL article URLs from {sources}."""
    },
    "hydrogen": {
        "card_count": 3,
        "long_read": False,
        "sources": "H2 Bulletin, Hydrogen Insight, H2Tech, Hydrogen Council",
        "prompt": """Search for the top 3 hydrogen economy developments this week ({date}).
Cover: project FIDs or milestones, electrolyser/cost developments, offtake agreements.
Find ACTUAL article URLs from {sources}."""
    },
    "nuclear": {
        "card_count": 3,
        "long_read": False,
        "sources": "World Nuclear News, NEI Magazine, Energy Storage News",
        "prompt": """Search for the top 3 nuclear & storage developments this week ({date}).
Cover: SMR milestones, grid-scale battery contracts, nuclear policy/financing news.
Find ACTUAL article URLs from {sources}."""
    },
    "ccus": {
        "card_count": 3,
        "long_read": False,
        "sources": "Carbon Capture Journal, Global CCS Institute, Carbon Brief",
        "prompt": """Search for the top 3 CCUS & carbon market developments this week ({date}).
Cover: capture project updates, carbon credit/ETS prices, DAC or industrial CCS news.
Find ACTUAL article URLs from {sources}."""
    },
    "clean-capital": {
        "card_count": 3,
        "long_read": False,
        "sources": "BloombergNEF, IRENA, RMI, IEEFA, Carbon Tracker",
        "prompt": """Search for the top 3 clean energy investment developments this week ({date}).
Cover: major deals or fundraises, green bond issuances, BNEF/IRENA investment flow data.
Find ACTUAL article URLs from {sources}."""
    },
}

# ── System prompt ──────────────────────────────────────────────────────────
SYSTEM_SHORT = """You are a senior energy intelligence analyst. Today is {date}.
Use web search to find REAL articles. Return ONLY valid JSON — no markdown, no backticks.

Return: {{"cards": [...]}}
Each card:
{{
  "title": "Specific analytical headline (max 12 words)",
  "source": "Publication name only",
  "source_url": "ACTUAL article URL you found via search (not homepage)",
  "body": "2-3 analytical sentences with specific numbers, dates, company names. May use <strong><em><u> tags."
}}"""

SYSTEM_LONG = """You are a senior energy intelligence analyst. Today is {date}.
Use web search to find the most important story. Return ONLY valid JSON — no markdown, no backticks.

Return: {{"cards": [{{
  "title": "Headline (max 15 words)",
  "source": "Publication name",
  "source_url": "ACTUAL article URL found via search",
  "body": "Full 8-10 minute read article (600+ words). Use <p> tags for paragraphs. Use <strong> for key terms. Structure with clear analytical sections. End with a 'Key Takeaways' paragraph."
}}]}}"""

# ── Generate one section ───────────────────────────────────────────────────
def generate_section(sid, config, max_retries=3):
    long_read = config["long_read"]
    n = config["card_count"]
    prompt = config["prompt"].format(
        date=date_str, n=n, sources=config["sources"]
    )
    system = (SYSTEM_LONG if long_read else SYSTEM_SHORT).format(date=date_str)

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt}
                ],
                tools=TOOLS,
                temperature=0.3,
                max_tokens=4000 if long_read else 2000,
            )

            # Extract text from response (may include tool_use blocks)
            raw = ""
            for block in resp.choices[0].message.content if isinstance(
                resp.choices[0].message.content, list
            ) else []:
                if hasattr(block, 'text'):
                    raw += block.text
            if not raw:
                raw = resp.choices[0].message.content or ""
            if isinstance(raw, list):
                raw = " ".join(
                    b.get("text","") if isinstance(b,dict) else str(b) for b in raw
                )
            raw = raw.strip()

            # Strip any markdown fences
            import re
            raw = re.sub(r'^```(?:json)?\s*', '', raw)
            raw = re.sub(r'\s*```$', '', raw)

            parsed = json.loads(raw)
            if isinstance(parsed, list):
                cards = parsed
            elif isinstance(parsed, dict):
                cards = parsed.get("cards") or next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
            else:
                cards = []

            clean = []
            for c in cards[:15]:
                if not isinstance(c, dict): continue
                clean.append({
                    "title":      str(c.get("title","Untitled")).strip(),
                    "source":     str(c.get("source","")).strip(),
                    "source_url": str(c.get("source_url","#")).strip(),
                    "body":       str(c.get("body","")).strip(),
                    "long_read":  long_read,
                })
            return clean

        except RateLimitError:
            wait = 30 * (attempt + 1)
            print(f"\n    Rate limit — waiting {wait}s...", flush=True)
            time.sleep(wait)
        except (APITimeoutError, APIConnectionError):
            wait = 15 * (attempt + 1)
            print(f"\n    Timeout — waiting {wait}s...", flush=True)
            time.sleep(wait)
        except json.JSONDecodeError as e:
            print(f"\n    JSON error attempt {attempt+1}: {e}", flush=True)
            if attempt == max_retries - 1: return []
            time.sleep(5)
        except Exception as e:
            print(f"\n    Error attempt {attempt+1}: {type(e).__name__}", flush=True)
            if attempt == max_retries - 1: return []
            time.sleep(10)
    return []

# ── Main loop ──────────────────────────────────────────────────────────────
print(f"The Energy Intelligence Brief — {date_str}")
print(f"Generating {len(SECTIONS)} sections with web search...\n")

output = {
    "last_updated": datetime.datetime.utcnow().isoformat() + "Z",
    "date":         date_iso,
    "date_display": date_str,
    "sections":     {}
}

DELAY = 4  # seconds between sections

for i, (sid, config) in enumerate(SECTIONS.items(), 1):
    label = "long-read" if config["long_read"] else f"{config['card_count']} cards"
    print(f"  [{i:02d}/{len(SECTIONS)}] {sid} ({label})...", end=" ", flush=True)
    cards = generate_section(sid, config)
    output["sections"][sid] = {"cards": cards}
    print(f"✓ {len(cards)}" if cards else "✗ empty")

    if not cards:
        print(f"         Retrying {sid} in 20s...")
        time.sleep(20)
        cards = generate_section(sid, config)
        output["sections"][sid] = {"cards": cards}
        print(f"         Retry: {len(cards)} cards")

    if i < len(SECTIONS):
        time.sleep(DELAY)

# ── Write output ───────────────────────────────────────────────────────────
out_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "content.json")
)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

total = sum(len(s["cards"]) for s in output["sections"].values())
empty = [k for k,v in output["sections"].items() if not v["cards"]]
print(f"\n{'='*50}")
print(f"Complete: {total} cards, {len(SECTIONS)} sections.")
if empty: print(f"Empty sections: {', '.join(empty)}")
else: print("All sections populated.")
print(f"Saved: {out_path}")
