#!/usr/bin/env python3
"""
Dyrenytt – daglig episode-generator.
Kjøres av GitHub Actions: henter nyheter -> lager manus -> syntetiserer (Piper)
-> lager MP3 -> oppdaterer episodes.json og RSS-feed (docs/feed.xml).

Miljøvariabler (settes av workflow):
  SITE_BASE        f.eks. https://brukernavn.github.io/dyrenytt-podcast
  GITHUB_REPOSITORY  "owner/repo" (settes automatisk av Actions)
  RELEASE_TAG      tag der lydfiler lastes opp som assets (default: episodes)
  KEEP_EPISODES    antall episoder å beholde i feeden (default: 30)
  ANTHROPIC_API_KEY (valgfri) – gir LLM-generert manus i stedet for mal
  ANTHROPIC_MODEL  (valgfri) – default claude-3-5-haiku-latest
  MOCK_NEWS        "1" for å bruke innebygde testnyheter (lokal testing)
"""
import os, sys, json, re, subprocess, tempfile, wave, tarfile, urllib.request, html
from datetime import datetime, timezone, timedelta
from email.utils import format_datetime

import numpy as np

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS   = os.path.join(ROOT, "docs")
OUT    = os.path.join(ROOT, "out")
MODEL_DIR = os.path.join(ROOT, "model", "vits-piper-no_NO-talesyntese-medium")
os.makedirs(OUT, exist_ok=True)

SITE_BASE   = os.environ.get("SITE_BASE", "https://example.github.io/dyrenytt-podcast").rstrip("/")
REPO        = os.environ.get("GITHUB_REPOSITORY", "owner/dyrenytt-podcast")
RELEASE_TAG = os.environ.get("RELEASE_TAG", "episodes")
KEEP        = int(os.environ.get("KEEP_EPISODES", "30"))
TZ          = timezone(timedelta(hours=2))  # Europe/Oslo (sommertid); vinter = +1

SR = 22050
LENGTH_SCALE = 1.23
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "tts-models/vits-piper-no_NO-talesyntese-medium.tar.bz2")

MONTHS = ["januar","februar","mars","april","mai","juni","juli","august",
          "september","oktober","november","desember"]
DAYS   = ["mandag","tirsdag","onsdag","torsdag","fredag","lørdag","søndag"]

# ---------------------------------------------------------------- nyheter
def _gnews(query, hl="no", gl="NO", ceid="NO:no", days=3):
    from urllib.parse import quote
    q = quote(f"{query} when:{days}d")
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

# Målrettede nyhetsstrømmer. cat styrer prioritet: chip/ID øverst, så kjæledyr,
# så regelverk/velferd. Vi henter både norsk og EU/engelsk (hl/gl/ceid).
# (cat, query, hl, gl, ceid, dager, ønsket antall)
TOPICS = [
    # --- Chip-/ID-merking og sporbarhet – Norge (bred tidsvindu, regelverk kommer sjelden) ---
    ("chip", 'chipmerking OR ID-merking OR mikrochip OR "obligatorisk merking" (hund OR katt OR kjæledyr OR sporbarhet OR DyreID OR regelverk OR forskrift)',
     "no", "NO", "NO:no", 30, 3),
    # --- Chip-/ID-merking og sporbarhet – EU/Europa (engelsk) ---
    ("chip", 'pet microchipping OR "compulsory microchipping" OR animal identification OR pet passport OR animal traceability (EU OR Europe OR regulation OR directive OR law)',
     "en-GB", "GB", "GB:en", 45, 3),
    # --- Kjæledyr-nyheter – Norge (kort vindu, ferske saker) ---
    ("pet", 'kjæledyr OR hund OR katt OR kanin (helse OR veterinær OR dyrlege OR advarsel OR tilbakekalling OR sykdom OR dyrevelferd)',
     "no", "NO", "NO:no", 4, 4),
    # --- Kjæledyr-nyheter – Europa/internasjonalt (engelsk) ---
    ("pet", 'dog OR cat OR pet (health OR recall OR disease OR welfare OR EU regulation)',
     "en-GB", "GB", "GB:en", 4, 2),
    # --- Regelverk og dyrevelferd – Norge ---
    ("reg", '(dyrevelferd OR Mattilsynet OR dyrehelse) (kjæledyr OR hund OR katt OR regelverk OR forskrift OR "ny lov" OR forbud OR krav)',
     "no", "NO", "NO:no", 14, 2),
]
GENERAL_TOPICS = [("Norge nyheter", "no", "NO", "NO:no", 1)]

def _clean(title):
    title = html.unescape(re.sub("<[^>]+>", "", title or "")).strip()
    source = ""
    if " - " in title:                     # Google News: "Tittel - Kilde"
        base, source = title.rsplit(" - ", 1)
        title = base.strip(); source = source.strip()
    return title, source

def fetch_news():
    if os.environ.get("MOCK_NEWS") == "1":
        return MOCK_ANIMAL, MOCK_GENERAL
    import feedparser
    seen = set()
    def norm(t): return re.sub(r"\s+", " ", t.lower()).strip()
    buckets = {"chip": [], "pet": [], "reg": []}
    for cat, q, hl, gl, ceid, days, want in TOPICS:
        try:
            d = feedparser.parse(_gnews(q, hl, gl, ceid, days))
        except Exception:
            continue
        got = 0
        for e in d.entries:
            title, tsrc = _clean(e.get("title", ""))
            if not title or len(title) < 8:
                continue
            k = norm(title)
            if k in seen:
                continue
            summary, _ = _clean(e.get("summary", ""))
            src = tsrc
            if not src and isinstance(e.get("source"), dict):
                src = e["source"].get("title", "")
            seen.add(k)
            buckets[cat].append({"title": title, "summary": summary[:300], "source": src, "cat": cat})
            got += 1
            if got >= want:
                break
    # Prioritert rekkefølge: chip/ID først, så kjæledyr, så regelverk.
    animal = buckets["chip"] + buckets["pet"] + buckets["reg"]

    general = []
    for q, hl, gl, ceid, days in GENERAL_TOPICS:
        try:
            d = feedparser.parse(_gnews(q, hl, gl, ceid, days))
        except Exception:
            continue
        for e in d.entries[:5]:
            title, tsrc = _clean(e.get("title", ""))
            if title and norm(title) not in seen:
                seen.add(norm(title))
                general.append({"title": title, "summary": _clean(e.get("summary", ""))[0][:200], "source": tsrc})
            if len(general) >= 2:
                break

    if not animal:                          # nødfallback
        return MOCK_ANIMAL, MOCK_GENERAL
    return animal[:8], general[:2]

MOCK_ANIMAL = [
    {"title":"EU strammer inn krav til chipmerking av hund og katt","summary":"Nytt europeisk regelverk vil kreve obligatorisk ID-merking og registrering for bedre sporbarhet og oppfølging av dyrevelferd. Endringene ventes å påvirke også norske kjæledyreiere.","source":"European Commission","cat":"chip"},
    {"title":"Krav om ID-merking av katt vurderes i Norge","summary":"Myndighetene ser på om obligatorisk chipmerking og registrering bør utvides fra hund til også å gjelde katt, for å redusere antall hjemløse dyr.","source":"Mattilsynet","cat":"chip"},
    {"title":"Fugleinfluensa hos villfugl – lav risiko for kjæledyr","summary":"Eiere bør likevel holde hund og katt unna syke og døde fugler.","source":"Veterinærinstituttet","cat":"pet"},
    {"title":"Veterinærer advarer mot varmen for hund","summary":"Hunder svetter ikke og overopphetes lett på varme dager.","source":"NKK","cat":"pet"},
]
MOCK_GENERAL = [
    {"title":"Mye regn meldt på Vestlandet","summary":"Meteorologene varsler kraftig nedbør.","source":"NRK"},
    {"title":"Signalfeil gir togtrøbbel i Oslo-området","summary":"Bane Nor melder om forsinkelser.","source":"NRK"},
]

# ---------------------------------------------------------------- vær (Oslo)
_WEATHER_NB = {
    "clearsky":"klart", "fair":"lettskyet", "partlycloudy":"delvis skyet", "cloudy":"skyet",
    "fog":"tåke", "lightrain":"lett regn", "rain":"regn", "heavyrain":"kraftig regn",
    "lightrainshowers":"lette regnbyger", "rainshowers":"regnbyger", "heavyrainshowers":"kraftige regnbyger",
    "lightsleet":"lett sludd", "sleet":"sludd", "heavysleet":"kraftig sludd",
    "lightsnow":"lett snø", "snow":"snø", "heavysnow":"kraftig snø",
    "snowshowers":"snøbyger", "lightsnowshowers":"lette snøbyger",
    "lightrainshowersandthunder":"regnbyger og torden", "rainandthunder":"regn og torden",
}
def _wsym(code):
    base = (code or "").split("_")[0]
    return _WEATHER_NB.get(base, "")

def fetch_weather(lat=59.913, lon=10.752):   # Oslo sentrum
    if os.environ.get("MOCK_NEWS") == "1":
        return "delvis skyet, mellom fjorten og tjueén grader"
    try:
        import urllib.request
        url = f"https://api.met.no/weatherapi/locationforecast/2.0/compact?lat={lat}&lon={lon}"
        contact = os.environ.get("FEED_EMAIL") or "dyrenytt@example.com"
        req = urllib.request.Request(url, headers={
            "User-Agent": f"Dyrenytt-podcast/1.0 ({contact})"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        ts = data["properties"]["timeseries"]
        today = datetime.now(TZ).date()
        temps, symbol = [], None
        for entry in ts:
            dt = datetime.fromisoformat(entry["time"].replace("Z", "+00:00")).astimezone(TZ)
            if dt.date() > today:
                break
            if dt.date() != today:
                continue
            temps.append(entry["data"]["instant"]["details"]["air_temperature"])
            if 11 <= dt.hour <= 15 and not symbol:
                nb = entry["data"].get("next_6_hours") or entry["data"].get("next_1_hours") or {}
                symbol = nb.get("summary", {}).get("symbol_code")
        if not temps:
            return None
        lo, hi = round(min(temps)), round(max(temps))
        desc = _wsym(symbol)
        return (f"{desc}, " if desc else "") + f"mellom {lo} og {hi} grader"
    except Exception as e:
        print(f"Vær utilgjengelig ({e}).")
        return None

# ---------------------------------------------------------------- manus
SYSTEM_PROMPT = """Du er manusforfatter for en kort, daglig norsk podkast som heter «Dyrenytt».
To programledere: Kari (lysere stemme) og Tom (dypere stemme). Målgruppen hører på mens de løper eller pendler til jobb.
Skriv en naturlig, vennlig dialog på norsk bokmål, ca. 1500–1800 ord, som varer rundt 9–10 minutter.
Hoveddelen skal handle om dyrehelse og kjæledyr; avslutt med 1–2 korte generelle nyheter og en kort værmelding for Oslo i dag (kun én setning helt til slutt).
PRIORITER saker om ID-merking og chipmerking av kjæledyr, og nytt regelverk – også fra EU og utlandet – samt sporbarhet og dyrevelferd. Dette er kjerneinteressen for lytterne. Ta gjerne med chip- og dyrevelferdsnyheter fra utlandet, og forklar hva EU-regler kan bety for norske kjæledyreiere. Gi mindre plass til produksjonsdyr, vilt og skadedyr med mindre saken er stor. Saker merket [ID/CHIP-MERKING] og [REGELVERK] i listen skal løftes fram først. IKKE nevn hvem som lager eller står bak podkasten.
Start med en kort intro med dagens dato, avslutt med en vennlig outro. Vær konkret og praktisk, gi gjerne råd til kjæledyreiere.

VIKTIG – teksten leses opp av en norsk talesyntese (Piper), som uttaler engelsk og forkortelser feil. Skriv derfor talesyntese-vennlig:
- Unngå engelske ord der det finnes norske.
- Skriv alle tall og årstall med bokstaver (f.eks. «2026» → «to tusen og tjueseks», «15 %» → «femten prosent», «kl. 07» → «klokka sju»).
- Egennavn, engelske ord og forkortelser som kan uttales feil, skriver du lydnært med norsk rettskriving. Eksempler: «Adequan» → «Adekvan», «Long Island» → «Long Æiland», «USA» → «U-ess-A», «WHO» → «Vé-Há-O», «zoonose» → «soonoose». Bruk skjønn på nye ord etter samme mønster.
- Skriv forkortelser helt ut (f.eks. «bl.a.» → «blant annet», «f.eks.» → «for eksempel»).
- Ikke bruk kolon, semikolon, parenteser, tankestrek eller spesialtegn. Bruk kun vanlige punktum, komma, spørsmålstegn og utropstegn.
- Del opp lange setninger i kortere, så tonefallet flyter naturlig.

Ikke bruk overskrifter, emojier eller punktlister.
Returner KUN gyldig JSON: en liste av objekter {"speaker","text"}, der speaker er "K" eller "T".
Sett "seg": true på replikker som starter et nytt tema (gir lengre pause)."""

# Naturlig variant – brukes med Gemini TTS, som uttaler ord riktig selv.
SYSTEM_PROMPT_NATURAL = """Du er manusforfatter for en kort, daglig norsk podkast som heter «Dyrenytt».
To programledere: Kari og Tom. Målgruppen hører på mens de løper eller pendler til jobb.
Skriv en naturlig, vennlig dialog på norsk bokmål, ca. 1500–1800 ord, som varer rundt 9–10 minutter.
Hoveddelen skal handle om dyrehelse og kjæledyr; avslutt med 1–2 korte generelle nyheter og en kort værmelding for Oslo i dag (kun én setning helt til slutt).
PRIORITER saker om ID-merking og chipmerking av kjæledyr, og nytt regelverk – også fra EU og utlandet – samt sporbarhet og dyrevelferd. Dette er kjerneinteressen for lytterne. Ta gjerne med chip- og dyrevelferdsnyheter fra utlandet, og forklar hva EU-regler kan bety for norske kjæledyreiere. Gi mindre plass til produksjonsdyr, vilt og skadedyr med mindre saken er stor. Saker merket [ID/CHIP-MERKING] og [REGELVERK] i listen skal løftes fram først. IKKE nevn hvem som lager eller står bak podkasten.
Start med en kort intro med dagens dato, avslutt med en vennlig outro. Vær konkret og praktisk, gi gjerne råd til kjæledyreiere.
Skriv helt naturlig norsk – du trenger IKKE lydskrive ord eller unngå tall og forkortelser, stemmen uttaler dette riktig.
Ikke bruk overskrifter, emojier eller punktlister.
Returner KUN gyldig JSON: en liste av objekter {"speaker","text"}, der speaker er "K" eller "T".
Sett "seg": true på replikker som starter et nytt tema (gir lengre pause)."""

def build_script_llm(dato_str, animal, general, natural=False, weather=None):
    import urllib.request
    key = os.environ["ANTHROPIC_API_KEY"]
    model = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
    system = SYSTEM_PROMPT_NATURAL if natural else SYSTEM_PROMPT
    def fmt(items):
        lab = {"chip":"[ID/CHIP-MERKING] ", "pet":"[KJÆLEDYR] ", "reg":"[REGELVERK] "}
        return "\n".join(f"- {lab.get(i.get('cat',''),'')}{i['title']}. {i['summary']} (Kilde: {i.get('source','')})" for i in items)
    weather_line = f"\n\nVÆR I OSLO I DAG (nevn kort, kun én setning helt til slutt): {weather}" if weather else ""
    user = (f"Dato: {dato_str}.\n\nDYREHELSE-/KJÆLEDYR-NYHETER:\n{fmt(animal)}\n\n"
            f"GENERELLE NYHETER (bruk 1–2 kort til slutt):\n{fmt(general)}"
            f"{weather_line}\n\n"
            "Skriv episoden nå som JSON.")
    body = json.dumps({
        "model": model, "max_tokens": 8000,
        "system": system,
        "messages": [{"role":"user","content":user}],
    }).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
        headers={"content-type":"application/json","x-api-key":key,"anthropic-version":"2023-06-01"})
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    txt = "".join(b.get("text","") for b in resp.get("content",[]))
    m = re.search(r"\[.*\]", txt, re.S)
    data = json.loads(m.group(0))
    out = []
    for d in data:
        spk = "K" if str(d.get("speaker","K")).upper().startswith("K") else "T"
        if d.get("seg"): spk = "SEG_" + spk
        t = str(d.get("text","")).strip()
        if t: out.append((spk, t))
    if len(out) < 6:
        raise ValueError("LLM-manus for kort")
    return out

LEADS = ["Vi starter med dagens viktigste sak.", "Neste sak.", "Vi går videre.",
         "Så til noe annet.", "Her er en sak til.", "Og enda en."]
REACTS = ["Godt å vite.", "Det var nyttig.", "Verdt å merke seg.",
          "Bra å ha med seg.", "Interessant.", "Takk, det var oppklarende."]
TIPS = [
 "Dagens tips handler om tenner. Tannstein og betennelse er blant de mest oversette helseproblemene hos hund og katt, og gir smerte og dårlig ånde. Daglig pussing med tannbørste og tannkrem laget for dyr hjelper mye, og be veterinæren sjekke tennene på den årlige helsesjekken.",
 "Dagens tips er om varme. Hunder svetter ikke slik vi gjør, de kvitter seg med varme ved å pese. På varme dager, legg turen til tidlig morgen eller sen kveld, ta med vann, og test asfalten med håndbaken. Er den for varm for hånda di, er den for varm for potene.",
 "Dagens tips gjelder ID-merking. Sørg for at hunden eller katten er ID-merket og registrert, og at kontaktopplysningene dine er oppdaterte. Det er den beste sjansen for å bli gjenforent hvis dyret kommer på avveie.",
 "Dagens tips er om vekt. Overvekt er et av de vanligste helseproblemene hos kjæledyr og sliter på ledd og indre organer. Vei fôret, vær forsiktig med godbiter, og kjenn jevnlig etter ribbeina. Du skal kunne kjenne dem uten å trykke hardt.",
 "Dagens tips handler om flått. I sesongen bør du sjekke pelsen etter turer, særlig rundt hode, ører og poter. Bruk gjerne forebyggende middel, og fjern flått raskt med en flåttfjerner rett mot huden.",
]

def build_script_template(dato_str, animal, general, weather=None):
    import random
    seed = sum(ord(c) for c in dato_str)
    rnd = random.Random(seed)
    D = []
    D.append(("K", f"God morgen, og velkommen til Dyrenytt, din daglige dose nyheter om dyrehelse og kjæledyr. Det er {dato_str}. Jeg heter Kari."))
    D.append(("T", "Og jeg heter Tom. Enten du snører løpeskoene eller står i kø ved kaffemaskinen, så tar vi det viktigste du bør vite i dag. Vi bruker noen minutter sammen, så er du oppdatert før du er framme på jobb."))
    speakers = ["T","K"]
    for idx, it in enumerate(animal):
        s = speakers[idx % 2]; other = speakers[(idx+1) % 2]
        lead = LEADS[idx] if idx < len(LEADS) else "Og en sak til."
        summ = it.get("summary") or ""
        D.append(("SEG_"+s, f"{lead} {it['title']}." + (f" {summ}" if summ else "")))
        D.append((other, rnd.choice(REACTS) + " Da går vi videre."))
    # roterende dagens tips
    D.append(("SEG_K", "Før vi runder av dyredelen, tar vi dagens lille tips."))
    D.append(("T", TIPS[seed % len(TIPS)]))
    D.append(("K", "Et godt og lavterskel råd."))
    if general:
        D.append(("SEG_K", "Og helt til slutt, litt fra nyhetsbildet ellers, for det skjer jo mer i verden enn bare dyr."))
        for it in general[:2]:
            summ = it.get("summary") or ""
            D.append(("T", f"{it['title']}." + (f" {summ}" if summ else "")))
    if weather:
        D.append(("SEG_K", f"Og været i Oslo i dag: {weather}."))
        D.append(("T", "Kle deg etter forholdene."))
    D.append(("SEG_K", f"Og det var Dyrenytt for i dag, {dato_str}. Takk for at du løp, eller gikk, sammen med oss."))
    D.append(("T", "Ha en riktig fin dag, så høres vi igjen i morgen tidlig."))
    return D

# ---------------------------------------------------------------- lyd
# Sikkerhetsnett for ord som talesyntesen ofte bommer på. Sonnet skriver
# stort sett talesyntese-vennlig selv, men dette fanger opp resten (og hjelper
# mal-manuset som leser rå nyhetstekst). Utvid gjerne med egne ord.
REPLACE = {
    "Adequan":"Adekvan","American Regent":"Amerikan Rídsjent","Long Island":"Long Æiland",
    "zoonose":"soonoose","zoonoser":"soonooser",
    "USA":"U-ess-A","EU":"E-U","FN":"Ef-En","WHO":"Vé-Há-O","NATO":"Nato",
    "NRK":"En-Err-Kå","NTNU":"En-Te-En-U","NKK":"En-Kå-Kå",
    "API":"Á-Pí-Í","RSS":"ærr-ess-ess","HPAI":"H-P-A-I","DNA":"De-En-A",
    "e.coli":"e-koli","E.coli":"e-koli",
}
def preprocess(text):
    for a,b in REPLACE.items():
        text = re.sub(r"\b"+re.escape(a)+r"\b", b, text)
    text = re.sub(r":\s*([a-zæøå])", lambda m: ". "+m.group(1).upper(), text)
    text = text.replace(":", ".")
    for d in (" – "," — "):
        text = text.replace(d, ", ")
    text = text.replace("–", ",").replace("—", ",")
    return text

FILT = {
    "K": f"asetrate={SR}*1.12,aresample={SR},atempo=0.8929",
    "T": f"asetrate={SR}*0.94,aresample={SR},atempo=1.0638",
}

def ensure_model():
    onnx = os.path.join(MODEL_DIR, "no_NO-talesyntese-medium.onnx")
    if os.path.exists(onnx):
        return
    os.makedirs(os.path.dirname(MODEL_DIR), exist_ok=True)
    tb = os.path.join(ROOT, "model", "voice.tar.bz2")
    print("Laster ned Piper-modell...")
    urllib.request.urlretrieve(MODEL_URL, tb)
    with tarfile.open(tb, "r:bz2") as t:
        t.extractall(os.path.join(ROOT, "model"))
    os.remove(tb)

def synthesize(dialog, mp3_path, title):
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    ensure_model()
    voice = PiperVoice.load(os.path.join(MODEL_DIR, "no_NO-talesyntese-medium.onnx"),
                            config_path=os.path.join(MODEL_DIR, "no_NO-talesyntese-medium.onnx.json"))
    def synth_line(text):
        fd, p = tempfile.mkstemp(suffix=".wav"); os.close(fd)
        with wave.open(p,"wb") as w:
            voice.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=LENGTH_SCALE))
        return p
    def process(p, spk):
        o = p.replace(".wav","_p.wav")
        subprocess.run(["ffmpeg","-y","-loglevel","error","-i",p,"-af",FILT[spk],o], check=True)
        with wave.open(o,"rb") as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)/32768.0
        os.remove(p); os.remove(o)
        return data
    pieces=[]; sil_s=np.zeros(int(SR*0.42),dtype=np.float32); sil_seg=np.zeros(int(SR*0.90),dtype=np.float32)
    for spk,text in dialog:
        seg = spk.startswith("SEG_"); s = spk.split("_")[-1] if seg else spk
        audio = process(synth_line(preprocess(text)), s)
        if pieces: pieces.append(sil_seg if seg else sil_s)
        pieces.append(audio)
    full = np.concatenate(pieces)
    full = full*(10**(-1.5/20)/(np.max(np.abs(full)) or 1.0))
    dur = len(full)/SR
    master = os.path.join(OUT,"master.wav")
    with wave.open(master,"wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(full,-1,1)*32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",master,
        "-af",f"afade=t=in:st=0:d=0.4,afade=t=out:st={round(dur-0.5,2)}:d=0.5",
        "-codec:a","libmp3lame","-b:a","128k",
        "-metadata",f"title={title}","-metadata","artist=Dyrenytt","-metadata","album=Dyrenytt",
        mp3_path], check=True)
    os.remove(master)
    return int(round(dur))

# ---------------------------------------------------------------- feed
def load_state():
    p = os.path.join(DOCS,"episodes.json")
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return []

def save_state(eps):
    json.dump(eps, open(os.path.join(DOCS,"episodes.json"),"w",encoding="utf-8"),
              ensure_ascii=False, indent=2)

FEED_EMAIL = os.environ.get("FEED_EMAIL", "dyrenytt@example.com")
FEED_DESC  = "Daglig dose nyheter om dyrehelse og kjæledyr – rundt ti minutter, laget for veien til jobb."

def _hms(sec):
    h, rem = divmod(int(sec), 3600); m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

def build_feed(eps):
    from xml.sax.saxutils import escape as esc
    items = []
    for ep in eps:  # eps er nyeste først – riktig rekkefølge i RSS
        items.append(f"""    <item>
      <title>{esc(ep['title'])}</title>
      <description>{esc(ep.get('desc',''))}</description>
      <itunes:summary>{esc(ep.get('desc',''))}</itunes:summary>
      <enclosure url="{esc(ep['url'])}" length="{ep['size']}" type="audio/mpeg"/>
      <guid isPermaLink="false">{esc(ep['url'])}</guid>
      <pubDate>{esc(ep['pubdate'])}</pubDate>
      <itunes:duration>{_hms(ep['duration'])}</itunes:duration>
      <itunes:explicit>no</itunes:explicit>
    </item>""")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:atom="http://www.w3.org/2005/Atom">
  <channel>
    <title>Dyrenytt</title>
    <link>{esc(SITE_BASE)}</link>
    <atom:link href="{esc(SITE_BASE)}/feed.xml" rel="self" type="application/rss+xml"/>
    <description>{esc(FEED_DESC)}</description>
    <language>no</language>
    <copyright>Dyrenytt</copyright>
    <itunes:author>Dyrenytt</itunes:author>
    <itunes:summary>{esc(FEED_DESC)}</itunes:summary>
    <itunes:type>episodic</itunes:type>
    <itunes:owner><itunes:name>Dyrenytt</itunes:name><itunes:email>{esc(FEED_EMAIL)}</itunes:email></itunes:owner>
    <itunes:image href="{esc(SITE_BASE)}/cover.png"/>
    <itunes:category text="News"/>
    <itunes:explicit>no</itunes:explicit>
    <image><url>{esc(SITE_BASE)}/cover.png</url><title>Dyrenytt</title><link>{esc(SITE_BASE)}</link></image>
{chr(10).join(items)}
  </channel>
</rss>
"""
    open(os.path.join(DOCS,"feed.xml"),"w",encoding="utf-8").write(xml)

# ---------------------------------------------------------------- Gemini TTS
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash-preview-tts")
V_KARI = os.environ.get("GEMINI_VOICE_KARI", "Kore")     # lysere/kvinnelig
V_TOM  = os.environ.get("GEMINI_VOICE_TOM",  "Charon")   # dypere/mannlig
GEMINI_STYLE = os.environ.get("GEMINI_STYLE",
    "Les dette som to varme, tydelige norske programledere i en morgenpodkast, med naturlig tempo og tonefall.")
GEMINI_SR = 24000

def _gemini_tts_single(text, voice, timeout=300, attempts=3):
    """Syntetiser ÉN stemme (enkelt-taler). Fast voiceName = samme stemme hver gang."""
    import urllib.request, urllib.error, base64, time
    key = os.environ["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
    body = json.dumps({
        "contents":[{"parts":[{"text":text}]}],
        "generationConfig":{
            "responseModalities":["AUDIO"],
            "speechConfig":{"voiceConfig":{"prebuiltVoiceConfig":{"voiceName":voice}}},
        },
    }).encode()
    last = None
    for a in range(attempts):
        try:
            req = urllib.request.Request(url, data=body, method="POST",
                headers={"content-type":"application/json","x-goog-api-key":key})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.load(r)
            return base64.b64decode(resp["candidates"][0]["content"]["parts"][0]["inlineData"]["data"])
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "ignore")[:400]
            last = f"HTTP {e.code}: {detail}"
            if e.code in (400, 401, 403, 404):  # klientfeil – nytter ikke å prøve igjen
                break
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        if a < attempts - 1:
            time.sleep(4 * (a + 1))
    raise RuntimeError(last or "ukjent Gemini-feil")

def _speaker_runs(dialog, max_chars=600):
    """Slå sammen påfølgende replikker fra samme person, men bytt aldri stemme midt i."""
    runs = []  # (spk 'K'/'T', text, is_seg)
    for spk, text in dialog:
        seg = spk.startswith("SEG_")
        s = spk.split("_")[-1]
        if runs and runs[-1][0] == s and not seg and len(runs[-1][1]) + len(text) < max_chars:
            p = runs[-1]; runs[-1] = (s, p[1] + " " + text, p[2])
        else:
            runs.append((s, text, seg))
    return runs

def build_audio_gemini(dialog, mp3_path, title):
    # Én fast stemme per person gjennom HELE episoden (enkelt-taler per replikk).
    runs = _speaker_runs(dialog)
    print(f"  Gemini: {len(runs)} replikker å syntetisere (Kari={V_KARI}, Tom={V_TOM})...")
    pcm = bytearray()
    sil       = b"\x00\x00" * int(GEMINI_SR * 0.30)
    sil_seg   = b"\x00\x00" * int(GEMINI_SR * 0.60)
    for i, (s, text, seg) in enumerate(runs):
        voice = V_KARI if s == "K" else V_TOM
        data = _gemini_tts_single(text, voice)
        if pcm: pcm += (sil_seg if seg else sil)
        pcm += data
        print(f"  replikk {i+1}/{len(runs)} ({'Kari' if s=='K' else 'Tom'}) ok")
    wav = os.path.join(OUT, "gemini.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(GEMINI_SR); w.writeframes(bytes(pcm))
    dur = len(pcm) / 2 / GEMINI_SR
    fo = max(0.1, round(dur - 0.5, 2))
    subprocess.run(["ffmpeg","-y","-loglevel","error","-i",wav,
        "-af",f"afade=t=in:st=0:d=0.3,afade=t=out:st={fo}:d=0.5",
        "-codec:a","libmp3lame","-b:a","128k",
        "-metadata",f"title={title}","-metadata","artist=Dyrenytt","-metadata","album=Dyrenytt",
        mp3_path], check=True)
    os.remove(wav)
    return int(round(dur))

# ---------------------------------------------------------------- main
def main():
    now = datetime.now(TZ)
    dato_str = f"{DAYS[now.weekday()]} {now.day}. {MONTHS[now.month-1]}"
    date_id = now.strftime("%Y-%m-%d")
    title = f"Dyrenytt – {dato_str}"
    fname = f"dyrenytt-{date_id}.mp3"
    mp3_path = os.path.join(OUT, fname)

    animal, general = fetch_news()
    weather = fetch_weather()
    print(f"Hentet {len(animal)} dyre-saker, {len(general)} generelle. Vær: {weather or 'utilgjengelig'}.")

    engine = "gemini" if os.environ.get("GEMINI_API_KEY") else "piper"

    try:
        if os.environ.get("ANTHROPIC_API_KEY"):
            dialog = build_script_llm(dato_str, animal, general, natural=(engine == "gemini"), weather=weather)
            print("Manus: LLM.")
        else:
            raise KeyError("ingen nøkkel")
    except Exception as e:
        print(f"Bruker mal-manus ({e}).")
        dialog = build_script_template(dato_str, animal, general, weather=weather)

    if engine == "gemini":
        try:
            dur = build_audio_gemini(dialog, mp3_path, title)
            print("Lyd: Gemini TTS.")
        except Exception as e:
            print(f"Gemini TTS feilet ({e}) – faller tilbake til Piper.")
            dur = synthesize(dialog, mp3_path, title)
    else:
        dur = synthesize(dialog, mp3_path, title)
    size = os.path.getsize(mp3_path)
    print(f"MP3: {fname} – {dur}s, {size} bytes.")

    publish = os.environ.get("PUBLISH", "1") == "1"

    if publish:
        url = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}/{fname}"
        desc = "Dagens dyrehelse- og kjæledyrnyheter: " + "; ".join(i["title"] for i in animal[:4])
        eps = [e for e in load_state() if e["file"] != fname]  # unngå duplikat samme dag
        eps.insert(0, {"date":date_id,"file":fname,"url":url,"title":title,
                       "desc":desc,"duration":dur,"size":size,
                       "pubdate":format_datetime(now)})
        eps = eps[:KEEP]
        save_state(eps)
        build_feed(eps)
        print("Publisert til feed.")
    else:
        print("TESTMODUS – feeden er IKKE endret. Last ned lydfila fra kjøringens artefakter for å lytte.")

    # fortell workflow hvilken fil som ble laget
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out,"a") as f:
            f.write(f"mp3_path={mp3_path}\n")
            f.write(f"mp3_name={fname}\n")
    print("Ferdig.")

if __name__ == "__main__":
    main()
