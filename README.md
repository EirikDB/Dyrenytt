# Dyrenytt – selvgående daglig podkast

Dette repoet lager en ny podkast-episode helt automatisk hver hverdagsmorgen og publiserer den slik at Spotify (og andre podkast-apper) plukker den opp av seg selv. Ingen servere å drifte, og det er gratis.

**Slik henger det sammen:** en planlagt GitHub Actions-jobb henter dagens nyheter (dyrehelse/kjæledyr + litt generelt), skriver et manus, lager lyden med den norske Piper-stemmen, legger MP3-en som en *release-asset*, og oppdaterer en **RSS-feed** som ligger på GitHub Pages. Spotify abonnerer på den feeden én gang, og viser nye episoder automatisk.

> Merk: dette bruker et **offentlig** GitHub-repo, så feeden og lydfilene er offentlig tilgjengelige (samme premiss som Spotify uansett har). Vil du ha det helt internt/privat, må du bruke en betalt host som Transistor i stedet – si fra, så lager jeg en variant.

## Engangs-oppsett (ca. 10 minutter)

### 1. Opprett repoet
Lag et nytt **offentlig** repo på GitHub (f.eks. `dyrenytt-podcast`), og last opp alle filene i denne mappa (behold mappestrukturen). Enten via GitHubs «Add file → Upload files», eller med git:
```
git init && git add . && git commit -m "Dyrenytt"
git branch -M main
git remote add origin https://github.com/<BRUKER>/dyrenytt-podcast.git
git push -u origin main
```

### 2. Gi Actions skrivetilgang
Settings → **Actions** → General → «Workflow permissions» → velg **Read and write permissions** → Save.
(Dette lar jobben legge ut lydfiler og oppdatere feeden.)

### 3. Slå på GitHub Pages
Settings → **Pages** → «Build and deployment» → Source: **Deploy from a branch** → Branch: **main**, mappe: **/docs** → Save.
Feeden din blir da liggende på: `https://<BRUKER>.github.io/dyrenytt-podcast/feed.xml`

### 4. (Anbefalt) Sett e-posten din
Spotify sender en verifiseringskode til e-postadressen i feeden. Settings → **Secrets and variables** → Actions → New repository secret:
- Navn: `FEED_EMAIL` – Verdi: din e-postadresse

### 5. Lag den første episoden nå
Actions-fanen → velg **«Dyrenytt daglig episode»** → **Run workflow**. Etter et par minutter skal det ligge en episode i feeden. Åpne `https://<BRUKER>.github.io/dyrenytt-podcast/feed.xml` i nettleseren og sjekk at den vises.

### 6. Koble feeden til Spotify
Gå til **creators.spotify.com** → Add a show → **«Find an existing show»** → lim inn feed-URL-en din → følg verifiseringen (koden sendes til `FEED_EMAIL`). Deretter henter Spotify inn nye episoder automatisk hver morgen.
(Du kan legge samme feed-URL inn i Apple Podcasts og andre apper på samme måte.)

Del gjerne Spotify-lenken (eller feed-URL-en) med kollegaene – de trykker «Følg» én gang og får resten automatisk.

## Endre tidspunktet
Tidspunktet står i `.github/workflows/daily.yml` under `cron`, og er **alltid i UTC**:
```
- cron: "30 3 * * 1-5"   # 03:30 UTC = 05:30 norsk sommertid, 04:30 vintertid. Man–fre.
```
Vil du ha den klar 06:30 om sommeren, sett `30 4 * * 1-5`. (GitHub kan starte cron-jobber noen minutter forsinket ved høy last, så legg inn litt margin.)

## Valgfritt: bedre manus med KI
Uten nøkkel bruker skriptet et enkelt mal-manus (leser overskriftene med faste overganger + et roterende «dagens tips»). Vil du ha et rikere, mer naturlig samtale-manus, legg inn en Anthropic-nøkkel:
- Secret-navn: `ANTHROPIC_API_KEY` – Verdi: nøkkelen din fra console.anthropic.com

Da genereres manuset av Claude ut fra dagens nyheter. Det koster noen få øre per episode.

## Stemmene
Episoden bruker den norske Piper-stemmen «talesyntese», med to programledere laget ved lett pitch-justering (Kari lysere, Tom dypere). Vil du oppgradere til ekte, naturlige stemmer (f.eks. ElevenLabs), kan det kobles på her senere – runnerne har åpent internett, så det er fullt mulig.

## Godt å vite
- **Kostnad:** gratis. Offentlige repo har ubegrenset GitHub Actions-tid.
- **Opprydding:** de siste 30 episodene beholdes i feeden (endre `KEEP_EPISODES` i `daily.yml`). Eldre lydfiler slettes automatisk.
- **Inaktivitet:** GitHub pauser planlagte jobber hvis repoet er helt inaktivt i 60 dager. En manuell «Run workflow» nullstiller det.
- **Filer:** `scripts/generate.py` er hjernet; `docs/` er nettstedet Spotify leser fra; `model/` lastes ned automatisk ved første kjøring.

## Feilsøking
- *Feeden er tom / 404:* sjekk at Pages er slått på (steg 3) og at workflowen har kjørt uten feil (Actions-fanen).
- *Jobben feiler på opplasting:* sjekk at «Read and write permissions» er på (steg 2).
- *Spotify finner ikke feeden:* åpne feed-URL-en i nettleseren først; den må laste som XML. Vent noen minutter etter at Pages er publisert.
