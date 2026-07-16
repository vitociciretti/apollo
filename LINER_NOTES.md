# TECH(NO) US

*Eight American technology stocks, 2010–2026. Every note, every drum hit, every
breakdown is computed from the daily price history. Nothing is composed by hand —
the market wrote these tracks; we only chose the instruments.*

---

## How to listen — the mapping legend

| What you hear | What it means in the market |
|---|---|
| **The kick drum (four-on-the-floor)** | A healthy trend regime. The kick plays only while the stock is out of deep drawdown — as long as you hear it, the trend is alive. Its punch scales with realized volatility: harder kicks = wilder markets. |
| **The kick disappears (breakdown)** | A deep drawdown — the worst quartile of the stock's own history. Bear markets are literally the breakdowns of each track. The longer the silence, the longer the pain. |
| **Clap roll (building up)** | A recovery forming: the last bar of a drawdown before the stock breaks back out. The roll's crescendo is the market finding its feet. |
| **Crash cymbal** | A single trading day beyond 4 standard deviations — an earnings shock, a flash crash, a short squeeze. Also marks the first bar of every recovery. |
| **The bassline's pitch** | Where price sits versus its own 252-day trend. Rolling on the root note = trading at trend; the harmony lifts when the stock runs above trend and darkens (to Phrygian mode) below it. |
| **Octave jumps in the bass** | Individual large-move days (>1σ) kicking the bassline up an octave. Momentum you can hear. |
| **The filter opening (bass gets brighter)** | Rising volatility. The classic techno filter sweep *is* the vol curve — quiet years sound muted and dark, 2020 sounds wide open. |
| **The stabs (syncopated chords)** | Active weeks: stabs only fire when the underlying days actually moved. The groove's irregularity is the market's own irregularity. |
| **Sharper, more detuned tone (timbre)** | Fat tails: rolling kurtosis morphs every voice from clean (calm, near-Gaussian regimes) to razor-edged (crash-prone regimes). Piano vs violin = thin tails vs fat tails. |
| **The pads (swelling in breakdowns)** | The stock's own volatility spectrum. Each pad note is a statistically significant volatility cycle (Wiener–Khinchin theorem), transposed into the audible range and tuned to the track's key. The half-year and one-quarter cycles are the earnings calendar itself. |
| **The tempo** | Annualized volatility, anchored by each track's genre. Calm compounders run at 115–120 BPM (dub, deep house); the wild ones at 140 (trance, acid). |
| **The key** | Each stock gets its own minor key — its identity across the album. |

## The tracks

| # | Title | Key | BPM | The story |
|---|---|---|---|---|
| 01 | Chipmaker | F minor | 131 | **Peak-time techno.** From graphics cards to the AI singularity, with 2018, 2022 and the DeepSeek scare as breakdowns |
| 02 | Cupertino | A minor | 117 | **Deep house.** The steadiest compounder gets the swung, unhurried groove |
| 03 | Redmond | D minor | 115 | **Dub techno.** Lowest vol in the basket: spacious offbeat chords in long echo |
| 04 | PageRank | G minor | 125 | **Progressive.** Long steady runs, arpeggios that never quite resolve |
| 05 | Everything Store | C minor | 130 | **Warehouse.** Relentless logistics: driving, percussion-forward, pads stripped back |
| 06 | Social Graph | B♭ minor | 127 | **Breaks.** The one that broke the four-on-the-floor (IPO 2012; 2018 and 2022 are its architecture) |
| 07 | Stream | E minor | 140 | **Uplifting trance.** Binge arcs, supersaw arps, the biggest builds on the record |
| 08 | Ludicrous | B minor | 140 | **Acid.** Highest vol, 303 squelch, accents fired by the big days. Obviously |
| 09 | Correlation One | F minor | 126 | **Techno duet.** Two stocks, one groove: the chipmaker plays bass, Cupertino plays lead, and their rolling 63-day correlation writes the harmony. When correlation spikes toward 1 — which is what markets do in a crisis — the two voices lock into unison; in calm, decoupled markets they drift apart into tension intervals. Breakdowns fire only when both are in deep drawdown at once. You are listening to diversification live and die |

Each genre is matched to the statistics: low-volatility compounders get the patient
grooves (deep house, dub), the high-volatility names get the aggressive ones
(trance, acid), and the genre's BPM anchor is then fine-tuned by each stock's
annualized vol.

## The videos — visual legend

Every track has a synced video: the price chart draws itself as the music plays
(~63 trading days per second of audio).

| On screen | Meaning | You hear |
|---|---|---|
| Blue line | growth of $1, log scale | the bassline's pitch (level vs trend) |
| Green lower panel | 21-day rolling volatility | filter brightness, kick punch |
| Red shaded regions | deep-drawdown regimes | the breakdown (kick drops out) |
| **Red dot** | **a single >4σ trading day** — earnings shock, flash crash | the crash cymbal, same instant |
| "BREAKDOWN" flag | the playhead is inside a bear regime | Phrygian mode, pads swelling |

*Composed by an LSTM's ancestors: prices → log-returns → spectra, regimes and moments → sound.
Pipeline: Python/NumPy/SciPy synthesis, MIDI export for production; data: Yahoo Finance
adjusted closes. Full mathematical specification in `docs/technical_documentation.pdf`.
All source at github.com/vitociciretti/apollo.*
