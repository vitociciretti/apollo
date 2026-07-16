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
| **The tempo** | Annualized volatility. Calm compounders run at 122 BPM; the wild ones at 134. |
| **The key** | Each stock gets its own minor key — its identity across the album. |

## The tracks

| # | Title | Key | BPM | The story |
|---|---|---|---|---|
| 01 | Chipmaker | F minor | 130 | From graphics cards to the AI singularity, with 2018, 2022 and the DeepSeek scare as breakdowns |
| 02 | Cupertino | A minor | 123 | The steadiest groove on the record — a compounder's four-on-the-floor |
| 03 | Redmond | D minor | 122 | The slowest BPM: lowest vol in the basket. Relentless |
| 04 | PageRank | G minor | 123 | Search never crashed hard — few breakdowns, long runs |
| 05 | Everything Store | C minor | 125 | 2014 doubt and the 2022 unwind as its two great silences |
| 06 | Social Graph | B♭ minor | 127 | The longest track (IPO 2012): the 2018 and 2022 collapses are its architecture |
| 07 | Stream | E minor | 131 | Qwikster, the 2022 subscriber shock — a track that keeps losing and finding its kick |
| 08 | Ludicrous | B minor | 134 | The fastest BPM on the album. Obviously |

*Composed by an LSTM's ancestors: prices → log-returns → spectra, regimes and moments → sound.
Pipeline: Python/NumPy/SciPy synthesis, MIDI export for production. All source at github.com/vitociciretti/apollo.*
