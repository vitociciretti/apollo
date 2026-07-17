# APOLLO

*God of music. Also of prophecy. This repository is both.*

**Apollo turns financial time series into techno.** Every note, drum hit, breakdown
and filter sweep on the album *TECH(NO) US* is computed from the real 2010–2026
daily price history of eight US technology stocks. Nothing is composed by hand —
the market wrote the tracks; we chose the instruments.

▶ Start here: `album/01_Chipmaker.mp4` — the price chart drawing itself in sync
with the music. Breakdowns shade in when the kick drops out; red dots flash on
4-sigma days. Every track has one.

Each track carries the genre its statistics call for — the calm compounders get
deep house and dub techno; the violent ones get trance and acid (see
[LINER_NOTES.md](LINER_NOTES.md) for the genre map and the full audio legend).
A complete mathematical specification of the pipeline — Wiener–Khinchin theory,
estimators, the three samplers, and the sonification grammar as a measurable map —
is in [`docs/technical_documentation.pdf`](docs/technical_documentation.pdf).

## The idea

1. **Wiener–Khinchin**: the power spectral density of a stationary process is the
   Fourier transform of its autocovariance. Daily *returns* are spectrally flat
   (white noise — no melody, as the martingale hypothesis demands), but *absolute
   returns* carry a long-memory 1/f spectrum (GPH d ≈ 0.4) with significant cycles
   at the earnings-calendar harmonics (~63 and ~126 trading days). The melody of
   markets lives in the volatility.
2. **Generative sampling**: that distribution can be sampled — by phase
   randomization of the spectrum (the max-entropy sampler given the
   autocovariance), by a fitted GJR-GARCH, or by an LSTM density model
   (`pipeline/train_rnn.py`) that learns the conditional Student-t of tomorrow's
   return and reproduces the fat tails (kurtosis 9.7 vs 9.8), volatility
   clustering and long memory of the real series.
3. **Sonification grammar**: a path — real or sampled — drives every musical
   parameter. The full mapping (kick = trend regime, breakdown = drawdown,
   crash = 4σ day, filter = volatility, timbre = kurtosis, tempo = annualized
   vol…) is documented in [LINER_NOTES.md](LINER_NOTES.md).

## v4 — the creativity grammar

v4 teaches the engine *form, melody and voice*, all still read off the data.
`regime.py` derives each track's song form from the series' own era structure:
change-points in rolling volatility split the history into sections
(intro/calm/build/peak/breakdown/recovery/outro), each with its own layer gates
and gain staging. `motif.py` gives every stock an 8-note **theme** from its
first trading year and develops it bar-by-bar with classic motivic operators —
inversion in drawdowns, transposition when price is stretched from trend,
diminution in high vol, augmentation in calm — the market picks the variation.
`volume_perc.py` opens a **trading-volume** channel: per-bar volume percentile
drives percussion density, and earnings-volume spikes trigger drum fills.
Sections that are bull end-to-end (z > 1 vs the 252-day trend) **modulate to
the relative major**. `choir.py` adds a singing voice: TTS pitch-mapped through
a phase vocoder onto data-derived melodies, stacked five voices deep and
vocoded onto a synth carrier — English narrations on tracks 01/07/09 and the
Latin liturgy of track 10, *Descensus Gradientis*, whose melody is the LSTM's
own training-loss curve (`descensus.py`). Finally `transformer_gen.py` adds a
**joint 8-ticker decoder-only transformer** sampler; its fidelity scorecard vs
the real series and the LSTM baseline is in
[`pipeline/transformer_scorecard.md`](pipeline/transformer_scorecard.md).

## Repository layout

```
album/            10 tracks (8 stocks + the correlation duet + the gradient-descent closer): mastered WAVs + MIDI + synced videos
LINER_NOTES.md    the album commentary: what every sound means in the market
docs/             technical_documentation.tex/.pdf — the full mathematical spec
pipeline/
  spectral.py       Wiener–Khinchin / Welch / multitaper analysis, Fisher g-test, GPH
  melody*.py        direct sonifications of the spectrum and its history
  sample_melody.py  phase-randomization + GARCH samplers
  train_rnn.py      LSTM conditional-density model: train, diagnose, sample
  render_song.py    first-generation composition grammar
  render_techno.py  techno grammar, single track
  techno_album.py   the album pipeline (v4): any return series -> events -> WAV + MIDI
  regime.py         v4 form: vol change-points + drawdown regimes -> sections
  motif.py          v4 melody: per-stock theme + development operators
  volume_perc.py    v4 volume channel: percussion density + earnings fills
  choir.py          v4 voice: TTS -> phase-vocoder pitch-mapping -> stacked vocoder
  vocals.py         choir passes for tracks 01/07/09 (lyrics in lyrics.md)
  descensus.py      track 10: the LSTM's training history as Latin-choral hard dance
  transformer_gen.py joint 8-ticker transformer sampler (see transformer_scorecard.md)
  make_video.py     price-evolution video synced to a track
  make_video3.py    v4 triptych video: Julia set + price + LSTM training panel
  r_*.npy           log-return series per ticker (from adjusted closes)
```

## Reproduce

```bash
pip install numpy scipy pandas yfinance mido arch torch imageio-ffmpeg matplotlib
python pipeline/techno_album.py            # renders the album (WAV + MIDI)
python pipeline/make_video.py NVDA 01_Chipmaker
python pipeline/train_rnn.py               # train the LSTM, sample a synthetic path
```

The MIDI files are the production path: drop them into a DAW, replace the
synthesis with real instruments, keep the market's composition.

*Not investment advice. A flat return spectrum is precisely the statement that
none of this predicts anything — that's the joke, and the point.*
