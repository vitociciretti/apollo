# TECH(NO) US — Release checklist

Everything below the "You do" line is prepared. The account steps legally require
your identity and payment details.

## Already done (in this repo / Downloads\TechnoUS)

- [x] 10 masters, distributor-compliant: 44.1 kHz / 16-bit / stereo WAV
- [x] Cover art `cover_3000.jpg`: 3000×3000 RGB JPG (min is 3000×3000; no logos,
      no URLs, no ticker symbols — all compliant)
- [x] Track titles clean of trademarks (no tickers/company names)
- [x] Data source: Yahoo Finance public adjusted closes (no licensed-data issue)
- [x] Liner notes + videos for socials/YouTube

## You do (≈ 30 minutes)

1. **Pick a distributor** — recommendation: DistroKid (~$23/yr, unlimited releases,
   keeps 100% of royalties). Alternatives: TuneCore, CD Baby, Amuse (free tier, slower).
2. **Create the account** — email, password, payment. On DistroKid choose the
   "Musician" plan (one artist name).
3. **Decide the artist name** — the one open decision. Candidates:
   - **APOLLO** — on-concept, but heavily used on Spotify (you'll share the name page
     until you claim your artist profile)
   - **Apollo Markets / Apollo Process** — distinctive, still on-concept
   - **Vito Ciciretti** — the quant-brand play: ties the album to your research identity
4. **Upload** — new album → "TECH(NO) US" → 10 tracks in order → cover JPG.
   Metadata per track below. Genre: Electronic / Techno. Language: instrumental
   except tracks 01/07/09 (English vocals) and 10 (Latin vocals).
   "AI-generated vocals?" — **Yes for tracks 01, 07, 09 and 10** (the choir is a
   vocoded synthetic voice — TTS pitch-mapped and vocoded, no human singer;
   distributors ask, so disclose); No for the rest. Songwriter: your legal name
   (composition = the mapping design; see docs).
5. **Release date** — set ≥ 2 weeks out if you want to pitch to editorial playlists
   via Spotify for Artists; "as soon as possible" otherwise.
6. **Claim Spotify for Artists** after the release goes live (verifies you, unlocks
   playlist pitching and the artist page — put the mapping legend in the bio).

## Track metadata (copy-paste)

| # | Title | Length | AI vocals? | ISRC |
|---|---|---|---|---|
| 1 | Chipmaker | 2:38 | **Yes** (vocoded synthetic voice) | auto-assigned by distributor |
| 2 | Cupertino | 2:56 | No | " |
| 3 | Redmond | 2:59 | No | " |
| 4 | PageRank | 2:45 | No | " |
| 5 | Everything Store | 2:39 | No | " |
| 6 | Social Graph | 3:30 | No | " |
| 7 | Stream | 2:27 | **Yes** (vocoded synthetic voice) | " |
| 8 | Ludicrous | 2:24 | No | " |
| 9 | Correlation One | 2:44 | **Yes** (vocoded synthetic voice) | " |
| 10 | Descensus Gradientis | 2:39 | **Yes** (vocoded synthetic voice, Latin) | " |

(Lengths measured from the mastered WAVs in `Downloads\TechnoUS`.)

Album description (for stores that take one):

> Eight American technology stocks, sixteen years of daily prices, no composer.
> Every kick, breakdown and filter sweep is computed from the data: the beat is
> the bull market, the breakdowns are the drawdowns, the crash cymbals are the
> four-sigma days. Composed by the market, rendered as techno.

## Honest quality note

These WAVs are algorithmic masters: numpy-oscillator synthesis, then a real
mastering chain (tonal EQ, bass-mono below 120 Hz, glue compression, ~-11 LUFS
integrated with a true-peak-safe limiter at -0.2 dBFS — Spotify normalizes to
-14 LUFS, so loudness is competitive). Unmastered premasters are preserved in
`premaster/`. What a DAW pass with the included MIDI files would still add:
professional synth voices and per-track mix decisions. Options: release now and
re-release a "producer edition" later, or produce first. Releasing now is the
faster way to learn if anyone cares.
