"""Tech(no) US — harmony.py: per-genre harmonic color for the album pipeline.

Standalone companion to techno_album.py (v2). Never imports or modifies it.
Three entry points, all pure functions on ints/lists:

  scale_for(genre, bear, build=False) -> 7 semitone offsets (the bar's mode)
  voicing(genre, root_midi, scale, degree) -> midi notes for the stab/chord layer
  bass_degree(genre, deg, step=0) -> scale degree for the bass line

Design contract carried over from v2: *bear regimes are always phrygian* for
every genre. The b2 darkness encodes drawdown data semantics, not stylistic
taste, so no genre override may touch it. Genre color only applies in
non-bear bars.
"""

# ---------------------------------------------------------------- scales ----
# All scales are 7 ascending semitone offsets from the root (index = degree).

# Aeolian (natural minor): the v2 default — dark but neutral, the album's home mode.
AEOLIAN = [0, 2, 3, 5, 7, 8, 10]
# Phrygian: aeolian with b2 — the flat second is the classic "menace" interval,
# reserved for bear/drawdown bars in every genre (data semantics, see module doc).
PHRYGIAN = [0, 1, 3, 5, 7, 8, 10]
# Dorian: minor with a natural 6th — the raised 6th removes aeolian's b6 gloom,
# giving the brighter, jazzier minor that deep house and progressive live on.
DORIAN = [0, 2, 3, 5, 7, 9, 10]
# Harmonic minor: aeolian with a raised 7th — the leading tone creates a strong
# pull to the tonic, i.e. built-in tension that resolves: perfect for trance builds.
HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]

# v2's trance/prog arpeggio degree cycle, re-exported verbatim so callers can
# keep the exact ARPSEQ behavior of techno_album.py (0-2-4 outlines the triad,
# the 7 tops it an octave up, then it walks back down — a symmetric trance arp).
ARPSEQ = [0, 2, 4, 7, 4, 2]

# genres known to the album (mirrors techno_album.STYLES keys)
GENRES = ("techno", "deephouse", "dub", "prog", "warehouse", "breaks", "trance", "acid")


def scale_for(genre, bear, build=False):
    """Return the 7-offset scale for one bar of `genre`.

    Priority order (first match wins):
      bear            -> phrygian for ALL genres: the b2 darkness is what the
                         drawdown data *means*, so style never overrides it.
      trance + build  -> harmonic minor: the raised 7th (leading tone) only
                         during builds, so the tension it creates resolves
                         exactly when the drop lands.
      deephouse       -> dorian: the natural 6th is the brighter minor color
                         that deep house chords (min9s) are voiced from.
      prog            -> dorian: progressive arps favor the nat-6 too — long
                         runs sound lifted rather than mournful.
      everything else -> aeolian, the v2 baseline minor.
    """
    if bear:
        return list(PHRYGIAN)
    if genre == "trance" and build:
        return list(HARMONIC_MINOR)
    if genre in ("deephouse", "prog"):
        return list(DORIAN)
    return list(AEOLIAN)


# --------------------------------------------------------------- voicing ----
def _step(root_midi, scale, s):
    """Diatonic step: scale degree `s` above root, wrapping octaves.

    Stacking chord tones as scale steps (d, d+2, d+4, ...) keeps every voicing
    diatonic to whatever mode the bar is in — the same shape darkens in
    phrygian and brightens in dorian for free.
    """
    return root_midi + scale[s % 7] + 12 * (s // 7)


def voicing(genre, root_midi, scale, degree):
    """MIDI notes for the stab/chord layer at `degree` of `scale`.

    Per-genre chord shapes (d = degree; steps are diatonic thirds unless noted):
      deephouse -> rootless min9 up an octave (b3, 5, b7, 9): dropping the root
                   is the classic Rhodes/garage voicing — the bass already owns
                   the root, so the keys float above it.
      dub       -> low close add9 (root, 5, 9): no 3rd = hollow and dark, the
                   9 adds haze without brightness — dub chords are texture.
      prog      -> sus2 / sus4 alternating by degree parity: suspensions have
                   no 3rd, so prog stabs stay ambiguous and never resolve —
                   perpetual forward motion.
      trance    -> minor triad + octave-doubled root (4 notes): the doubled
                   root is the supersaw thickener; trance chords are anthems,
                   not jazz.
      techno    -> bare fifth dyad: no 3rd at all — maximum tunnel, harmony
                   reduced to pure power interval.
      warehouse -> root + fifth + octave: the fifth dyad with the octave on
                   top, a slightly bigger slab for the bigger room.
      breaks    -> min7 with no 5th (root, b3, b7): shell voicing — the 5th
                   adds nothing, the b7 adds funk; leaner fits busier drums.
      acid      -> single note an octave up: a 303 line is monophonic by
                   definition — its "chord" is just the accent note.
    """
    d = degree
    if genre == "deephouse":
        # rootless min9: 3rd, 5th, 7th, 9th as diatonic steps, lifted +12
        return [_step(root_midi, scale, s) + 12 for s in (d + 2, d + 4, d + 6, d + 8)]
    if genre == "dub":
        # add9 close: root, 5th (step d+4), 9th (step d+8 = 2nd an octave up)
        return [_step(root_midi, scale, s) for s in (d, d + 4, d + 8)]
    if genre == "prog":
        # even degree -> sus2 (root, 2nd, 5th); odd -> sus4 (root, 4th, 5th)
        mid = d + 1 if d % 2 == 0 else d + 3
        return [_step(root_midi, scale, s) for s in (d, mid, d + 4)]
    if genre == "trance":
        # triad (d, d+2, d+4) + root doubled an octave up (step d+7)
        return [_step(root_midi, scale, s) for s in (d, d + 2, d + 4, d + 7)]
    if genre == "techno":
        return [_step(root_midi, scale, s) for s in (d, d + 4)]
    if genre == "warehouse":
        return [_step(root_midi, scale, s) for s in (d, d + 4, d + 7)]
    if genre == "breaks":
        return [_step(root_midi, scale, s) for s in (d, d + 2, d + 6)]
    if genre == "acid":
        return [_step(root_midi, scale, d) + 12]
    raise ValueError("unknown genre: %r" % (genre,))


# ----------------------------------------------------------- bass degree ----
def bass_degree(genre, deg, step=0):
    """Scale degree the bass plays for bar-degree `deg` at 16th-step `step`.

    Pass-through for most genres: the bass states the bar's harmony plainly.
    prog/trance lift the degree through ARPSEQ exactly as v2 does
    (deg + ARPSEQ[step % 6]) % 7 — the arp cycle IS the bassline in those
    styles, walking the triad instead of pedaling the root.
    With the default step=0 the lift is ARPSEQ[0] == 0, so 2-arg calls behave
    as pure pass-through (mod 7) for every genre.
    """
    if genre in ("prog", "trance"):
        return (deg + ARPSEQ[step % len(ARPSEQ)]) % 7
    return deg


# -------------------------------------------------------------- self-test ---
_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name(m):
    return "%s%d" % (_NAMES[m % 12], m // 12 - 1)


if __name__ == "__main__":
    ROOT, DEG = 53, 0  # F3, tonic
    for g in GENRES:
        sc = scale_for(g, bear=False)
        v = voicing(g, ROOT, sc, DEG)
        assert all(isinstance(m, int) and 20 <= m <= 100 for m in v), (g, v)
        print("%-9s %s" % (g, " ".join(_note_name(m) for m in v)))
    # bear override is universal
    for g in GENRES:
        assert scale_for(g, bear=True) == PHRYGIAN, g
        assert scale_for(g, bear=True, build=True) == PHRYGIAN, g
    # trance build raises the 7th only when asked
    assert scale_for("trance", False, build=True) == HARMONIC_MINOR
    assert scale_for("trance", False) == AEOLIAN
    # bass degree: pass-through vs arp lift
    assert bass_degree("techno", 5) == 5
    assert bass_degree("trance", 5, 3) == (5 + ARPSEQ[3]) % 7
    assert bass_degree("prog", 3) == 3  # step=0 -> ARPSEQ[0]==0
    print("harmony.py self-test OK")
