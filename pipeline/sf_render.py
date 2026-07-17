"""Offline General-MIDI rendering via tinysoundfont + bundled TimGM6mb.sf2.

The soundfont (TimGM6mb.sf2, GM bank, ~6 MB) was extracted from the
pretty_midi sdist on PyPI and lives next to this module.

Public API
----------
render_notes(events, program, sr=44100) -> (left, right)
    events  : list of (t_s, dur_s, midi, vel) tuples.
              t_s/dur_s in seconds, midi 0-127, vel either int 1-127
              or float in (0, 1].
    program : GM program number, 0-indexed (0=Grand Piano ... 127).
              Pass drums=True to route to the GM drum bank instead.
    Returns two float64 numpy arrays (L, R), peak-normalised to <= 0.9
    only if the raw render clips.
"""
import os
import numpy as np
import tinysoundfont

SR = 44100
SF2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "TimGM6mb.sf2")
_RELEASE_TAIL = 2.0  # seconds rendered after the last note-off
_CHUNK = 4096


def _vel127(v):
    if isinstance(v, float) and v <= 1.0:
        v = v * 127.0
    return int(max(1, min(127, round(v))))


def render_notes(events, program, sr=SR, drums=False, gain=0.0):
    """Render (t_s, dur_s, midi, vel) events with a GM program to stereo."""
    if not events:
        z = np.zeros(1)
        return z, z.copy()

    synth = tinysoundfont.Synth(gain=gain, samplerate=sr)
    sfid = synth.sfload(SF2_PATH)
    synth.program_select(0, sfid, 0, int(program), is_drums=drums)

    # (sample, order, kind, midi, vel); offs (order 0) before ons (order 1)
    # at the same sample so repeated notes retrigger cleanly.
    sched = []
    for t_s, dur_s, midi, vel in events:
        on = max(0, int(round(t_s * sr)))
        off = max(on + 1, int(round((t_s + dur_s) * sr)))
        sched.append((on, 1, "on", int(midi), _vel127(vel)))
        sched.append((off, 0, "off", int(midi), 0))
    sched.sort()

    total = sched[-1][0] + int(_RELEASE_TAIL * sr)
    out = np.zeros(2 * total, dtype=np.float32)

    cursor = 0
    for smp, _order, kind, midi, vel in sched:
        while cursor < smp:
            n = min(_CHUNK, smp - cursor)
            buf = synth.generate(n)
            out[2 * cursor:2 * (cursor + n)] = np.frombuffer(buf, dtype=np.float32)
            cursor += n
        if kind == "on":
            synth.noteon(0, midi, vel)
        else:
            synth.noteoff(0, midi)
    while cursor < total:
        n = min(_CHUNK, total - cursor)
        buf = synth.generate(n)
        out[2 * cursor:2 * (cursor + n)] = np.frombuffer(buf, dtype=np.float32)
        cursor += n

    synth.sfunload(sfid)
    st = out.astype(np.float64).reshape(-1, 2)
    peak = np.max(np.abs(st)) if st.size else 0.0
    if peak > 1.0:
        st *= 0.9 / peak
    return st[:, 0].copy(), st[:, 1].copy()


if __name__ == "__main__":
    # Self-test: C minor scale on a warm pad (GM 89) and a pizzicato pluck (GM 45).
    scale = [60, 62, 63, 65, 67, 68, 70, 72]
    events = [(i * 0.25, 0.4, m, 100) for i, m in enumerate(scale)]
    for name, prog in (("pad", 88), ("pluck", 45)):
        L, R = render_notes(events, prog)
        rms = float(np.sqrt(np.mean(L ** 2 + R ** 2)))
        peak = float(max(np.max(np.abs(L)), np.max(np.abs(R))))
        assert rms > 1e-4, f"{name} render is silent (rms={rms})"
        assert peak <= 1.0, f"{name} render clips (peak={peak})"
        assert len(L) == len(R) > SR, f"{name} render too short"
        print(f"{name:5s} prog={prog:3d}  len={len(L)}  rms={rms:.4f}  peak={peak:.3f}")
    print("sf_render self-test OK")
