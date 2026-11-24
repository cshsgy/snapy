"""#348 harness (94e4601e) restricted to its eight face decks: on the W2 head the eight cell decks are refused
(gravity-work: cell removed by I-7) and h348.py stops at the first refusal, so produce/compare run here with
h348.decks() filtered to work == 'face'. Same decks, same run_deck, same compare."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness348"))
import h348
_decks = h348.decks
def face_decks():
    return [d for d in _decks() if d[2] == "face"]
h348.decks = face_decks
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--out"); ap.add_argument("--nstep", type=int, default=10); ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare: sys.exit(0 if h348.compare(*args.compare) else 1)
    h348.produce(args)
