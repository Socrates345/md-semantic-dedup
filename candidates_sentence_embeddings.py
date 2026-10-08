# Stage 1: pairs of notes that look alike -> candidates.json
#   python candidates_sentence_embeddings.py <file.md> [folder] [floor]
# folder is where candidates.json goes (default: here), floor the lowest similarity kept (default 0.70).
# run.py calls this with a run folder and a low floor, then settles the FLOOR to keep.
# Nothing is sent to huggingface.co: this process runs with its offline switch on. If the model is not on this
# machine yet, it is downloaded once by a separate process that never reads the file.
# pip install sentence-transformers numpy
import os, re, sys, json, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")   # has to be set before the import below, which reads it
import numpy as np
from sentence_transformers import SentenceTransformer

OUT = os.path.join(sys.argv[2] if len(sys.argv) > 2 else ".", "candidates.json")
lines = open(sys.argv[1], encoding="utf-8").read().splitlines()

# One unit per line: in these notes a newline is a deliberate break, so lines are never re-joined
# (a hard-wrapped paragraph ends up as separate lines). Bullet, heading and checkbox markers are stripped.
MARK = re.compile(r"^\s*(#{1,6}\s*|[*+>•–]\s*|-(?!\d)\s*|\d+[.)]\s+)")
units = []
for i, line in enumerate(lines, 1):
    t = line.strip()
    if not t or re.fullmatch(r"[-*_=#\s]+", t): continue     # blank lines, --- separators, empty bullets
    if t.startswith("#") and len(t) < 40: continue           # short section titles ("# notes") repeat everywhere
    units.append((i, re.sub(r"^\[[ xX]\]\s*", "", MARK.sub("", t)).strip()))

# Short notes stay whole; only long paragraphs are split into sentences. Drop tiny fragments ("ok", "idk")
sents = [(ln, s) for ln, u in units
         for s in (re.split(r"(?<=[.!?])\s+", u) if len(u) > 200 else [u]) if len(s) >= 15]
print(f"{len(sents)} sentences")

MODEL = "BAAI/bge-m3"   # multilingual
try: model = SentenceTransformer(MODEL)
except OSError:         # offline and not in the cache yet
    print(f"{MODEL} is not on this machine yet: downloading it from huggingface.co (2.3 GB, once). Only the model is requested.")
    fetch = f"from sentence_transformers import SentenceTransformer; SentenceTransformer('{MODEL}')"
    if subprocess.run([sys.executable, "-c", fetch], env={**os.environ, "HF_HUB_OFFLINE": "0"}).returncode:
        sys.exit("the download failed, see above")
    model = SentenceTransformer(MODEL)
emb =model.encode([s for _, s in sents], normalize_embeddings=True,
                   batch_size=32, show_progress_bar=True)
sim = emb @ emb.T
np.fill_diagonal(sim, -1)

K, FLOOR = 5, float(sys.argv[3]) if len(sys.argv) > 3 else 0.70   # each sentence's 5 closest matches, if similarity >= FLOOR
pairs = set()
for i in range(len(sents)):
    for j in np.argsort(-sim[i])[:K]:
        if round(float(sim[i, j]), 3) >= FLOOR:   # the rounded score is the one written out and the one run.py's FLOOR applies to
            pairs.add((min(i, j), max(i, j)))

out = sorted(({"score": round(float(sim[i, j]), 3),
               "a_line": sents[i][0], "a": sents[i][1],
               "b_line": sents[j][0], "b": sents[j][1]} for i, j in pairs),
             key=lambda c: -c["score"])
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"{len(out)} candidate pairs scoring {FLOOR:.2f} or more -> {OUT}")