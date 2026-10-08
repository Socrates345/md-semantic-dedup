# Everything at once: Stage 1, the choice of FLOOR, Stage 2 and, when the file has an answer key, the score
#   python run.py sample\sample.md qwen/qwen3.5-9b [floor] [earlier run's folder]      model id as `lms ls` shows it
# Each run gets its own folder: runs\<file name>, then runs\<file name> (1), (2), ... so nothing is overwritten.
# FLOOR: with an answer key (the sample) the program takes the highest one that keeps every true duplicate Stage 1
# found, i.e. the most duplicates for the fewest pairs. Without one it cannot know, and asks. A floor given on the
# command line overrules both.
# The pairs under the FLOOR are not lost: they go to below_floor.json, and Stage 2 reads those that have a line
# in a group of duplicates it found.
# With the folder of an earlier run of the same file, Stage 2 takes that run's verdicts over and asks the judge only
# the pairs it did not judge. FLOOR is then that run's, unless one is given. For the same judge and question only.
# The stages run as separate processes, so the embedding model has left the GPU before the judge is loaded.
import re, sys, json, pathlib, subprocess

WIDE = 0.50   # Stage 1 keeps every pair from here up; the FLOOR is then settled from what it found
def as_floor(text):                           # "0.65" or "0,65"; None unless it is a number from WIDE to 1
    try: f = float(text.strip().replace(",", "."))
    except ValueError: return None
    return f if WIDE <= f <= 1 else None
def down(score): return round(score * 1000) // 10 / 100     # down to two decimals

sys.stdout.reconfigure(line_buffering=True)   # keeps these lines in order with the scripts' when output is piped to a file
if len(sys.argv) not in (3, 4, 5): sys.exit("usage: python run.py <file.md> <model id> [floor] [earlier run's folder]     (`lms ls` lists the models)")
src, model = pathlib.Path(sys.argv[1]), sys.argv[2]
floor = earlier = None
for arg in sys.argv[3:]:                      # a number is the floor, a folder with a verdicts.json in it an earlier run
    if floor is None and as_floor(arg) is not None: floor = as_floor(arg)
    elif earlier is None and pathlib.Path(arg, "verdicts.json").is_file(): earlier = pathlib.Path(arg)
    else: sys.exit(f"{arg} is neither a floor (a number between {WIDE:.2f} and 1) nor the folder of an earlier run (no verdicts.json in it)")
if not src.is_file(): sys.exit(f"{src}: file not found")
here = pathlib.Path(__file__).parent
truth = src.with_name("ground_truth.json")   # the answer key, which only the sample has

out, n = pathlib.Path("runs", src.stem), 0
while out.exists(): n += 1; out = pathlib.Path("runs", f"{src.stem} ({n})")
out.mkdir(parents=True)
print(f"results folder: {out}\n")

def step(script, *args, then=""):
    p = subprocess.Popen([sys.executable, str(here / script), *map(str, args)])
    while p.returncode is None:
        try: p.wait()
        except KeyboardInterrupt: pass       # Ctrl+C reaches the script too: wait while it stops what it started in LM Studio
    if p.returncode: sys.exit(f"\n{script} stopped, nothing after it was run. Folder: {out}{then}")

step("candidates_sentence_embeddings.py", src, out, WIDE)
cands = json.load(open(out / "candidates.json", encoding="utf-8"))
if floor is None and earlier and (earlier / "candidates.json").is_file():
    was = [c["score"] for c in json.load(open(earlier / "candidates.json", encoding="utf-8"))]
    if was:
        floor = max(WIDE, down(min(was)))
        print(f"\nFLOOR {floor:.2f} taken from {earlier}: its lowest-scoring pair, down to two decimals.\n"
              f'To overrule it, add a floor: python run.py {src} {model} 0.70 "{earlier}"')
if truth.exists():
    step("score.py", out, truth, f"{src.name}, before choosing FLOOR")
    low = re.search(r"Lowest-scoring true duplicate: ([\d.]+)", (out / "score.md").read_text(encoding="utf-8"))   # score.py's line
    if floor is None and low:
        floor = down(float(low.group(1)))
        print(f"\nFLOOR {floor:.2f} chosen by the program: the highest one that keeps every true duplicate Stage 1 found.\n"
              f"To overrule it, add a floor at the end: python run.py {src} {model} 0.70")
else:
    print("\n FLOOR   pairs for the judge")
    for h in range(round(WIDE * 100), 100, 5): print(f"  {h / 100:.2f}   {sum(c['score'] >= h / 100 for c in cands):>8}")
    if floor is None: print("No answer key for this file, so the program cannot choose the FLOOR itself.\n"
                            "Type the one it chose on your sample run, or a higher one if that is too many pairs.")
while floor is None:
    try: floor = as_floor(input(f"\nFLOOR to use, a number between {WIDE:.2f} and 1: "))
    except (KeyboardInterrupt, EOFError): sys.exit(f"\nstopped before the judge. Folder: {out}")
below = [c for c in cands if c["score"] < floor]   # not thrown away: Stage 2 reads those that have a line in a group
cands = [c for c in cands if c["score"] >= floor]
if not cands: sys.exit(f"no pair scores {floor} or more. Folder: {out}")
for name, part in (("candidates.json", cands), ("below_floor.json", below)):
    json.dump(part, open(out / name, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"\nFLOOR {floor}: {len(cands)} pairs go to the judge. The {len(below)} below it are kept in below_floor.json:\n"
      "the judge reads one of those only when one of its lines turns out to be in a group of duplicates.\n")

note = f"{src.name}, judge {model}, FLOOR {floor}"
step("llm_filtering.py", model, out, *[earlier] * bool(earlier), then=f'\nOnce it has finished, the score is made with:\n  python score.py "{out}" "{truth}" "{note}"' * truth.exists())
if truth.exists(): step("score.py", out, truth, note)
print(f"\nFINAL REVIEW, the file to read and merge from:  {out / 'review.md'}")
if truth.exists(): print(f"Score against the answer key:                   {out / 'score.md'}")
