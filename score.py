# Score Stage 1 / Stage 2 output against the planted answer key (sample only, no dependencies)
#   python score.py [run folder] [ground_truth.json] [note]
# defaults: this folder, sample/ground_truth.json. Reads candidates.json and, if it is there, verdicts.json
# from the run folder. The report goes to score.md in that folder; the console only gets the headlines and
# the FLOOR table. The note ends up in the report's title (run.py puts the judge and the FLOOR there).
import re, os, sys, json, itertools, collections

args = sys.argv[1:] + [None] * 3
OUT = args[0] or "."
truth = json.load(open(args[1] or "sample/ground_truth.json", encoding="utf-8"))
cands = json.load(open(os.path.join(OUT, "candidates.json"), encoding="utf-8"))
verd_path = os.path.join(OUT, "verdicts.json")
sys.stdout.reconfigure(encoding="utf-8")
CATS = truth["categories"]

md = [f"# Score: {args[2] or os.path.basename(os.path.abspath(OUT))}", "",
      "A check of the method against the sample's answer key. "
      "**This is not the file to merge from: that is `review.md`.**"]
say = md.append
def section(title, items=()):                 # heading, then the headline facts as bullets (also shown on the console)
    md.extend(["", f"## {title}", "", *("- " + i for i in items)])
    if items: print("\n" + title, *("  " + i for i in items), sep="\n")
def block(lines):                             # listings keep their fixed-width layout
    if lines: md.extend(["", "```text", *lines, "```"])
def finish(problem=None):
    if problem: md.extend(["", problem])
    path = os.path.join(OUT, "score.md")
    with open(path, "w", encoding="utf-8") as f: f.write("\n".join(md) + "\n")
    print(f"\nfull report -> {path}")
    sys.exit(problem)

def norm(s):
    return " ".join(re.sub(r"[^\w\s]", " ", s.lower().replace("_", " ")).split())

def same_text(sentence, span):
    a, b = norm(sentence), norm(span)
    if not a or not b: return False
    if a in b or b in a: return True          # Stage 1 may cut a span short or glue extras onto it
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / min(len(ta), len(tb)) >= 0.8

spans = [(it["id"], member, s["line"], s["text"])
         for it in truth["items"] for member, ss in it["members"].items() for s in ss]

# Stage 1 reports the line the text starts on; the window allows for a span that starts lower in a paragraph
def who(line, sentence):
    return {(i, m) for i, m, ln, text in spans if 0 <= ln - line <= 10 and same_text(sentence, text)}

def planted_pair(c):
    for i, m in who(c["a_line"], c["a"]):
        for j, n in who(c["b_line"], c["b"]):
            if i == j and m != n: return i, tuple(sorted((m, n)))

def short(s, n=110): return s if len(s) <= n else s[:n - 1] + "…"
def show(c): return f"    L{c['a_line']}: {short(c['a'])}\n    L{c['b_line']}: {short(c['b'])}"
def tags(c):
    t = sorted({f"{i}.{m}" for side in ("a", "b") for i, m in who(c[side + "_line"], c[side])})
    return "  [touches " + ", ".join(t) + "]" if t else ""

pairs = [(it["id"], pair) for it in truth["items"]
         for pair in itertools.combinations(sorted(it["members"]), 2)]
members = {it["id"]: it["members"] for it in truth["items"]}
found, unlabelled = {}, []
for c in cands:
    key = planted_pair(c)
    if key: found[key] = max(found.get(key, 0), c["score"])
    else: unlabelled.append(c)

def too_short(iid, pair):     # same 15-char minimum as Stage 1
    return any(all(len(s["text"]) < 15 for s in members[iid][m]) for m in pair)

# ---------------------------------------------------------------- Stage 1
if not cands: finish("candidates.json is empty: FLOOR is too high, or Stage 1 found no sentences")
scores = [c["score"] for c in cands]
dups = [p for p in pairs if CATS[p[0][:2]]["kind"] == "dup"]
negs = [p for p in pairs if CATS[p[0][:2]]["kind"] == "neg"]
kept = [found[p] for p in dups if p in found]
section("Stage 1", [f"{len(cands)} candidate pairs, scores {min(scores):.3f} to {max(scores):.3f}",
                    f"True duplicates found: {len(kept)}/{len(dups)}",
                    f"Hard negatives let through (Stage 2 has to reject them): {sum(p in found for p in negs)}/{len(negs)}",
                    f"Unlabelled pairs (not in the answer key): {len(unlabelled)}"]
                   # run.py reads this line back to choose the FLOOR: keep its wording in step with the search there
                   + [f"Lowest-scoring true duplicate: {min(kept):.3f} (any FLOOR up to that keeps all {len(kept)})"] * bool(kept))

table = ["| FLOOR | true dups kept | hard negatives kept | unlabelled | pairs for the judge |",
         "|------:|---------------:|--------------------:|-----------:|--------------------:|"]
for h in range(max(40, round(min(scores) * 1000) // 50 * 5), 96, 5):      # whole hundredths: 0.05 steps drift as floats
    t = h / 100
    d = sum(found.get(p, 0) >= t for p in dups); n = sum(found.get(p, 0) >= t for p in negs)
    u = sum(c["score"] >= t for c in unlabelled)
    sent = sum(c["score"] >= t for c in cands)        # can exceed d + n + u: two candidates may be the same planted pair
    table.append(f"| {t:5.2f} | {f'{d}/{len(dups)}':>14} | {f'{n}/{len(negs)}':>19} | {u:>10} | {sent:>19} |")
section("FLOOR table")
md.extend(["What each FLOOR would keep. The rows start at the lowest score in this folder's candidates.json.", "", *table])
print("", *table, sep="\n")

section("Stage 1 by kind of duplicate")
say("Found / planted for each kind, with every found pair's similarity underneath.")
out = []
for cat, c in CATS.items():
    mine = [p for p in pairs if p[0][:2] == cat]
    hit = [p for p in mine if p in found]
    if c["kind"] == "neg": note = "hard negatives let through (Stage 2 has to reject them)"
    elif c["stage1"] == "miss": note = "expected miss"
    elif c["stage1"] == "partial": note = "partial expected"
    else: note = "ok" if len(hit) == len(mine) else "<-- look here"
    out.append(f"{cat}  {len(hit):>2}/{len(mine):<2}  {c['description']}  ({note})")
    if any(len(members[i]) > 2 for i, _ in mine):      # repeated ideas: what matters is that all versions end up linked
        for iid in sorted({i for i, _ in mine}):
            link = {m: m for m in members[iid]}
            def root(m):
                while link[m] != m: m = link[m]
                return m
            got = [p for p in hit if p[0] == iid]
            for _, (m, n) in got: link[root(m)] = root(n)
            groups = len({root(m) for m in link})
            lost = ["-".join(p[1]) + "*" * too_short(*p) for p in mine if p[0] == iid and p not in found]
            out.append(f"        {iid}  {len(got)}/{len(got) + len(lost)} pairs, {len(link)} versions "
                       + ("all linked" if groups == 1 else f"SPLIT into {groups} groups") + ":  "
                       + "  ".join(f"{'-'.join(p[1])} {found[p]:.3f}" for p in got)
                       + ("   missing: " + " ".join(lost) if lost else "")
                       + ("   (* = under 15 chars, never embedded)" if any("*" in l for l in lost) else ""))
        continue
    if hit: out.append("        " + "   ".join(f"{p[0]} {found[p]:.3f}" for p in hit))
    for p in mine:
        if p not in found:
            why = "under 15 chars, dropped before embedding" if too_short(*p) else "below FLOOR or outside top-K"
            out.append(f"        {p[0]} missing: {why}")
block(out)

section("Top unlabelled pairs")
say("Not in the answer key, so presumed DIFFERENT; check that they really are.")
block([f"  {c['score']:.3f}{tags(c)}\n{show(c)}" for c in sorted(unlabelled, key=lambda c: -c["score"])[:20]])

# ---------------------------------------------------------------- Stage 2
if not os.path.exists(verd_path):
    section("Stage 2")
    say("Not scored: the judge has not run in this folder yet (no verdicts.json).")
    finish()
verdicts = json.load(open(verd_path, encoding="utf-8"))
LABELS = ["SAME", "OVERLAP", "DIFFERENT", "UNCLEAR"]
conf, per_cat, bg = collections.Counter(), collections.defaultdict(lambda: [0, 0]), collections.Counter()
hidden, noise, bg_same, strict, ok, judged = [], [], [], 0, 0, 0
for v in verdicts:
    key = planted_pair(v)
    if not key:
        bg[v["verdict"]] += 1
        if v["verdict"] == "SAME": bg_same.append(v)
        continue
    cat = CATS[key[0][:2]]
    judged += 1
    conf[cat["expected"], v["verdict"]] += 1
    strict += v["verdict"] == cat["expected"]
    good = v["verdict"] in cat["acceptable"]
    ok += good; per_cat[key[0][:2]][0] += good; per_cat[key[0][:2]][1] += 1
    if cat["kind"] == "dup" and v["verdict"] == "DIFFERENT": hidden.append((key[0], v))
    if cat["kind"] == "neg" and not good: noise.append((key[0], v))

reached = f"{len(verdicts)} pairs judged, {judged} of them labelled ({len(pairs) - judged} labelled pairs never reached the judge)"
if not judged:
    section("Stage 2", [reached])
    finish("no labelled pair in the verdicts, nothing to score")
section("Stage 2", [reached,
                    f"Exact agreement with the answer key: {strict}/{judged} ({100 * strict / judged:.0f}%)",
                    f"Acceptable verdicts: {ok}/{judged} ({100 * ok / judged:.0f}%)",
                    f"True duplicates the judge hid as DIFFERENT (these vanish from review.md): {len(hidden)}",
                    f"Hard negatives the judge did not reject: {len(noise)}",
                    "Unlabelled pairs: " + ", ".join(f"{bg[l]} {l}" for l in LABELS)])
md.extend(["", "| answer key \\ judge | " + " | ".join(LABELS) + " |", "|---|" + "---:|" * len(LABELS)])
for e in LABELS[:3]:
    say(f"| {e} | " + " | ".join(str(conf[e, l]) for l in LABELS) + " |")
md.extend(["", "Acceptable per kind: " + "   ".join(f"{cat} {a}/{n}" for cat, (a, n) in per_cat.items())])

section(f"True duplicates the judge hid as DIFFERENT: {len(hidden)}")
say("These pairs never reach review.md. The list should be empty or nearly so.")
block([f"  {iid}  score {v['score']:.3f}\n{show(v)}" for iid, v in hidden])
section(f"Hard negatives the judge did not reject: {len(noise)}")
say("Look-alikes that are not duplicates but still show up in review.md.")
block([f"  {iid}  judged {v['verdict']}\n{show(v)}" for iid, v in noise])
section(f"Unlabelled pairs judged SAME: {len(bg_same)}")
say("Not in the answer key. The first 15, best score first.")
block([f"  score {v['score']:.3f}{tags(v)}\n{show(v)}" for v in bg_same[:15]])
finish()
