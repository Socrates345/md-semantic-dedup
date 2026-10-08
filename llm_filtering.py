# Stage 2: local LLM judge through LM Studio, driven headless with its own `lms` CLI (localhost only)
#   python llm_filtering.py qwen/qwen3.5-9b [folder] [earlier run's folder]      model id as `lms ls` shows it
# Reads candidates.json from the folder (default: here) and writes verdicts.json and review.md into it.
# The judge reads one pair at a time, in three rounds:
#   1. every pair of candidates.json
#   2. near-misses: the pairs of below_floor.json (those run.py found under the FLOOR) that have a line in a group
#   3. cross-check: the pairs inside each group that Stage 1 had not paired
# A group is the lines joined by SAME verdicts, directly or through another line (A = B and B = C: A, B and C).
# review.md lists each group once, with every line in it.
# Starts the server if it is not up, unloads every model LM Studio has in memory so the judge is the only one
# loaded, judges every pair, then unloads it and stops the server if it started it, also on an error or Ctrl+C.
# The LM Studio window never has to open.
# A pair LM Studio fails on is asked again, reloading the judge if need be. Every verdict is saved as it comes
# (verdicts_so_far.jsonl), so if the run still dies, the same command on the same folder picks up where it stopped.
# With an earlier run's folder as last argument, the SAME, OVERLAP and DIFFERENT of its verdicts.json are taken over:
# a pair with the same two texts is not asked again, wherever its lines now are. Only for the same judge and the
# same PROMPT, which is not checked. This is how a run from before the groups gets them without being judged again.
# One-time setup in the app: run it once (that puts `lms` on PATH) and untick Thinking for the model.
# With LLM_BASE_URL set, nothing is started or stopped: the script just talks to that server.
import os, re, sys, json, time, shutil, itertools, collections, subprocess, requests
MANAGE = "LLM_BASE_URL" not in os.environ
PORT, CONTEXT, TTL = 1234, 4096, 1800   # TTL: LM Studio unloads the model itself after 30 idle minutes if this script dies
TRIES, WAIT = 4, 15                     # tries per pair and seconds between them; from the second failure on, the judge is reloaded first
CROSS_ALL = 8                           # round 3 judges every pair of a group up to this many lines; in a larger one, each line against the best-linked line
BASE = os.environ.get("LLM_BASE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
MODEL = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
def read(name): return json.load(open(os.path.join(OUT, name), encoding="utf-8"))
cands = read("candidates.json")
below = read("below_floor.json") if os.path.exists(os.path.join(OUT, "below_floor.json")) else []   # only run.py makes it
if any("a_id" not in c for c in cands + below):          # a folder from before the notes were numbered: number them here
    ids = {}
    for c in cands + below:
        for s in "ab": c[s + "_id"] = ids.setdefault((c[s + "_line"], c[s]), len(ids))
for via, part in (("floor", cands), ("near", below)):
    for c in part: c["via"] = via                        # the round a pair is from; round 3 makes its own, "cross"
note = {c[s + "_id"]: (c[s + "_line"], c[s]) for c in cands + below for s in "ab"}   # id -> (line, text)
def key(c): return tuple(sorted((c["a_id"], c["b_id"])))
judged = []                                              # every pair that has its verdict, in the order they got it
KEPT = {}                                                # an earlier run's verdicts by the pair's two texts: taken over, not asked again
if len(sys.argv) > 3:
    for v in json.load(open(os.path.join(sys.argv[3], "verdicts.json"), encoding="utf-8")):
        if v["verdict"] != "UNCLEAR": KEPT[tuple(sorted((v["a"], v["b"])))] = v["verdict"]   # an UNCLEAR is worth asking again
    print(f"earlier run {sys.argv[3]}: its verdicts on {len(KEPT)} pairs of texts are taken over where the same pair comes up")

SO_FAR = os.path.join(OUT, "verdicts_so_far.jsonl")      # one line per judged pair; removed once verdicts.json is written
earlier = {}                                             # what an interrupted run left there: pair -> (its line numbers, verdict, kept)
if os.path.exists(SO_FAR):
    for line in open(SO_FAR, encoding="utf-8"):
        try: d = json.loads(line); earlier[tuple(d["pair"])] = (d["lines"], d["verdict"], d.get("kept", False))
        except (ValueError, KeyError, TypeError): continue   # a line cut short when the run died, or one from before the rounds
    print(f"picking up an interrupted run: {len(earlier)} pairs are already judged")
def saved(): return (f"{len(judged)} verdicts are saved. To continue from there:\n"
                     f'  python llm_filtering.py {MODEL or "<model id>"} "{OUT}"')

def lms(*args, timeout=60):
    exe = shutil.which("lms")
    if not exe: sys.exit("lms (LM Studio's CLI) is not on PATH: run the LM Studio app once, it installs it.")
    # lms prints UTF-8 spinner glyphs that Windows' default codec chokes on, so the encoding is pinned
    return subprocess.run([exe, *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)

def server_up():
    try: return requests.get(f"{BASE}/v1/models", timeout=5).ok
    except requests.RequestException: return False

def load_judge():      # two models in memory at once overheat the machine: empty LM Studio first, whatever was loaded
    global loaded_model
    print("lms unload --all (so the judge is the only model in memory)")
    r = lms("unload", "--all")
    if r.returncode != 0: sys.exit("lms unload --all failed, not loading the judge on top:\n" + r.stdout + r.stderr)
    print(f"loading {MODEL} (context {CONTEXT})")
    loaded_model = True      # from here on the cleanup at the end unloads, even if the load stops half-way
    r = lms("load", MODEL, "--context-length", str(CONTEXT), "--ttl", str(TTL), timeout=300)
    if r.returncode != 0: sys.exit("lms load failed:\n" + r.stdout + r.stderr)

def ask(c):             # the judge's raw answer for one pair; raises with LM Studio's own words when it gives none
    r = requests.post(f"{BASE}/v1/chat/completions", timeout=600, json={
        "model": MODEL, "messages": [{"role": "user", "content": PROMPT.format(**c)}],
        "temperature": 0, "max_tokens": 1024, "stream": False})
    try: return r.json()["choices"][0]["message"].get("content") or ""
    except (ValueError, KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

PROMPT = """Do these two sentences express the same meaning or information, so that one is redundant?
A: {a}
B: {b}
Answer with exactly one word: SAME, OVERLAP, or DIFFERENT."""

def judge(pairs):       # a verdict for each of these pairs, saved as it comes; one an interrupted or an earlier run already judged is not asked again
    for n, c in enumerate(pairs, 1):
        k = key(c); lines = [note[x][0] for x in k]; texts = tuple(sorted((c["a"], c["b"])))
        if earlier.get(k, [None])[0] == lines: c["verdict"], c["kept"] = earlier[k][1:]
        else:
            c["kept"] = texts in KEPT                    # kept: the verdict is the earlier run's, the judge was not asked
            if c["kept"]: c["verdict"] = KEPT[texts]
            else:
                for attempt in range(1, TRIES + 1):
                    try: resp = ask(c); break
                    except (requests.RequestException, RuntimeError) as e:
                        if attempt == TRIES: sys.exit(f"\nLM Studio failed {TRIES} times on pair {n} of {len(pairs)}. The last answer:\n{e}\n\n{saved()}")
                        print(f"pair {n}: LM Studio failed ({str(e)[:160]}). Try {attempt + 1} of {TRIES} in {WAIT} s")
                        time.sleep(WAIT)
                        if MANAGE and attempt >= 2: load_judge()
                resp = re.sub(r"<think>.*?(</think>|$)", "", resp, flags=re.S).upper()   # drop reasoning, it names every label
                said = re.findall(r"SAME|OVERLAP|DIFFERENT", resp)
                c["verdict"] = said[-1] if said else "UNCLEAR"
            so_far.write(json.dumps({"pair": k, "lines": lines, "verdict": c["verdict"], "kept": c["kept"]}) + "\n")
            so_far.flush()
        judged.append(c)
        if n % 100 == 0: print(n, "/", len(pairs))

def groups():           # the lines joined by SAME verdicts, directly or through another line: one list of ids per group, in file order
    link = {}
    def root(x):
        link.setdefault(x, x)
        while link[x] != x: link[x] = link[link[x]]; x = link[x]
        return x
    for c in judged:
        if c["verdict"] == "SAME": link[root(c["a_id"])] = root(c["b_id"])
    found = collections.defaultdict(list)
    for x in sorted(link, key=lambda x: (note[x][0], x)): found[root(x)].append(x)
    return list(found.values())

started_server = loaded_model = False
try:
    if MANAGE and not MODEL: sys.exit("usage: python llm_filtering.py <model id>     (`lms ls` lists them)")
    if MANAGE and server_up(): print("server already up: leaving it running when done")
    elif MANAGE:
        r = lms("server", "start", "--port", str(PORT), "--bind", "127.0.0.1")
        if r.returncode != 0 and not server_up(): sys.exit("lms server start failed:\n" + r.stdout + r.stderr)
        started_server = True
        print(f"started LM Studio's server on port {PORT}")
    if MANAGE: load_judge()
    if not MODEL:
        ids = [m["id"] for m in requests.get(f"{BASE}/v1/models", timeout=30).json()["data"]]
        if len(ids) != 1: sys.exit("Pass the model name as argument, one of:\n  " + "\n  ".join(ids))
        MODEL = ids[0]
    print("judge:", MODEL)

    with open(SO_FAR, "a", encoding="utf-8") as so_far:
        judge(cands)
        while True:     # round 2. A line that joins a group brings its own near-misses along, hence the loop
            grouped = {x for g in groups() for x in g}
            near = [c for c in below if "verdict" not in c and (c["a_id"] in grouped or c["b_id"] in grouped)]
            if not near: break
            print(f"near-misses: {len(near)} pairs under the FLOOR have a line in a group")
            judge(near)
        have = {key(c) for c in judged}
        same = collections.Counter(x for c in judged if c["verdict"] == "SAME" for x in key(c))
        cross = []      # round 3. Every pair Stage 1 listed with a line in a group is judged by now, so these have no score
        for g in groups():
            hub = max(g, key=lambda x: same[x])
            want = itertools.combinations(g, 2) if len(g) <= CROSS_ALL else ((hub, x) for x in g if x != hub)
            cross += [{"score": None, "a_id": a, "a_line": note[a][0], "a": note[a][1],
                       "b_id": b, "b_line": note[b][0], "b": note[b][1], "via": "cross"}
                      for a, b in map(sorted, want) if (a, b) not in have]
        if cross: print(f"cross-check: {len(cross)} pairs inside groups that Stage 1 had not paired")
        judge(cross)
except KeyboardInterrupt: sys.exit(f"\nstopped (Ctrl+C). {saved()}")
finally:
    for mine, args in ((loaded_model, ("unload", "--all")), (started_server, ("server", "stop"))):
        if not mine: continue
        print("lms", *args)
        try: lms(*args)
        except (OSError, subprocess.SubprocessError) as e: print("  failed, LM Studio's idle timeout will clean up:", e)

count = {v: sum(c["verdict"] == v for c in judged) for v in ("SAME", "OVERLAP", "DIFFERENT", "UNCLEAR")}
with open(os.path.join(OUT, "verdicts.json"), "w", encoding="utf-8") as f: json.dump(judged, f, ensure_ascii=False, indent=1)

gs = groups()
of = {x: g for g, lines in enumerate(gs) for x in lines}   # line -> its group
inside, around = [[] for _ in gs], [set() for _ in gs]     # per group: the judged pairs inside it, the lines outside it that OVERLAP one of its lines
loose = {"OVERLAP": [], "UNCLEAR": []}                     # the pairs outside the groups that are still worth a look
for c in judged:
    a, b = of.get(c["a_id"]), of.get(c["b_id"])
    if a is not None and a == b: inside[a].append(c)
    elif c["verdict"] == "OVERLAP" and (a, b) != (None, None):
        if a is not None: around[a].add(c["b_id"])
        if b is not None: around[b].add(c["a_id"])
    elif c["verdict"] in loose: loose[c["verdict"]].append(c)
order = sorted(range(len(gs)), key=lambda g: any(c["verdict"] != "SAME" for c in inside[g]))   # fully agreed first, file order within
num = {g: n for n, g in enumerate(order, 1)}
left_out = count["DIFFERENT"] - sum(c["verdict"] == "DIFFERENT" for ins in inside for c in ins)
kept = sum(c["kept"] for c in judged)
if kept: print(f"{kept} verdicts taken over from the earlier run, {len(judged) - kept} asked")
print(f"{len(gs)} groups of lines that say the same thing, {len(of)} lines in them")
print(count, "-> verdicts.json and review.md in", os.path.abspath(OUT))
if count["UNCLEAR"] > len(judged) / 10:
    print("Many UNCLEAR: the model is probably spending its tokens on reasoning. Untick Thinking for it in LM Studio.")

scores, by = [c["score"] for c in cands] or [0], collections.Counter(c["via"] for c in judged)
won = collections.Counter(c["via"] for c in judged if c["verdict"] == "SAME")   # what rounds 2 and 3 brought, for the cost they had
with open(os.path.join(OUT, "review.md"), "w", encoding="utf-8") as f:
    f.write(f"# FINAL REVIEW: {os.path.basename(os.path.abspath(OUT))}\n\n"
            "This is the file to read and merge from. Line numbers are those of the source file (Ctrl+G).\n\n"
            f"Judge {MODEL} read {len(judged)} pairs: {by['floor']} that Stage 1 scored {min(scores):.3f} to {max(scores):.3f}, "
            f"{by['near']} under that with a line in a group ({won['near']} of them SAME), "
            f"and {by['cross']} inside groups that Stage 1 had not paired ({won['cross']} of them SAME)."
            + f" {kept} of these verdicts are an earlier run's, taken over without asking again." * bool(kept) + "\n\n"
            "Lines the judge called SAME, directly or through another line, form one group. The groups it fully agreed on "
            "come first, then those where it did not call every pair SAME. A line that only OVERLAPs a group is listed under it.\n\n"
            f"After the groups come the {len(loose['OVERLAP'])} OVERLAP and {len(loose['UNCLEAR'])} UNCLEAR pairs outside them. "
            f"{left_out} DIFFERENT are left out.\n\n"
            f"## SAME: {len(gs)} groups, {len(of)} lines\n\n")
    for g in order:
        size, same = len(gs[g]), sum(c["verdict"] == "SAME" for c in inside[g])
        f.write(f"### Group {num[g]}: {size} lines"
                + f", {same} of {len(inside[g])} {'judged ' * (len(inside[g]) < size * (size - 1) // 2)}pairs SAME" * (size > 2) + "\n\n")
        for x in gs[g]: f.write(f"**line {note[x][0]}** {note[x][1]}\n\n")
        odd = [f"line {c['a_line']} / line {c['b_line']} {c['verdict']}" for c in inside[g] if c["verdict"] != "SAME"]
        if odd: f.write("Not SAME according to the judge: " + ", ".join(odd) + "\n\n")
        if around[g]: f.write("Overlaps with this group:\n\n")
        for x in sorted(around[g], key=lambda x: (note[x][0], x)):
            f.write(f"**line {note[x][0]}** {note[x][1]}" + f" (in group {num.get(of.get(x))})" * (x in of) + "\n\n")
        f.write("---\n\n")
    for v in loose:
        f.write(f"## {v} ({len(loose[v])})\n\n")
        for c in sorted(loose[v], key=lambda c: -c["score"]):
            f.write(f"**line {c['a_line']}** {c['a']}\n\n**line {c['b_line']}** {c['b']}\n\n---\n\n")
os.remove(SO_FAR)
