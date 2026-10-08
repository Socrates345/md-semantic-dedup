# Stage 2: local LLM judge through LM Studio, driven headless with its own `lms` CLI (localhost only)
#   python llm_filtering.py qwen/qwen3.5-9b [folder]      model id as `lms ls` shows it
# Reads candidates.json from the folder (default: here) and writes verdicts.json and review.md into it.
# Starts the server if it is not up, unloads every model LM Studio has in memory so the judge is the only one
# loaded, judges every pair, then unloads it and stops the server if it started it, also on an error or Ctrl+C.
# The LM Studio window never has to open.
# A pair LM Studio fails on is asked again, reloading the judge if need be. Every verdict is saved as it comes
# (verdicts_so_far.jsonl), so if the run still dies, the same command on the same folder picks up where it stopped.
# One-time setup in the app: run it once (that puts `lms` on PATH) and untick Thinking for the model.
# With LLM_BASE_URL set, nothing is started or stopped: the script just talks to that server.
import os, re, sys, json, time, shutil, subprocess, requests
MANAGE = "LLM_BASE_URL" not in os.environ
PORT, CONTEXT, TTL = 1234, 4096, 1800   # TTL: LM Studio unloads the model itself after 30 idle minutes if this script dies
TRIES, WAIT = 4, 15                     # tries per pair and seconds between them; from the second failure on, the judge is reloaded first
BASE = os.environ.get("LLM_BASE_URL", f"http://127.0.0.1:{PORT}").rstrip("/")
MODEL = sys.argv[1] if len(sys.argv) > 1 else None
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
cands = json.load(open(os.path.join(OUT, "candidates.json"), encoding="utf-8"))
SO_FAR = os.path.join(OUT, "verdicts_so_far.jsonl")      # one line per judged pair; removed once verdicts.json is written
if os.path.exists(SO_FAR):
    for line in open(SO_FAR, encoding="utf-8"):
        try: d = json.loads(line)
        except ValueError: continue                      # a line cut short when the run died
        if d["i"] < len(cands) and d["pair"] == [cands[d["i"]]["a_line"], cands[d["i"]]["b_line"]]:
            cands[d["i"]]["verdict"] = d["verdict"]
    print(f"picking up an interrupted run: {sum('verdict' in c for c in cands)} of {len(cands)} pairs are already judged")
def saved(): return (f"{sum('verdict' in c for c in cands)} of {len(cands)} verdicts are saved. To continue from there:\n"
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
        for n, c in enumerate(cands, 1):
            if "verdict" in c: continue                  # judged before the run was interrupted
            for attempt in range(1, TRIES + 1):
                try: resp = ask(c); break
                except (requests.RequestException, RuntimeError) as e:
                    if attempt == TRIES: sys.exit(f"\nLM Studio failed {TRIES} times on pair {n} of {len(cands)}. The last answer:\n{e}\n\n{saved()}")
                    print(f"pair {n}: LM Studio failed ({str(e)[:160]}). Try {attempt + 1} of {TRIES} in {WAIT} s")
                    time.sleep(WAIT)
                    if MANAGE and attempt >= 2: load_judge()
            resp = re.sub(r"<think>.*?(</think>|$)", "", resp, flags=re.S).upper()   # drop reasoning, it names every label
            said = re.findall(r"SAME|OVERLAP|DIFFERENT", resp)
            c["verdict"] = said[-1] if said else "UNCLEAR"
            so_far.write(json.dumps({"i": n - 1, "pair": [c["a_line"], c["b_line"]], "verdict": c["verdict"]}) + "\n")
            so_far.flush()
            if n % 100 == 0: print(n, "/", len(cands))
except KeyboardInterrupt: sys.exit(f"\nstopped (Ctrl+C). {saved()}")
finally:
    for mine, args in ((loaded_model, ("unload", "--all")), (started_server, ("server", "stop"))):
        if not mine: continue
        print("lms", *args)
        try: lms(*args)
        except (OSError, subprocess.SubprocessError) as e: print("  failed, LM Studio's idle timeout will clean up:", e)

results = {v: [c for c in cands if c["verdict"] == v] for v in ("SAME", "OVERLAP", "DIFFERENT", "UNCLEAR")}
with open(os.path.join(OUT, "verdicts.json"), "w", encoding="utf-8") as f: json.dump(cands, f, ensure_ascii=False, indent=1)
count = {v: len(results[v]) for v in results}
print(count, "-> verdicts.json and review.md in", os.path.abspath(OUT))
if count["UNCLEAR"] > len(cands) / 10:
    print("Many UNCLEAR: the model is probably spending its tokens on reasoning. Untick Thinking for it in LM Studio.")

scores = [c["score"] for c in cands] or [0]
with open(os.path.join(OUT, "review.md"), "w", encoding="utf-8") as f:
    f.write(f"# FINAL REVIEW: {os.path.basename(os.path.abspath(OUT))}\n\n"
            "This is the file to read and merge from. Line numbers are those of the source file (Ctrl+G).\n\n"
            f"Judge {MODEL} read {len(cands)} pairs (similarity {min(scores):.3f} to {max(scores):.3f}): "
            f"{count['SAME']} SAME, {count['OVERLAP']} OVERLAP and {count['UNCLEAR']} UNCLEAR are listed below, "
            f"{count['DIFFERENT']} DIFFERENT are left out.\n\n")
    for v in ("SAME", "OVERLAP", "UNCLEAR"):
        f.write(f"## {v} ({len(results[v])})\n\n")
        for c in results[v]:
            f.write(f"**line {c['a_line']}** {c['a']}\n\n**line {c['b_line']}** {c['b']}\n\n---\n\n")
os.remove(SO_FAR)
