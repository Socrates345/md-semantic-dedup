# md-semantic-dedup

De-duplicate semantically similar sentences within a large unstructured Markdown file, locally for privacy (bge-m3 embeddings shortlist, LM Studio model)

Find notes that say the same thing twice in a large, messy Markdown file, so you can merge them by hand. Your file is only read, never changed. Everything runs on your machine: the embedding model on your GPU, the judge in LM Studio.

One command does everything:
```
python run.py <file.md> <model id>
```
Each run gets its own new folder, `runs\<file name>`. **`review.md` in that folder is the final file: the one you read to merge.**

What the command runs, in order:
- **Stage 1** (`candidates_sentence_embeddings.py`): compares every note with every other and keeps the pairs that look alike -> `candidates.json`
- **FLOOR**, the similarity a pair needs to be sent to the judge: on the sample the program chooses it, on a file without an answer key it asks you
- **Stage 2** (`llm_filtering.py`): a local model reads each kept pair and says SAME, OVERLAP or DIFFERENT -> `review.md`
- **Scorer** (`score.py`), sample only: compares what the two stages found with the answer key -> `score.md`

**Sample** (`sample\sample.md`): an invented diary with 150 planted and labelled pairs, to try all of this before touching the real file.

It was written and used on Windows with an NVIDIA GPU, Python 3.12 and LM Studio. All commands below are PowerShell, run from this folder.

## Privacy by default
- Stage 1 is computed locally, with Hugging Face's offline switch on (`HF_HUB_OFFLINE=1`).
- One download from Hugging Face, on the first run: the bge-m3 model.
- LM Studio runs the judge locally, on `127.0.0.1`.
- Nothing from your file leaves the machine, in either stage.

**Warning:** that holds with the defaults only. Before a confidential run, check that:
- `LLM_BASE_URL` is not set. If it is, Stage 2 sends your notes to that address.
- no proxy applies to `127.0.0.1` (`HTTP_PROXY`, `ALL_PROXY`). If one does, set `NO_PROXY` to `127.0.0.1`, or your notes pass through it.
- LM Studio's server, if it is already running, is not served on the local network. The program uses it as it finds it.
- `HF_HUB_OFFLINE` is not set to `0`. If it is, Stage 1 contacts huggingface.co about the model, though not with your text.


## 1. Setup (once)

### venv
```
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

### packages
```
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install sentence-transformers numpy requests
python -c "import torch; print(torch.cuda.is_available())"
```
The last line should print `True`. `cu126` is for NVIDIA cards up to the RTX 40 series; for an RTX 50 card replace it with `cu130`. If it prints `False`, Stage 1 still works, only slower (CPU).

### LM Studio (prerequisite)
1. Open the LM Studio app once. That installs its `lms` command.
2. Download the judge model.
3. Untick **Thinking** for that model.
4. Close the app if you like, then check: `lms ls` should list the model. The name in that list is what you pass to `run.py`.

## 2. Try it on the sample

```
python run.py sample\sample.md qwen/qwen3.5-9b
```

What happens, in order:

| Step | Takes | What you see |
|---|---|---|
| Folder | instant | `results folder: runs\sample` (or `runs\sample (1)`, `(2)`, ... when that name is taken) |
| Stage 1 | the first run downloads the bge-m3 model (2.3 GB), after that seconds on the GPU | `439 sentences`, then about `1611 candidate pairs scoring 0.50 or more` |
| FLOOR | instant | a short `Stage 1` summary, then `FLOOR 0.60 chosen by the program` and `982 pairs go to the judge` |
| Stage 2 | depends on the model and the GPU; roughly a second per pair as a first guess | `started LM Studio's server`, `lms unload --all`, `loading ...`, `judge: ...`, then `lms unload --all`, `lms server stop`, and the counts per verdict |
| Score | instant | the `Stage 1` summary again, now for that FLOOR, and a `Stage 2` summary |
| End | | `FINAL REVIEW, the file to read and merge from: runs\sample\review.md` |

Note that Stage 2 starts LM Studio's server if it is not running, then **unloads every model LM Studio has in memory** before loading the judge, so the judge is the only model loaded. When it is done, also after an error or Ctrl+C, it unloads again and stops the server if it started it.


## 3. Reading the results

What is in a run folder:

| File | What it is |
|---|---|
| `review.md` | The pairs judged SAME, OVERLAP or UNCLEAR, with line numbers. Its first lines name the judge and give the counts. |
| `score.md` | Sample only. How the run did against the answer key. It checks the method; you do not merge from it. |
| `candidates.json` | the pairs sent to the judge (those at or above the FLOOR), best score first |
| `verdicts.json` | the same pairs with the judge's verdict |
| `verdicts_so_far.jsonl` | only while the judge is running, or after a run that stopped: the verdicts given so far. It is removed when the run completes. |

`sample\answer_key.md` lists every planted pair in the sample, by kind.

`score.md`, section by section:
- **Stage 1**: how many pairs were kept and how many of the planted duplicates are among them.
- **FLOOR table**: the same table as on screen, starting at the FLOOR that was used.
- **Stage 1 by kind of duplicate**: one line per kind, e.g. `PP  14/14  Paraphrase ...`: 14 of the 14 planted paraphrases were found, with each pair's similarity score underneath. `(<-- look here)` marks a kind where something expected was missed. `(expected miss)` is a known limit. The four kinds at the bottom (`NF`, `CF`, `NG`, `TP`) are look-alikes that are **not** duplicates. Stage 1 lets some through on purpose; rejecting them is the judge's job.
- **Top unlabelled pairs**: the best-scoring pairs that are not in the answer key.
- **Stage 2**: `Acceptable verdicts: x/y` is the judge's score against the answer key.
- **True duplicates the judge hid as DIFFERENT** is the list that matters most: those pairs never reach `review.md`. It should be empty or nearly so.
- **Hard negatives the judge did not reject** is noise you would have to dismiss by hand, e.g. "rent goes up to 1,150" against "rent goes up to 1,250".

## 4. Comparing two judge models

Run the command once per model:
```
python run.py sample\sample.md qwen/qwen3.5-9b
python run.py sample\sample.md <another model id from lms ls>
```
Both get the same FLOOR, since the program chooses it from Stage 1 alone. Each run prints its folder on the first line. Open the two `score.md`: the title of each names the judge and the FLOOR. Pick the model with the fewest hidden duplicates.

## 5. Tuning

- **FLOOR**: the lowest similarity a pair needs to be sent to the judge. Lower finds more duplicates and gives the judge more to read.
  - **On the sample the program chooses**: the highest FLOOR that keeps every true duplicate Stage 1 found, which is the most duplicates for the fewest pairs. In the example, that is 0.60: 124 duplicates, 982 pairs.
  - The rule always goes for the most duplicates, whatever they cost. The last 9 duplicates seen from the logs (0.65 to 0.60) cost 587 more pairs, and a single planted pair with a low score would pull FLOOR down further. To overrule it, add a floor at the end: `python run.py sample\sample.md qwen/qwen3.5-9b 0.70`.
  - **On the real file the program asks**, because without an answer key it cannot tell a true duplicate from a look-alike. It shows how many pairs each FLOOR would send to the judge. Type the FLOOR chosen on the sample, or a higher one if that is more pairs than you want to wait for. You can also give it on the command line as above, and then nothing is asked.
  - To compare two values, run again with the other one.
- **Your own test case:** in `sample\sample_source.md`, wrap two wordings as `{{PP15.a}}first wording{{/}}` and `{{PP15.b}}second wording{{/}}`, then `python build_sample.py` and rerun. The two-letter prefixes are listed at the top of `build_sample.py`. Use invented text only: the sample is part of the repository.
- **The judge's question:** `PROMPT` in `llm_filtering.py`.

## 6. To process your real file

`candidates.json`, `verdicts.json` and `review.md` contain the file's text word for word. If the file is confidential, do the real run in a folder that nothing else reads: not a cloud-synced folder, not a folder shared with a dev container or a coding agent, not a git checkout you push from.

1. Make such a folder.
2. Copy `run.py`, `candidates_sentence_embeddings.py` and `llm_filtering.py` into it, plus the real file.
3. In that folder: create a venv and install the packages as in section 1. bge-m3 is already on this machine from the sample run, so nothing is downloaded.
4. Run it. It shows the pair counts and asks for FLOOR (section 5):
   ```
   python run.py yourfile.md qwen/qwen3.5-9b
   ```
   There is no answer key, so no `score.md` is written.
5. Open `runs\yourfile\review.md` next to the real file and merge. Line numbers point to the exact line (Ctrl+G).
6. Delete the `runs` folder when you are done.


## Running one part by itself

`run.py` is the normal way. The parts also run alone on a run folder, which is useful after changing `score.py`:
```
python score.py "runs\sample (1)"
```
The first lines of each script give its arguments. `llm_filtering.py <model id> <folder>` continues a run that stopped part-way. On a folder whose run had finished, it judges everything again and replaces `verdicts.json` and `review.md`.


More background on the method, and troubleshooting, is in `docs\method.md`.
