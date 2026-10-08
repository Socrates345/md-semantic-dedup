# md-semantic-dedup

De-duplicate semantically similar sentences within a large unstructured Markdown file, locally for privacy (bge-m3 embeddings shortlist, LM Studio model)

Find notes that say the same thing more than once in a large, messy Markdown file, so you can merge them by hand. A thought written three times comes out as one group of three lines, not as three loose pairs. Your file is only read, never changed. Everything runs on your machine: the embedding model on your GPU, the judge in LM Studio.

One command does everything:
```
python run.py <file.md> <model id>
```
Each run gets its own new folder, `runs\<file name>`. **`review.md` in that folder is the final file: the one you read to merge.**

What the command runs, in order:
- **Stage 1** (`candidates_sentence_embeddings.py`): compares every note with every other and keeps the pairs that look alike -> `candidates.json`
- **FLOOR**, the similarity a pair needs to be sent to the judge: on the sample the program chooses it, on a file without an answer key it asks you
- **Stage 2** (`llm_filtering.py`): a local model reads each kept pair and says SAME, OVERLAP or DIFFERENT. Lines it calls SAME form a group; it then reads the pairs under the FLOOR that touch a group, and the pairs still unread inside each group -> `review.md`
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
| FLOOR | instant | a short `Stage 1` summary, then `FLOOR 0.60 chosen by the program` and `982 pairs go to the judge`; the pairs below it are kept in `below_floor.json` |
| Stage 2 | depends on the model and the GPU; roughly a second per pair as a first guess | `started LM Studio's server`, `lms unload --all`, `loading ...`, `judge: ...`, then `near-misses: ... pairs under the FLOOR have a line in a group` (once or several times) and `cross-check: ... pairs inside groups that Stage 1 had not paired`, then `lms unload --all`, `lms server stop`, the number of groups and the counts per verdict |
| Score | instant | the `Stage 1` summary again, now for that FLOOR, and a `Stage 2` summary |
| End | | `FINAL REVIEW, the file to read and merge from: runs\sample\review.md` |

Note that Stage 2 starts LM Studio's server if it is not running, then **unloads every model LM Studio has in memory** before loading the judge, so the judge is the only model loaded. When it is done, also after an error or Ctrl+C, it unloads again and stops the server if it started it.


## 3. Reading the results

What is in a run folder:

| File | What it is |
|---|---|
| `review.md` | The groups of lines judged SAME, then the OVERLAP and UNCLEAR pairs outside them, with line numbers. Its first lines name the judge and give the counts. |
| `score.md` | Sample only. How the run did against the answer key. It checks the method; you do not merge from it. |
| `candidates.json` | the pairs at or above the FLOOR, best score first: the judge reads them all |
| `below_floor.json` | the pairs Stage 1 scored between 0.50 and the FLOOR: the judge reads those that have a line in a group |
| `verdicts.json` | every pair the judge read, with its verdict, where the pair came from (`via`: `floor`, `near` or `cross`) and whether the verdict was taken over from an earlier run (`kept`) |
| `verdicts_so_far.jsonl` | only while the judge is running, or after a run that stopped: the verdicts given so far. It is removed when the run completes. |

`sample\answer_key.md` lists every planted pair in the sample, by kind.

`review.md`, how to read it:
- **A group** is a set of lines the judge called SAME, directly or through another line: if it says A = B and B = C, then A, B and C are one group. Each group is listed once, its lines in file order. That is one merge decision.
- **`3 lines, 3 of 3 pairs SAME`**: the judge has read every pair inside the group, also those Stage 1 never paired, and agreed on all of them. Groups like this come first. A group of 2 lines has no count: it is one pair, judged SAME.
- **`4 lines, 4 of 6 pairs SAME`**, followed by `Not SAME according to the judge: line 60 / line 556 DIFFERENT`: the group holds together through some lines only. Either one SAME was wrong and two ideas got joined, or the judge was too strict on the named pair. These groups come last; read them before merging.
- **`Overlaps with this group:`** lists lines judged OVERLAP with a line of the group: they share part of it, usually with a detail more or less. OVERLAP never puts a line into a group.
- **OVERLAP** and **UNCLEAR**, after the groups: the pairs of that kind whose lines are in no group.
- A group of more than 8 lines says `judged pairs`: there the judge read each line against the group's best-linked line, not every pair.

`score.md`, section by section:
- **Stage 1**: how many pairs were kept and how many of the planted duplicates are among them.
- **FLOOR table**: the same table as on screen, starting at the FLOOR that was used.
- **Stage 1 by kind of duplicate**: one line per kind, e.g. `PP  14/14  Paraphrase ...`: 14 of the 14 planted paraphrases were found, with each pair's similarity score underneath. `(<-- look here)` marks a kind where something expected was missed. `(expected miss)` is a known limit. The four kinds at the bottom (`NF`, `CF`, `NG`, `TP`) are look-alikes that are **not** duplicates. Stage 1 lets some through on purpose; rejecting them is the judge's job.
- **Top unlabelled pairs**: the best-scoring pairs that are not in the answer key.
- **Stage 2**: `True duplicates shown together in review.md: x/129` is the result of the whole run: the planted duplicates whose two lines `review.md` puts side by side, in a group or as a listed pair. The brackets say what got them there: a pair at or above the FLOOR, a near-miss, a cross-check, or only the group they share. `Acceptable verdicts: x/y` is the judge's score against the answer key, pair by pair.
- **Ideas written 3 times or more that came out as one group**: one line per such idea, e.g. `GR07  7 versions: all in one group`, or which versions landed in which group and which in none.
- **Groups mixing different planted ideas** should be empty. An entry is a group that holds two different planted ideas, or both sides of a look-alike: a wrong SAME joined them.
- **True duplicates the judge hid as DIFFERENT** is the list that matters most: the judge said DIFFERENT and no group holds both lines, so the pair never reaches `review.md`. It should be empty or nearly so.
- **Hard negatives the judge did not reject** is noise you would have to dismiss by hand, e.g. "rent goes up to 1,150" against "rent goes up to 1,250".

## 4. Comparing two judge models

Run the command once per model:
```
python run.py sample\sample.md qwen/qwen3.5-9b
python run.py sample\sample.md <another model id from lms ls>
```
Both get the same FLOOR, since the program chooses it from Stage 1 alone. Each run prints its folder on the first line. Open the two `score.md`: the title of each names the judge and the FLOOR. Pick the model with the fewest hidden duplicates and no mixed groups.

## 5. Tuning

- **FLOOR**: the lowest similarity a pair needs to be sent to the judge. Lower finds more duplicates and gives the judge more to read.
  - **On the sample the program chooses**: the highest FLOOR that keeps every true duplicate Stage 1 found, which is the most duplicates for the fewest pairs. In the example, that is 0.60: 124 duplicates, 982 pairs.
  - The rule always goes for the most duplicates, whatever they cost. The last 9 duplicates seen from the logs (0.65 to 0.60) cost 587 more pairs, and a single planted pair with a low score would pull FLOOR down further. To overrule it, add a floor at the end: `python run.py sample\sample.md qwen/qwen3.5-9b 0.70`.
  - **On the real file the program asks**, because without an answer key it cannot tell a true duplicate from a look-alike. It shows how many pairs each FLOOR would send to the judge. Type the FLOOR chosen on the sample, or a higher one if that is more pairs than you want to wait for. You can also give it on the command line as above, and then nothing is asked.
  - To compare two values, run again with the other one. A higher FLOOR loses less than the FLOOR table says, because the judge still reads the pairs under it that touch a group: compare `True duplicates shown together in review.md` in the two `score.md`.
- **Groups**: `CROSS_ALL` in `llm_filtering.py` (8). In a group of up to that many lines the judge reads every pair; in a larger one, each line against the best-linked line, so that a thought repeated 200 times costs 199 pairs and not 19,900.
- **Your own test case:** in `sample\sample_source.md`, wrap two wordings as `{{PP15.a}}first wording{{/}}` and `{{PP15.b}}second wording{{/}}`, then `python build_sample.py` and rerun. The two-letter prefixes are listed at the top of `build_sample.py`. Use invented text only: the sample is part of the repository.
- **The judge's question:** `PROMPT` in `llm_filtering.py`.

## 6. To process your real file

`candidates.json`, `below_floor.json`, `verdicts.json` and `review.md` contain the file's text word for word. If the file is confidential, do the real run in a folder that nothing else reads: not a cloud-synced folder, not a folder shared with a dev container or a coding agent, not a git checkout you push from.

1. Make such a folder.
2. Copy `run.py`, `candidates_sentence_embeddings.py` and `llm_filtering.py` into it, plus the real file.
3. In that folder: create a venv and install the packages as in section 1. bge-m3 is already on this machine from the sample run, so nothing is downloaded.
4. Run it. It shows the pair counts and asks for FLOOR (section 5):
   ```
   python run.py yourfile.md qwen/qwen3.5-9b
   ```
   There is no answer key, so no `score.md` is written.
5. Open `runs\yourfile\review.md` next to the real file and merge, one group at a time. Line numbers point to the exact line (Ctrl+G).
6. Delete the `runs` folder when you are done.


## Starting from an earlier run

The judge is the slow part, and its verdict on a pair depends only on the two texts. So a run can take over the verdicts of an earlier run of the same file and ask the judge only what is new. Add the earlier run's folder at the end:
```
python run.py yourfile.md qwen/qwen3.5-9b "runs\yourfile"
```
- A new folder is made as always, here `runs\yourfile (1)`. The earlier one is only read.
- Stage 1 runs again; that is the quick part. FLOOR is the earlier run's (its lowest-scoring pair, down to two decimals) and nothing is asked. To use another, add it: `python run.py yourfile.md qwen/qwen3.5-9b 0.65 "runs\yourfile"`.
- A pair with the same two texts as in the earlier `verdicts.json` keeps that verdict, wherever its lines are now. So it also works on a file you have edited since: only the pairs with a changed line are new. The earlier run's UNCLEAR pairs are asked again.
- The first lines of `review.md` say how many verdicts were taken over.
- Use the same judge and the same `PROMPT` as in the earlier run. The program cannot check that, and would mix the verdicts of two judges.

What it is for:
- **A run made before the groups existed**, with pairs only: the command above gives it the groups. The judge reads only the near-misses and the cross-checks.
- **Another FLOOR**: with a lower one, only the pairs between the two FLOORs are new, plus what they bring in.

Without running Stage 1 again, on the earlier folder itself:
```
python llm_filtering.py qwen/qwen3.5-9b "runs\yourfile" "runs\yourfile"
```
The first folder is the one to work in, the second the one to take verdicts from. A run from before the groups did not keep the pairs under the FLOOR, so there are no near-misses to read then: the judge reads the cross-checks only. This replaces `review.md` and `verdicts.json` in that folder, so copy the folder first if you want to keep the old ones.

## Running one part by itself

`run.py` is the normal way. The parts also run alone on a run folder, which is useful after changing `score.py`:
```
python score.py "runs\sample (1)"
```
The first lines of each script give its arguments. `llm_filtering.py <model id> <folder>` continues a run that stopped part-way. On a folder whose run had finished, it judges everything again and replaces `verdicts.json` and `review.md`. On a folder without `below_floor.json` (Stage 1 run by hand) there are no near-misses to read; the groups are still built and cross-checked.


More background on the method, and troubleshooting, is in `docs\method.md`.
