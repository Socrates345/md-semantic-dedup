# Method notes

Background for `readme.md`: why the pipeline has two stages, what each one does to the text, and what to expect from the result.

Kind of file this program is meant for: A large Markdown file with hundreds of thousands of lines, written over years, full of notes that may repeat each other in different words
and above all, a confidential file you cannot feed a conventional LLM. The aim is to find duplicates within the md file and sort them with a standard local GPU (the machine runs local models of 9 to 27 billion parameters at best). Additionally, no local model could hold the whole file in its context, so the work is split:
a small embedding model compares every note with every other and shortlists the pairs that look alike, and a local LLM reads only that shortlist, one pair at a time.

## Stage 0: test on the synthetic sample first
`sample\sample.md` is an invented diary of about 6,400 words, shaped like a real notes file: no dates, `#` and `---` splits, mostly `-` bullets, loose lines, some French, some code words. It holds 150 labelled pairs: 129 true duplicates of 15 kinds and 21 look-alikes that must not be merged. `sample\answer_key.md` lists them all.

```
python build_sample.py                             # only needed after editing sample\sample_source.md
python run.py sample\sample.md qwen/qwen3.5-9b     # everything else, into a new folder runs\sample, runs\sample (1), ...
```
`run.py` runs Stage 1 with a wide net (every pair scoring 0.50 or more), scores it against the answer key, shows the FLOOR table and chooses the FLOOR itself: the highest one that keeps every true duplicate Stage 1 found (0.60 on the sample today). It then cuts `candidates.json` to that FLOOR, runs Stage 2 on it and scores again. A third argument, `python run.py sample\sample.md <model id> 0.70`, overrules the choice. For a file with no answer key the program cannot choose, so it shows how many pairs each FLOOR keeps and asks. A run never writes outside its own folder, so earlier runs are kept.

`sample\sample_source.md` is the master copy of the sample and is never fed to the pipeline. In it every planted duplicate is wrapped in markers, e.g. `{{PP03.a}}one wording{{/}}` and `{{PP03.b}}another wording{{/}}`. `build_sample.py` strips the markers to produce `sample.md` and the answer key. To add a tricky case of your own, wrap both versions the same way (the prefixes are listed at the top of `build_sample.py`) and rebuild. Only invented text goes in there.

## Stage 1 script: candidates_sentence_embeddings.py
The embedding model is bge-m3, which is multilingual. With the CUDA build of torch installed (see the readme's setup), Stage 1 uses the NVIDIA GPU on its own, no code change.
The first run downloads bge-m3 (about 2.3 GB) from huggingface.co. That is the only contact with it: Stage 1 turns on Hugging Face's offline switch (`HF_HUB_OFFLINE=1`) for itself before it reads the file, and when the model is missing a separate process, which never reads the file, fetches it.
Every non-blank line of the file is treated as one note: bullet, heading and checkbox markers are stripped, `---` lines and short `#` titles are skipped, and notes under 15 characters are dropped. Only paragraphs over 200 characters are split into sentences. Lines are never joined back together, so a sentence hard-wrapped over two lines is seen as two notes.
Each note keeps its 5 closest matches, if they score at or above the FLOOR.
Run alone, the script's FLOOR defaults to 0.70. On the sample 0.70 keeps 107 of the 129 true duplicates and sends 156 pairs to the judge; 0.75 keeps 94 and sends 109; 0.65 keeps 115 but sends 395; 0.60 keeps 124 but sends 982, and that is the one `run.py` chooses.
On a file without an answer key `run.py` asks for FLOOR. Start from the value it chose on the sample. To check it against the document itself, open candidates.json in the run folder while the question is waiting and skim it:
real duplicates cluster at the top, and somewhere down the list they stop appearing.
The score at that point is the FLOOR for that document.

## Stage 2 script: llm_filtering.py
The judge runs in LM Studio, and the script drives it with LM Studio's own `lms` command so the window never has to open: `python llm_filtering.py qwen/qwen3.5-9b` (the model id as `lms ls` shows it).
It starts the server on 127.0.0.1:1234 if it is not up, and loads and unloads the models as the readme's section 2 describes. A pair LM Studio fails on is tried up to 4 times in all, 15 seconds apart, with the judge reloaded from the second failure on. Each verdict is appended to `verdicts_so_far.jsonl` as it comes, so a run that still dies is continued by running the same command on the same folder. A server the script starts is bound to this machine only; one that was already running is used as it is (see the warning in the readme's privacy section).
One-time setup in the LM Studio app: open it once (that installs `lms`), download the model, and untick Thinking for it. The judge only has to answer one word, and reasoning makes every pair many times slower.
Besides review.md it writes verdicts.json, which score.py reads; both go into the run folder when `run.py` drives it. To use a server you manage yourself, set `LLM_BASE_URL` and the script starts and stops nothing. Your notes then go to that address, so for a confidential file keep it on this machine.

## Things to expect
- The same idea may appear three or four times, which shows up as several overlapping pairs. When you review, handle it as one group with a single merge decision.
- Some redundancy works at a different scale. For example, one sentence might summarize a whole paragraph elsewhere. Sentence-level embeddings will catch some of these but not all. A second pass with paragraphs as units instead of sentences would pick up more of them; the scripts do not do that today.

## Troubleshooting

| What you see | What to do |
|---|---|
| `lms (LM Studio's CLI) is not on PATH` | Open the LM Studio app once, then open a new PowerShell window. |
| `lms load failed` | The model id is not the one `lms ls` shows. |
| `Many UNCLEAR` at the end of Stage 2 | Thinking is still on for the model: untick it in LM Studio and rerun. |
| `the download failed, see above` on the first Stage 1 run | bge-m3 could not be fetched from huggingface.co: check the internet connection and run again. |
| `pair N: LM Studio failed (...). Try 2 of 4 in 15 s` | Nothing to do: it asks again by itself, and reloads the judge if that is not enough. |
| `LM Studio failed 4 times on pair ...` | LM Studio would not come back. The verdicts so far are saved. Let the machine cool down, close other GPU-heavy programs, then run the `python llm_filtering.py ...` line it printed (and, on the sample, the `python score.py ...` line after it). |
| `lms unload --all failed` | LM Studio could not empty its memory, so the judge was not loaded on top. Open the app, eject the models there, and run again. |
| `torch.cuda.is_available()` prints `False` | The CPU build of torch got installed: `python -m pip uninstall torch`, then the `--index-url` line again. |
| `... stopped, nothing after it was run` | The script named there failed or was interrupted; its own message is just above. Fix that and run again: you get a new folder. |
| `no pair scores ... or more` | The FLOOR you typed is above every pair's score. Run again with a lower one. |
| A run folder without `review.md` | That run was stopped before the judge finished. With a `verdicts_so_far.jsonl` in it, `python llm_filtering.py <model id> "<folder>"` continues it; otherwise it is safe to delete. |
