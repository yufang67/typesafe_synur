# SYNUR observation extraction with JEV

A Python notebook experiment using **JEV directly** to classify/extract observations
from clinical text against SYNUR's 193-concept source schema. The notebook currently
enables **162 concepts across SINGLE_SELECT, MULTI_SELECT, and NUMERIC**.
STRING prediction and scoring are disabled; original STRING references remain
visible as `SKIP` in reports. JEV is the only inference model; this is not a
small-model / verifier / larger-model cascade.

**Data is downloaded separately. The notebook reads local files only.** Model
credentials are not included. **The current saved notebook has `LIVE_CALLS = True`
and selects all 101 dev transcripts.** Review its configuration before running
cells. For credential-free inspection, turn off both live calls and run-artifact
export as described below; dataset diagnostics, source candidates, the status-question
plan, and example value questions do not require inference.

## Current status and features

Snapshot: **2026-09-20**. The recorded full-dev run of `jev-1.13.0` has exact
observation micro **precision 83.74%, recall 77.35%, and F1 80.42%**, with
101/101 completed transcripts. These are local development-set results, not a
held-out benchmark or a clinical performance claim. See [Recorded results](#recorded-results).

| Area | Implemented behavior |
| --- | --- |
| Local data | Pinned download, manifest/checksum verification, split-local IDs, source files left unchanged |
| Schema scope | Runtime type selection; 130 single-select, 12 multi-select, and 20 numeric concepts currently enabled |
| Extraction | Status-first JEV Choice questions, exact enum selection, per-member Noul decisions, source-grounded scalar selection |
| Optional STRING support | Engine supports hierarchical clause/token/span selection; 31 STRING concepts are disabled in the current experiment |
| Offline inspection | Reference diagnostics, normalization audit, numeric candidates, request preview, fixture-driven extraction tests |
| Reliability | Validated typed answers, bounded request batches, confidence-based review, explicit partial/failure states |
| Evaluation | Raw/normalized exact micro scores, concept/per-type scores, COR/DEL/INS/SUB alignment, audit and candidate coverage |
| Reports | Full format-v3 JSON plus a short JSON with ID, transcript, comparisons, and metrics; no `raw_expected_observations` |
| Run artifacts | Predictions, full audit/provenance, failures, metrics, dataset/model/settings metadata in new local directories |

## Setup (PowerShell)

Run from this repository's root. Python 3.11 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[notebook,dev]"
```

Download once, **outside Jupyter**:

```powershell
.\.venv\Scripts\python.exe scripts\download_synur.py --output data\synur
```

The local `data\synur\` directory contains the original JSONL files, observation
schema, upstream dataset card, and a revision/checksum manifest. Files are pinned
to Hugging Face revision `c7f79af4dcc8e5fb175c40cef0592d85a76bf11c`.
The setup script verifies cached files instead of silently trusting them.
Dataset files and experiment results are git-ignored.
Loading also checks manifest completeness, byte and row counts, schema/row
structure, duplicate JSON keys and row IDs, UTF-8 validity, and unsafe
symlink/junction paths. Invalid snapshots fail explicitly; they are not silently
repaired or downloaded by the notebook.

| Split | Rows | Role |
| --- | ---: | --- |
| `mediqa_synur_train` | 122 | Development |
| `mediqa_synur_dev` | 101 | Initial evaluation |
| `mediqa_synur_test` | 199 | Held-out evaluation |
| `original` | 223 | Original release; overlaps train/dev |

Do not combine `original` with train/dev as additional independent examples.
IDs are split-local. The dataset contains synthetic **nurse dictations**, not
doctor-patient conversations. The workflow also accepts conversation text, but
SYNUR scores alone do not establish performance on dialogue.

Open the notebook:

```powershell
.\.venv\Scripts\python.exe -m jupyterlab notebooks\synur_observation_extraction.ipynb
```

Use the project environment's Python kernel. Start Jupyter from the repository
root or `notebooks` directory. Optional `SYNUR_DATA_DIR` selects another copy of the
local dataset. Missing/corrupt local files produce an error; the notebook never
downloads replacements.

`ENABLED_VALUE_TYPES` in the configuration cell controls the experiment scope.
The notebook selects the model schema and copies a matching in-memory label view
for every split before reference diagnostics or evaluation. Disabled types do not
produce model questions, source candidates, false negatives, or coverage/review
denominators. The source dataset and full schema are unchanged; excluded label
counts are shown in the data summary. After changing scope, restart the kernel
and run from the configuration cell so older predictions are not reused.

### Current notebook configuration

These are the values in the saved configuration cell, not guaranteed defaults
for a future revision:

| Setting | Current value | Meaning |
| --- | --- | --- |
| `SPLIT` | `'mediqa_synur_dev'` | Development split |
| `ROW_ID` | `None` | Select by sample limit rather than exact ID |
| `SAMPLE_LIMIT` | `101` | First 101 rows in source order, currently the entire dev split |
| `ENABLED_VALUE_TYPES` | `('SINGLE_SELECT', 'MULTI_SELECT', 'NUMERIC')` | Exclude STRING from inference and scoring |
| `MODEL` | `TYPESAFE_MODEL`, falling back to `'jev-1.13.0'` | Requested inference model |
| `LIVE_CALLS` | `True` | Run JEV after API key setup |
| `SAVE_RESULTS` | `True` | Export predictions, audit, failures, metrics, and run metadata |
| `SAVE_REPORT` | `True` | Export full and short per-transcript reports |
| `SETTINGS` | `Settings()` | Confidence and batching defaults documented below |

For a safe, small **offline walkthrough**, change the configuration before
running all cells:

```python
ROW_ID = '152'
LIVE_CALLS = False
SAVE_RESULTS = False
SAVE_REPORT = False
```

Disabling `SAVE_RESULTS` matters: that export intentionally raises when there
are no predictions. `SAVE_REPORT` may instead be left on to save reports with
explicitly unavailable model metrics; it makes no model calls.

`ROW_ID = '152'` selects the exact split-local ID, **not the 152nd row**.
A missing ID is an error, never a fallback to the first row. `ROW_ID = None`
uses `SAMPLE_LIMIT`; there is no random sampling in this selection step.
The walkthrough displays the transcript and filtered reference labels first,
then live model observations, then error counts, observation comparisons, and
precision/recall/F1. Enter credentials only in the masked setup prompt.
The notebook retains saved outputs, which may reflect an earlier configuration;
use the timestamped run artifacts as the source of truth for recorded results.

## Design and module boundaries

The model selects among typed, schema-defined choices and source candidates.
Python owns data integrity, candidate enumeration, answer validation, output
assembly, and scoring. Reference labels are used for diagnostics/evaluation,
never as model input.

```text
Separate download -> verified local dataset -> selected schema + source rows
                                                    |
                        transcript + active schema only
                                                    v
                   source candidates + native question plan
                                                    |
                          JEV status questions for every concept
                                                    |
                 supported concepts -> enum/member/candidate questions
                                                    |
                     validated observations + audit + failures
                                                    |
reference labels --------------------------> local evaluation
                                                    |
                                full/short reports + optional run export
```

| Module | Responsibility |
| --- | --- |
| `scripts\download_synur.py` | Download allowlisted pinned files, verify persisted bytes, publish manifest last |
| `src\synur\dataset.py` | Verify and load the local snapshot without network access |
| `src\synur\observations.py` | Schema registry, strict observation validation, separate auditable reference normalization |
| `src\synur\candidates.py` | Transcript-only numeric and contiguous-text candidates, source offsets, candidate diagnostics |
| `src\synur\questions.py` | Native state and Choice/Noul compilation, option hierarchy, request packing |
| `src\synur\jev.py` | Explicit live-call gate, masked API key setup, TypeSafe SDK lifecycle and model-error translation |
| `src\synur\experiment.py` | Offline preview, extraction orchestration, thresholds, validation, audit and run export |
| `src\synur\evaluation.py` | Row-local matching, exact scores, edit alignment, coverage and error diagnostics |
| `src\synur\reporting.py` | Per-transcript report construction, recorded provenance, compact projection, JSON export |
| `notebooks\synur_observation_extraction.ipynb` | Configuration, local inspection, sequential row extraction, result display and export |

The SDK adapter is the live-inference boundary. Tests substitute controlled
adapters; candidate discovery, report generation, and scoring are local and can
be repeated from saved audit records without another model run. Independent
questions are batched within a transcript; the notebook processes rows
sequentially rather than launching concurrent row jobs.

## Does JEV need a system prompt?

**No.** Its documented interface is `state`, typed `questions`, and `model`, not
chat messages with system/user roles.

```python
from typesafe_sdk import Choice

state = {"context": transcript, "schema": registry.to_entries()}
questions = {
    "nausea_value": Choice(
        instructions={
            "rules": "Use only explicit current-patient evidence; do not infer missing findings.",
            "concept": {"id": "30", "name": "Nausea", "value_type": "SINGLE_SELECT"},
            "question": "Which nausea value is supported in context?",
        },
        criteria={
            "yes": "Nausea is explicitly present.",
            "no": "Nausea is explicitly denied or absent.",
            "not_stated": "No nausea finding is stated.",
            "ambiguous": "Evidence is insufficient to determine a value.",
            "conflicting": "There are unresolved contradictory nausea findings.",
        },
    )
}
# Inside an explicitly enabled, credentialed TypeSafe client:
# response = client.system_one(state=state, questions=questions, model="jev-1.13.0")
```

The full implementation compiles questions from the schema rather than
hand-writing individual concepts. Question-map keys only correlate answers;
concept definitions and rules appear in `instructions`. A key named `system`
inside state would be ordinary data, not a privileged system message. The
`system_message` in the linked cascade cookbook describes another extractor's
instructions for its verifier; it is not a required JEV parameter.

## Extraction workflow

1. **Assess every enabled concept.** A Choice for each concept in the selected schema classifies
   source support, absence, ambiguity, or conflict. An explicit negative finding
   is source support, not an unmentioned concept.
2. **Resolve supported values.** Single-select uses Choice over the exact enum;
   multi-select uses an independent Noul per enum member. Numeric values are
   selected from source scalars. STRING extraction remains disabled; if explicitly
   re-enabled in `ENABLED_VALUE_TYPES`, text values are selected through a source
   clause, starting token, and complete contiguous span.
3. **Assemble and validate.** Python maps choices to schema values, copies/parses
   selected source spans, and validates IDs, names, types, and enums. JEV is not
   asked to generate arbitrary JSON.

The emitted observations preserve SYNUR's shape:

```json
[
  {"id": "30", "name": "Nausea", "value_type": "SINGLE_SELECT", "value": "No"}
]
```

This is an **illustrative format**, not a model result. Confidence, probabilities,
source offsets for selected spans, per-concept status, and errors are separate
audit records. Enum outputs have decisions but do not claim a source span that
JEV did not select.

Observations must have exactly `id`, `name`, `value_type`, and `value`, with
canonical schema metadata. Select values must belong to the exact enum;
multi-select lists must be nonempty and duplicate-free; numbers must be finite
and cannot be booleans; STRING values, when enabled, must be nonempty. Duplicate
predicted concept IDs are invalid. Confidence and evidence belong in the audit,
not extra observation fields.

The entire selected registry is considered for each selected transcript. Independent
questions are batched, and follow-ups depend on completed decisions. Source
labels never enter the request or candidate generator.

### Status and provenance

Every concept starts as `pending` and must finish as `emitted`, `absent`,
`review`, or `failed`. `absent` means the finding was not stated; an explicit
negative finding can instead emit a valid observation such as `Nausea = No`.
Low confidence, ambiguity, conflict, missing candidates, and uncertain
multi-select members go to review rather than becoming invented observations.
An empty supported-member set is also reviewed rather than emitted as an empty
multi-select value.

Row results distinguish `complete`, `partial`, and `failed`. A complete row
means extraction finished, **not** that every concept was emitted or that every
prediction is correct. A missing prediction record is a separate evaluation
state. Model-call failures and malformed, omitted, or unexpected answers are
recorded as failures, not converted into clinical absence. Choice probability
maps and selections are validated before assembly.

Audit records retain per-concept status/reason, decisions, available evidence
text and offsets, and linked question IDs. Request records include question
hashes, request IDs, returned model names, and available usage data. Extraction
metadata records `prompt_version`, state/schema hashes, settings, concept
count, and candidate policy. These support investigation without inventing
reasoning or claiming evidence the model did not select.

### Uncertainty and limits

`Settings()` uses these defaults:

| Setting | Value |
| --- | ---: |
| `choice_confidence` | `0.8` |
| `support_probability` | `0.8` |
| `reject_probability` | `0.2` |
| `batch_size` | `32` questions |
| `max_request_bytes` | `60000` |
| `max_state_question_bytes` | `30000` |

- Default Choice acceptance: confidence **>= 0.8**. Default Noul support:
  probability **>= 0.8**; non-support: **<= 0.2**. Between these values, abstain.
  These configurable heuristics are **not clinically calibrated**. Inspect
  Choice confidence separately from option probabilities.
- Ambiguous/conflicting values go to review. An uncertain multi-select member
  prevents emitting a misleadingly complete list for that concept.
- Missing candidates, failed calls, and malformed responses are distinct from
  clinical absence. Partial runs retain explicit failures and coverage.
- When numeric extraction is enabled, numbers retain their original offsets, even when values repeat. Only scalar
  numbers and unambiguous supported spoken-number forms are parsed. Composite
  values such as `120/80`, ranges, and fractions are reported as unsupported,
  not silently reduced to one component. No implicit unit conversion is applied.
- When enabled, text extraction copies a contiguous source span. It cannot generate a
  paraphrase or combine disjoint evidence. Candidate coverage is reported
  separately from model selection quality.
- Choices never exceed **255 options**, including abstention alternatives.
  Larger candidate sets use hierarchical selection without dropping options.
  This hierarchy can lose recall; the decision path remains in the audit.
- Requests use configurable conservative UTF-8 byte budgets (60,000 per
  request, 30,000 for state plus the largest question, including safety overhead).
  These are **not exact token counts**. Oversized requests fail explicitly;
  context and candidates are never silently truncated. JEV's documented model
  limits remain authoritative (64k total, 32k state plus longest question).

## Live JEV calls and credentials

The TypeSafe SDK integration is implemented and pinned to the verified SDK
version. Only credentials and explicit opt-in are needed for a live experiment.
`.env.example` is documentation, not an automatically loaded secrets file.

```powershell
# Placeholder only: replace privately before starting the notebook kernel.
$env:TYPESAFE_API_KEY = "REPLACE_WITH_YOUR_KEY"
$env:TYPESAFE_MODEL = "jev-1.13.0"
# Optional authorized deployment override:
# $env:TYPESAFE_BASE_URL = "https://api.typesafe.ai/"
```

With `LIVE_CALLS = True` (the current saved setting), run the **API key setup**
cell before inference. It reuses a non-placeholder environment key or opens a
masked input prompt. Paste the key into the prompt, never into cell source. The
entered key is stored only in the running kernel's environment, not in a file or
notebook output. Restart the kernel to discard a key entered through the prompt.
With live calls disabled, this cell does not request a key. Setup does not call
the service or verify account access.

Placeholder keys are rejected before sending requests. To limit live inference
to a smoke run, set `ROW_ID = '152'`; the current `ROW_ID = None` and
`SAMPLE_LIMIT = 101` configuration instead processes the full dev split.
Changing environment variables in a separate terminal does not update an
already-running kernel; restart it as needed.

Model calls use bounded SDK retries/timeouts. Model failures are surfaced rather
than converted into empty observations. No live calls are made by the test
fixtures, and fixture decisions are not presented as JEV performance.

**Research only, not clinical decision support.** Only send synthetic or
appropriately authorized data. Do not paste credentials into notebook cells.
Avoid SDK body-level debug logging for clinical data; this project does not
create external transcript-sharing/playground links.

## Reference normalization and evaluation

Raw files are preserved. Only STRING labels are currently filtered out
before both raw and normalized evaluation; NUMERIC labels are included.
Numeric values compare exactly (for example, 150 equals 150.0), without
tolerance or unit conversion. Per-type scores show only enabled
types (plus any unexpected invalid labels). Coverage and review rates use the
selected schema. Within that scope, a separate, auditable evaluation view permits only:

- Zero-padded concept IDs when the canonical ID exists and name/type agree.
- Numeric scalar to text conversion for STRING references, or for a select
  reference when its exact string representation is an allowed enum.
- A narrowly scoped encoding repair only when exactly one valid enum results.

Unresolved values, mismatched metadata, and repeated/conflicting annotations
remain visible. Apart from the explicit type exclusion, no reference label is
silently dropped or resolved by guessing.
Prediction validation stays strict; reference cleanup is not a model-output
repair mechanism. There is no general trimming, case folding, synonym mapping,
or permissive value coercion.

The notebook reports raw and normalized exact observation micro
precision/recall/F1, concept scores, per-type summaries, review/failure counts,
and coverage. Observation order and multi-select order do not matter.
Repeated reference observations use multiset matching. Free text uses exact
comparison, not an unreported semantic judge. Missing predictions, abstentions,
and failed rows remain in end-to-end recall. Empty-denominator conventions are
documented in the evaluation module; no model run means unavailable metrics.
These are local baseline metrics, **not the official shared-task scorer**.

Evaluation identities are `(split, id)`. Duplicate source rows or prediction
records are rejected; an unsplit prediction whose ID occurs in multiple source
splits must be disambiguated. Unrequested predictions and invalid labels remain
visible in diagnostics. Concept-level scores ignore value differences but
still require valid observations; they are distinct from exact-observation
scores.

Coverage includes requested/complete/partial/failed/missing row counts, audited
concept counts, review/failure rates, and schema validity. Candidate coverage
asks whether a reference numeric/text value is representable in source
candidates; it is a post-hoc diagnostic, not model accuracy or input to candidate
generation.

Both reference views include disjoint observation-level edit counts and details:
**correct (C)** for an exact valid match, **substitution (S)** for a remaining
reference/prediction pair with the same concept ID, **insertion (I)** for an
unpaired prediction, and **deletion (D)** for an unpaired reference. Exact matches
are paired first, and matching never crosses rows. Invalid observations cannot
be correct; same-ID malformed pairs count as substitutions. Multi-select lists
are single observations with order-insensitive exact values, not member-level
scores. Thus `TP=C`, `FP=I+S`, and `FN=D+S`:

- Precision: `C / (C + I + S)`
- Recall: `C / (C + D + S)`
- F1: `2C / (2C + I + D + 2S)`

The notebook shows model error counts and scores only when an extraction is
available. A no-call or all-failed run is explicitly unavailable, not a perfect
score or a set of claimed model deletions. Full evaluation diagnostics remain
available in `metrics`; existing false-positive/missing lists include
substitutions, while `alignment` and `edit_counts` partition them without double
counting.

### Per-transcript JSON report

`SAVE_REPORT = True` (currently enabled) saves a full `transcript_report.json` and a
`transcript_report_short.json` in a new local `results\report_<unique_id>\`
directory after notebook evaluation.
It includes every selected transcript (currently all 101 dev rows), not unevaluated
rows from other splits. Original STRING reference labels are retained as `SKIP`
for context but remain excluded from scoring; NUMERIC observations are scored.
Report generation makes no model calls.

Each entry in `transcripts` contains the row ID, split, original transcript,
`expected_observations`, `predicted_observations`, `error_counts`, and its own
`precision`, `recall`, and `f1`. Observation entries contain the observation
object, an `error_type` (`COR`, `DEL`, `INS`, `SUB`, or `SKIP`), and a `comparison_index`
linking to `comparisons`. Each comparison pairs an expected observation and a
prediction. An insertion has a null expected observation and tags only the extra
prediction; a deletion has a null prediction. Substitutions tag both observations
but count once in the error breakdown. Skipped references have no matching
prediction and do not count as deletions, substitutions, or score denominators.
Their count is reported separately as `skipped_expected_count`.

Each predicted observation also has `provenance`: matching concept audit entries
(including any recorded evidence text/offsets, reason, choices, confidences, and
probabilities), linked model request records, and prediction metadata.
`audit_status` is `recorded`, `missing`, or `ambiguous`; duplicate audits are
retained, not silently resolved. The report never invents spans or explanations.
In particular, enum decisions usually have no selected source span and emitted
observations may have no textual reason; their recorded decisions remain available.

Reports use normalized scored reference labels by default and retain
normalization changes and validation issues for audit, without duplicating the
source labels in `raw_expected_observations`. Original labels remain in the
local dataset. The notebook passes the unfiltered source rows to the exporter;
callers of the Python report functions should likewise pass original rows with
the active schema registry.
Disabled reference types are retained as `SKIP` without normalization.
The Python `save_transcript_report` function also accepts `reference_view="raw"`.
Missing/failed predictions are marked unavailable, with null error tags, counts,
and metrics rather than invented model results. A successful empty extraction,
in contrast, correctly reports deletions. The export uses a new directory each
time and prints both full paths; it never overwrites earlier reports.
`save_transcript_report` returns the full report's path.

The short report contains only `id`, `transcript`, `comparisons`, and `metrics`
for each entry in `transcripts`. Its per-transcript `metrics` contain `available`,
`error_counts`, `skipped_expected_count`, `precision`, `recall`, and `f1`.
Comparisons and scores are copied from the full report, including `SKIP` entries
and null metrics for unavailable predictions; metadata, provenance, and duplicate
observation lists are omitted. Both reports end with the same `micro_metrics`.

The full report uses format version 3. The final field, `micro_metrics`, contains pooled
`tp`, `fp`, `fn`, `precision`, `recall`, and `f1` for the selected reference view.
These rates are calculated from total observation counts, not the mean of
per-transcript rates. `SKIP` labels are excluded. As in the evaluator, missing and
failed rows contribute reference misses to end-to-end micro recall when at least
one requested extraction is available; their individual rates remain unavailable.
With no available extraction, micro counts/rates are null. Zero-denominator rates
also remain null.

`SAVE_RESULTS = True` (currently enabled) creates a new `results\run_...\`
directory after inference containing:

- `predictions.jsonl`: observation outputs and row status.
- `audit.jsonl`: full decisions, evidence, request metadata, and failure context.
- `failures.jsonl`: explicit errors.
- `metrics.json` and `run.json`: scores, enabled observation types, source revision/checksums, selected rows,
  model versions, thresholds, request hashes, and available usage information.

Use `audit.jsonl` for a full replay record; `predictions.jsonl` contains the
compact `id`, `status`, and `observations` view. Dataset/split selection and
runtime versions are in `run.json`; detailed per-record settings, hashes, and
usage are in the audit. Both export paths refuse existing directories rather
than overwriting prior runs.

## Recorded results

These results are from existing local artifacts, not synthetic test fixtures.
The metrics and both report views were cross-checked against the recorded audit
outputs and pinned local dataset, without new model calls. Artifact directories
are git-ignored; the summary below records the experiment even when a fresh
checkout does not contain those local files.

### Full development split: 101 transcripts

| Run property | Recorded value |
| --- | --- |
| Run directory | `results\run_ad975c76b78d` |
| Report directory | `results\report_752944a7399f` |
| Run export time | `2026-09-20T19:29:11.639060+00:00` |
| Requested and returned model | `jev-1.13.0` |
| Runtime | Python `3.14.2`, TypeSafe SDK `0.7.0` |
| Prompt / settings | `direct-jev-v1`; `Settings()` defaults shown above |
| Dataset | Pinned SYNUR revision above; `mediqa_synur_dev`, all 101 unique row IDs |
| Scope | 162 enabled concepts; SINGLE_SELECT, MULTI_SELECT, NUMERIC |
| References | 1,245 scored observations; 70 STRING labels retained only as `SKIP` |
| Predictions | 1,150 observations; all schema-valid |
| Completion | 101 complete, 0 partial, 0 failed, 0 missing |

Raw and normalized **aggregate** exact-observation metrics are identical for
this run. Percentages below are rounded to two decimal places; saved JSON
retains full precision.

| Score view | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact observations, raw | 963 | 187 | 282 | 83.74% | 77.35% | 80.42% |
| Exact observations, normalized | 963 | 187 | 282 | 83.74% | 77.35% | 80.42% |
| Concept IDs, normalized | 977 | 173 | 268 | 84.96% | 78.47% | 81.59% |
| SINGLE_SELECT, normalized | 715 | 123 | 236 | 85.32% | 75.18% | 79.93% |
| MULTI_SELECT, normalized | 130 | 37 | 27 | 77.84% | 82.80% | 80.25% |
| NUMERIC, normalized | 118 | 27 | 19 | 81.38% | 86.13% | 83.69% |

The normalized edit breakdown is **963 COR, 268 DEL, 173 INS, and 14 SUB**.
Thus FP is `173 + 14 = 187` and FN is `268 + 14 = 282`; a substitution is
not an additional insertion/deletion in the edit breakdown.

| Diagnostic | Recorded result |
| --- | --- |
| Concept audit coverage | 16,362 / 16,362 (100%) |
| Concept outcomes | 1,150 emitted, 14,196 absent, 1,016 review, 0 failed |
| Review rate | 6.21% of enabled concept assessments |
| Review reasons | 995 low Choice confidence, 18 uncertain-member sets, 2 candidate-not-stated, 1 ambiguous |
| Model request records | 891 |
| Numeric candidate coverage | 132 / 137 expected numeric labels (96.35%); 5 uncovered |
| Prediction issues / recorded failures | 0 / 0 |
| Reference normalization / unresolved issues | 1 change / 1 issue |

The normalization converted row `86`, concept `42`, from numeric `3` to the
exact allowed string `"3"`. Row `112` retains a conflicting repeated
`Orientation` reference (concept `40`); it was not silently deduplicated.
Normalization did not change the aggregate scores.

The largest deletion groups were `Cognitive status` (21), `Dyspnea` (18), and
`Mobility` (15). The largest insertion groups were `Memory status` (25),
`Gastrointestinal symptoms` (20), and `Work of breathing` (16). These are
reference-alignment error counts, not a claim that every disagreement is a
clinically incorrect statement. The dominance of deletions and low-confidence
reviews identifies recall and abstention behavior as areas to investigate;
no causal attribution or threshold improvement has been established.

### Earlier single-row smoke result

`results\report_e4b629085cf0`, exported at
`2026-09-20T19:14:15.588427+00:00`, records dev row `152` with `jev-1.13.0`:
**10 COR, 1 DEL, 1 INS, 1 SUB**, hence TP=10, FP=2, FN=2 and
precision=recall=F1=**83.33%**. It has no skipped labels or recorded failures.
The full and short report files both exist locally. This is an earlier
single-row observation, not an independent test set; do not pool it with the
101-row dev run or assume a later run reproduces it.

### Interpretation and remaining gaps

There is no saved train/test-split benchmark in the current result artifacts,
no repeated-run variability estimate, and no validated latency/cost benchmark.
STRING extraction is implemented but unmeasured in these runs. Exact scoring
does not credit paraphrases or partial multi-select matches; source annotation
issues and unsupported numeric forms remain visible rather than being repaired
to improve scores. Full schema validity and audit coverage do not imply
semantic correctness. The 80.42% F1 is an in-scope, single-run development
baseline, not evidence of generalization to held-out data, doctor-patient
dialogue, or clinical use.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m pyright
```

Unit tests use temporary data and controlled model responses, not JEV accuracy
fixtures. Coverage includes source integrity, schema validation, permitted
reference repairs, exact/multiset alignment, scope gating, no reference leakage
into requests, option limits and byte budgets, threshold boundaries, corrupt
responses, model failures, report projection/provenance, and overwrite protection.
Tests requiring the local dataset are skipped when it is absent.

**Known notebook-test configuration mismatch:** the offline integration test in
`tests\test_notebook.py` still assumes `ROW_ID = '152'` and a single-row sample.
It forces `LIVE_CALLS` and `SAVE_REPORT` off, but does not disable the now-enabled
`SAVE_RESULTS`. Its configuration overrides need to be aligned with the current
full-dev notebook before that integration test can pass; this is not a model
result failure. The credential-cell hygiene test also expects an unexecuted
setup cell with no outputs, unlike the current saved notebook state.

The current notebook contains saved outputs and execution state. Clear outputs
before sharing a newly executed notebook if they contain material you do not
intend to publish, and never put credentials in cell source or output.

## Sources and attribution

[Microsoft SYNUR](https://huggingface.co/datasets/microsoft/SYNUR) is licensed
**CDLA-Permissive-2.0**. Its downloaded dataset card retains attribution and paper
citations. See Corbeil et al., *Empowering Healthcare Practitioners with Language
Models: Structuring Speech Transcripts in Two Real-World Clinical Applications*
([EMNLP 2025](https://aclanthology.org/2025.emnlp-industry.58/)), and the
MEDIQA-SYNUR 2026 shared-task citation in that card.

TypeSafe references:
[native request format](https://docs.typesafe.ai/api),
[state](https://docs.typesafe.ai/concepts/state),
[Python SDK](https://docs.typesafe.ai/sdk/python),
[pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook),
and the [SDE cascade](https://docs.typesafe.ai/cookbooks/sde_cascade)
that motivated question decomposition, but not this workflow's model roles.
