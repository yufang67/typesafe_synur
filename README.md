# SYNUR observation extraction with JEV

A Python notebook experiment using **JEV directly** to classify/extract observations
from clinical text against SYNUR's 193-concept source schema. The notebook currently
enables **SINGLE_SELECT, MULTI_SELECT, and NUMERIC**; only STRING prediction,
reference labels, and evaluation are excluded. JEV is the only inference model;
this is not a small-model / verifier / larger-model cascade.

**Data is downloaded separately. The notebook reads local files only.** Model
credentials are not included, and live calls are disabled by default. Without
credentials you can inspect the dataset, annotation issues, source candidates,
and the complete question plan, but no JEV predictions or accuracy are claimed.

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

The notebook currently selects **row ID `152` in `mediqa_synur_dev`**, not the
152nd row. `ROW_ID` is matched exactly within `SPLIT`; a missing ID is an error,
never a fallback to the first row. Set `ROW_ID = None` to use `SAMPLE_LIMIT` again.
The walkthrough displays the transcript and filtered reference labels first,
then live model observations, then error counts, observation comparisons, and
precision/recall/F1. Enter credentials only in the masked setup prompt.

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

The entire selected registry is considered for each selected transcript. Independent
questions are batched, and follow-ups depend on completed decisions. Source
labels never enter the request or candidate generator.

### Uncertainty and limits

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

## Enable live JEV calls later

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

Alternatively, set `LIVE_CALLS = True` in the notebook and run the **API key setup**
cell before inference. It reuses a non-placeholder environment key or opens a
masked input prompt. Paste the key into the prompt, never into cell source. The
entered key is stored only in the running kernel's environment, not in a file or
notebook output. Restart the kernel to discard a key entered through the prompt.
With live calls disabled, this cell does not request a key. Setup does not call
the service or verify account access.

Placeholder keys are rejected before sending requests. The default walkthrough
uses dev row ID `152`; choose another `ROW_ID` explicitly or set it to `None`
before increasing `SAMPLE_LIMIT`.
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
repair mechanism.

The notebook reports raw and normalized exact observation micro
precision/recall/F1, concept scores, per-type summaries, review/failure counts,
and coverage. Observation order and multi-select order do not matter.
Repeated reference observations use multiset matching. Free text uses exact
comparison, not an unreported semantic judge. Missing predictions, abstentions,
and failed rows remain in end-to-end recall. Empty-denominator conventions are
documented in the evaluation module; no model run means unavailable metrics.
These are local baseline metrics, **not the official shared-task scorer**.

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

`SAVE_REPORT = True` (the default) saves a new local
`results\report_<unique_id>\transcript_report.json` after notebook evaluation.
It includes every selected transcript (currently dev row `152`), not unevaluated
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

Reports use normalized scored reference labels by default and retain all original
labels, normalization changes, and validation issues for audit. The notebook
passes the unfiltered source rows to the exporter; callers of the Python report
functions should likewise pass original rows with the active schema registry.
Disabled reference types are retained as `SKIP` without normalization.
The Python `save_transcript_report` function also accepts `reference_view="raw"`.
Missing/failed predictions are marked unavailable, with null error tags, counts,
and metrics rather than invented model results. A successful empty extraction,
in contrast, correctly reports deletions. The export uses a new directory each
time and prints its full path; it never overwrites earlier reports.

The final field in the version-2 JSON is `micro_metrics`, containing pooled
`tp`, `fp`, `fn`, `precision`, `recall`, and `f1` for the selected reference view.
These rates are calculated from total observation counts, not the mean of
per-transcript rates. `SKIP` labels are excluded. As in the evaluator, missing and
failed rows contribute reference misses to end-to-end micro recall when at least
one requested extraction is available; their individual rates remain unavailable.
With no available extraction, micro counts/rates are null. Zero-denominator rates
also remain null.

Set `SAVE_RESULTS = True` after inference to create a new `results\run_...\`
directory containing:

- `predictions.jsonl`: observation outputs and row status.
- `audit.jsonl`: full decisions, evidence, request metadata, and failure context.
- `failures.jsonl`: explicit errors.
- `metrics.json` and `run.json`: scores, enabled observation types, source revision/checksums, selected rows,
  model versions, thresholds, request hashes, and available usage information.

## Development

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src scripts tests
.\.venv\Scripts\python.exe -m pyright
```

Download the dataset first to include the offline notebook integration test.
Unit tests use temporary data and controlled model responses. Committed notebook
cells have no saved outputs or execution counts.

The modules are local loading (`dataset`), schema/reference handling
(`observations`), transcript candidate discovery (`candidates`), native question
compilation (`questions`), the SDK connection (`jev`), orchestration/export
(`experiment`), local scoring (`evaluation`), and transcript JSON reports (`reporting`).

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
