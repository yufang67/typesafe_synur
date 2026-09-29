# Dataset v5: Pi Scorer vs TypeSafe - 2026-09-29

Comparison of the completed Pi Scorer v5/v4 run on September 29 with the saved
TypeSafe JEV v5/v4 run on September 21 in `run_datasetv5_d9726017250f`.
This report covers the full **422-row local collection**, not the 101-row dev
comparison in `comp_dev_pi_typesafe_0929.md`. No new inference was performed.
All deltas are **Pi minus TypeSafe**; **pp** means percentage points. Deltas are
calculated before rounding, so subtracting displayed percentages can differ by
0.01 pp.

## Run scope

| Item | TypeSafe | Pi Scorer |
| --- | --- | --- |
| Returned model | `jev-1.13.0` | `pi-scorer` |
| Run date (UTC) | 2026-09-21 | 2026-09-29 |
| Saved run identifier | `d9726017250f` | `pi-scorer_2026-09-29_110619_409a728f403f` |
| Dataset | `synur_dataset.v5.json` | `synur_dataset.v5.json` |
| Explicit schema | `synur_schema.v4.json` | `synur_schema.v4.json` |
| Dataset split | `local` | `local` |
| Processed transcripts | 422 / 422 | 422 / 422 |
| Complete / partial / failed / missing rows | 416 / 6 / 0 / 0 | 422 / 0 / 0 / 0 |
| Source schema concepts | 198 | 198 |
| Enabled concepts per transcript | 166 | 166 |
| Audited concept decisions | 70,052 / 70,052 | 70,052 / 70,052 |
| Enabled value types | SINGLE_SELECT, MULTI_SELECT, NUMERIC | SINGLE_SELECT, MULTI_SELECT, NUMERIC |
| Prompt version | `direct-jev-v1` | `direct-jev-v1` |
| TypeSafe SDK version | 0.7.0 | 0.7.0 |
| Python version | 3.13.15 | 3.13.15 |
| Returned model requests | 3,722 | 3,782 |
| Request failures | 0 | 0 |
| Response-validation failures | 6 | 0 |
| Recorded execution commit | Not recorded | `961cdb56a5a7da4646079eb7b14ecb5e60f216ca` |

The six TypeSafe partial rows each contain a rejected Choice answer whose
selected option was not a highest-probability option. They remain in the
full-scope evaluation; neither the rows nor their failures were removed or
silently repaired. They are response-validation failures, not failed HTTP/model
requests. Pi has no recorded request or validation failures.

The selected transcript IDs and order, normalization policy, per-transcript
state and schema hashes, candidate policies, and settings match. Both runs use
Choice confidence >= 0.8, Noul support >= 0.8, Noul rejection <= 0.2, batch size
32, a 60,000-byte request budget, and a 30,000-byte state-plus-question budget.
The 31 STRING concepts and one unsupported Date concept are excluded from the
166-concept inference/scoring scope. Neither source manifest records any Date
reference observations.

### Input provenance

Both runs record identical filenames, bytes, row counts, and SHA-256 hashes:

| File | Rows / concepts | Bytes | SHA-256 |
| --- | ---: | ---: | --- |
| `synur_dataset.v5.json` | 422 rows | 3,888,576 | `3e9fc163f2c3d83e324c30014d5ffae72ee27fd929e2764f7f3b887499e52249` |
| `synur_schema.v4.json` | 198 source concepts | 177,623 | `19fc69b61b153ab16dc90d74a7e10777c85465dc3fe43aff911d12d224c16912` |

The full dataset manifests differ **only in the two local input-file paths**.
The bytes used for the inputs are the same. The explicit v4 schema, rather than
the dataset's embedded schema, controls both runs.

## Overall exact observation comparison

These are micro-averaged exact observation metrics, not concept-presence metrics.
All 422 requested rows are included, including TypeSafe's six partial rows.
Raw and normalized observation metrics are identical within each run.

| Metric | TypeSafe `jev-1.13.0` | Pi `pi-scorer` | Delta |
| --- | ---: | ---: | ---: |
| Precision | 83.22% | 76.37% | -6.86 pp |
| Recall | 77.10% | 79.08% | +1.97 pp |
| F1 | 80.05% | 77.70% | -2.35 pp |
| True positives | 4,068 | 4,172 | +104 |
| False positives | 820 | 1,291 | +471 |
| False negatives | 1,208 | 1,104 | -104 |
| Predicted observations | 4,888 | 5,463 | +575 |
| Reference observations | 5,276 | 5,276 | 0 |

## Exact observation metrics by value type

P = precision; R = recall. A MULTI_SELECT list is scored as one complete
observation, not as independently scored enum members.

| Value type | TypeSafe P | TypeSafe R | TypeSafe F1 | Pi P | Pi R | Pi F1 | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SINGLE_SELECT | 84.72% | 74.48% | 79.27% | 76.69% | 85.76% | 80.97% | +1.71 pp |
| MULTI_SELECT | 79.28% | 82.47% | 80.85% | 75.15% | 71.26% | 73.16% | -7.69 pp |
| NUMERIC | 79.50% | 88.89% | 83.93% | 74.46% | 42.01% | 53.72% | -30.22 pp |

## Recorded timing: wall clock, not per-request latency

**Wall-clock durations are available, but individual request and server-only
model latencies were not recorded.** Every saved request's `usage` object
contains only `input_tokens` and `output_tokens`; neither audit records request
start/end timestamps or duration fields.

| Timing item | TypeSafe | Pi Scorer |
| --- | --- | --- |
| Timing source | Historical notebook `run-extraction` cell metadata | Run metadata `started_at` / `finished_at` |
| Recorded start (UTC) | 2026-09-21 14:50:07.028289 | 2026-09-29 15:06:20.075547 |
| Recorded end (UTC) | 2026-09-21 15:06:10.203774 | 2026-09-29 16:15:45.842560 |
| Elapsed wall-clock time | 963.18 s (16m 03.18s) | 4,165.77 s (69m 25.77s) |
| Per-request latency, including mean / median / p95 | Not recorded | Not recorded |
| Server-only model latency | Not recorded | Not recorded |

TypeSafe's duration is the difference between `iopub.status.busy` and
`iopub.status.idle` for the historical extraction cell in the
[executed notebook](..\notebooks\synur_observation_extraction.ipynb).
The notebook's saved export output identifies `run_d9726017250f`, now stored in
`run_datasetv5_d9726017250f`. Its extraction/export execution metadata and outputs
match the historical artifact snapshot `d3be3a1`; these are the September 21
execution times, not the notebook's later edit or merge times.

The two timing windows have **different boundaries**:

- TypeSafe includes client setup, extraction requests and local processing,
  checkpoint writes, and displaying the predictions. Credential setup and
  evaluation/export are separate notebook cells and excluded.
- Pi includes the interval from awaiting the masked key through client setup,
  extraction, checkpoint/status writes, and evaluation. Any credential-entry
  wait is included. Its `finished_at` is recorded before final run/report export,
  so it is not the timestamp when every output file finished saving.

Both windows cover processing all 422 rows, including TypeSafe's six partial
rows. They are descriptive elapsed times, **not an apples-to-apples model
latency benchmark**. The artifacts do not isolate credential wait, local I/O,
network, or server time. Dividing these intervals by request count would not
recover measured per-request latency, and no model-speed ratio is claimed.

## Could code-version changes explain the difference?

**No extraction or evaluation logic change was found that explains the score
gap.** The code comparison inspected the snapshot containing the historical
TypeSafe v5/v4 artifacts (`d3be3a1`) against Pi's recorded execution commit
(`961cdb5`). TypeSafe's artifact commit is not proof of its exact execution
commit, because that run did not record an execution SHA. Pi additionally
records its notebook and launcher hashes and a clean tracked worktree at launch.

| Area | Change in the Pi execution snapshot | Expected effect on accuracy |
| --- | --- | --- |
| Provider, endpoint, and model selection | Added Pi configuration | Intentionally changes the model/service called |
| Service-export loading and input conversion | Loader unchanged; default paths made portable | No change to the observed input bytes or state hashes |
| Prompts and question construction | Unchanged | None from code changes |
| Numeric candidate extraction and parsing | Unchanged | None from code changes |
| Confidence thresholds, answer validation, and value assembly | Unchanged | None from code changes |
| Evaluation and reference normalization | Unchanged | None from code changes |
| Artifact naming, provider metadata, and error reporting | Changed | No effect on successful predictions |

### Evidence from the saved requests

- All 422 transcripts have identical state hashes, schema hashes, settings,
  numeric candidate pools, and candidate-issue diagnostics across the two runs.
- All 2,954 initial screening batches, covering 70,052 concept questions, have
  identical ordered question IDs and question hashes.
- All 3,155 shared batches have identical question hashes. This total includes
  the 2,954 screening batches. A shared batch is identified by its ordered
  question IDs within the same transcript.
- Other follow-up batches differ because subsequent questions depend on which
  concepts and candidate branches each model selects. Batching questions
  together does not make their model decisions joint.

### Confidence behavior differs despite the same threshold

| Final per-concept audit outcome | TypeSafe | Pi Scorer |
| --- | ---: | ---: |
| `low_choice_confidence` | 4,403 | 47,633 |
| Total audited concepts | 70,052 | 70,052 |
| Choice acceptance threshold | 0.8 | 0.8 |

Low-confidence outcomes can arise during either screening or value selection.
They are not request failures and do not all represent missed reference
observations; many schema concepts are not mentioned in a transcript. The same
threshold behaves very differently across providers, but these counts alone do
not establish that confidence calibration causes the entire accuracy gap.

**Conclusion:** the available evidence is more consistent with model/service
behavior, confidence calibration, and returned-answer validity than with
application logic changes. Python and SDK versions match for these v5 runs,
unlike the earlier dev comparison. Run dates still differ, and TypeSafe lacks
an exact execution SHA. A stronger control would rerun TypeSafe and Pi from the
same recorded commit and environment, retaining failures under the same policy.
No additional model calls were made for this report.

## Interpretation and limitations

- Pi finds 104 more exact observations overall, but adds 471 false positives;
  higher recall does not offset the precision reduction.
- SINGLE_SELECT F1 is higher for Pi by 1.71 pp. MULTI_SELECT F1 is lower by
  7.69 pp, and NUMERIC F1 is lower by 30.22 pp.
- The largest per-type difference is NUMERIC recall: 88.89% for TypeSafe versus
  42.01% for Pi. Numeric true positives are 512 versus 242, respectively.
- Pi completes every row without validation failures; TypeSafe has six partial
  rows. This reliability difference is reported rather than hidden by filtering.
  Completion does not mean every concept was emitted or every prediction was correct.
- This is one saved run per model over a combined local collection with shared
  thresholds, not a held-out or repeated benchmark. Provider-specific calibration,
  statistical significance, and a general model ranking are not established.
- Only differently scoped wall-clock timings are available, as detailed above;
  comparable model-serving latency and monetary cost are not established by
  these artifacts and are not inferred from request counts.

## Source artifacts

Paths below are relative to this comparison file. The renamed TypeSafe folder
specified for this comparison is `run_datasetv5_d9726017250f`.

| Artifact | TypeSafe | Pi Scorer |
| --- | --- | --- |
| Metrics | [metrics.json](run_datasetv5_d9726017250f\metrics.json) | [metrics.json](run_pi-scorer_2026-09-29_110619_409a728f403f\metrics.json) |
| Run metadata | [run.json](run_datasetv5_d9726017250f\run.json) | [run.json](run_pi-scorer_2026-09-29_110619_409a728f403f\run.json) |
| Decision audit | [audit.jsonl](run_datasetv5_d9726017250f\audit.jsonl) | [audit.jsonl](run_pi-scorer_2026-09-29_110619_409a728f403f\audit.jsonl) |
| Recorded failures | [failures.jsonl](run_datasetv5_d9726017250f\failures.jsonl) | [failures.jsonl](run_pi-scorer_2026-09-29_110619_409a728f403f\failures.jsonl) |
| Short transcript report | [transcript_report_short.json](report_c0406bb820c0\transcript_report_short.json) | [transcript_report_short.json](report_pi-scorer_2026-09-29_110619_409a728f403f\transcript_report_short.json) |
| Full transcript report | [transcript_report.json](report_c0406bb820c0\transcript_report.json) | [transcript_report.json](report_pi-scorer_2026-09-29_110619_409a728f403f\transcript_report.json) |

The earlier Pi attempt (`pi-scorer_2026-09-29_110345_1a2cb8429d98`) and all
101-row dev runs are excluded from this comparison. The existing dev report and
all original run artifacts are unchanged.
