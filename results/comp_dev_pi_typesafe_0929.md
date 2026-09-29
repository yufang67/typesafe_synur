# Pi Scorer vs TypeSafe - 2026-09-29

Comparison of the completed Pi Scorer run on September 29 with the saved TypeSafe
JEV run on September 20. No new inference was performed for this comparison.
All deltas are **Pi minus TypeSafe**; **pp** means percentage points.

## Run scope

| Item | TypeSafe | Pi Scorer |
| --- | --- | --- |
| Returned model | `jev-1.13.0` | `pi-scorer` |
| Run date (UTC) | 2026-09-20 | 2026-09-29 |
| Run ID | `ad975c76b78d` | `pi-scorer_2026-09-29_094450_b4c0bcb5211a` |
| Dataset split | `mediqa_synur_dev` | `mediqa_synur_dev` |
| Completed transcripts | 101 / 101 | 101 / 101 |
| Enabled concepts per transcript | 162 | 162 |
| Audited concept decisions | 16,362 / 16,362 | 16,362 / 16,362 |
| Enabled value types | SINGLE_SELECT, MULTI_SELECT, NUMERIC | SINGLE_SELECT, MULTI_SELECT, NUMERIC |
| Prompt version | `direct-jev-v1` | `direct-jev-v1` |
| TypeSafe SDK version | 0.7.0 | 0.7.0 |
| Python version | 3.14.2 | 3.13.15 |
| Returned model requests | 891 | 903 |
| Request failures | 0 | 0 |

The dataset manifest, selected transcript IDs and order, normalization policy,
and per-transcript state hashes, schema hashes, candidate policies, and settings
match. Both runs use Choice confidence >= 0.8, Noul support >= 0.8, Noul rejection
<= 0.2, batch size 32, a 60,000-byte request budget, and a 30,000-byte
state-plus-question budget. STRING concepts are excluded.

## Overall exact observation comparison

These are micro-averaged exact observation metrics, not concept-presence metrics.
Raw and normalized observation metrics are identical within each run.

| Metric | TypeSafe `jev-1.13.0` | Pi `pi-scorer` | Delta |
| --- | ---: | ---: | ---: |
| Precision | 83.74% | 76.55% | -7.19 pp |
| Recall | 77.35% | 78.39% | +1.04 pp |
| F1 | 80.42% | 77.46% | -2.96 pp |
| True positives | 963 | 976 | +13 |
| False positives | 187 | 299 | +112 |
| False negatives | 282 | 269 | -13 |
| Predicted observations | 1,150 | 1,275 | +125 |
| Reference observations | 1,245 | 1,245 | 0 |

## Exact observation metrics by value type

P = precision; R = recall. A MULTI_SELECT list is scored as one complete
observation, not as independently scored enum members.

| Value type | TypeSafe P | TypeSafe R | TypeSafe F1 | Pi P | Pi R | Pi F1 | F1 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SINGLE_SELECT | 85.32% | 75.18% | 79.93% | 77.26% | 84.65% | 80.78% | +0.85 pp |
| MULTI_SELECT | 77.84% | 82.80% | 80.25% | 74.52% | 74.52% | 74.52% | -5.72 pp |
| NUMERIC | 81.38% | 86.13% | 83.69% | 71.05% | 39.42% | 50.70% | -32.98 pp |

## Could code-version changes explain the difference?

**No extraction or evaluation logic change was found that explains the score
gap.** The comparison inspected the snapshot containing the saved TypeSafe
artifacts (`f2b2f8d`) against the Pi support commit (`001adff`). These identify
the compared repository snapshots, not recorded execution SHAs: neither run's
metadata captures the exact Git commit executed.

| Area | Change in the newer code | Expected effect on accuracy |
| --- | --- | --- |
| Provider, endpoint, and model selection | Added Pi configuration | Intentionally changes the model/service called |
| Prompts and question construction | Unchanged | None from code changes |
| Numeric candidate extraction and parsing | Unchanged | None from code changes |
| Confidence thresholds and value assembly | Unchanged | None from code changes |
| Evaluation and reference normalization | Unchanged | None from code changes |
| Artifact names, provider metadata, and error reporting | Changed | No effect on successful predictions |

### Evidence from the saved requests

The saved audits provide evidence beyond the code diff:

- All 101 transcripts have identical state hashes, schema hashes, settings,
  numeric candidate pools, and candidate-issue diagnostics across the two runs.
- All 707 initial screening batches, covering 16,362 concept questions, have
  identical ordered question IDs and question hashes.
- All 756 batches shared by the runs within the same transcript have identical
  question hashes. This includes the 707 screening batches, not an additional
  756 batches. A shared batch is identified by its ordered question IDs.
- Other follow-up batches differ because subsequent questions depend on which
  concepts and candidate branches each model selects. Batching questions
  together does not make their model decisions joint.

### Confidence behavior differs despite the same threshold

| Final per-concept audit outcome | TypeSafe | Pi Scorer |
| --- | ---: | ---: |
| `low_choice_confidence` | 995 | 10,648 |
| Total audited concepts | 16,362 | 16,362 |
| Choice acceptance threshold | 0.8 | 0.8 |

Low-confidence outcomes can arise during either screening or value selection.
These counts are not request failures and do not all represent missed reference
observations; many schema concepts are not mentioned in a transcript. They show
that the same threshold behaves very differently across providers. They do not
by themselves prove that confidence calibration causes the entire accuracy gap.

**Conclusion:** the available evidence is more consistent with differences in
model/service behavior and confidence calibration than with application changes.
This is not a fully controlled experiment: run dates and Python versions differ,
and exact execution SHAs were not recorded. The cleanest control would be to
rerun TypeSafe on the current commit using the same Python environment as Pi,
with the same data, questions, and thresholds. No such rerun was performed for
this report.

## Interpretation and limitations

- Pi finds 13 more exact observations overall, but adds 112 false positives;
  higher recall does not offset the precision reduction.
- SINGLE_SELECT F1 is slightly higher for Pi; MULTI_SELECT and especially NUMERIC
  F1 are lower. NUMERIC recall falls from 86.13% to 39.42%.
- Both runs completed without request failures. Completion does not mean every
  concept was emitted or every prediction was correct.
- This is one saved development-set run per model, on different dates and Python
  versions, with shared thresholds rather than provider-specific calibration.
  It is not a repeated, held-out, or statistically established model ranking.
- These artifacts do not establish a comparable latency or monetary-cost
  benchmark, so neither is inferred here.

## Source artifacts

Paths below are relative to this comparison file.

| Artifact | TypeSafe | Pi Scorer |
| --- | --- | --- |
| Metrics | [metrics.json](run_ad975c76b78d\metrics.json) | [metrics.json](run_pi_0929\run_pi-scorer_2026-09-29_094450_b4c0bcb5211a\metrics.json) |
| Run metadata | [run.json](run_ad975c76b78d\run.json) | [run.json](run_pi_0929\run_pi-scorer_2026-09-29_094450_b4c0bcb5211a\run.json) |
| Decision audit | [audit.jsonl](run_ad975c76b78d\audit.jsonl) | [audit.jsonl](run_pi_0929\run_pi-scorer_2026-09-29_094450_b4c0bcb5211a\audit.jsonl) |

The earlier failed Pi attempt (`pi-scorer_2026-09-29_094348_81090619ee59`) is
excluded from all comparison scores above.
