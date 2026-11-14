# Language lane: open-vocab eval, normalization and sealed re-checks

Numbers copied from the team handoff notes (items 25, 27, 32) and the sealed normalize run. These are offline text/parsing evals (instruction -> bin mapping), not Isaac episodes, except the last table (sort_language on box E).

## Results

| # | experiment | before | after | notes |
|---|---|---|---|---|
| 25 | Felix open-vocab eval with Qwen3-VL (box E) | 14 pts without the model | **87 ok / 24 wrong / 13 skip = 150 pts (60.5 %)** | 22 of 24 wrong = `nearest_bin`. Confidence is uncalibrated (always >= 0.95). |
| 27 | Felix geometry features added to the world JSON (`brain_lang`) | 150 pts | **221/248 (89.1 %), 3 wrong** | Architecture confirmed by Felix. Order of work: typos/synonyms before brain ON; finals build must be measured end-to-end. |
| sealed | `normalize.py` (`ea24b7a7`) + resolver `21af8498`, Gemini held-out n=150 | EV 84.5 % | **EV 90.4 %** (+5.9 pp) | Sealed set, not used for tuning. |
| sealed | same, `blind_para` n=358 | 96.8 % | **96.8 %** | 0 regressions. |
| dev | normalization on known failure sentences (17) | 9/17 | **15/17** | Dev corpus n=336: no sentence rewritten wrongly. |
| 32 | brain re-test, box E, `sort_language` seeds 0-7 (commit f0df515) | submission: 28 correct / 8 counter / 0 wrong | brain_llm: 27 / 5 / **4 wrong** | Gate PASS for both. That build lacks the guard "never contradict a known bin", so brain stays OFF by default. |

## Conclusion

- The typo/synonym normalization layer (`instruction.normalize`, from `felix/language-lane`) is a clear gain on the sealed held-out set (+5.9 pp) and neutral on the 358-sentence blind set, so it goes into the release candidate.
- The Qwen3-VL brain improved a lot with geometric features (150 -> 221/248) but still produced wrong bins in the sim re-test; decision D1 = brain OFF unless it beats the base on >= 60 episodes with <= 2 % wrong.
- Sealed sets (Gemini held-out, blind_para) were not used for tuning.
