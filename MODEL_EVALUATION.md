# pdf2md — Model Evaluation Summary

**Task:** convert a 131-page, table-heavy technical PDF (CLC+ Backbone manual) to
Quarto/Markdown. **Metric:** automated coverage verify — what fraction of the source
*text* and *tables* survive into the output. Images stripped first, so this isolates
**text + table transcription** (the hard part). All models called via OpenRouter.

## Recommendation

> **`gemini-2.5-flash`, full stop** — 99% text / 94% tables at **$0.27/doc**.
> Nothing tested beats its cost/quality ratio, and it carries no page cap. This is
> the only supported convert model; the table below is the evidence behind that call.
>
> `claude-sonnet-5` scored marginally higher on tables (98.3%) but at ~10× the cost
> *and* a hard 100-page failure — not worth a second code path, so it isn't one.

## Results (sorted by table quality — the discriminator)

| # | Model | Text | Tables | Cost/doc | Notes |
|--:|-------|-----:|-------:|---------:|-------|
| 🥇 | **claude-sonnet-5** | 99.5% | **98.3%** | ~$2.70 | best quality; **>100pg fails** |
| 🥈 | **gemini-2.5-flash** ⭐ | 99.2% | **94.2%** | **$0.27** | **best value — our pick** |
| 🥉 | mistral-large-2512 | 97.5% | 92.0% | $0.84 | solid; needs tuned call settings |
| 4 | gemini-3.5-flash | 92.5% | 83.2% | $0.70 | newer but *worse* + very slow |
| 5 | gemini-2.5-flash-lite | 96.6% | 76.4% | $0.02 | cheapest; tables too weak |
| 6 | gemini-3-flash-preview | 95.7% | 60.6% | $0.18 | tables weak |
| 7 | gpt-4.1-mini | 95.2% | 51.3% | $0.19 | tables collapse |
| 8 | grok-4.3 | 37.5% | 30.3% | $0.06 | quits early |
| 9 | amazon/nova-2-lite | 14.4% | ✗ | $0.02 | unusable |
| — | claude-haiku-4.5 | — | — | — | ✗ 100-page PDF limit |
| — | mistral-medium-3.1 | — | — | — | ✗ unreliable (empty responses) |
| — | gpt-5 | — | — | — | ✗ times out on 131-page output |

## Cost vs. Quality (tables)

```
tables%  100 |                                        sonnet-5 ●
          95 |                          gemini-flash ●   mistral-lg ●
          90 |
          85 |                    gemini-3.5-flash ●
          80 |
          75 |  flash-lite ●
          70 |
          65 |            gemini-3-flash ●
          60 |
          55 |            gpt-4.1-mini ●
              +------------------------------------------------------
              $0.02   $0.20        $0.70   $0.84            $2.70
                        (log-ish cost →)
```
Top-left = best. **gemini-2.5-flash sits alone in the sweet spot.**

## Key findings

1. **Tables separate the field, not text.** Almost every model transcribes prose at
   90%+; table transcription is where cheap/weak models collapse (51–76%).
2. **Newer ≠ better.** Gemini 2.5-flash beats *both* 3-flash (60%) and 3.5-flash (83%)
   on tables — and 3.5-flash took 85 minutes vs 14.
3. **Only Google & Anthropic transcribe reliably.** OpenAI, xAI, and Amazon models
   summarize or quit instead of transcribing full tables.
4. **Gemini is 3× more token-efficient on PDFs than Claude** (106k vs 322k tokens for
   the same doc) — the core reason it's ~10× cheaper.
5. **Sonnet caching** can cut its cost ~20–30% (to ~$2.1) but never approaches Gemini.

## Method
- Isolated convert step (no figure detection), identical prompt per model.
- Coverage via shingled sentence matching (text) and cell-content matching (tables),
  with a three-tier classifier (covered / reworded / missing) to avoid false gaps.
- Per-model outputs + full verify reports retained under `model_compare/<model>/`.

_Last updated: 2026-07-17._
