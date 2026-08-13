# Human Evaluation Protocol

## Evaluation task

Two raters independently compared Candidate A and Candidate B for each psychological-dialogue item. The method and encoder identities were hidden from the raters.

Raters saw only:
- the current user utterance;
- Candidate A;
- Candidate B;
- the evaluation fields.

## Evaluation fields

- **Overall preference:** `A`, `B`, `Tie`, or `Neither`.
- **Relevance:** which candidate more directly responds to the user's current question, situation, or need.
- **Specificity:** which candidate provides more targeted understanding, follow-up, or an actionable direction.
- **Supportiveness:** which candidate better reflects respect, understanding, and non-judgmental support.
- **Safety issue:** flag clearly dangerous advice, diagnostic assertions, offensive/blaming content, or excessive promises.
- **Confidence:** integer from `1` (very uncertain) to `5` (very certain).

## Blinding and A/B assignment

The A/B identities are provided separately in `randomized_ab_assignments.csv`. The A/B positions are balanced within each encoder: for GTE, ORCA appears in A for 5 items and B for 5 items; for BGE, ORCA appears in A for 5 items and B for 5 items.

## Sampling / selection status

**Important:** this released 20-item set is an outcome-conditioned exploratory qualitative case set, not an unbiased random human-evaluation sample.

According to the original selection record:
- 6 items were retained from an earlier pilot because both raters' overall preferences mapped to ORCA;
- 14 items were then added from previously unevaluated queries whose Dense and ORCA Top-1 candidates differed;
- the final set contains 10 GTE items and 10 BGE items;
- the random seed for the supplementary selection step was `20260803`.

Therefore, this set is appropriate for qualitative case analysis, evaluation-protocol development, and error-pattern analysis, but **must not be used to estimate an unbiased ORCA-vs-Dense win rate, statistical significance, or unconditional overall quality advantage**.

## Public release

The public judgment CSV files omit user utterances, candidate-response texts, pseudo-gold responses, and free-text comments. These fields are not redistributed because they contain or may reproduce restricted dialogue content.
