# Model cards

One card per learned `EXPERIMENTAL_CANDIDATE`. Every card is limited to what was executed. Training data is
synthetic in every case, and no learned module is the runtime default (ADR-0004). Numbers come from
[research/EXPERIMENT_RESULTS_2026-09-18.md](../research/EXPERIMENT_RESULTS_2026-09-18.md) (CPU, seeds 2026201,
2026202, 2026203, registry outcome INCONCLUSIVE).

| Card | Module | Executed | Claim status |
|---|---|---|---|
| [ecmer_encoders.md](ecmer_encoders.md) | `conrad.core.ecmer` | unit tests only | IMPLEMENTED |
| [learned_association.md](learned_association.md) | `conrad.core.association_scorer` | CORE-ASSOC-E001 | IMPLEMENTED; KILL_CANDIDATE |
| [learned_buo.md](learned_buo.md) | `conrad.core.buo` | CORE-BUO-E001 | IMPLEMENTED; KILL_CANDIDATE |
| [learned_rbp.md](learned_rbp.md) | `conrad.core.rbp` | CORE-RBP-E001 | IMPLEMENTED; KILL_CANDIDATE |
| [gru_tbd.md](gru_tbd.md) | `conrad.core.tbd.GruTBD` | CORE-TBD-E001, training smoke | IMPLEMENTED |
| [learned_tcdp.md](learned_tcdp.md) | `conrad.domains.technical.learned_tcdp` | unit tests only | IMPLEMENTED |
| [learned_cefd.md](learned_cefd.md) | `conrad.domains.ecological.learned` | unit tests only | IMPLEMENTED |
| [egdc_learned_scorer.md](egdc_learned_scorer.md) | `conrad.decision.learned` | unit tests only | IMPLEMENTED |
| [mcbr_learned_ranker.md](mcbr_learned_ranker.md) | `conrad.active.learned` | unit tests only | IMPLEMENTED |
| [baac_value_heads.md](baac_value_heads.md) | `conrad.communication.learned` | unit tests only | IMPLEMENTED |

"KILL_CANDIDATE" is the verdict written in the results record. It is not a registry outcome, and no ADR has
retired the module.
