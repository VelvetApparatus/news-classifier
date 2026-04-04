# Model Artifacts

Place the RuBERT clustering artifacts required for online inference in this directory. The batch inference worker expects at least the following files:

- `bert_config.json` — configuration used to initialize `RuBERTEmbedder`.
- `clusterer.pkl` — serialized `TextClusterer`.

Any additional files (for example, `clustering_config.json`, `topics.json`) can also be stored here for reference.
