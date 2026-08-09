# Siamese Backbone Serialization Fix — TODO

## Steps

- [x] Analyze root cause (Lambda layer + HDF5 save incompatibility in TF 2.16)
- [x] Add `save_embedding_branch` / `load_embedding_branch` helpers in `src/siamese_model.py`
- [x] Update `src/train_siamese.py` to save embedding weights (not full .h5 model)
- [x] Update `src/feature_extraction.py` to rebuild branch for `--model siamese`
- [x] Syntax check all three edited files (valid)
- [x] Fix venv environment (upgraded setuptools 65.5.0 -> 83.0.0; tensorflow 2.16.2 now imports successfully)
- [ ] Re-run Siamese training: `venv\Scripts\python.exe src/train_siamese.py --data_dir sample_data --epochs 3 --out models/siamese_backbone.h5`
- [ ] Re-run Siamese extraction: `python src/feature_extraction.py --data_dir sample_data --model siamese --weights models/siamese_backbone.h5 --out models/embeddings/siamese.npz`
- [ ] Verify embedding artifact shape
- [ ] Run evaluation across all three model outputs
- [ ] Launch Streamlit UI and confirm it stays alive
