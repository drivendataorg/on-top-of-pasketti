# Release Notes

## v1.0.0

First public open-source release of the DrivenData Pasketti phonetic
solution.

### Included in this release

* Standalone training and inference repository with a minimal compatibility
  layer under `src/_compat/` so the original project code can run outside the
  author's internal monorepo.
* Exact final 11-model ensemble definition in `src/models.txt`.
* Packaging flow for the official DrivenData runtime via `make pack` and
  `scripts/pack_submission.sh`.
* Published Hugging Face model repository containing the final online
  checkpoints and CatBoost reranker artifacts.
* Download helper script `scripts/download_weights.sh` for retrieving the
  public release weights.

### Public artifacts

| Artifact | Link |
| --- | --- |
| GitHub repository | <https://github.com/chenghuige/pasketti-phonetic-solution> |
| Hugging Face weights | <https://huggingface.co/huigecheng/pasketti-phonetic-weights> |

### Recommended reproduction flow

```bash
make setup
make data
HF_REPO_ID=huigecheng/pasketti-phonetic-weights bash scripts/download_weights.sh
make pack
```

### Notes

* Large neural-network checkpoints live in Hugging Face rather than GitHub.
* Competition data is not redistributed in this repository.
* The word-track solution is out of scope for this release.