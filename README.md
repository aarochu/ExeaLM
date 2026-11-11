# ExeaLM

A language model built in the open by high schoolers. Custom architecture, open code, open weights. Run by [Exea Labs](https://exealabs.org).

**Status:** early. The repo is just getting started, and there are no released weights yet.

## Join

Sign up here so we can add you to the contributor list: **https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform**

The repo is public, so you can open an issue or pull request without signing up. The form just helps us keep a running record of who worked on the project. See [CONTRIBUTING.md](CONTRIBUTING.md) to get started.

## Links

- Code: https://github.com/exea-labs/ExeaLM
- Models: https://huggingface.co/exea-labs/ExeaLM
- Sign-up form: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform

## The plan

- **Architecture:** custom, designed and validated by contributors. Anything we scale up first has to beat a plain Llama-style baseline at equal compute in small-scale experiments.
- **Data:** about 2 trillion tokens selected from open corpora, mainly the [NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4) and the data released with [K2 Horizon](https://ifm.ai/blog/k2) (Institute of Foundation Models). We document every source, license, and filter.
- **Training:** one central training run on cloud compute arranged by Exea. Contributors work on data pipelines, architecture experiments, evals, docs, and review.
- **Release:** code under Apache 2.0, weights and model cards on Hugging Face.

## Roadmap

1. Data pipeline: download, dedupe, filter, and mix
2. Tokenizer
3. Architecture experiments at small scale
4. Evaluation harness
5. Full training run
6. Post-training and release

Each phase gets an issue with an owner. Check the [issues](https://github.com/exea-labs/ExeaLM/issues) for open tasks.

## Ways to contribute

- Data curation and filtering
- Model architecture and research
- Training and infrastructure
- Evaluation and benchmarks
- Documentation and writing
- Community and outreach

## Community

Please read our [Code of Conduct](CODE_OF_CONDUCT.md). Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
