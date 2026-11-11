# ExeaLM

An open language model built by high schoolers, run by [Exea Labs](https://exealabs.org). Code and weights are public under Apache 2.0.

No weights are released yet. We are at the start: the data pipeline, tokenizer, and first architecture experiments are open tasks.

## Join

Fill out the sign-up form so we can list you as a contributor: https://docs.google.com/forms/d/e/1FAIpQLSdPHVZhuXxBDUf-cJM2zDbj1JZPsduu95Y_kTJ_g4gKAXdDVw/viewform

The repo is public, so you can open an issue or pull request without signing up. The form gives us a record of who worked on the project. [CONTRIBUTING.md](CONTRIBUTING.md) explains how to start.

## Architecture

ExeaLM uses TONN, a sequence model that builds its own architecture from data. It starts from a large random network of small circuits (delays, running sums, products, attention) and keeps a circuit only if switching it off raises the error on data the network has not trained on.

Paper: [TONN: Networks That Build Their Own Architecture from Data](https://www.preprints.org/manuscript/202611.0994) (Avneh Singh Bhatia, Preprints.org, November 2026). It is a concept paper and has not been peer reviewed. Its larger-scale results are mixed, and we test TONN against a plain transformer at equal compute on small models before any large run.

## Data

About 100 billion tokens, selected from open corpora: the [NII LLM-jp Corpus v4](https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v4), the data released with [K2 Horizon](https://ifm.ai/blog/k2), FineWeb-Edu, DCLM-Baseline, and others. The count comes from our compute budget, about 850 MI300X GPU-hours for a model of roughly 1 billion parameters. [DATA.md](DATA.md) has the full mix, every dataset link, and the math behind the number. Every source, license, and filter gets documented.

## Training

One central run on cloud compute credits arranged by Exea. Contributors work on data pipelines, architecture experiments, evals, docs, and review. Weights will be released at https://huggingface.co/exea-labs/ExeaLM.

## Roadmap

1. Data pipeline: download, dedupe, filter, and mix
2. Tokenizer
3. Small-scale architecture experiments
4. Evaluation harness
5. Full training run
6. Post-training and release

Each phase gets an issue with an owner. Open tasks are in [issues](https://github.com/exea-labs/ExeaLM/issues). We need people on data curation, architecture research, training infrastructure, evals, docs, and community.

## Community

Read the [Code of Conduct](CODE_OF_CONDUCT.md) before contributing. Contributors are listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

Apache 2.0. See [LICENSE](LICENSE).
