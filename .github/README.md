# psAI ops <img src="images/logo.png" alt="apehex logo" width="32" height="32">

[![License][shield-license]][github-license]
[![Latest][shield-release]][github-release]

A collection of experimental web apps to inspect & engineer activations.

It is a WIP, some apps are not functional yet.

The [human / LLM detector][hface-human] is improving though.

## Installation

The package is available on pypi:

```shell
pip install -U psaiops
```

All the apps run on a single GPU and can be launched from a Google Colab notebook.

They are showcased in the [demo notebook][colab-demo].

## Overview

To run a given application all you need is to call the associated `app.py`:

```python
python psaiops/compose/contrast/app.py
```

All the apps run with the model `gpt-oss-20b` by default so it is highly recommanded to use a GPU.

Some of the apps are specific to `gpt-oss-20b` but most can be used with another model.
You can look at the section `__main__` at the bottom of the file `app.py` for more details on the setup.

### Contrastive Steering

A straightforward implementation of the technique [contrastive activation addition][arxiv-caa] is available in `psaiops.compose.contrast.app`.

### Latent Maths

Pushing the idea of CAA further, `psaiops.compose.maths.app` allows to compose several prompts in the latent space with maths operators.

Like CAA you can do the difference between prompts, but also multiply, project, average, etc.

### Dataset Combination

The app `psaiops.combine.app` allows to draw from several datasets to form new samples and datasets.

It is useful to create pairs of prompts and form specific latent directions with the contrastive steering technique.

### Debugging And Scoring

To support the creation of apps that operate in the latent space, I've used several tools that allow to view the internals of the models.

In particular, you can take apart LLM generated text from human text:

- in the [Hugging Face hub][hface-human]
- or by running the app `psaiops.score.human.app`

It is using techniques scattered over several other apps that:

- use a LLM as critic to estimate how surprising each token is
- score the input tokens according to the attention they get during the generation
- view the expert logits and associate the routing with the input tokens
- view the flow of residuals and assess the contribution of the layers to the final output 

## License

Licensed under the [AGPLv3][github-license].

[arxiv-caa]: https://arxiv.org/pdf/2312.06681
[colab-demo]: https://github.com/apehex/psai-ops/blob/main/notebooks/demo.ipynb
[hface-human]: https://huggingface.co/spaces/apehex/human-scores

[github-license]: LICENSE.md
[github-release]: https://github.com/apehex/psai-ops/releases/latest

[shield-license]: https://img.shields.io/badge/license-aGPLv3-green?style=flat-square
[shield-release]: https://img.shields.io/github/release/apehex/psai-ops.svg?style=flat-square
