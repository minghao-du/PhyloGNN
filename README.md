# PhyloGNN

PhyloGNN converts phylogenetic trees into PyTorch Geometric graph data and
provides model and training utilities for graph neural network workflows on
those data.

The fastest first-use path is:

1. Install the package in the existing project environment.
2. Read the local Sphinx quickstart.
3. Convert a small tree into a `torch_geometric.data.Data` object.

Install the core package from a checkout:

```bash
python -m pip install -e .
```

Install documentation tooling when you want to build the local docs:

```bash
python -m pip install -e ".[docs,all]"
cd docs
make html SPHINXOPTS="-W"
```

Open `docs/source/index.rst` for the source documentation, or browse
`docs/_build/html/index.html` after the HTML build. Start with
`docs/source/installation.rst` and `docs/source/quickstart.rst`; the user guide
covers tree input, graph conversion, feature engineering, training, metrics,
tracking, and troubleshooting.

Maintainers can validate the documentation with:

```bash
python -m sphinx -b doctest -W docs/source docs/_build/doctest
```

## Documentation Deployment

Documentation is deployed to GitHub Pages by the `Deploy documentation`
workflow when changes land on `main`. The workflow installs
`.[docs,all]`, builds the Sphinx HTML with warnings treated as errors, validates
documentation links, uploads `docs/_build/html`, and deploys the uploaded
artifact only after the build and link validation steps succeed. Failed builds
or failed link validation appear as failed GitHub Actions runs/status checks,
which notify maintainers through the repository's normal GitHub notification
settings. Because artifact upload and deployment depend on successful validation,
a failed run stops before deployment and the previously published Pages version
remains online.

Pull requests targeting `main` run the `Validate documentation` workflow. This
performs the same install, Sphinx build, and link validation checks without any
deployment steps. The deployment workflow can also be started manually from the
GitHub Actions tab by selecting `Deploy documentation` and using `Run workflow`.

To enable publishing, open the repository settings in GitHub and confirm that
Pages is enabled with `GitHub Actions` selected as the source. The first
successful deployment will publish the Pages URL in the workflow summary.

Deployment validation notes:

- SC-002: GitHub-hosted workflow duration must be checked on the first
  successful run; target is under 5 minutes.
- SC-004: Compare the deployed Pages homepage, navigation, and styling against
  the local `docs/_build/html/index.html` build after the first deployment.
