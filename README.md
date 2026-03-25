# Truss Push Action

This action deploys a [Truss](https://github.com/basetenlabs/truss) model or [chain](https://docs.baseten.co/development/chain/deploy) to [Baseten](https://baseten.co). It pushes the deployment, waits for it to become active, optionally validates it with a predict request, and promotes it to production.

**Models** are detected when `truss-directory` points to a directory containing `config.yaml`. **Chains** are detected when `truss-directory` points to a `.py` file containing a `@chains.mark_entrypoint` class.

## Usage

### Model

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
```

### Chain

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my_chain.py"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    predict-payload: '{"max_value": 5}'
```

## Inputs

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    # Path to a model directory containing config.yaml,
    # or a .py file for chain deployments
    # Required
    truss-directory: ""

    # Baseten API key
    # Required
    baseten-api-key: ""

    # Override the model/chain name
    # For models: maps to truss push --model-name
    # For chains: sets the chain_name
    # Default: '' (uses model_name from config.yaml, or entrypoint class name for chains)
    model-name: ""

    # Whether to promote the deployment to production after validation
    # Default: false
    promote: ""

    # Deploy to a specific environment (implies publish, ignores promote)
    # Default: '' (no environment)
    environment: ""

    # Preserve previous production deployment's autoscaling setting
    # Only meaningful with promote: true
    # Default: false
    preserve-previous-production-deployment: ""

    # Attach git versioning info (sha, branch, tag) to the deployment
    # Default: true
    include-git-info: ""

    # JSON string of labels as key-value pairs
    # Default: ''
    labels: ""

    # Name of the deployment. Defaults to 'PR-<number>_<sha>' on pull
    # requests or '<sha>' otherwise
    # Default: '' (auto-generated)
    deployment-name: ""

    # Whether to deactivate the deployment after validation
    # Default: true
    cleanup: ""

    # JSON predict payload. For models, defaults to
    # model_metadata.example_model_input from config.yaml.
    # For chains, must be provided explicitly.
    # Default: ''
    predict-payload: ""

    # Max minutes to wait for deployment to become active
    # Default: 45
    deploy-timeout-minutes: ""

    # Timeout in seconds for predict request
    # Default: 300
    predict-timeout: ""
```

> **Note:** For multi-team organizations, configure the team in your `.trussrc` file. The action uses the team configured in `.trussrc` automatically.

## Scenarios

- [Deploy a model and promote to production](#deploy-a-model-and-promote-to-production)
- [Deploy a chain](#deploy-a-chain)
- [Deploy a model without cleanup](#deploy-a-model-without-cleanup)
- [Deploy with a custom predict payload](#deploy-with-a-custom-predict-payload)
- [Deploy to a specific environment](#deploy-to-a-specific-environment)
- [Deploy with labels](#deploy-with-labels)
- [Run in CI on pull requests](#run-in-ci-on-pull-requests)
- [Deploy multiple models](#deploy-multiple-models)

### Deploy a model and promote to production

By default the deployment is deactivated after validation. Set `promote: true` to push the deployment live.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    promote: true
    cleanup: false
```

### Deploy a chain

Deploy a Baseten chain from a Python source file. The action auto-detects chains when the path ends in `.py`.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./chains/my_chain.py"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    model-name: "my-rag-chain"
    promote: true
    cleanup: false
    predict-payload: '{"query": "What is Baseten?"}'
```

### Deploy a model without cleanup

Keep the deployment running after validation for further inspection or manual promotion.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    cleanup: false
```

### Deploy with a custom predict payload

Override the example input defined in `config.yaml` with an inline JSON payload.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    predict-payload: '{"prompt": "Hello, world!", "max_new_tokens": 128}'
    predict-timeout: 60
```

### Deploy to a specific environment

Push to a named environment (e.g., staging). This implies publish and ignores the `promote` flag.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    environment: "staging"
    cleanup: false
```

### Deploy with labels

Attach metadata labels to track deployments in your CI pipeline.

```yaml
- uses: basetenlabs/action-truss-push@v0.1.0
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    labels: '{"team": "ml-platform", "triggered-by": "ci"}'
```

### Run in CI on pull requests

Validate model changes on every pull request without promoting to production.

```yaml
name: Validate model

on:
  pull_request:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: basetenlabs/action-truss-push@v0.1.0
        with:
          truss-directory: "./my-model"
          baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
          promote: false
          cleanup: true
```

### Deploy multiple models

Deploy several models in parallel using a matrix strategy.

```yaml
name: Deploy models

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        model:
          - path: llm/gpt-oss-20b/latency
          - path: video/stable-diffusion/quality
    steps:
      - uses: actions/checkout@v4

      - uses: basetenlabs/action-truss-push@v0.1.0
        with:
          truss-directory: ${{ matrix.model.path }}
          baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
          promote: true
          cleanup: false
```

## Recommended secrets configuration

Store your Baseten API key as an [encrypted secret](https://docs.github.com/en/actions/security-guides/encrypted-secrets) in your repository or organization. Never hardcode it in your workflow file.

```yaml
permissions:
  contents: read
```

The action does not require any additional GitHub token permissions beyond reading repository contents.

## License

This project is released under the [MIT License](LICENSE).
