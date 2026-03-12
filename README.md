# Truss Push Action

This action deploys a [Truss](https://github.com/basetenlabs/truss) model to [Baseten](https://baseten.co). It pushes the model, waits for the deployment to become active, optionally validates it with a predict request, and promotes it to production.

## Usage

```yaml
- uses: basetenlabs/action-truss-push@v1
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
```

## Inputs

```yaml
- uses: basetenlabs/action-truss-push@v1
  with:
    # Path to directory containing config.yaml
    # (e.g., llm/gpt-oss-20b/latency, video/model-name/quality)
    # Required
    truss-directory: ""

    # Baseten API key
    # Required
    baseten-api-key: ""

    # Whether to promote the deployment to production after validation
    # Default: false
    promote: ""

    # Whether to deactivate the deployment after validation
    # Default: true
    cleanup: ""

    # Max seconds to wait for deployment to become active
    # Default: 2700
    deploy-timeout: ""

    # JSON override for predict payload. If empty, reads
    # model_metadata.example_model_input from config.yaml
    # Default: ''
    predict-payload: ""

    # Timeout in seconds for predict request
    # Default: 300
    predict-timeout: ""
```

## Scenarios

- [Deploy a model and promote to production](#deploy-a-model-and-promote-to-production)
- [Deploy a model without cleanup](#deploy-a-model-without-cleanup)
- [Deploy with a custom predict payload](#deploy-with-a-custom-predict-payload)
- [Run in CI on pull requests](#run-in-ci-on-pull-requests)
- [Deploy multiple models](#deploy-multiple-models)

### Deploy a model and promote to production

By default the deployment is deactivated after validation. Set `promote: true` to push the deployment live.

```yaml
- uses: basetenlabs/action-truss-push@v1
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    promote: true
    cleanup: false
```

### Deploy a model without cleanup

Keep the deployment running after validation for further inspection or manual promotion.

```yaml
- uses: basetenlabs/action-truss-push@v1
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    cleanup: false
```

### Deploy with a custom predict payload

Override the example input defined in `config.yaml` with an inline JSON payload.

```yaml
- uses: basetenlabs/action-truss-push@v1
  with:
    truss-directory: "./my-model"
    baseten-api-key: ${{ secrets.BASETEN_API_KEY }}
    predict-payload: '{"prompt": "Hello, world!", "max_new_tokens": 128}'
    predict-timeout: 60
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

      - uses: basetenlabs/action-truss-push@v1
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

      - uses: basetenlabs/action-truss-push@v1
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
