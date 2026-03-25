#!/usr/bin/env python3
"""Deploy and validate a Truss model or chain on Baseten.

Used by the truss-push composite action. Reads configuration from environment
variables set by action.yml and writes results to GITHUB_OUTPUT/GITHUB_STEP_SUMMARY.
"""

import json
import os
import re
import subprocess
import sys
import threading
import time

import requests
import truss
import yaml

BASETEN_API_URL = "https://api.baseten.co/v1"
IN_GITHUB_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"
_TRUSS_TIMESTAMP_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]: ")

PHASE_TO_STATUS = {
    "config": "deploy_failed",
    "deploy": "deploy_failed",
    "predict": "predict_failed",
}

CHAIN_FAILED_STATUSES = {
    "DEPLOY_FAILED", "BUILD_FAILED", "FAILED", "BUILD_STOPPED",
}
CHAIN_READY_STATUSES = {"ACTIVE", "SCALED_TO_ZERO", "MODEL_READY"}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def load_config(truss_directory):
    config_path = os.path.join(truss_directory, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_predict_payload(config, payload_override):
    if payload_override:
        return json.loads(payload_override)
    metadata = config.get("model_metadata", {})
    return metadata.get("example_model_input")


def build_deployment_name():
    """Build deployment name like 'PR-42_abc1234' or just 'abc1234'."""
    short_sha = os.environ.get("GITHUB_SHA", "unknown")[:7]
    ref = os.environ.get("GITHUB_REF", "")
    # PR refs look like refs/pull/42/merge
    if ref.startswith("refs/pull/"):
        pr_number = ref.split("/")[2]
        return f"PR-{pr_number}_{short_sha}"
    return short_sha


def deploy(
    truss_directory,
    api_key,
    promote,
    deployment_name,
    model_name=None,
    environment=None,
    preserve_previous_production_deployment=False,
    include_git_info=False,
    labels=None,
    deploy_timeout_minutes=None,
):
    truss.login(api_key)
    return truss.push(
        truss_directory,
        publish=True,
        promote=promote,
        deployment_name=deployment_name,
        model_name=model_name or None,
        environment=environment or None,
        preserve_previous_production_deployment=preserve_previous_production_deployment,
        include_git_info=include_git_info,
        labels=labels,
        deploy_timeout_minutes=deploy_timeout_minutes,
    )


def wait_for_active(deployment, timeout):
    """Wait for deployment to become active, with a configurable timeout."""
    start = time.time()
    deployment.wait_for_active(timeout_seconds=timeout)
    return time.time() - start


def predict(model_id, deployment_id, api_key, payload, timeout):
    """Run a predict request. Handles both streaming and non-streaming."""
    headers = {"Authorization": f"Api-Key {api_key}"}
    url = (
        f"https://model-{model_id}.api.baseten.co"
        f"/deployment/{deployment_id}/predict"
    )
    streaming = payload.get("stream", False)

    start = time.time()
    if streaming:
        return _predict_streaming(url, headers, payload, timeout, start)
    return _predict_sync(url, headers, payload, timeout, start)


def _predict_sync(url, headers, payload, timeout, start):
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    elapsed = time.time() - start
    body = resp.text[:4096]
    return {
        "response": body,
        "total_time": elapsed,
        "ttfb": elapsed,
        "tokens": 0,
        "tokens_per_sec": 0,
        "streaming": False,
    }


def _predict_streaming(url, headers, payload, timeout, start):
    """Parse OpenAI-compatible SSE stream."""
    resp = requests.post(
        url, headers=headers, json=payload, timeout=timeout, stream=True
    )
    resp.raise_for_status()

    ttfb = None
    token_count = 0
    chunks = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue

        data = line[len("data: "):]
        if data.strip() == "[DONE]":
            break

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue

        choices = parsed.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if ttfb is None:
                    ttfb = time.time() - start
                token_count += 1
                chunks.append(content)

    elapsed = time.time() - start
    full_response = "".join(chunks)[:4096]

    return {
        "response": full_response,
        "total_time": elapsed,
        "ttfb": ttfb or elapsed,
        "tokens": token_count,
        "tokens_per_sec": token_count / elapsed if elapsed > 0 else 0,
        "streaming": True,
    }


def _forward_logs(proc):
    """Read from proc stdout, strip the truss-added timestamp prefix, print."""
    for line in proc.stdout:
        line = _TRUSS_TIMESTAMP_RE.sub("", line)
        sys.stdout.write(line)
        sys.stdout.flush()


def start_log_stream(model_id, deployment_id):
    """Start streaming deployment logs via truss CLI in the background."""
    try:
        proc = subprocess.Popen(
            [
                "truss", "model-logs",
                "--model-id", model_id,
                "--deployment-id", deployment_id,
                "--tail",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        thread = threading.Thread(
            target=_forward_logs, args=(proc,), daemon=True)
        thread.start()
        return proc
    except Exception as e:
        print(f"  Warning: could not start log stream - {e}")
        return None


def stop_log_stream(proc):
    """Stop the background log stream process."""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def deactivate_deployment(model_id, deployment_id, api_key):
    headers = {"Authorization": f"Api-Key {api_key}"}
    url = f"{BASETEN_API_URL}/models/{model_id}/deployments/{deployment_id}/deactivate"
    resp = requests.post(url, headers=headers, timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def deploy_chain(source_file, chain_name, api_key, promote):
    """Deploy a chain and return the ChainService handle."""
    from pathlib import Path

    from truss_chains import framework
    from truss_chains.deployment import deployment_client
    from truss_chains import private_types as chains_def

    truss.login(api_key)
    source_path = Path(source_file)

    with framework.ChainletImporter.import_target(source_path) as entrypoint_cls:
        resolved_name = (
            chain_name
            or entrypoint_cls.meta_data.chain_name
            or entrypoint_cls.display_name
        )
        print(f"Entrypoint class: {entrypoint_cls.__name__}")
        print(f"Chain name: {resolved_name}")

        options = chains_def.PushOptionsBaseten.create(
            chain_name=resolved_name,
            promote=promote,
            publish=True,
            only_generate_trusses=False,
            remote="baseten",
            include_git_info=False,
            working_dir=source_path.parent,
        )
        return deployment_client.push(entrypoint_cls, options)


def wait_for_chain_active(chain_service, timeout):
    """Poll chain_service.get_info() until all chainlets are ready."""
    start = time.time()
    poll_interval = 10
    prev_statuses = {}
    logs_printed = False

    while True:
        elapsed = time.time() - start
        if elapsed > timeout:
            raise TimeoutError(
                f"Chain did not become active within {timeout}s"
            )

        chainlets = chain_service.get_info()
        statuses = {c.name: c.status for c in chainlets}

        # Print logs URLs once on first successful poll
        if not logs_printed:
            for c in chainlets:
                print(f"  {c.name}: {c.logs_url}", flush=True)
            logs_printed = True

        failed = [
            name for name, s in statuses.items()
            if s in CHAIN_FAILED_STATUSES
        ]
        if failed:
            raise RuntimeError(
                f"Chain deployment failed. Chainlet statuses: {statuses}"
            )

        if all(s in CHAIN_READY_STATUSES for s in statuses.values()):
            print(f"All chainlets ready ({elapsed:.0f}s)", flush=True)
            return time.time() - start

        # Only print when statuses change
        if statuses != prev_statuses:
            print(
                f"  Chainlet statuses ({elapsed:.0f}s): {statuses}",
                flush=True,
            )
            prev_statuses = statuses

        time.sleep(poll_interval)


def predict_chain(chain_id, deployment_id, api_key, payload, timeout):
    """Run a predict request against a chain deployment."""
    headers = {"Authorization": f"Api-Key {api_key}"}
    url = (
        f"https://chain-{chain_id}.api.baseten.co"
        f"/deployment/{deployment_id}/run_remote"
    )

    start = time.time()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    elapsed = time.time() - start
    body = resp.text[:4096]

    return {
        "response": body,
        "total_time": elapsed,
        "ttfb": elapsed,
        "tokens": 0,
        "tokens_per_sec": 0,
        "streaming": False,
    }


def deactivate_chain(chain_id, deployment_id, api_key):
    headers = {"Authorization": f"Api-Key {api_key}"}
    url = (
        f"{BASETEN_API_URL}/chains/{chain_id}"
        f"/deployments/{deployment_id}/deactivate"
    )
    resp = requests.post(url, headers=headers, timeout=30)
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def log_group(title):
    """Print a collapsible group marker for GitHub Actions logs."""
    if IN_GITHUB_ACTIONS:
        print(f"::group::{title}")


def log_endgroup():
    if IN_GITHUB_ACTIONS:
        print("::endgroup::")


def write_output(name, value):
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a") as f:
        value_str = str(value)
        if "\n" in value_str:
            f.write(f"{name}<<EOF\n{value_str}\nEOF\n")
        else:
            f.write(f"{name}={value_str}\n")


def write_summary(
    name, status, deployment_id, entity_id, deploy_time, predict_result,
    is_chain=False,
):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    deploy_type = "Chain" if is_chain else "Model"
    ok = status == "success"
    id_label = "Chain ID" if is_chain else "Model ID"
    lines = [
        f"## {'✅' if ok else '❌'} Truss Deploy ({deploy_type}): {name}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Status** | `{status}` |",
        f"| **{id_label}** | `{entity_id or 'N/A'}` |",
        f"| **Deployment ID** | `{deployment_id or 'N/A'}` |",
        f"| **Deploy Time** | {deploy_time:.1f}s |",
    ]

    if predict_result:
        lines.append(
            f"| **Predict Total Time** | {predict_result['total_time']:.2f}s |"
        )
        if predict_result["streaming"]:
            lines.append(f"| **TTFB** | {predict_result['ttfb']:.2f}s |")
            lines.append(f"| **Tokens** | {predict_result['tokens']} |")
            lines.append(
                f"| **Tokens/sec** | {predict_result['tokens_per_sec']:.1f} |"
            )

    if entity_id and deployment_id and not is_chain:
        lines.append("")
        lines.append(
            f"[View logs](https://app.baseten.co/models/{entity_id}"
            f"/logs/{deployment_id})"
        )

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Model flow
# ---------------------------------------------------------------------------

def run_model(truss_directory, api_key, model_name_override, should_promote,
              environment, preserve_prev, include_git_info, labels,
              deployment_name, should_cleanup, payload_override,
              deploy_timeout_minutes, predict_timeout):
    deploy_timeout_seconds = deploy_timeout_minutes * 60

    status = "success"
    phase = "config"
    deployment_id = None
    model_id = None
    model_name = None
    deploy_start = None
    deploy_time = 0.0
    predict_result = None

    try:
        # Phase 1: Load config
        log_group("Load config")
        print(f"Loading config from {truss_directory}/config.yaml")
        config = load_config(truss_directory)
        model_name = config.get("model_name", truss_directory)
        payload = get_predict_payload(config, payload_override)
        log_endgroup()

        # Phase 2: Deploy
        phase = "deploy"
        deploy_start = time.time()
        log_group(f"Deploy {model_name}")
        print(f"Deploying {model_name}...")
        deployment = deploy(
            truss_directory,
            api_key,
            should_promote,
            deployment_name,
            model_name=model_name_override,
            environment=environment,
            preserve_previous_production_deployment=preserve_prev,
            include_git_info=include_git_info,
            labels=labels,
            deploy_timeout_minutes=deploy_timeout_minutes,
        )
        deployment_id = deployment.model_deployment_id
        model_id = deployment.model_id
        print(f"Deployment ID: {deployment_id}")
        print(f"Model ID: {model_id}")
        print(
            f"Logs: https://app.baseten.co/models/{model_id}/logs/{deployment_id}"
        )
        log_endgroup()

        log_group(f"Wait for active (timeout: {deploy_timeout_minutes}m)")
        log_proc = start_log_stream(model_id, deployment_id)
        try:
            deploy_time = wait_for_active(deployment, deploy_timeout_seconds)
        finally:
            stop_log_stream(log_proc)
        print(f"Deployment active in {deploy_time:.1f}s")
        log_endgroup()

        # Phase 3: Predict
        if payload:
            phase = "predict"
            log_group("Predict")
            print(f"Running predict (timeout: {predict_timeout}s)...")
            predict_result = predict(
                model_id, deployment_id, api_key, payload, predict_timeout
            )
            print(f"Predict completed in {predict_result['total_time']:.2f}s")
            if predict_result["streaming"]:
                print(f"  TTFB: {predict_result['ttfb']:.2f}s")
                print(f"  Tokens: {predict_result['tokens']}")
                print(f"  Tokens/sec: {predict_result['tokens_per_sec']:.1f}")
            log_endgroup()
        else:
            print("No predict payload configured, skipping predict check")

    except TimeoutError as e:
        status = "deploy_timeout"
        elapsed = time.time() - deploy_start if deploy_start else 0
        print(f"\nERROR: {status} after {elapsed:.0f}s - {e}")
    except Exception as e:
        status = PHASE_TO_STATUS.get(phase, "deploy_failed")
        print(f"\nERROR: {status} - {e}")

    finally:
        # Cleanup
        if deployment_id and should_cleanup:
            log_group("Cleanup")
            print(f"Deactivating deployment {deployment_id}...")
            try:
                deactivate_deployment(model_id, deployment_id, api_key)
                print("Deployment deactivated")
            except Exception as e:
                print(f"WARNING: Cleanup failed - {e}")
                if status == "success":
                    status = "cleanup_failed"
            log_endgroup()

        # Write outputs
        write_output("deployment-id", deployment_id or "")
        write_output("model-id", model_id or "")
        write_output("chain-id", "")
        write_output("model-name", model_name or "")
        write_output("deploy-time-seconds", f"{deploy_time:.1f}")
        write_output(
            "predict-response",
            predict_result["response"] if predict_result else "",
        )
        write_output("status", status)

        write_summary(
            model_name or "unknown", status, deployment_id, model_id,
            deploy_time, predict_result, is_chain=False,
        )

        print(f"\nFinal status: {status}")
        if status != "success":
            sys.exit(1)


# ---------------------------------------------------------------------------
# Chain flow
# ---------------------------------------------------------------------------

def run_chain(source_file, api_key, model_name_override, should_promote,
              should_cleanup, payload_override, deploy_timeout_minutes,
              predict_timeout):
    deploy_timeout_seconds = deploy_timeout_minutes * 60

    status = "success"
    phase = "config"
    chain_id = None
    deployment_id = None
    chain_name = None
    deploy_start = None
    deploy_time = 0.0
    predict_result = None

    try:
        # Phase 1: Deploy chain
        phase = "deploy"
        deploy_start = time.time()
        log_group("Deploy chain")
        chain_service = deploy_chain(
            source_file, model_name_override, api_key, should_promote,
        )
        chain_name = chain_service.name
        handle = chain_service._chain_deployment_handle
        chain_id = handle.chain_id
        deployment_id = handle.chain_deployment_id
        print(f"Chain ID: {chain_id}")
        print(f"Deployment ID: {deployment_id}")
        print(f"Status page: {chain_service.status_page_url}")
        log_endgroup()

        # Phase 2: Wait for active
        log_group(f"Wait for active (timeout: {deploy_timeout_minutes}m)")
        deploy_time = wait_for_chain_active(
            chain_service, deploy_timeout_seconds,
        )
        print(f"Chain active in {deploy_time:.1f}s")
        log_endgroup()

        # Phase 3: Predict
        payload = json.loads(payload_override) if payload_override else None
        if payload:
            phase = "predict"
            log_group("Predict")
            print(f"Running predict (timeout: {predict_timeout}s)...")
            predict_result = predict_chain(
                chain_id, deployment_id, api_key, payload, predict_timeout,
            )
            print(f"Predict completed in {predict_result['total_time']:.2f}s")
            log_endgroup()
        else:
            print("No predict payload configured, skipping predict check")

    except TimeoutError as e:
        status = "deploy_timeout"
        elapsed = time.time() - deploy_start if deploy_start else 0
        print(f"\nERROR: {status} after {elapsed:.0f}s - {e}")
    except Exception as e:
        status = PHASE_TO_STATUS.get(phase, "deploy_failed")
        print(f"\nERROR: {status} - {e}")

    finally:
        # Cleanup
        if deployment_id and should_cleanup:
            log_group("Cleanup")
            print(f"Deactivating chain deployment {deployment_id}...")
            try:
                deactivate_chain(chain_id, deployment_id, api_key)
                print("Chain deployment deactivated")
            except Exception as e:
                print(f"WARNING: Cleanup failed - {e}")
                if status == "success":
                    status = "cleanup_failed"
            log_endgroup()

        # Write outputs
        write_output("deployment-id", deployment_id or "")
        write_output("model-id", "")
        write_output("chain-id", chain_id or "")
        write_output("model-name", chain_name or "")
        write_output("deploy-time-seconds", f"{deploy_time:.1f}")
        write_output(
            "predict-response",
            predict_result["response"] if predict_result else "",
        )
        write_output("status", status)

        write_summary(
            chain_name or "unknown", status, deployment_id, chain_id,
            deploy_time, predict_result, is_chain=True,
        )

        print(f"\nFinal status: {status}")
        if status != "success":
            sys.exit(1)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    truss_directory = os.environ["TRUSS_DIRECTORY"]
    api_key = os.environ["BASETEN_API_KEY"]
    model_name_override = os.environ.get("MODEL_NAME", "").strip() or None
    should_promote = os.environ.get("PROMOTE", "false").lower() == "true"
    should_cleanup = os.environ.get("CLEANUP", "false").lower() == "true"
    payload_override = os.environ.get("PREDICT_PAYLOAD", "").strip()
    deploy_timeout_minutes = int(os.environ.get("DEPLOY_TIMEOUT_MINUTES", "45"))
    predict_timeout = int(os.environ.get("PREDICT_TIMEOUT", "300"))

    is_chain = truss_directory.endswith(".py")

    if is_chain:
        print(f"Detected chain source file: {truss_directory}")
        run_chain(
            truss_directory, api_key, model_name_override, should_promote,
            should_cleanup, payload_override, deploy_timeout_minutes,
            predict_timeout,
        )
    else:
        environment = os.environ.get("ENVIRONMENT", "").strip() or None
        preserve_prev = (
            os.environ.get(
                "PRESERVE_PREVIOUS_PRODUCTION_DEPLOYMENT", "false"
            ).lower() == "true"
        )
        include_git_info = (
            os.environ.get("INCLUDE_GIT_INFO", "true").lower() == "true"
        )
        labels_raw = os.environ.get("LABELS", "").strip()
        labels = json.loads(labels_raw) if labels_raw else None
        deployment_name = os.environ.get("DEPLOYMENT_NAME", "").strip()
        if not deployment_name:
            deployment_name = build_deployment_name()

        print(f"Detected model directory: {truss_directory}")
        run_model(
            truss_directory, api_key, model_name_override, should_promote,
            environment, preserve_prev, include_git_info, labels,
            deployment_name, should_cleanup, payload_override,
            deploy_timeout_minutes, predict_timeout,
        )


if __name__ == "__main__":
    main()
