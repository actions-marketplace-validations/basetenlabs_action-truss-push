#!/usr/bin/env python3
"""Deploy and validate a Truss model on Baseten.

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
    model_name, status, deployment_id, model_id, deploy_time, predict_result
):
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return

    ok = status == "success"
    lines = [
        f"## {'✅' if ok else '❌'} Truss Deploy: {model_name}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| **Status** | `{status}` |",
        f"| **Model ID** | `{model_id or 'N/A'}` |",
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

    if model_id and deployment_id:
        lines.append("")
        lines.append(
            f"[View logs](https://app.baseten.co/models/{model_id}"
            f"/logs/{deployment_id})"
        )

    with open(summary_file, "a") as f:
        f.write("\n".join(lines) + "\n")


def main():
    truss_directory = os.environ["TRUSS_DIRECTORY"]
    api_key = os.environ["BASETEN_API_KEY"]
    model_name_override = os.environ.get("MODEL_NAME", "").strip() or None
    should_promote = os.environ.get("PROMOTE", "false").lower() == "true"
    environment = os.environ.get("ENVIRONMENT", "").strip() or None
    preserve_prev = (
        os.environ.get("PRESERVE_PREVIOUS_PRODUCTION_DEPLOYMENT", "false").lower()
        == "true"
    )
    include_git_info = (
        os.environ.get("INCLUDE_GIT_INFO", "true").lower() == "true"
    )
    labels_raw = os.environ.get("LABELS", "").strip()
    labels = json.loads(labels_raw) if labels_raw else None
    deployment_name = os.environ.get("DEPLOYMENT_NAME", "").strip()
    if not deployment_name:
        deployment_name = build_deployment_name()
    should_cleanup = os.environ.get("CLEANUP", "false").lower() == "true"
    payload_override = os.environ.get("PREDICT_PAYLOAD", "").strip()
    deploy_timeout_minutes = int(os.environ.get("DEPLOY_TIMEOUT_MINUTES", "45"))
    deploy_timeout_seconds = deploy_timeout_minutes * 60
    predict_timeout = int(os.environ.get("PREDICT_TIMEOUT", "300"))

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
        # Cleanup: deactivate deployment if requested
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
        write_output("model-name", model_name or "")
        write_output("deploy-time-seconds", f"{deploy_time:.1f}")
        write_output(
            "predict-response",
            predict_result["response"] if predict_result else "",
        )
        write_output("status", status)

        # Write summary
        write_summary(
            model_name or "unknown",
            status,
            deployment_id,
            model_id,
            deploy_time,
            predict_result,
        )

        print(f"\nFinal status: {status}")
        if status != "success":
            sys.exit(1)


if __name__ == "__main__":
    main()
