"""Resume the DANDI data pull after a machine restart.

1. Read _pull_audit.json (produced by audit_dandi_pull.py).
2. Delete truncated files (logged to _pull_resume.log).
3. Download every entry in the queue in parallel (3-way), resumable via
   `curl -C -`, with retries.
4. Verify each file's final size against the audit's expected size.
5. Send notifications at start, every 5 completions, and on final result.

Run detached:
    cd /home/rohit/Desktop/website/personalwebsite
    nohup /home/rohit/miniconda3/envs/ml/bin/python \
        scripts/resume_dandi_pull.py \
        > data/external/_pull_stdout.log 2>&1 &
"""
from __future__ import annotations
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

BASE = Path("/home/rohit/Desktop/website/personalwebsite/data/external")
AUDIT = BASE / "_pull_audit.json"
LOG = BASE / "_pull_resume.log"
NOTIFY = "/home/rohit/bin/notify"
TITLE = "C. elegans data pull"
MAX_PARALLEL = 3
CURL_RETRIES = 5
CURL_RETRY_DELAY = 10


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    with open(LOG, "a") as f:
        f.write(line)
    sys.stdout.write(line)
    sys.stdout.flush()


def notify(msg: str, priority: str = "default") -> None:
    try:
        subprocess.run([NOTIFY, msg, TITLE, priority], timeout=10, check=False)
    except Exception as e:
        log(f"[NOTIFY FAIL] {e}")


def human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def delete_truncated(truncated: list[dict]) -> None:
    if not truncated:
        log("No truncated files to delete.")
        return
    freed = 0
    for rec in truncated:
        p = BASE / rec["subdir"] / rec["local_filename"]
        if p.exists():
            freed += p.stat().st_size
            p.unlink()
            log(f"DELETED {rec['subdir']}/{rec['local_filename']}")
        else:
            log(f"SKIP-DELETE (not found) {rec['subdir']}/{rec['local_filename']}")
    log(f"Freed {human(freed)} from {len(truncated)} partial files")


def target_name(rec: dict) -> str:
    """Preserve existing filename if the file was truncated (was already named
    by the previous session); use canonical name only for genuinely new files."""
    return rec.get("local_filename") or rec["canonical_name"]


def download_one(rec: dict) -> tuple[dict, int, str]:
    url = f"https://api.dandiarchive.org/api/assets/{rec['asset_id']}/download/"
    target = BASE / rec["subdir"] / target_name(rec)
    target.parent.mkdir(parents=True, exist_ok=True)
    # -L follow redirects (DANDI → S3 presigned)
    # -C - resume if partial exists
    # --fail exit non-zero on HTTP errors (otherwise curl writes error body to file)
    # --retry handles transient failures (includes presign re-resolve since curl re-requests)
    # --connect-timeout / --max-time prevent hangs
    cmd = [
        "curl", "-L", "--fail", "-C", "-",
        "--retry", str(CURL_RETRIES),
        "--retry-delay", str(CURL_RETRY_DELAY),
        "--retry-all-errors",
        "--connect-timeout", "30",
        "-sS",
        "-o", str(target),
        url,
    ]
    # Run without capturing stdout (curl silent with -sS, errors go to stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return rec, proc.returncode, (proc.stderr or "")[-400:]


def main() -> int:
    if not AUDIT.exists():
        log(f"ERROR: {AUDIT} not found — run audit_dandi_pull.py first.")
        return 2

    with open(AUDIT) as f:
        audit = json.load(f)

    truncated = audit["truncated"]
    queue = audit["download_queue"]
    total = len(queue)
    total_bytes = sum(r["expected_size"] for r in queue)

    log("=" * 60)
    log(f"Resume pull: {total} files, {human(total_bytes)} expected")
    log(f"Truncated to delete: {len(truncated)}")
    log(f"Missing to download: {len(audit['missing'])}")
    log("=" * 60)

    notify(f"Start: {total} files, {human(total_bytes)} to pull "
           f"(parallel={MAX_PARALLEL})")

    delete_truncated(truncated)

    done_ok = 0
    done_fail = 0
    done_bytes = 0
    fail_list: list[dict] = []
    t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL) as ex:
        futures = {ex.submit(download_one, r): r for r in queue}
        for fut in concurrent.futures.as_completed(futures):
            rec, rc, stderr = fut.result()
            fn = target_name(rec)
            target = BASE / rec["subdir"] / fn
            local_size = target.stat().st_size if target.exists() else 0
            exp = rec["expected_size"]

            if rc == 0 and local_size == exp:
                done_ok += 1
                done_bytes += exp
                log(f"OK  {rec['subdir']}/{fn} ({human(exp)})")
            else:
                done_fail += 1
                fail_list.append({"rec": rec, "rc": rc, "size": local_size,
                                  "expected": exp, "stderr": stderr})
                log(f"FAIL rc={rc} size={local_size}/{exp} "
                    f"{rec['subdir']}/{fn} :: {stderr[:200]}")

            n = done_ok + done_fail
            if n % 5 == 0 or n == total:
                elapsed = time.time() - t0
                rate = done_bytes / max(elapsed, 1)
                remaining = (total_bytes - done_bytes) / max(rate, 1)
                notify(
                    f"Progress {n}/{total} "
                    f"({done_ok} ok, {done_fail} fail) — "
                    f"{human(done_bytes)}, "
                    f"~{remaining/60:.0f} min left"
                )

    elapsed = time.time() - t0
    log("=" * 60)
    log(f"DONE: {done_ok}/{total} ok, {done_fail} failed in {elapsed/60:.1f} min")
    if fail_list:
        log("FAILURES:")
        for f in fail_list:
            log(f"  {f['rec']['subdir']}/{target_name(f['rec'])} "
                f"rc={f['rc']} size={f['size']}/{f['expected']}")

    if done_fail:
        notify(
            f"Pull complete: {done_ok}/{total} ok, {done_fail} FAILED "
            f"in {elapsed/60:.0f} min — check _pull_resume.log",
            "urgent",
        )
        return 1
    notify(
        f"Pull COMPLETE: all {total} files OK ({human(done_bytes)}) "
        f"in {elapsed/60:.0f} min",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
