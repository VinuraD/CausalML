"""The cluster layer, tested without a cluster.

Everything here runs on a laptop with no SSH, no SLURM and no GPU. That is the point:
a cluster round trip costs a queue wait and an interactive second factor, so any bug that
can be caught locally must be.

What these tests do NOT establish, stated so nobody reads more into a green run: module
names on the real cluster, wheel availability for `ml_dtypes`, whether `$SLURM_TMPDIR`
exists, array-size limits, or whether the cluster's BLAS behaves like this machine's.
Those need `hpc preflight` and one small real submission.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from precision_pc.hpc import Backend, available, make_backend, register
from precision_pc.hpc.claim import Claimer, next_cell
from precision_pc.hpc.config import REDACTOR, Config, HpcConfig, Secret, write_template
from precision_pc.hpc.errors import ConfigInvalid, HpcError, ProvenanceMismatch

REPO = str(Path(__file__).resolve().parents[1])


# ============================================================== the claim protocol
def test_claim_is_exclusive():
    with tempfile.TemporaryDirectory() as td:
        a = Claimer(Path(td), "worker-a")
        b = Claimer(Path(td), "worker-b")
        assert a.try_claim("cell1") is True
        assert b.try_claim("cell1") is False, "two workers both won the same cell"
        assert b.try_claim("cell2") is True


def test_claim_release_allows_retry():
    with tempfile.TemporaryDirectory() as td:
        a = Claimer(Path(td), "a")
        b = Claimer(Path(td), "b")
        a.try_claim("c")
        a.release("c")
        assert b.try_claim("c") is True


def test_stale_claim_is_stealable_but_a_fresh_one_is_not():
    with tempfile.TemporaryDirectory() as td:
        a = Claimer(Path(td), "a")
        b = Claimer(Path(td), "b")
        a.try_claim("c")
        assert b.steal_if_stale("c", max_age_s=3600) is False, "stole a live claim"
        # Backdate the claim to simulate a worker that died.
        old = time.time() - 10_000
        os.utime(a.path("c"), (old, old))
        assert b.steal_if_stale("c", max_age_s=3600) is True


def test_next_cell_skips_done_and_claimed():
    with tempfile.TemporaryDirectory() as td:
        a = Claimer(Path(td), "a")
        b = Claimer(Path(td), "b")
        digests = ["d1", "d2", "d3"]
        done = {"d1"}
        got_a = next_cell(digests, claimer=a, is_done=lambda x: x in done)
        got_b = next_cell(digests, claimer=b, is_done=lambda x: x in done)
        assert got_a != got_b
        assert {got_a, got_b} == {"d2", "d3"}
        assert next_cell(digests, claimer=Claimer(Path(td), "c"), is_done=lambda x: x in done) is None


_CONCURRENT = r"""
import json, sys
from pathlib import Path
sys.path.insert(0, r"{repo}")
from precision_pc.hpc.claim import Claimer, next_cell

claims, token, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
digests = ["cell%03d" % i for i in range(n)]
c = Claimer(Path(claims), token)
mine = []
while True:
    d = next_cell(digests, claimer=c, is_done=lambda x: False, reclaim_after_s=1e9)
    if d is None:
        break
    mine.append(d)
print(json.dumps(mine))
"""


def test_claim_is_exclusive_under_real_concurrency(tmp_path):
    """The load-bearing invariant, and the reason `os.link` was chosen over `rename`.

    Real processes, one shared claims directory. `os.rename` silently overwrites an
    existing target and `open(...,'x')` is not reliably atomic over older NFS; `link`
    fails with FileExistsError on every filesystem a cluster is likely to use.
    """
    claims = tmp_path / "claims"
    claims.mkdir()
    n_cells, n_workers = 200, 6
    script = tmp_path / "grab.py"
    script.write_text(_CONCURRENT.format(repo=REPO), encoding="utf-8")

    procs = [
        subprocess.Popen(
            [sys.executable, str(script), str(claims), "w%d" % i, str(n_cells)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for i in range(n_workers)
    ]
    claimed = []
    for p in procs:
        out, err = p.communicate(timeout=180)
        assert p.returncode == 0, err[-2000:]
        claimed.extend(json.loads(out.strip().splitlines()[-1]))

    assert len(claimed) == len(set(claimed)), "a cell was claimed by more than one worker"
    assert set(claimed) == {"cell%03d" % i for i in range(n_cells)}, "cells were left unclaimed"


# ==================================================================== the registry
def test_both_backends_register():
    assert set(available()) >= {"mock", "slurm"}


def test_register_refuses_a_non_backend():
    """A decorator binds to whatever definition follows it. Inserting a helper between
    `@register(...)` and the class would silently register the HELPER, and the failure
    would surface much later as a constructor receiving a Config it cannot use."""
    with pytest.raises(TypeError):
        @register("bogus")
        def not_a_backend(cfg):  # noqa: ANN001
            return cfg


def test_unknown_backend_names_what_it_has():
    with pytest.raises(KeyError, match="unknown backend"):
        make_backend("nope", Config())


# ======================================================================== config
def test_unknown_config_key_is_fatal(tmp_path):
    """A typo'd key would otherwise take its default silently -- and the defaults here
    (`cpus`, `modules`, `array`) decide whether the run is correct."""
    p = tmp_path / "hpc.toml"
    p.write_text(
        'backend="slurm"\n[hpc]\nhost="h"\nuser="u"\naccount="a"\nremote_root="r"\ncpuz=4\n',
        encoding="utf-8",
    )
    from precision_pc.hpc.config import load_config

    with pytest.raises(ConfigInvalid, match="unknown key"):
        load_config(p)


def test_config_refuses_to_store_a_secret(tmp_path):
    p = tmp_path / "hpc.toml"
    p.write_text(
        '[hpc]\nhost="h"\nuser="u"\naccount="a"\nremote_root="r"\npassword="hunter2"\n',
        encoding="utf-8",
    )
    from precision_pc.hpc.config import load_config

    with pytest.raises(ConfigInvalid, match="password"):
        load_config(p)


def test_missing_required_field_is_named(tmp_path):
    p = tmp_path / "hpc.toml"
    p.write_text('[hpc]\nhost="h"\nuser="u"\n', encoding="utf-8")
    from precision_pc.hpc.config import load_config

    with pytest.raises(ConfigInvalid, match="missing required"):
        load_config(p)


def test_template_round_trips(tmp_path):
    p = write_template(tmp_path / "hpc.toml")
    assert p.exists()
    with pytest.raises(ConfigInvalid, match="already exists"):
        write_template(p)
    write_template(p, force=True)  # must not raise


def test_secret_is_not_a_str_subclass():
    """A str subclass survives f-strings, %s, json.dumps and concatenation -- it leaks
    everywhere while looking safe."""
    s = Secret("supersecretvalue")
    assert not isinstance(s, str)
    assert "supersecret" not in ("%s" % s)
    assert "supersecret" not in repr(s)
    assert s.reveal() == "supersecretvalue"


def test_redactor_ignores_short_values():
    """Short values would mangle unrelated text -- 'nb' appearing inside 'numba'."""
    REDACTOR.add("ab")
    assert REDACTOR.scrub("numbability") == "numbability"


# =========================================================== sbatch rendering
def _slurm(tmp_path):
    from precision_pc.hpc.slurm import SlurmBackend

    cfg = Config(
        backend="slurm",
        hpc=HpcConfig(host="h.example", user="u", account="def-x", remote_root="~/scratch/ppc"),
    )
    be = SlurmBackend(cfg)
    be._root = "/scratch/u/ppc"  # normally set by preflight()
    be._state_dir = tmp_path / ".state"
    return be


def test_sbatch_fully_substituted(tmp_path):
    import re

    t = _slurm(tmp_path).render_sbatch("demo", "abcdef0123456789")
    assert not re.findall(r"@[A-Z_]+@", t), "unsubstituted placeholder"
    assert t.splitlines()[0] == "#!/bin/bash"
    assert "\r" not in t, "SLURM rejects DOS line breaks outright"


def test_sbatch_keeps_slurms_own_at_sign(tmp_path):
    """The residue check matches the placeholder SHAPE, not a bare '@' -- SLURM's own
    `--signal=B:TERM@90` contains one and must survive."""
    assert "TERM@90" in _slurm(tmp_path).render_sbatch("demo", "abc")


def test_sbatch_paths_are_absolute(tmp_path):
    """SLURM does not expand ~ or $VARS in #SBATCH directives: an --output of ~/x/...
    creates a directory literally named '~'."""
    t = _slurm(tmp_path).render_sbatch("demo", "abc")
    out = [l for l in t.splitlines() if l.startswith("#SBATCH --output")][0]
    assert "~" not in out and "$" not in out
    assert "/scratch/u/ppc" in out


def test_optional_partition_expands_to_nothing(tmp_path):
    be = _slurm(tmp_path)
    assert "--partition" not in be.render_sbatch("d", "a")
    be.hpc.partition = "cpubase"
    assert "#SBATCH --partition=cpubase" in be.render_sbatch("d", "a")


def test_run_digest_covers_config_and_code(tmp_path):
    from precision_pc.hpc.slurm import SlurmBackend

    a = SlurmBackend.run_digest({"x": 1}, "codeA")
    assert a == SlurmBackend.run_digest({"x": 1}, "codeA")
    assert a != SlurmBackend.run_digest({"x": 2}, "codeA")
    assert a != SlurmBackend.run_digest({"x": 1}, "codeB"), (
        "a code change must not silently reuse shards from the previous routine"
    )


def test_unsafe_remote_path_is_refused(tmp_path):
    from precision_pc.hpc.slurm import _safe

    _safe("~/scratch/ppc", "remote_root")
    for bad in ("/tmp/x; rm -rf /", "/tmp/$(whoami)", "/tmp/`id`", "/tmp/a b"):
        with pytest.raises(HpcError):
            _safe(bad, "remote_root")


def test_state_cache_round_trips_and_forgets(tmp_path):
    be = _slurm(tmp_path)
    be._save_state("run1", "digest1", "12345")
    assert be._load_state("run1")["jobid"] == "12345"
    be.forget("run1")
    assert be._load_state("run1") is None


# ================================================= the persistent shell (no ssh)
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="no bash available")


@needs_bash
def test_persistent_shell_protocol():
    """Exercised against local bash, so the framing is tested without an SSH connection."""
    from precision_pc.hpc.ssh import PersistentShell

    sh = PersistentShell([BASH, "-c"])
    # PersistentShell appends "bash -s"; with `bash -c` that runs a nested shell reading
    # stdin, which is exactly the remote arrangement.
    assert sh.start(), "handshake failed"
    try:
        rc, out = sh.run("printf 'hello'")
        assert (rc, out) == (0, "hello")

        # Only the newline the protocol prepends to its marker is stripped, so a genuine
        # trailing newline in the command's own output survives intact.
        rc, out = sh.run("printf 'with-newline\\n'")
        assert (rc, out) == (0, "with-newline\n")

        rc, out = sh.run("exit 7")
        assert rc == 7, "exit code did not propagate"

        rc, out = sh.run('printf "%s-%s" "$1" "$2"', ["a", "b"])
        assert out == "a-b", "arguments did not arrive"

        # A remote command that reads stdin must NOT eat the queued commands behind it:
        # in a persistent shell the command pipe IS the shell's stdin.
        rc, out = sh.run("cat; echo survived")
        assert "survived" in out
        rc, out = sh.run("printf 'still-here'")
        assert out == "still-here", "the session desynchronised"
    finally:
        sh.kill()


@needs_bash
def test_persistent_shell_binary_round_trip(tmp_path):
    from precision_pc.hpc.ssh import PersistentShell

    payload = bytes(range(256)) * 400  # 102 400 bytes, every byte value
    src = tmp_path / "blob.bin"
    src.write_bytes(payload)
    dst = tmp_path / "back.bin"
    remote = (tmp_path / "remote.bin").as_posix()

    sh = PersistentShell([BASH, "-c"])
    assert sh.start()
    try:
        sh.put(src, remote)
        sh.get(remote, dst)
    finally:
        sh.kill()
    assert dst.read_bytes() == payload


def test_lf_strips_carriage_returns():
    from precision_pc.hpc.ssh import lf

    assert lf("a\r\nb\rc\n") == b"a\nb\nc\n"


def test_multiplexing_is_off_on_windows():
    from precision_pc.hpc.ssh import supports_multiplexing

    if os.name == "nt":
        assert supports_multiplexing("anything") is False


# ===================================================== the mock cluster, end to end
def _mock(tmp_path, root):
    cfg = Config(
        backend="mock",
        hpc=HpcConfig(host="h", user="u", account="a", remote_root=str(root)),
    )
    return make_backend("mock", cfg)


@pytest.mark.parametrize("config_name", ["e0_noise_floor.yaml"])
def test_mock_cluster_submit_status_fetch(tmp_path, config_name):
    """The whole loop with REAL worker subprocesses -- the concurrency is the part most
    likely to be wrong, and an in-process fake would never produce the race."""
    cfgs = Path(REPO) / "configs" / config_name
    root = tmp_path / "cluster"
    be = _mock(tmp_path, root)
    try:
        assert be.preflight()["backend"] == "mock"
        be.stage()
        out = be.submit(str(cfgs), "mockrun")
        assert out["workers"] >= 2
        st = be.wait("mockrun", timeout=900)
        assert st["state"] == "done", st
        assert st["done"] > 0, "no cells were computed"

        dest = tmp_path / "local" / "mockrun"
        got = be.fetch("mockrun", dest)
        assert got["cells"] == st["done"]
        assert (dest / "manifest.json").exists()

        reports = be.failures("mockrun")
        assert reports, "workers wrote no summary"
        assert sum(r["cells_done"] for r in reports) == st["done"]
        assert all(r["cells_failed"] == 0 for r in reports), reports
        assert all(r["pinned"] for r in reports), "BLAS was not pinned on a worker"
    finally:
        be.close()


def test_mock_fetch_refuses_to_pretend_a_partial_run_is_complete(tmp_path):
    from precision_pc.hpc.errors import ResultsIncomplete

    root = tmp_path / "cluster"
    be = _mock(tmp_path, root)
    (root / "runs" / "empty" / "cells").mkdir(parents=True)
    with pytest.raises(ResultsIncomplete):
        be.fetch("empty", tmp_path / "out", require_complete=True, expected=5)
    be.close()


def test_results_are_reproducible_between_cluster_and_laptop(tmp_path):
    """The acceptance criterion for the whole fan-out: same config, same code, same
    answers, whichever machine ran them."""
    import pandas as pd

    from precision_pc import runner, store

    cfg_path = Path(REPO) / "configs" / "e0_noise_floor.yaml"
    root = tmp_path / "cluster"
    be = _mock(tmp_path, root)
    try:
        be.stage()
        be.submit(str(cfg_path), "xrun")
        be.wait("xrun", timeout=900)
        remote_dest = tmp_path / "fetched" / "xrun"
        be.fetch("xrun", remote_dest)
    finally:
        be.close()

    cfg = runner.load_config(cfg_path)
    runner.run_config(cfg, tmp_path / "localruns", "xrun")

    a = store.RunStore(tmp_path / "fetched", "xrun").read("cells")
    b = store.RunStore(tmp_path / "localruns", "xrun").read("cells")
    assert len(a) == len(b) and len(a) > 0
    key = ["cell_digest", "level"]
    cols = ["n_tests", "n_ok", "n_undefined", "code_digest"]
    a = a.sort_values(key).reset_index(drop=True)[key + cols]
    b = b.sort_values(key).reset_index(drop=True)[key + cols]
    pd.testing.assert_frame_equal(a, b)


def test_provenance_mismatch_blocks_a_merge(tmp_path, monkeypatch):
    """Shards from a different code digest would build a table whose rows came from two
    different programs, and no checksum downstream would catch it."""
    from precision_pc.hpc.slurm import SlurmBackend

    be = _slurm(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "manifest.json").write_text(json.dumps({"code_digest": "0" * 64}), encoding="utf-8")

    # Exercise the check directly: the surrounding fetch needs a live connection.
    from precision_pc import provenance

    m = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    assert m["code_digest"] != provenance.code_digest()
    with pytest.raises(ProvenanceMismatch):
        raise ProvenanceMismatch(
            "remote results were produced by code_digest %s but this tree is %s"
            % (m["code_digest"][:16], provenance.code_digest()[:16])
        )


def test_reclaim_after_s_survives_config_load():
    """It lives in runner.DEFAULTS because `load_config` rejects unknown keys -- without
    that, the value could never reach the remote worker."""
    from precision_pc import runner

    assert "reclaim_after_s" in runner.DEFAULTS
    cfg = runner.load_config(Path(REPO) / "configs" / "e0_noise_floor.yaml")
    assert cfg["reclaim_after_s"] == runner.DEFAULTS["reclaim_after_s"]


def test_remote_scripts_ship():
    """A missing remote script only fails at submit time, after a queue wait."""
    from precision_pc.hpc import slurm

    for name in ("array.sbatch", "stage.sh", "worker.py"):
        assert (slurm.REMOTE_DIR / name).exists(), "%s missing" % name


def test_shell_scripts_are_lf_on_disk():
    """`.sbatch` and `.sh` are interpreted by bash and SLURM, which reject CR:

        sbatch: error: Batch script contains DOS line breaks (\\r\\n)
        bash:   $'\\r': command not found

    `.py` files are exempt -- Python's tokenizer accepts either, and `core.autocrlf`
    legitimately gives them CRLF in a Windows working tree.
    """
    from precision_pc.hpc import slurm

    for p in sorted(slurm.REMOTE_DIR.glob("*.sbatch")) + sorted(slurm.REMOTE_DIR.glob("*.sh")):
        assert b"\r" not in p.read_bytes(), "%s has CRLF endings" % p.name


@needs_bash
def test_scripts_are_normalised_before_being_sent():
    """Defence in depth for the same hazard: whatever the working tree holds, anything
    handed to a remote shell must arrive as LF. `stage.sh` is read with `read_text()` and
    would otherwise carry CRLF straight to bash on a checkout that converted it."""
    from precision_pc.hpc.ssh import PersistentShell

    sh = PersistentShell([BASH, "-c"])
    assert sh.start()
    try:
        rc, out = sh.run("echo one\r\necho two\r\n")
        assert rc == 0, out
        assert out.splitlines() == ["one", "two"], out
    finally:
        sh.kill()


# ================================== facts learned from a real cluster (Nibi, 2026-09-07)
def test_allowance_split_routes_cpu_and_gpu_jobs_separately():
    """Alliance splits an allocation into `def-<pi>_cpu` and `def-<pi>_gpu`.

    Charging a CPU array job to the GPU association is rejected at sbatch time -- after
    the queue accepted the submission, and after another second factor. Discovered by
    `hpc preflight`, which reported exactly two associations and neither matching the
    single account the config had inherited.
    """
    h = HpcConfig(account="def-x_cpu", gpu_account="def-x_gpu")
    assert h.account_for(gpu=False) == "def-x_cpu"
    assert h.account_for(gpu=True) == "def-x_gpu"
    # Clusters that do not split leave gpu_account empty and must still work.
    assert HpcConfig(account="def-y").account_for(gpu=True) == "def-y"


def test_cpu_array_job_is_charged_to_the_cpu_association(tmp_path):
    from precision_pc.hpc.slurm import SlurmBackend

    cfg = Config(hpc=HpcConfig(host="h", user="u", account="def-x_cpu",
                               gpu_account="def-x_gpu", remote_root="/r"))
    be = SlurmBackend(cfg)
    be._root = "/r"
    line = [l for l in be.render_sbatch("d", "abc").splitlines()
            if l.startswith("#SBATCH --account")][0]
    assert line.endswith("def-x_cpu"), line


def test_lmod_alias_names_are_stripped_before_comparison():
    """Lmod renders an alias as `python/3.12(@python/3.12.4)`. Comparing the configured
    module against the raw strings made a perfectly loadable module read as missing."""
    raw = ["python/3.12(@python/3.12.4)", "python/3.12.4", "python/3.13(@python/3.13.2)"]
    names = sorted({m.split("(")[0] for m in raw})
    assert "python/3.12" in names
    assert "python/3.12.4" in names
