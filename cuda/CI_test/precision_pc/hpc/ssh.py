"""SSH transport: subprocess around the system client, not paramiko.

The organising constraint is **multi-factor authentication**. Every new TCP connection
costs an interactive second factor, so the whole module exists to minimise the number of
connections rather than the number of bytes.

Three invariants, each of which was a bug before it was a rule:

* **stderr is inherited, never piped.** Capturing it swallows the "Duo push sent" line and
  the user stares at a hung terminal with no idea a phone is buzzing.
* **stdin is inherited** for per-command connections, so the prompt is never starved.
* **Pipes are binary, not text mode.** A Windows text-mode pipe rewrites \\n to \\r\\n and
  the remote bash then reports `$'exit\\r': command not found`.

`PersistentShell` is the centrepiece: one `ssh -T host bash -s` held open for the life of
the client, with every command written to its stdin and read back to a nonce marker. One
authentication, many commands.
"""

from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .config import REDACTOR, HpcConfig
from .errors import AuthRequired, HpcError

log = logging.getLogger("precision_pc.hpc")

_CANDIDATES = [
    r"C:\Program Files\Git\usr\bin\ssh.exe",  # MSYS OpenSSH
    r"C:\Windows\System32\OpenSSH\ssh.exe",  # Win32 OpenSSH
    "/usr/bin/ssh",
    "/bin/ssh",
]


def find_ssh(explicit: str = "") -> str:
    """Resolve the ssh binary by absolute path, never by bare name."""
    if explicit:
        return explicit
    for c in _CANDIDATES:
        if os.path.exists(c):
            return c
    import shutil

    found = shutil.which("ssh")
    if not found:
        raise HpcError("no ssh client found; install OpenSSH or set hpc.ssh_exe")
    return found


def supports_multiplexing(ssh_exe: str) -> bool:
    """False on Windows, always.

    Win32-OpenSSH never implemented ControlMaster, and the MSYS build relies on an
    emulated AF_UNIX socket that fails in practice with
    `mux_client_request_session: read from master failed`. Authentication succeeds and
    then the session dies -- costing a second factor and producing nothing.
    """
    if os.name == "nt":
        return False
    try:
        out = subprocess.run(
            [ssh_exe, "-V"], capture_output=True, text=True, timeout=10
        ).stderr
        return "for_windows" not in out.lower()
    except Exception:
        return False


def lf(text: str) -> bytes:
    """UTF-8 bytes with UNIX line endings.

    `Path.write_text` on Windows translates each newline to CRLF, which produces a
    valid-looking sbatch file that SLURM rejects outright:
        sbatch: error: Batch script contains DOS line breaks (\\r\\n)
    """
    return text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8")


class PersistentShell:
    """One `ssh -T host bash -s`, reused for the life of the client."""

    def __init__(self, argv: List[str], *, timeout: float = 600.0):
        self.argv = argv
        self.timeout = timeout
        self.proc: Optional[subprocess.Popen] = None
        self.ok = False

    def start(self) -> bool:
        nonce = uuid.uuid4().hex[:12]
        try:
            self.proc = subprocess.Popen(
                self.argv + ["bash -s"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,  # inherited: the MFA prompt must stay visible
            )
        except Exception as exc:
            log.debug("persistent shell failed to start: %r", exc)
            return False
        self._write("echo READY-%s\n" % nonce)
        deadline_marker = ("READY-" + nonce).encode()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                return False
            if deadline_marker in line:
                self.ok = True
                return True

    def _write(self, s: str) -> None:
        assert self.proc and self.proc.stdin
        self.proc.stdin.write(s.encode("utf-8"))
        self.proc.stdin.flush()

    def run(self, script: str, args: Sequence[str] = ()) -> Tuple[int, str]:
        """Run a script remotely, returning (returncode, stdout)."""
        assert self.proc and self.proc.stdout
        nonce = uuid.uuid4().hex[:12]
        # Normalise HERE, not at every call site. `stage.sh` is read with read_text() and
        # a checkout that converted it would otherwise deliver CRLF straight to bash,
        # which answers with a command-not-found error on every single line.
        b64 = base64.b64encode(lf(script)).decode()
        quoted = " ".join("'" + a + "'" for a in args)
        # Three deliberate details:
        #  (a) the script goes to a temp FILE, not sourced inline, so `set -e`, traps and
        #      heredocs inside it behave as they would standalone;
        #  (b) `< /dev/null` -- in a persistent shell the command pipe IS the shell's
        #      stdin, so any remote command that reads stdin would swallow the commands
        #      queued behind it and desynchronise the session;
        #  (c) the marker carries a LEADING newline so it always starts a line, otherwise
        #      `base64 -w0` output (no trailing newline) merges into it and the reader
        #      blocks forever.
        cmd = (
            "t=$(mktemp); printf %%s '%s' | base64 -d > $t; "
            "bash $t %s < /dev/null; __rc=$?; rm -f $t; "
            "printf '\\nEND-%s %%d\\n' $__rc\n" % (b64, quoted, nonce)
        )
        self._write(cmd)
        marker = ("END-" + nonce).encode()
        chunks: List[bytes] = []
        rc = 1
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise HpcError("remote shell closed unexpectedly")
            if marker in line:
                try:
                    rc = int(line.split()[-1])
                except (ValueError, IndexError):
                    rc = 1
                break
            chunks.append(line)
        text = b"".join(chunks).decode("utf-8", errors="replace")
        # Strip the newline that was prepended to the marker.
        return rc, text[:-1] if text.endswith("\n") else text

    def put(self, local: Path, remote: str) -> None:
        """Send a file down the same pipe. No scp, so no second connection."""
        nonce = uuid.uuid4().hex[:12]
        data = base64.b64encode(Path(local).read_bytes()).decode()
        chunks = [data[i : i + 76] for i in range(0, len(data), 76)]
        script = "\n".join(
            [
                "set -eu",
                'mkdir -p "$(dirname "%s")" 2>/dev/null || true' % remote,
                "base64 -d > '%s' <<'B64-%s'" % (remote, nonce),
                *chunks,
                "B64-%s" % nonce,
            ]
        )
        rc, out = self.run(script)
        if rc != 0:
            raise HpcError("failed to upload %s: %s" % (remote, out[-500:]))

    def get(self, remote: str, local: Path) -> None:
        rc, out = self.run("base64 -w0 '%s'" % remote)
        if rc != 0:
            raise HpcError("failed to download %s: %s" % (remote, out[-500:]))
        Path(local).write_bytes(base64.b64decode(out.strip()))

    def kill(self) -> None:
        if self.proc:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
            try:
                self.proc.terminate()
            except Exception:
                pass
            self.proc = None
            self.ok = False


class Ssh:
    """Per-cluster wrapper. Lazily opens one persistent shell and reuses it."""

    def __init__(self, cfg: HpcConfig):
        self.cfg = cfg
        self.exe = find_ssh(cfg.ssh_exe)
        self.mux = supports_multiplexing(self.exe)
        self._shell: Optional[PersistentShell] = None
        self._shell_tried = False

    def _base_opts(self) -> List[str]:
        c = self.cfg
        opts = [
            "-i",
            os.path.expanduser(c.key_path),
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=4",
            "-o",
            "NumberOfPasswordPrompts=1",
        ]
        kh = os.path.expanduser(c.known_hosts)
        if os.path.exists(kh):
            # Strict checking only when a known_hosts file actually exists; otherwise fall
            # back to the user's default rather than TOFU-accepting an unknown key.
            opts += ["-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + kh]
        return opts

    def _target(self) -> str:
        return self.cfg.user + "@" + self.cfg.host

    @staticmethod
    def _require_console() -> None:
        if not sys.stdin.isatty():
            raise AuthRequired(
                "no interactive console. This cluster needs a second factor typed at a "
                "terminal, so it cannot run headless -- start it from a real shell."
            )

    def shell(self) -> Optional[PersistentShell]:
        if self._shell is not None:
            return self._shell if self._shell.ok else None
        if self._shell_tried:
            return None
        self._shell_tried = True
        self._require_console()
        sh = PersistentShell([self.exe, "-T"] + self._base_opts() + [self._target()])
        if sh.start():
            self._shell = sh
            return sh
        sh.kill()
        return None

    def run_script(self, script: str, args: Sequence[str] = (), *, scrub: bool = True):
        """Run a bash script remotely.

        `scrub=False` for output this code PARSES. Redaction keeps identifiers out of
        logs and artifacts; it must never corrupt the program's own inputs. `remote_root`
        is a registered value, so a scrubbed `pwd -P` returns `***` and every path
        derived from it is wrong.
        """
        sh = self.shell()
        if sh is not None:
            rc, out = sh.run(script, args)
        else:
            self._require_console()
            b64 = base64.b64encode(lf(script)).decode()
            quoted = " ".join("'" + a + "'" for a in args)
            cmd = (
                "t=$(mktemp); printf %%s '%s' | base64 -d > $t; "
                "bash $t %s; __rc=$?; rm -f $t; exit $__rc" % (b64, quoted)
            )
            argv = [self.exe, "-T"] + self._base_opts() + [self._target(), cmd]
            p = subprocess.run(argv, stdout=subprocess.PIPE, stderr=None, text=True)
            rc, out = p.returncode, p.stdout or ""
        return rc, (REDACTOR.scrub(out) if scrub else out)

    def push(self, local: Path, remote: str) -> None:
        sh = self.shell()
        if sh is None:
            raise HpcError("cannot upload without a persistent shell")
        sh.put(local, remote)

    def pull(self, remote: str, local: Path) -> None:
        sh = self.shell()
        if sh is None:
            raise HpcError("cannot download without a persistent shell")
        sh.get(remote, local)

    def close(self) -> None:
        if self._shell is not None:
            self._shell.kill()
            self._shell = None
