"""Work claiming, via hard links.

The fan-out here is a **work-stealing pull**, not a static partition. A shard manifest
lists every cell; each array task loops "claim the next unclaimed cell, compute it, write
the shard". That choice is not stylistic:

* Cell costs vary by orders of magnitude -- an FP8 cell hits the pseudo-inverse fallback
  on most tests and runs far slower than the FP64 cell beside it -- so any static
  `k::N` split leaves some tasks idle for most of the allocation.
* It decouples the number of array tasks from the number of cells, so `--array=0-63%16`
  becomes a pure throughput knob and `MaxArraySize` (commonly 1001) is never a limit.
* It is robust to preemption: a killed task simply stops claiming, and its unfinished
  cell is picked up by another worker once the claim goes stale.

`os.link` is the primitive because it is the one atomic create-if-absent operation that
behaves correctly on the shared filesystems clusters actually use. `os.rename` silently
overwrites an existing target, and `open(..., 'x')` is not reliably atomic over older
NFS. `link` fails with FileExistsError when the target exists, on every one of them.

**A claim is not a completion marker.** The only evidence a cell is done is its output
shard. A claim says "someone is working on this"; a stale claim with no shard means that
worker died, and the cell is fair game again.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass
class Claimer:
    """Claims cells for one worker."""

    claims_dir: Path
    token: str  # identifies this worker; recorded in the claim file
    _src: Optional[Path] = None

    def __post_init__(self):
        self.claims_dir = Path(self.claims_dir)
        self.claims_dir.mkdir(parents=True, exist_ok=True)

    def _source(self) -> Path:
        """One source file per worker, hard-linked to each claim it wins.

        Creating a fresh temp file per attempt would be wasteful and would leave litter
        behind on a hard kill. Linking a single file many times costs nothing; the link
        count rising is exactly the intended semantics.
        """
        if self._src is None:
            p = self.claims_dir / (".src-" + self.token)
            p.write_text(self.token + "\n" + str(time.time()), encoding="utf-8")
            self._src = p
        return self._src

    def path(self, digest: str) -> Path:
        return self.claims_dir / (digest + ".claim")

    def try_claim(self, digest: str) -> bool:
        """Atomically claim `digest`. False means someone else already holds it."""
        target = self.path(digest)
        try:
            os.link(self._source(), target)
            return True
        except FileExistsError:
            return False
        except OSError:
            # Some filesystems refuse hard links entirely. Fall back to O_EXCL, which is
            # atomic locally and on most modern NFS; the worst case is two workers
            # computing the same cell, which wastes time but cannot corrupt anything --
            # both write the same content-addressed shard atomically.
            try:
                fd = os.open(str(target), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                return False
            with os.fdopen(fd, "w") as fh:
                fh.write(self.token)
            return True

    def release(self, digest: str) -> None:
        """Give a claim back after a failure, so another worker can retry it."""
        try:
            self.path(digest).unlink()
        except FileNotFoundError:
            pass

    def age(self, digest: str) -> Optional[float]:
        try:
            return time.time() - self.path(digest).stat().st_mtime
        except FileNotFoundError:
            return None

    def is_stale(self, digest: str, max_age_s: float) -> bool:
        a = self.age(digest)
        return a is not None and a > max_age_s

    def steal_if_stale(self, digest: str, max_age_s: float) -> bool:
        """Re-claim a cell whose holder appears to have died.

        Only ever called on cells with no output shard, and only after `max_age_s`, which
        must exceed the longest plausible single-cell runtime. Setting it too low makes
        two workers duplicate work; it cannot make them corrupt anything.
        """
        if not self.is_stale(digest, max_age_s):
            return False
        self.release(digest)
        return self.try_claim(digest)

    def cleanup(self) -> None:
        if self._src is not None:
            try:
                self._src.unlink()
            except FileNotFoundError:
                pass
            self._src = None


def next_cell(
    digests: Iterable[str],
    *,
    claimer: Claimer,
    is_done,
    reclaim_after_s: float = 3600.0,
) -> Optional[str]:
    """The pull loop's single step: the next cell this worker should compute.

    Two passes, in order. First every unclaimed cell, so workers spread out instead of
    contending. Only when nothing is unclaimed does it consider stealing stale claims,
    which is the recovery path for a preempted worker.
    """
    pending: List[str] = [d for d in digests if not is_done(d)]
    for d in pending:
        if claimer.try_claim(d):
            return d
    for d in pending:
        if claimer.steal_if_stale(d, reclaim_after_s):
            return d
    return None
