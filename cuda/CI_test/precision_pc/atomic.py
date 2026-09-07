"""Atomic file writes that are safe against concurrent writers.

The obvious atomic-write idiom -- write `foo.tmp`, then `os.replace(foo.tmp, foo)` -- is
atomic with respect to *readers*, but it is not safe when two processes write the same
destination at once, because they share the one temp name.

That case is not hypothetical here. Every array task calls `RunStore.init()`, and any two
tasks handling different cells of the same seed write the same test-set artifact. With a
fixed temp name the two failure modes are:

  Windows   `os.replace` raises PermissionError [WinError 32] -- the other process still
            has the temp file open. Loud, and it kills the cell.
  POSIX     `rename` succeeds. Both writers were appending to the SAME temp file, so what
            gets renamed into place can be an interleaved, corrupt Parquet. Silent.

The POSIX case is the dangerous one: a corrupt shard renamed into place looks done, and
`is_done()` counts it. So the temp name must be unique per writer, and the destination is
content-addressed anyway -- whichever writer lands last, the bytes are equivalent.
"""

from __future__ import annotations

import os
import time
import uuid
from pathlib import Path


def tmp_sibling(path, suffix: str = ".tmp") -> Path:
    """A temp path beside `path`, unique to this process and call.

    Beside, not in the system temp directory, because `os.replace` is only atomic within
    a single filesystem and a cluster's scratch is rarely the same mount as /tmp.
    """
    path = Path(path)
    token = "%d-%s" % (os.getpid(), uuid.uuid4().hex[:8])
    return path.with_name(path.name + "." + token + suffix)


def write_bytes(path, data: bytes, *, retries: int = 5, delay_s: float = 0.05) -> Path:
    """Atomically write `data` to `path`, tolerating a concurrent writer.

    `manifest.json` is written by every array task, so several processes genuinely race
    for it. Unlike the content-addressed shards their contents are not identical (each
    records its own worker id), but any one of them is a valid manifest, so last-writer-
    wins is the correct semantics.

    Windows holds a brief lock rather than allowing the replace, and that lock is
    transient, so retry a few times before giving up. Only if the destination still does
    not exist is this a real failure worth raising.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = tmp_sibling(path)
    try:
        tmp.write_bytes(data)
        for attempt in range(retries):
            try:
                os.replace(tmp, path)
                return path
            except PermissionError:
                if attempt == retries - 1:
                    break
                time.sleep(delay_s * (attempt + 1))
        if path.exists():
            return path  # someone else's equally valid version landed
        os.replace(tmp, path)  # re-raise the real error
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        tmp.unlink(missing_ok=True)
    return path


def write_text(path, text: str, encoding: str = "utf-8") -> Path:
    return write_bytes(path, text.encode(encoding))


def finalize(tmp, path) -> Path:
    """Move a temp file this process wrote into its final place, cleaning up on failure.

    For callers that must hand a real path to a third-party writer (`pyarrow.write_table`,
    `np.savez`) rather than produce bytes themselves.
    """
    tmp, path = Path(tmp), Path(path)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows refuses to replace a destination another process holds open. Every
        # destination here is CONTENT-ADDRESSED, so a rival writer producing the same
        # path is producing equivalent bytes -- losing that race is success, not failure.
        # Re-raise only if the destination genuinely is not there.
        if path.exists():
            tmp.unlink(missing_ok=True)
            return path
        tmp.unlink(missing_ok=True)
        raise
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return path
