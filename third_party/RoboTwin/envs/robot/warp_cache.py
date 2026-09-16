"""Keep Warp's generated kernels private to each planner process."""

import atexit
import os
import shutil
import tempfile


_cache_pid = None
_cache_path = None


def configure_warp_cache():
    global _cache_pid, _cache_path
    pid = os.getpid()
    if _cache_pid == pid:
        return _cache_path

    import warp as wp
    import warp.build

    # Warp 0.10.1 writes shared PTX files without inter-process locking.
    # Use its API: this version does not read WARP_CACHE_PATH.
    path = tempfile.mkdtemp(prefix=f"robotwin-warp-{pid}-")
    wp.build.init_kernel_cache(path)
    _cache_pid, _cache_path = pid, path

    def cleanup():
        # A forked child must never remove its parent's cache.
        if os.getpid() == pid:
            shutil.rmtree(path, ignore_errors=True)

    atexit.register(cleanup)
    return path
