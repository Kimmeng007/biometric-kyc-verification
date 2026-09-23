import zlib


def hash_str(s: str) -> int:
    """Stable (process-independent) string hash, used to seed per-sample degradations."""
    return zlib.crc32(s.encode())
