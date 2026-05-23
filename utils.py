import sys
from collections.abc import Iterable


def progress_bar(current: int | Iterable[int], total: int, bar_length: int = 100) -> None:
    """Affiche une barre de progression sur une seule ligne."""
    if not isinstance(current, int):
        current = sum(current)
    fraction = current / total
    filled_length = int(bar_length * fraction)
    bar = "#" * filled_length + "-" * (bar_length - filled_length)
    percent = fraction * 100
    sys.stdout.write(f"\r|{bar}| {percent:.2f}% ({current}/{total})")
    if current >= total:
        print()
