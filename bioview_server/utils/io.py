from pathlib import Path


def get_cache_file(file_name):
    cache_file = Path.home() / ".bioview" / file_name

    if not cache_file.exists():
        cache_file.parent.mkdir(parents=False, exist_ok=True)
        cache_file.touch()

    return cache_file

# NOTE:DEPRECATED
def get_unique_path(dirname, filename):
    f_path = Path(dirname) / filename
    base, ext = f_path.stem, f_path.suffix
    counter = 1
    while f_path.exists():
        counter += 1
        f_path = f_path.with_name(f"{base}_{counter}{ext}")
    return f_path
