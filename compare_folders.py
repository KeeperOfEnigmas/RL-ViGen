#!/usr/bin/env python3
"""Compare file names and relative paths, without reading file contents.

Usage:
    python compare_folders.py "D:\\FolderA" "E:\\FolderB"
    python compare_folders.py "D:\\FolderA" "E:\\FolderB" --ignore-case
    python compare_folders.py "D:\\FolderA" "E:\\FolderB" --exclude cache logs "results/temp"

Excluded folder names (e.g. cache) match at any depth. Relative folder paths
(e.g. results/temp) match from each comparison root. Both skip all descendants.
Use / or \\ separators; wildcards are not supported. --ignore-case also applies
to exclusions. Repeat --exclude or list several folders after it.

Run without arguments to enter the two folder paths interactively.
Exit codes: 0 = same file paths, 1 = differences, 2 = error.
Empty folders are ignored. Directory symlinks/junctions are not followed.
"""

import argparse
import ntpath
import os
from pathlib import Path, PurePosixPath
import sys


def normalize_excludes(groups, ignore_case):
    exclusions = set()
    for group in groups:
        for value in group:
            value = value.replace("\\", "/")
            parts = PurePosixPath(value).parts
            if (not parts or value.startswith("/") or ntpath.splitdrive(value)[0]
                    or ".." in parts):
                raise ValueError("Exclusions must be folder names or relative paths: " + value)
            normalized = "/".join(parts)
            exclusions.add(normalized.casefold() if ignore_case else normalized)
    return exclusions


def collect_files(root, ignore_case, exclusions=()):
    files = {}

    def fail(error):
        raise error

    def excluded(directory, name):
        relative = (Path(directory) / name).relative_to(root).as_posix()
        if ignore_case:
            relative, name = relative.casefold(), name.casefold()
        return name in exclusions or relative in exclusions

    for directory, subfolders, names in os.walk(root, onerror=fail, followlinks=False):
        # Avoid traversing junctions as well as directory symlinks on Windows.
        subfolders[:] = [
            name for name in subfolders
            if not excluded(directory, name)
            and not os.path.islink(os.path.join(directory, name))
            and not getattr(os.path, "isjunction", lambda path: False)(
                os.path.join(directory, name)
            )
        ]
        for name in names:
            relative = (Path(directory) / name).relative_to(root).as_posix()
            key = relative.casefold() if ignore_case else relative
            files.setdefault(key, []).append(relative)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder_a", nargs="?")
    parser.add_argument("folder_b", nargs="?")
    parser.add_argument("--ignore-case", action="store_true",
                        help="Treat uppercase/lowercase names as equivalent.")
    parser.add_argument("--exclude", nargs="+", action="append", default=[],
                        metavar="FOLDER",
                        help="Skip folder names at any depth, or paths relative to each root. "
                             "Applies to both folders; can be repeated.")
    args = parser.parse_args()

    try:
        exclusions = normalize_excludes(args.exclude, args.ignore_case)
        roots = []
        for label, value in (("A", args.folder_a), ("B", args.folder_b)):
            if value is None:
                value = input("Folder " + label + ": ").strip().strip('"')
                if not value:
                    raise ValueError("Folder path cannot be empty.")
            root = Path(value).expanduser().resolve(strict=True)
            if not root.is_dir():
                raise ValueError("Not a folder: " + str(root))
            roots.append(root)

        files_a = collect_files(roots[0], args.ignore_case, exclusions)
        files_b = collect_files(roots[1], args.ignore_case, exclusions)
    except (OSError, ValueError, EOFError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 2

    only_a = sorted(files_a.keys() - files_b.keys())
    only_b = sorted(files_b.keys() - files_a.keys())
    print("Folder A: " + str(roots[0]))
    print("Folder B: " + str(roots[1]))
    if exclusions:
        print("Excluded folders: " + ", ".join(sorted(exclusions)))
    print("Files in A: " + str(sum(map(len, files_a.values()))))
    print("Files in B: " + str(sum(map(len, files_b.values()))))
    print("Matching relative paths: " + str(len(files_a.keys() & files_b.keys())))
    print("\nComparison uses file names and relative paths only.")

    if not only_a and not only_b:
        print("SAME: Both folders contain the same file paths.")
        return 0

    print("DIFFERENT: Some file paths exist in only one folder.")
    for label, keys, files in (("A", only_a, files_a), ("B", only_b, files_b)):
        print("\nOnly in folder " + label + ":")
        if not keys:
            print("  (none)")
        for key in keys:
            for relative in sorted(files[key]):
                print("  " + relative)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
