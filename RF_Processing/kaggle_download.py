#!/usr/bin/env python3
"""
Download only selected files from a Kaggle dataset.

Default behavior:
- Lists all files in the dataset
- Filters file names containing "noise"
- Downloads only matched files
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError


DEFAULT_DATASET = "sgluege/noisy-drone-rf-signal-classification-v2"
DEFAULT_MATCH = "noise"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a subset of files from a Kaggle dataset by filename keyword."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Kaggle dataset slug (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--match",
        default=DEFAULT_MATCH,
        help=f"Case-insensitive keyword to match in file path (default: {DEFAULT_MATCH})",
    )
    parser.add_argument(
        "--out",
        default=".",
        help="Output directory for downloaded files (default: current directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list matched files, do not download",
    )
    parser.add_argument(
        "--list-out",
        default=None,
        help="Optional output .txt file to save file names (matched list)",
    )
    parser.add_argument(
        "--save-all-names",
        action="store_true",
        help="When used with --list-out, save all dataset file names instead of matched only",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Page size for listing dataset files (default: 200)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="Max retries for rate-limited requests (default: 8)",
    )
    parser.add_argument(
        "--backoff-base",
        type=float,
        default=1.5,
        help="Base seconds for exponential backoff (default: 1.5)",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.2,
        help="Small delay between successful API requests in seconds (default: 0.2)",
    )
    parser.add_argument(
        "--print-matched-names",
        action="store_true",
        help="Print all matched file names before download",
    )
    return parser.parse_args()


def _call_with_retry(api_call, *, max_retries: int, backoff_base: float, what: str):
    for attempt in range(max_retries + 1):
        try:
            return api_call()
        except HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code != 429 or attempt >= max_retries:
                raise
            delay = backoff_base * (2 ** attempt)
            print(
                f"[429] {what} rate-limited. Retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s..."
            )
            time.sleep(delay)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.out).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    # Kaggle API is paginated (default page_size=20). Collect all pages.
    files = []
    page_token = None
    while True:
        response = _call_with_retry(
            lambda: api.dataset_list_files(
                dataset=args.dataset,
                page_token=page_token,
                page_size=args.page_size,
            ),
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            what="ListDatasetFiles",
        )
        page_files = getattr(response, "files", None) or []
        files.extend(page_files)
        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

        next_page_token = (
            getattr(response, "next_page_token", None)
            or getattr(response, "nextPageToken", None)
            or getattr(response, "nextpage_token", None)
        )
        if not next_page_token:
            break
        page_token = next_page_token
    if not files:
        print(f"No files found in dataset: {args.dataset}")
        return

    keyword = args.match.lower()
    matched = [f for f in files if keyword in f.name.lower()]

    print(f"Dataset: {args.dataset}")
    print(f"Total files: {len(files)}")
    print(f"Matched files (keyword='{args.match}'): {len(matched)}")

    if args.list_out:
        list_path = Path(args.list_out).resolve()
        list_path.parent.mkdir(parents=True, exist_ok=True)
        names_to_save = files if args.save_all_names else matched
        with list_path.open("w", encoding="utf-8") as f:
            for item in names_to_save:
                f.write(f"{item.name}\n")
        print(
            f"Saved {len(names_to_save)} file names to: {list_path} "
            f"({'all' if args.save_all_names else 'matched'})"
        )

    if not matched:
        print("No matched files. Sample file names:")
        for sample in files[:20]:
            print(f"- {sample.name}")
        return

    if args.print_matched_names:
        for f in matched:
            print(f.name)

    if args.dry_run:
        return

    for idx, f in enumerate(matched, start=1):
        print(f"[{idx}/{len(matched)}] Downloading: {f.name}")
        _call_with_retry(
            lambda: api.dataset_download_file(
                dataset=args.dataset,
                file_name=f.name,
                path=str(output_dir),
                force=False,
                quiet=False,
            ),
            max_retries=args.max_retries,
            backoff_base=args.backoff_base,
            what=f"Download file {f.name}",
        )
        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

    print(f"Done. Files downloaded to: {output_dir}")


if __name__ == "__main__":
    main()
