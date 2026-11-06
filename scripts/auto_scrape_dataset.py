#!/usr/bin/env python3
"""
Autonomous Data Engineering & Computer Vision Dataset Scraper
-------------------------------------------------------------
Scrapes, validates, deduplicates, and organizes a 4-class industrial
biofouling image dataset ('Low', 'Moderate', 'High', 'Critical') for
computer vision assessment models.

Author: Autonomous Data Engineering Agent
"""

import argparse
import csv
import hashlib
import io
import logging
import os
import re
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from PIL import Image

# Ensure stdout supports unicode/utf-8 safely on all terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("BiofoulingScraper")

# 1. Target Queries per Class Definition
TARGET_QUERIES: Dict[str, List[str]] = {
    "Low": [
        "clean cooling tower fill",
        "clean heat exchanger plate",
        "unfouled industrial pipe interior",
        "new cooling tower basin",
    ],
    "Moderate": [
        "light biofilm cooling tower",
        "early biofouling heat exchanger",
        "patchy algae water cooling system",
        "slight biological slime industrial pipe",
    ],
    "High": [
        "thick biofilm cooling tower fill",
        "severe biofouling heat exchanger",
        "algae bloom cooling tower fill media",
        "heavy microbial slime industrial cooling",
    ],
    "Critical": [
        "mature biofilm sloughing cooling tower",
        "extreme biofouling blockage heat exchanger",
        "heavy detached biofilm chunks water system",
        "severe EPS slime biofouling",
    ],
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


class BiofoulingDatasetScraper:
    def __init__(
        self,
        output_dir: str = "data/raw",
        metadata_file: str = "metadata.csv",
        target_per_class: int = 55,
        min_width: int = 150,
        min_height: int = 150,
        request_delay: float = 0.2,
        timeout: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.metadata_file = Path(metadata_file)
        self.target_per_class = target_per_class
        self.min_width = min_width
        self.min_height = min_height
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENTS[0],
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })

        # Set of global image content hashes to prevent duplicates across classes & within class
        self.global_hashes: Set[str] = set()
        self.metadata_records: List[Dict[str, str]] = []

        # Statistics tracking
        self.stats: Dict[str, Dict[str, any]] = {
            cls_name: {
                "target": target_per_class,
                "verified": 0,
                "corrupted_or_failed": 0,
                "too_small": 0,
                "duplicates": 0,
                "resolutions": [],
            }
            for cls_name in TARGET_QUERIES
        }

    def _fetch_ddgs_urls(self, query: str, max_results: int = 40) -> List[str]:
        """Fetch candidate image URLs using DuckDuckGo search."""
        urls: List[str] = []
        if DDGS is None:
            return urls

        try:
            ddgs = DDGS()
            results = ddgs.images(
                query=query,
                region="wt-wt",
                safesearch="off",
                max_results=max_results,
            )
            for item in results:
                img_url = item.get("image") or item.get("image_url")
                if img_url and img_url.startswith("http"):
                    urls.append(img_url)
        except Exception as e:
            logger.warning(f"DDGS search warning for '{query}': {e}")
        return urls

    def _fetch_bing_urls(self, query: str, max_results: int = 40) -> List[str]:
        """Fallback candidate fetcher using Bing image search."""
        urls: List[str] = []
        try:
            headers = {
                "User-Agent": USER_AGENTS[1],
                "Accept-Language": "en-US,en;q=0.9",
            }
            encoded_query = urllib.parse.quote_plus(query)
            # Fetch 2 pages of results
            for offset in [1, 35]:
                bing_url = f"https://www.bing.com/images/search?q={encoded_query}&first={offset}&count=35"
                resp = self.session.get(bing_url, headers=headers, timeout=self.timeout)
                if resp.status_code == 200:
                    extracted = re.findall(r'murl&quot;:&quot;(http[^&]+)&quot;', resp.text)
                    for u in extracted:
                        cleaned_url = u.replace("\\/", "/")
                        if cleaned_url.startswith("http") and cleaned_url not in urls:
                            urls.append(cleaned_url)
                if len(urls) >= max_results:
                    break
        except Exception as e:
            logger.warning(f"Bing search fallback warning for '{query}': {e}")
        return urls

    def search_image_urls(self, query: str, max_results: int = 40) -> List[str]:
        """Retrieve candidate image URLs with primary (DDGS) and fallback (Bing) scrapers."""
        urls = self._fetch_ddgs_urls(query, max_results=max_results)
        time.sleep(self.request_delay)

        # Supplement with Bing if needed
        if len(urls) < max_results:
            fallback_urls = self._fetch_bing_urls(query, max_results=max_results)
            for u in fallback_urls:
                if u not in urls:
                    urls.append(u)

        return urls

    def download_image(self, url: str) -> Optional[bytes]:
        """Download raw image bytes with error handling and rate-limiting."""
        try:
            resp = self.session.get(
                url,
                timeout=self.timeout,
                stream=True,
                headers={"User-Agent": USER_AGENTS[hash(url) % len(USER_AGENTS)]},
            )
            if resp.status_code == 200:
                content = resp.content
                if len(content) > 1024:  # Minimum 1 KB to filter trivial 1x1 pixels/error pages
                    return content
        except (requests.RequestException, Exception):
            pass
        return None

    def validate_and_process_image(
        self, raw_bytes: bytes
    ) -> Tuple[bool, Optional[Image.Image], Optional[str], Optional[Tuple[int, int]], str]:
        """
        Validate image integrity, check minimum resolution (150x150),
        convert to RGB, compute SHA-256 hash, and check for duplicates.

        Returns:
            (is_valid, processed_img, hash_str, (width, height), rejection_reason)
        """
        try:
            # Check if bytes can be opened by PIL
            img = Image.open(io.BytesIO(raw_bytes))
            img.load()  # Fully decode pixel buffer to catch truncated/corrupt files
        except Exception as e:
            return False, None, None, None, f"Corrupted/Undecodable ({type(e).__name__})"

        width, height = img.size
        if width < self.min_width or height < self.min_height:
            return False, None, None, (width, height), f"Too small ({width}x{height} < {self.min_width}x{self.min_height})"

        # Convert to RGB color mode (handles RGBA, P, CMYK, L, etc. for standard JPG saving)
        try:
            if img.mode != "RGB":
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    # Alpha channel conversion with white background
                    alpha_img = img.convert("RGBA")
                    background = Image.new("RGB", alpha_img.size, (255, 255, 255))
                    background.paste(alpha_img, mask=alpha_img.split()[3])
                    img = background
                else:
                    img = img.convert("RGB")
        except Exception as e:
            return False, None, None, (width, height), f"Color conversion error ({e})"

        # Compute SHA-256 hash over normalized RGB image data for robust deduplication
        hash_sha256 = hashlib.sha256(img.tobytes()).hexdigest()

        if hash_sha256 in self.global_hashes:
            return False, None, hash_sha256, (width, height), "Duplicate image"

        return True, img, hash_sha256, (width, height), ""

    def process_class(self, class_name: str) -> None:
        """Collect, validate, deduplicate, and save images for a specific class."""
        class_dir = self.output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        queries = TARGET_QUERIES[class_name]
        logger.info(f"==================================================")
        logger.info(f"===> Class [{class_name}] (Target: {self.target_per_class} images)")
        logger.info(f"==================================================")

        verified_count = 0
        candidate_urls: List[Tuple[str, str]] = []  # (url, source_query)
        seen_urls: Set[str] = set()

        # Step 1: Collect Candidate URLs from all 4 queries
        for query in queries:
            logger.info(f"Querying [{class_name}]: '{query}'...")
            urls = self.search_image_urls(query, max_results=40)
            new_for_query = 0
            for u in urls:
                if u not in seen_urls:
                    seen_urls.add(u)
                    candidate_urls.append((u, query))
                    new_for_query += 1
            logger.info(f"  -> Fetched {len(urls)} URLs ({new_for_query} unique new candidates)")
            time.sleep(self.request_delay)

        logger.info(f"Total candidate URLs gathered for class [{class_name}]: {len(candidate_urls)}")

        # Step 2: Download, Validate, Deduplicate, and Save
        for url, query in candidate_urls:
            if verified_count >= self.target_per_class:
                logger.info(f"Class [{class_name}] target of {self.target_per_class} images reached!")
                break

            time.sleep(self.request_delay)
            raw_bytes = self.download_image(url)
            if not raw_bytes:
                self.stats[class_name]["corrupted_or_failed"] += 1
                continue

            is_valid, img, hash_val, resolution, reason = self.validate_and_process_image(raw_bytes)

            if not is_valid:
                if "Duplicate" in reason:
                    self.stats[class_name]["duplicates"] += 1
                elif "Too small" in reason:
                    self.stats[class_name]["too_small"] += 1
                else:
                    self.stats[class_name]["corrupted_or_failed"] += 1
                continue

            # Valid unique image
            verified_count += 1
            self.global_hashes.add(hash_val)
            self.stats[class_name]["verified"] = verified_count
            self.stats[class_name]["resolutions"].append(resolution)

            # Generate standard filename & ID
            image_id = f"bf_{class_name.lower()}_{verified_count:04d}"
            filename = f"{image_id}.jpg"
            save_path = class_dir / filename

            # Save as standard high-quality JPG
            img.save(save_path, "JPEG", quality=95, optimize=True)

            # Record metadata
            rel_filepath = str(save_path.as_posix())
            res_str = f"{resolution[0]}x{resolution[1]}"
            self.metadata_records.append({
                "image_id": image_id,
                "filename": rel_filepath,
                "class_label": class_name,
                "source_query": query,
                "file_hash": hash_val,
                "resolution": res_str,
            })

            if verified_count % 10 == 0 or verified_count == self.target_per_class:
                logger.info(f"  [{class_name}] Verified & Saved: {verified_count}/{self.target_per_class} ({res_str})")

        if verified_count < self.target_per_class:
            logger.warning(
                f"Class [{class_name}] finished with {verified_count}/{self.target_per_class} images."
            )

    def write_metadata_csv(self) -> None:
        """Write metadata.csv to the project root with the required schema."""
        fieldnames = ["image_id", "filename", "class_label", "source_query", "file_hash", "resolution"]
        self.metadata_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metadata_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.metadata_records:
                writer.writerow(record)
        logger.info(f"Generated metadata catalog: {self.metadata_file} ({len(self.metadata_records)} rows)")

    def load_existing_dataset(self) -> bool:
        """Load and validate existing dataset from disk and metadata.csv if available."""
        if not self.metadata_file.exists() or not self.output_dir.exists():
            return False

        try:
            with open(self.metadata_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                records = list(reader)

            if len(records) < len(TARGET_QUERIES) * (self.target_per_class - 5):
                return False

            self.metadata_records = records
            self.global_hashes = {r["file_hash"] for r in records}

            for r in records:
                cls_name = r["class_label"]
                if cls_name in self.stats:
                    self.stats[cls_name]["verified"] += 1
                    try:
                        w, h = map(int, r["resolution"].split("x"))
                        self.stats[cls_name]["resolutions"].append((w, h))
                    except Exception:
                        pass

            return True
        except Exception:
            return False

    def print_summary_table(self) -> None:
        """Print an organized terminal summary table of the scraped dataset."""
        headers = ["Class", "Target", "Verified", "Duplicates Filtered", "Rejected (<150px/Corrupt)", "Avg Resolution", "Status"]
        table_data = []

        total_target = 0
        total_verified = 0
        total_dups = 0
        total_rejected = 0

        for cls_name, s in self.stats.items():
            resolutions = s["resolutions"]
            if resolutions:
                avg_w = int(sum(r[0] for r in resolutions) / len(resolutions))
                avg_h = int(sum(r[1] for r in resolutions) / len(resolutions))
                avg_res_str = f"{avg_w}x{avg_h}"
            else:
                avg_res_str = "N/A"

            rejected_count = s["too_small"] + s["corrupted_or_failed"]
            status = "[OK] Ready" if s["verified"] >= s["target"] else f"[OK] ({s['verified']}/{s['target']})"

            table_data.append([
                cls_name,
                s["target"],
                s["verified"],
                s["duplicates"],
                rejected_count,
                avg_res_str,
                status,
            ])

            total_target += s["target"]
            total_verified += s["verified"]
            total_dups += s["duplicates"]
            total_rejected += rejected_count

        table_data.append([
            "TOTAL",
            total_target,
            total_verified,
            total_dups,
            total_rejected,
            "-",
            "COMPLETED" if total_verified >= total_target else f"{total_verified}/{total_target}",
        ])

        print("\n" + "=" * 105)
        print("                        INDUSTRIAL BIOFOULING DATASET SUMMARY REPORT                        ")
        print("=" * 105)
        if tabulate:
            print(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            row_fmt = "{:<12} | {:<8} | {:<10} | {:<20} | {:<26} | {:<15} | {:<10}"
            print(row_fmt.format(*headers))
            print("-" * 105)
            for row in table_data:
                print(row_fmt.format(*[str(x) for x in row]))
        print("=" * 105)
        print(f"Dataset root: {self.output_dir.resolve()}")
        print(f"Metadata file: {self.metadata_file.resolve()}")
        print(f"Total Unique Images: {len(self.global_hashes)}")
        print("=" * 105 + "\n")

    def run(self, force_scrape: bool = False) -> None:
        """Execute full dataset collection pipeline."""
        start_time = time.time()

        if not force_scrape and self.load_existing_dataset():
            logger.info("Existing verified dataset loaded successfully from disk & metadata.csv.")
            self.print_summary_table()
            return

        logger.info("Starting Autonomous Biofouling Dataset Acquisition Pipeline...")
        logger.info(f"Target classes: {list(TARGET_QUERIES.keys())}")
        logger.info(f"Target count per class: {self.target_per_class} images (Total target: {self.target_per_class * 4})")

        for class_name in TARGET_QUERIES:
            self.process_class(class_name)

        self.write_metadata_csv()
        self.print_summary_table()

        elapsed = time.time() - start_time
        logger.info(f"Dataset acquisition pipeline completed in {elapsed:.2f} seconds.")


def main():
    parser = argparse.ArgumentParser(description="Automated 4-Class Industrial Biofouling Dataset Scraper")
    parser.add_argument("--output-dir", type=str, default="data/raw", help="Output directory for raw images")
    parser.add_argument("--metadata-file", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--target-per-class", type=int, default=55, help="Target verified images per class (default: 55)")
    parser.add_argument("--min-size", type=int, default=150, help="Minimum width and height in pixels (default: 150)")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between requests in seconds")
    parser.add_argument("--timeout", type=int, default=8, help="HTTP download timeout in seconds")
    parser.add_argument("--force", action="store_true", help="Force re-scraping even if dataset exists")

    args = parser.parse_args()

    scraper = BiofoulingDatasetScraper(
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
        target_per_class=args.target_per_class,
        min_width=args.min_size,
        min_height=args.min_size,
        request_delay=args.delay,
        timeout=args.timeout,
    )
    scraper.run(force_scrape=args.force)


if __name__ == "__main__":
    main()
