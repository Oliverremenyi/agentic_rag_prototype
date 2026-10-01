"""One-time, offline data preparation (constitution Principle V: never run in Docker).

Downloads the public Enterprise RAG Challenge dataset via kagglehub (requires the
operator's own Kaggle account/API token), selects a small, topically varied curated
subset of companies, and copies their report markdown + simplified table-of-contents
metadata into data/docs/ and data/toc/ for src/rag/ingest.py to consume.

See specs/001-agentic-rag-chatbot/research.md decision 4 for the curation rationale.

Usage: uv run python scripts/prepare_dataset.py
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "data" / "docs"
TOC_DIR = REPO_ROOT / "data" / "toc"

DATASET_HANDLE = "rrr3try/enterprise-rag-markdown"

# Curated for industry variety plus at least one same-currency (USD) pair with
# comparable financial-performance data, so a comparison question has a clean
# answer. Toshiba Corporation is included for topic variety but its report in
# this dataset is (oddly) tagged currency AUD rather than JPY/USD, so it is
# deliberately NOT used as one half of the comparison eval questions.
CURATED_COMPANIES = [
    "Microsoft Corporation",
    "Toshiba Corporation",
    "HCA Healthcare, Inc.",
    "Insperity, Inc.",
    "1-800-FLOWERS.COM, INC.",
    "MGM Resorts International",
]


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _find_dataset_root() -> Path:
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def _simplify_table_of_contents(raw_toc: list[dict]) -> list[dict]:
    simplified = []
    for entry in raw_toc:
        title = (entry.get("title") or "").strip()
        page_id = entry.get("page_id")
        if not title or page_id is None:
            continue
        simplified.append({"title": title, "start_page": page_id})
    return simplified


def prepare_dataset() -> list[dict]:
    dataset_root = _find_dataset_root()
    subset_path = dataset_root / "subset.json"
    markdown_root = dataset_root / "EnterpriseRAG_2025_02_markdown"

    with open(subset_path, encoding="utf-8") as fh:
        subset = json.load(fh)
    by_name = {entry["company_name"]: entry for entry in subset}

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    TOC_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for company_name in CURATED_COMPANIES:
        entry = by_name.get(company_name)
        if entry is None:
            raise ValueError(f"Company not found in dataset subset.json: {company_name}")

        sha1 = entry["sha1"]
        company_dir = markdown_root / sha1
        md_src = company_dir / f"{sha1}.md"
        meta_src = company_dir / f"{sha1}_meta.json"
        if not md_src.exists() or not meta_src.exists():
            raise FileNotFoundError(f"Missing markdown/meta for {company_name} ({sha1})")

        slug = _slugify(company_name)
        md_dst = DOCS_DIR / f"{slug}.md"
        toc_dst = TOC_DIR / f"{slug}.json"

        shutil.copyfile(md_src, md_dst)

        with open(meta_src, encoding="utf-8") as fh:
            meta = json.load(fh)
        toc = _simplify_table_of_contents(meta.get("table_of_contents", []))
        with open(toc_dst, "w", encoding="utf-8") as fh:
            json.dump(toc, fh, indent=2)

        manifest.append(
            {
                "slug": slug,
                "company_name": company_name,
                "industry": entry.get("major_industry"),
                "currency": entry.get("cur"),
                "sha1": sha1,
                "sections": len(toc),
            }
        )
        print(f"Prepared {company_name!r} -> {md_dst.name} ({len(toc)} sections)")

    manifest_path = DOCS_DIR.parent / "docs_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nWrote manifest for {len(manifest)} companies to {manifest_path}")
    return manifest


if __name__ == "__main__":
    prepare_dataset()
