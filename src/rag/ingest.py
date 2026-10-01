"""Document loading and section-aware chunking for the RAG subgraph.

Ingests the curated annual reports under data/docs/ (see
specs/001-agentic-rag-chatbot/data-model.md CompanyReport/ReportSection/Chunk)
using each report's own table-of-contents metadata (data/toc/<slug>.json,
produced by scripts/prepare_dataset.py) to chunk by section instead of a
naive fixed word count.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from src import config

_PAGE_MARKER_RE = re.compile(r"^\{(\d+)\}-+\s*$")
_IMAGE_PLACEHOLDER_RE = re.compile(r"^!\[\]\(_page_\d+_[A-Za-z]+_\d+\.jpeg\)\s*$")


@dataclass
class CompanyReport:
    doc_id: str
    company_name: str
    path: str
    toc_path: str
    content: str


@dataclass
class ReportSection:
    doc_id: str
    title: str
    start_page: int
    end_page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    company_name: str
    section_title: str
    text: str


def _load_manifest(manifest_path: Path) -> dict[str, str]:
    if not manifest_path.exists():
        return {}
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return {entry["slug"]: entry["company_name"] for entry in manifest}


def load_reports(
    docs_dir: Path | None = None,
    toc_dir: Path | None = None,
    manifest_path: Path | None = None,
) -> list[CompanyReport]:
    docs_dir = docs_dir or config.DOCS_DIR
    toc_dir = toc_dir or config.TOC_DIR
    manifest_path = manifest_path or config.DOCS_MANIFEST_PATH
    company_names = _load_manifest(manifest_path)

    reports: list[CompanyReport] = []
    for path in sorted(docs_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        slug = path.stem
        toc_path = toc_dir / f"{slug}.json"
        company_name = company_names.get(slug, slug.replace("-", " ").title())
        reports.append(
            CompanyReport(
                doc_id=slug,
                company_name=company_name,
                path=str(path),
                toc_path=str(toc_path),
                content=content,
            )
        )
    return reports


def _split_into_pages(content: str) -> dict[int, str]:
    """Split page-marked markdown (e.g. "{12}---...") into {page_num: text}."""
    pages: dict[int, list[str]] = {}
    current_page = 0
    for line in content.splitlines():
        match = _PAGE_MARKER_RE.match(line)
        if match:
            current_page = int(match.group(1))
            pages.setdefault(current_page, [])
            continue
        if _IMAGE_PLACEHOLDER_RE.match(line):
            continue
        pages.setdefault(current_page, []).append(line)
    return {page: "\n".join(lines) for page, lines in pages.items()}


def _load_toc(toc_path: Path) -> list[dict]:
    if not toc_path.exists():
        return []
    with open(toc_path, encoding="utf-8") as fh:
        return json.load(fh)


def build_sections(report: CompanyReport) -> list[ReportSection]:
    """Group a report's pages into sections using its table of contents.

    Multiple headings that land on the same page (common with marker-pdf
    extraction) are merged into one section for that page, titled by joining
    their headings, so we never duplicate a page's content across sections.
    """
    pages = _split_into_pages(report.content)
    if not pages:
        return []
    last_page = max(pages)

    toc = _load_toc(Path(report.toc_path))
    titles_by_page: dict[int, list[str]] = {}
    for entry in toc:
        start_page = entry.get("start_page")
        title = entry.get("title")
        if start_page is None or not title:
            continue
        titles_by_page.setdefault(start_page, []).append(title)

    boundary_pages = sorted(titles_by_page)
    if not boundary_pages or boundary_pages[0] > 0:
        boundary_pages = [0] + boundary_pages
        titles_by_page.setdefault(0, ["Front Matter"])

    sections: list[ReportSection] = []
    for i, start_page in enumerate(boundary_pages):
        end_page = boundary_pages[i + 1] - 1 if i + 1 < len(boundary_pages) else last_page
        end_page = max(end_page, start_page)
        section_text = "\n\n".join(
            pages[p] for p in range(start_page, end_page + 1) if p in pages
        ).strip()
        if not section_text:
            continue
        title = " / ".join(titles_by_page[start_page])
        sections.append(
            ReportSection(
                doc_id=report.doc_id,
                title=title,
                start_page=start_page,
                end_page=end_page,
                text=section_text,
            )
        )
    return sections


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows by character count (not word count).

    Financial tables in this corpus often contain long runs of non-whitespace
    (e.g. glued-together pipe-delimited numbers, or markdown table separator
    rows), so a "800 words" window can still exceed an embedding model's token
    limit if a handful of those "words" are each hundreds of characters long.
    Chunking by character count instead gives a hard, predictable upper bound.
    """
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    step = max(chunk_size - overlap, 1)
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size].strip()
        if not piece:
            break
        chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks


def build_chunks(reports: list[CompanyReport]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for report in reports:
        sections = build_sections(report)
        for section_index, section in enumerate(sections):
            pieces = chunk_text(section.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            for piece_index, piece in enumerate(pieces):
                chunks.append(
                    Chunk(
                        chunk_id=f"{report.doc_id}::{section_index}::{piece_index}",
                        doc_id=report.doc_id,
                        company_name=report.company_name,
                        section_title=section.title,
                        text=f"Section: {section.title}\n\n{piece}",
                    )
                )
    return chunks
