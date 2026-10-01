import json

from src.rag.ingest import CompanyReport, build_chunks, build_sections


def _write_report(tmp_path, slug: str, pages: list[str], toc: list[dict]):
    content_lines = []
    for i, page_text in enumerate(pages):
        content_lines.append(f"{{{i}}}------------------------------------------------")
        content_lines.append(page_text)
    content = "\n".join(content_lines)

    docs_dir = tmp_path / "docs"
    toc_dir = tmp_path / "toc"
    docs_dir.mkdir(exist_ok=True)
    toc_dir.mkdir(exist_ok=True)

    md_path = docs_dir / f"{slug}.md"
    md_path.write_text(content, encoding="utf-8")
    toc_path = toc_dir / f"{slug}.json"
    toc_path.write_text(json.dumps(toc), encoding="utf-8")

    return CompanyReport(
        doc_id=slug,
        company_name="Acme Corp",
        path=str(md_path),
        toc_path=str(toc_path),
        content=content,
    )


def test_build_sections_groups_pages_by_toc_boundaries(tmp_path):
    report = _write_report(
        tmp_path,
        "acme",
        pages=["Cover page text", "Intro section text", "Financials text page 2", "Financials text page 3"],
        toc=[
            {"title": "Introduction", "start_page": 1},
            {"title": "Financial Highlights", "start_page": 2},
        ],
    )

    sections = build_sections(report)

    titles = [s.title for s in sections]
    assert titles == ["Front Matter", "Introduction", "Financial Highlights"]
    assert sections[0].start_page == 0 and sections[0].end_page == 0
    assert sections[2].start_page == 2 and sections[2].end_page == 3
    assert "Financials text page 2" in sections[2].text
    assert "Financials text page 3" in sections[2].text


def test_build_sections_merges_headings_sharing_one_page(tmp_path):
    report = _write_report(
        tmp_path,
        "acme",
        pages=["Cover", "Multiple headings live here"],
        toc=[
            {"title": "Heading A", "start_page": 1},
            {"title": "Heading B", "start_page": 1},
        ],
    )

    sections = build_sections(report)

    assert len(sections) == 2
    merged = sections[1]
    assert merged.title == "Heading A / Heading B"
    assert merged.text.count("Multiple headings live here") == 1


def test_build_sections_strips_image_placeholders(tmp_path):
    report = _write_report(
        tmp_path,
        "acme",
        pages=["![](_page_0_Picture_0.jpeg)\nReal content here"],
        toc=[],
    )

    sections = build_sections(report)

    assert "jpeg" not in sections[0].text
    assert "Real content here" in sections[0].text


def test_build_chunks_prefixes_section_title_and_carries_company_name(tmp_path):
    report = _write_report(
        tmp_path,
        "acme",
        pages=["Cover", "Some financial detail text"],
        toc=[{"title": "Financial Highlights", "start_page": 1}],
    )

    chunks = build_chunks([report])

    financial_chunks = [c for c in chunks if c.section_title == "Financial Highlights"]
    assert len(financial_chunks) == 1
    chunk = financial_chunks[0]
    assert chunk.company_name == "Acme Corp"
    assert chunk.text.startswith("Section: Financial Highlights\n\n")
    assert "Some financial detail text" in chunk.text
