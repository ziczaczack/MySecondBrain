"""PDF and .docx files are walked and their text extracted for indexing.

These exercise the extraction/dispatch layer directly (no embedding model) so
they stay fast and deterministic.
"""

from pathlib import Path

from kb.source import FileSource, _load_file


def _make_pdf(path: Path, text: str) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=18)
    pdf.cell(0, 10, text)
    path.write_bytes(bytes(pdf.output()))


def _make_docx(path: Path, paragraphs: list[str]) -> None:
    from docx import Document as Docx

    doc = Docx()
    for p in paragraphs:
        doc.add_paragraph(p)
    doc.save(str(path))


def test_load_file_extracts_pdf_text(tmp_path):
    p = tmp_path / "paper.pdf"
    _make_pdf(p, "Marmoset primates glimmer xyzzy")
    text = _load_file(p)
    assert text is not None, "PDF should be extracted, not skipped as binary"
    assert "marmoset" in text.lower() and "xyzzy" in text.lower(), (
        f"extracted PDF text missing expected words: {text!r}"
    )


def test_load_file_extracts_docx_text(tmp_path):
    p = tmp_path / "memo.docx"
    _make_docx(p, ["Quokka logistics", "Wallaby spreadsheet plumbus"])
    text = _load_file(p)
    assert text is not None, "DOCX should be extracted, not skipped as binary"
    assert "quokka" in text.lower() and "plumbus" in text.lower(), (
        f"extracted DOCX text missing expected words: {text!r}"
    )


def test_filesource_walks_pdf_and_docx(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    _make_pdf(docs / "a.pdf", "alpha pdf body")
    _make_docx(docs / "b.docx", ["beta docx body"])
    (docs / "c.md").write_text("gamma markdown body\n", encoding="utf-8")

    keys = FileSource(docs).candidate_keys()
    names = {Path(k).name for k in keys}
    assert {"a.pdf", "b.docx", "c.md"} <= names, (
        f"FileSource must include PDF and DOCX, got {sorted(names)}"
    )
