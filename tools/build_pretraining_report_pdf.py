#!/usr/bin/env python3
"""Render the public Modujo pretraining report as a polished Chinese PDF."""

from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1] / "docs"
SOURCE = ROOT / "pretraining-report.md"
OUTPUT = ROOT / "Modujo-1B-A0.75B-Pretraining-Report.pdf"
FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

NAVY = colors.HexColor("#132238")
BLUE = colors.HexColor("#2367A8")
CYAN = colors.HexColor("#16A3A5")
TEXT = colors.HexColor("#263648")
MUTED = colors.HexColor("#66788A")
LIGHT = colors.HexColor("#EDF3F7")
GRID = colors.HexColor("#CEDAE3")


def inline(text: str) -> str:
    """Convert the small inline-Markdown subset used by the report."""
    tokens: list[str] = []

    def stash(value: str) -> str:
        tokens.append(value)
        return f"\x00{len(tokens) - 1}\x00"

    text = re.sub(
        r"\[([^]]+)\]\((https?://[^)]+)\)",
        lambda m: stash(f'<link href="{html.escape(m.group(2), quote=True)}" color="#2367A8"><u>{html.escape(m.group(1))}</u></link>'),
        text,
    )
    text = re.sub(
        r"\[([^]]+)\]\(([^)]+)\)",
        lambda m: stash(f'<font color="#2367A8">{html.escape(m.group(1))}</font>'),
        text,
    )
    def code_markup(match: re.Match) -> str:
        value = match.group(1)
        font = "CJK" if re.search(r"[\u3400-\u9fff]", value) else "Mono"
        return stash(f'<font name="{font}" color="#1F5D78">{html.escape(value)}</font>')

    text = re.sub(r"`([^`]+)`", code_markup, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    for index, value in enumerate(tokens):
        text = text.replace(f"\x00{index}\x00", value)
    return text


def make_styles():
    pdfmetrics.registerFont(TTFont("CJK", FONT))
    pdfmetrics.registerFont(TTFont("Mono", MONO))
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=base["Title"],
            fontName="CJK",
            fontSize=25,
            leading=34,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            fontName="CJK",
            fontSize=10,
            leading=16,
            textColor=MUTED,
            spaceAfter=7 * mm,
        ),
        "h2": ParagraphStyle(
            "H2CN",
            fontName="CJK",
            fontSize=17,
            leading=24,
            textColor=NAVY,
            spaceBefore=7 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3CN",
            fontName="CJK",
            fontSize=12.5,
            leading=19,
            textColor=BLUE,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            fontName="CJK",
            fontSize=9.6,
            leading=16.5,
            textColor=TEXT,
            alignment=TA_LEFT,
            spaceAfter=3.2 * mm,
            wordWrap="CJK",
        ),
        "note": ParagraphStyle(
            "NoteCN",
            fontName="CJK",
            fontSize=8.4,
            leading=14,
            textColor=MUTED,
            leftIndent=4 * mm,
            rightIndent=3 * mm,
            borderColor=CYAN,
            borderWidth=0,
            borderPadding=4 * mm,
            backColor=LIGHT,
            spaceBefore=3 * mm,
            spaceAfter=3 * mm,
            wordWrap="CJK",
        ),
        "table": ParagraphStyle(
            "TableCN",
            fontName="CJK",
            fontSize=7.5,
            leading=11.5,
            textColor=TEXT,
            wordWrap="CJK",
        ),
        "table_head": ParagraphStyle(
            "TableHeadCN",
            fontName="CJK",
            fontSize=7.7,
            leading=11.5,
            textColor=colors.white,
            wordWrap="CJK",
        ),
        "caption": ParagraphStyle(
            "CaptionCN",
            fontName="CJK",
            fontSize=8,
            leading=12,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceAfter=4 * mm,
        ),
    }


def parse_table(lines: list[str], styles, width: float):
    rows = []
    for raw in lines:
        protected = raw.strip().strip("|").replace(r"\|", "\x00PIPE\x00")
        cells = [cell.strip().replace("\x00PIPE\x00", "|") for cell in protected.split("|")]
        if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        style = styles["table_head"] if not rows else styles["table"]
        rows.append([Paragraph(inline(cell), style) for cell in cells])

    columns = len(rows[0])
    if columns == 2:
        col_widths = [width * 0.29, width * 0.71]
    elif columns == 4:
        col_widths = [width * 0.13, width * 0.20, width * 0.45, width * 0.22]
    else:
        col_widths = [width / columns] * columns

    table = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.35, GRID),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for row in range(1, len(rows)):
        if row % 2 == 0:
            commands.append(("BACKGROUND", (0, row), (-1, row), colors.HexColor("#F6F9FB")))
    table.setStyle(TableStyle(commands))
    return table


def build_story(styles, content_width: float):
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    paragraph: list[str] = []

    def flush_paragraph():
        if paragraph:
            story.append(Paragraph(inline(" ".join(x.strip() for x in paragraph)), styles["body"]))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            i += 1
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            story.append(Spacer(1, 10 * mm))
            story.append(Paragraph(inline(stripped[2:]), styles["title"]))
            story.append(Paragraph("PUBLIC RESEARCH REPORT · MODUJO", styles["subtitle"]))
            story.append(HRFlowable(width="100%", thickness=1.2, color=CYAN, spaceAfter=4 * mm))
        elif stripped.startswith("## "):
            flush_paragraph()
            if stripped == "## 下一步重点":
                story.append(PageBreak())
            story.append(Paragraph(inline(stripped[3:]), styles["h2"]))
        elif stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(stripped[4:]), styles["h3"]))
        elif stripped.startswith("!["):
            flush_paragraph()
            match = re.match(r"!\[([^]]*)\]\(([^)]+)\)", stripped)
            if match:
                image_path = (SOURCE.parent / match.group(2)).resolve()
                if image_path.exists():
                    image = Image(str(image_path))
                    scale = min(content_width / image.imageWidth, 150 * mm / image.imageHeight)
                    image.drawWidth = image.imageWidth * scale
                    image.drawHeight = image.imageHeight * scale
                    story.append(KeepTogether([image, Paragraph(inline(match.group(1)), styles["caption"])]))
        elif stripped.startswith("|"):
            flush_paragraph()
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            story.append(parse_table(table_lines, styles, content_width))
            story.append(Spacer(1, 3 * mm))
            continue
        elif stripped.startswith(">"):
            flush_paragraph()
            note_lines = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                note_lines.append(lines[i].strip()[1:].strip())
                i += 1
            story.append(Paragraph(inline(" ".join(note_lines)), styles["note"]))
            continue
        elif re.match(r"^[-*] ", stripped):
            flush_paragraph()
            story.append(Paragraph("• " + inline(stripped[2:]), styles["body"]))
        else:
            paragraph.append(stripped)
        i += 1
    flush_paragraph()
    return story


def decorate_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(colors.HexColor("#D9E2E9"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, height - 16 * mm, width - doc.rightMargin, height - 16 * mm)
    canvas.setFont("CJK", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(doc.leftMargin, height - 12 * mm, "Modujo-1B-A0.75B · 预训练报告")
    canvas.drawRightString(width - doc.rightMargin, 10 * mm, f"{canvas.getPageNumber()}")
    canvas.setFillColor(CYAN)
    canvas.rect(doc.leftMargin, 9.5 * mm, 18 * mm, 0.8, fill=1, stroke=0)
    canvas.restoreState()


def main():
    styles = make_styles()
    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=21 * mm,
        bottomMargin=17 * mm,
        title="Modujo-1B-A0.75B 预训练报告",
        author="Modujo",
        subject="Modujo-1B-A0.75B pretraining report",
        creator="ModujoMoe",
    )
    story = build_story(styles, A4[0] - doc.leftMargin - doc.rightMargin)
    doc.build(story, onFirstPage=decorate_page, onLaterPages=decorate_page)
    print(OUTPUT)


if __name__ == "__main__":
    main()
