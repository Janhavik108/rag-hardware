"""
Generates a 3-page fake datasheet that deliberately reproduces the failure
modes seen on real datasheets:
  - a repeated running header ("ACME9000(L)") on every page
  - a decorative bordered box (mimics a pinout/block diagram) that
    pdfplumber's table finder tends to misdetect as a table
  - a run of bold numbered list items (mimics revision-history entries)
    that could snowball heading-merging if not capped

Usage:
    python scripts/make_test_pdf.py
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                 TableStyle, PageBreak)

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13)
body = styles["BodyText"]
header_style = ParagraphStyle("Header", parent=styles["Normal"], fontSize=9,
                               textColor=colors.black, fontName="Helvetica-Bold")
bold_item = ParagraphStyle("BoldItem", parent=styles["BodyText"], fontName="Helvetica-Bold")

doc = SimpleDocTemplate("data/raw/test_datasheet.pdf", pagesize=letter)
story = []

RUNNING_HEADER = "ACME9000(L)"


def add_header(page_num):
    story.append(Paragraph(RUNNING_HEADER, header_style))
    story.append(Paragraph(str(page_num), header_style))  # bold page number, same as real datasheets
    story.append(Spacer(1, 10))


# --- Page 1: normal section + real register table ---
add_header(1)
story.append(Paragraph(
    "8. USART - Universal Synchronous Asynchronous Receiver Transmitter", h1))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "The USART is a highly flexible serial communication device. The baud rate "
    "is set by the UBRR register according to the formula BAUD = FOSC / (16 * "
    "(UBRR + 1)) for asynchronous normal mode.", body))
story.append(Spacer(1, 14))

story.append(Paragraph("8.1 UBRRH and UBRRL - USART Baud Rate Registers", h2))
story.append(Spacer(1, 8))
story.append(Paragraph(
    "These two registers combine to form a 12-bit value, UBRR. UBRRH contains "
    "the four most significant bits, and UBRRL contains the eight least "
    "significant bits.", body))
story.append(Spacer(1, 14))

table_data = [
    ["Bit", "7", "6", "5", "4", "3", "2", "1", "0"],
    ["UBRRH", "-", "-", "-", "-", "UBRR11", "UBRR10", "UBRR9", "UBRR8"],
    ["UBRRL", "UBRR7", "UBRR6", "UBRR5", "UBRR4", "UBRR3", "UBRR2", "UBRR1", "UBRR0"],
]
t = Table(table_data)
t.setStyle(TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ("FONTSIZE", (0, 0), (-1, -1), 8),
]))
story.append(t)
story.append(PageBreak())

# --- Page 2: decorative bordered box (fake pinout diagram) ---
add_header(2)
story.append(Paragraph("Pin Configuration", h1))
story.append(Spacer(1, 8))
decorative = Table([["", ""], ["", ""]], colWidths=[150, 150], rowHeights=[80, 80])
decorative.setStyle(TableStyle([
    ("BOX", (0, 0), (-1, -1), 1.5, colors.black),
    ("INNERGRID", (0, 0), (-1, -1), 1.0, colors.black),
]))
story.append(decorative)
story.append(PageBreak())

# --- Page 3: bold numbered list (fake revision history) ---
add_header(3)
story.append(Paragraph("Revision History", h1))
story.append(Spacer(1, 8))
for i, change in enumerate([
    "Updated Ordering Information on page 12.",
    "Removed note from Feature list in Analog to Digital Converter section.",
    "Corrected typo in Table 42.",
    "Added Errata section on page 336.",
], start=1):
    story.append(Paragraph(f"{i}. {change}", bold_item))
    story.append(Spacer(1, 4))

doc.build(story)
print("wrote data/raw/test_datasheet.pdf")
