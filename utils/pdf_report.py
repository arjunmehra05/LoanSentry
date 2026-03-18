import io
import re
import numpy as np
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

FEATURE_LABELS = {
    "annual_inc":            "Annual Income",
    "loan_amnt":             "Loan Amount",
    "int_rate":              "Interest Rate",
    "dti":                   "Debt-to-Income Ratio",
    "FICO_AVG":              "FICO Score",
    "emp_length":            "Employment Length",
    "revol_util":            "Revolving Utilization",
    "LOAN_TO_INCOME":        "Loan-to-Income Ratio",
    "delinq_2yrs":           "Delinquencies (2yr)",
    "pub_rec":               "Public Records",
    "inq_last_6mths":        "Credit Inquiries (6mo)",
}

PURPOSE_LABELS = {
    0: "Car", 1: "Credit Card", 2: "Debt Consolidation",
    3: "Home Improvement", 4: "Major Purchase", 5: "Medical",
    6: "Moving", 7: "Small Business", 8: "Vacation", 9: "Other"
}

GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}


def generate_pdf_report(input_dict, prediction_results, explanation,
                        shap_values, feature_names):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=15*mm
    )

    styles = getSampleStyleSheet()

    # ── Custom styles ─────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "Title",
        parent=styles["Normal"],
        fontSize=20,
        textColor=colors.white,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.white,
        alignment=TA_CENTER,
        fontName="Helvetica",
    )
    section_style = ParagraphStyle(
        "Section",
        parent=styles["Normal"],
        fontSize=13,
        textColor=colors.HexColor("#1e3c72"),
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        fontName="Helvetica",
        leading=14,
        spaceAfter=4,
    )
    bold_style = ParagraphStyle(
        "Bold",
        parent=body_style,
        fontName="Helvetica-Bold",
    )

    story = []

    # ── Header banner ─────────────────────────────────────────────────────
    header_data = [[
        Paragraph("LoanSentry - Assessment Report", title_style),
    ], [
        Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
            f"Confidential - For Internal Use Only",
            subtitle_style),
    ]]
    header_table = Table(header_data, colWidths=[170*mm])
    header_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#1e3c72")),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("TOPPADDING",    (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8*mm))

    # ── Risk summary card ─────────────────────────────────────────────────
    rc   = prediction_results["risk_category"]
    prob = prediction_results["prob_ensemble"]
    conf = prediction_results["confidence"]
    pred = prediction_results["prediction"]

    risk_colors_map = {
        "Low":    colors.HexColor("#00C851"),
        "Medium": colors.HexColor("#FF8800"),
        "High":   colors.HexColor("#C6613F"),
    }
    rc_color = risk_colors_map.get(rc, colors.gray)
    if rc == "Low":
        rec_text = "Approve"
    elif rc == "Medium":
        rec_text = "Review"
    else:
        rec_text = "Decline"

    story.append(Paragraph("Risk Assessment Summary", section_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 3*mm))

    risk_data = [
        ["Risk Category", "Default Probability", "Model Confidence", "Recommendation"],
        [rc, f"{prob:.1%}", f"{conf:.1f}%", rec_text],
    ]
    risk_table = Table(risk_data, colWidths=[42*mm]*4)
    risk_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#f0f2f6")),
        ("BACKGROUND",    (0, 1), (0, 1),   rc_color),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.HexColor("#333333")),
        ("TEXTCOLOR",     (0, 1), (0, 1),   colors.white),
        ("TEXTCOLOR",     (1, 1), (-1, 1),  colors.HexColor("#333333")),
        ("FONTNAME",      (0, 0), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
    ]))
    story.append(risk_table)
    story.append(Spacer(1, 6*mm))

    # ── Applicant profile ─────────────────────────────────────────────────
    story.append(Paragraph("Applicant Profile", section_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 3*mm))

    purpose = PURPOSE_LABELS.get(input_dict.get("purpose", 9), "Other")
    grade   = GRADE_LABELS.get(input_dict.get("grade", 6), "G")

    profile_data = [
        ["Field", "Value", "Field", "Value"],
        ["Annual Income",    f"${input_dict.get('annual_inc', 0):,.0f}",
         "FICO Score",       f"{input_dict.get('FICO_AVG', 0):.0f}"],
        ["Loan Amount",      f"${input_dict.get('loan_amnt', 0):,.0f}",
         "Employment Years", f"{input_dict.get('emp_length', 0):.0f}"],
        ["Interest Rate",    f"{input_dict.get('int_rate', 0):.1f}%",
         "Loan Grade",       grade],
        ["DTI Ratio",        f"{input_dict.get('dti', 0):.1f}%",
         "Loan Purpose",     purpose],
        ["Revolving Util.",  f"{input_dict.get('revol_util', 0):.1f}%",
         "Loan Term",        f"{input_dict.get('term', 36)} months"],
    ]
    profile_table = Table(profile_data, colWidths=[45*mm, 40*mm, 45*mm, 40*mm])
    profile_table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#f0f2f6")),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTNAME",    (0, 1), (0, -1),  "Helvetica-Bold"),
        ("FONTNAME",    (2, 1), (2, -1),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 6*mm))

    # ── Top SHAP features ─────────────────────────────────────────────────
    story.append(Paragraph("Top Risk Factors (SHAP)", section_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 3*mm))

    shap_vals   = shap_values[0] if len(np.array(shap_values).shape) > 1 \
                  else shap_values
    top_indices = np.abs(shap_vals).argsort()[-8:][::-1]

    shap_data = [["#", "Feature", "Impact", "Direction"]]
    for rank, i in enumerate(top_indices, 1):
        feat  = FEATURE_LABELS.get(feature_names[i], feature_names[i])
        val   = float(shap_vals[i])
        direc = "Increases Risk" if val > 0 else "Reduces Risk"
        shap_data.append([str(rank), feat, f"{val:+.4f}", direc])

    shap_table = Table(shap_data, colWidths=[12*mm, 80*mm, 35*mm, 43*mm])
    shap_style = TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#f0f2f6")),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
        ("ALIGN",       (0, 0), (0, -1),  "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING",   (0, 0), (-1, -1), 8),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#fafafa")]),
    ])
    # Color direction column
    for row_idx, i in enumerate(top_indices, 1):
        val = float(shap_vals[i])
        cell_color = colors.HexColor("#ffe8e8") if val > 0 \
                     else colors.HexColor("#e8f5e9")
        shap_style.add("BACKGROUND", (3, row_idx), (3, row_idx), cell_color)

    shap_table.setStyle(shap_style)
    story.append(shap_table)
    story.append(Spacer(1, 6*mm))

    # ── AI explanation ────────────────────────────────────────────────────
    story.append(Paragraph("AI-Generated Risk Explanation", section_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=colors.HexColor("#e0e0e0")))
    story.append(Spacer(1, 3*mm))

    # Strip markdown formatting for PDF
    clean = _clean_markdown(explanation)
    for line in clean:
        if line["type"] == "heading":
            story.append(Paragraph(line["text"], bold_style))
            story.append(Spacer(1, 1*mm))
        elif line["type"] == "bullet":
            story.append(Paragraph(f"• {line['text']}", body_style))
        elif line["type"] == "text" and line["text"].strip():
            story.append(Paragraph(line["text"], body_style))
    story.append(Spacer(1, 6*mm))

    # ── Footer ────────────────────────────────────────────────────────────
    footer_style = ParagraphStyle(
        "Footer",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.HexColor("#999999"),
        alignment=TA_CENTER,
        fontName="Helvetica-Oblique",
    )
    story.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        "This report is generated by LoanSentry AI and should be reviewed "
        "by a qualified credit analyst before making lending decisions.",
        footer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer


def _clean_markdown(text):
    """Convert markdown text to structured lines for ReportLab."""
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove emoji characters
        line = re.sub(r'[^\x00-\x7F]+', '', line).strip()
        if not line:
            continue

        if line.startswith("### "):
            lines.append({"type": "heading", "text": line[4:]})
        elif line.startswith("## "):
            lines.append({"type": "heading", "text": line[3:]})
        elif line.startswith("- ") or line.startswith("* "):
            # Remove bold markers
            txt = re.sub(r'\*\*(.+?)\*\*', r'\1', line[2:])
            lines.append({"type": "bullet", "text": txt})
        elif line.startswith("---"):
            continue
        else:
            txt = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
            txt = re.sub(r'\*(.+?)\*', r'\1', txt)
            lines.append({"type": "text", "text": txt})
    return lines