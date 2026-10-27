"""Generates a professional, self-contained HTML Plagiarism Verification Report."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .detector import DetectionResult


def generate_html_report(
    result: DetectionResult,
    doc_a_name: str = "Document A (Reference)",
    doc_b_name: str = "Document B (Suspect)",
    backend_name: str = "SBERT",
) -> str:
    """Generate an interactive, styled HTML report for download."""
    date_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M:%S")

    # Colors for verdict badge
    badge_colors = {
        "Severe Plagiarism": "#dc2626",
        "High Plagiarism": "#ea580c",
        "Moderate Similarity": "#ca8a04",
        "Low Similarity": "#16a34a",
        "Original": "#2563eb",
    }
    badge_color = badge_colors.get(result.verdict, "#475569")

    # Flagged sentence table rows
    rows_html = ""
    for idx, match in enumerate(result.sentence_matches, 1):
        type_badge = (
            f"<span style='background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;'>{match.match_type}</span>"
            if match.match_type == "Verbatim Match"
            else f"<span style='background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600;'>{match.match_type}</span>"
        )
        conf_val = getattr(match, "confidence", "High")
        if conf_val == "High":
            conf_badge = "<span style='background:#dcfce7;color:#15803d;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;'>High</span>"
        elif conf_val == "Moderate":
            conf_badge = "<span style='background:#fef3c7;color:#b45309;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;'>Moderate</span>"
        else:
            conf_badge = "<span style='background:#f1f5f9;color:#64748b;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:700;'>Tentative</span>"

        source_val = getattr(match, "source_label", "") or f"Ref Sentence #{match.index_a + 1}"
        source_badge = f"<span style='background:#e0f2fe;color:#0369a1;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;'>{source_val}</span>"

        synonyms_str = ""
        if match.synonyms:
            syn_pairs = [f"<code>{wa} &harr; {wb}</code>" for wa, wb in match.synonyms]
            synonyms_str = "<br><small style='color:#64748b;'>Synonyms swapped: " + ", ".join(syn_pairs) + "</small>"

        rows_html += f"""
        <tr style="border-bottom:1px solid #e2e8f0;">
            <td style="padding:12px;text-align:center;font-weight:600;">{idx}</td>
            <td style="padding:12px;text-align:center;">{source_badge}</td>
            <td style="padding:12px;">{match.sentence_a}</td>
            <td style="padding:12px;">{match.sentence_b}{synonyms_str}</td>
            <td style="padding:12px;text-align:center;font-weight:700;color:{badge_color};">{match.score * 100:.1f}%</td>
            <td style="padding:12px;text-align:center;">{type_badge}</td>
            <td style="padding:12px;text-align:center;">{conf_badge}</td>
        </tr>
        """

    if not rows_html:
        rows_html = "<tr><td colspan='7' style='padding:20px;text-align:center;color:#64748b;'>No sentence pairs exceeded the match threshold.</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Plagiarism Verification Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: #ffffff;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            padding: 36px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .title {{
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
        }}
        .badge {{
            display: inline-block;
            padding: 8px 18px;
            border-radius: 9999px;
            color: #ffffff;
            font-weight: 700;
            font-size: 16px;
            background-color: {badge_color};
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 28px;
        }}
        .metric-card {{
            background: #f1f5f9;
            padding: 16px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: 800;
            color: #0f172a;
        }}
        .metric-label {{
            font-size: 13px;
            color: #64748b;
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 16px;
            font-size: 14px;
        }}
        th {{
            background: #f8fafc;
            padding: 12px;
            text-align: left;
            font-weight: 700;
            color: #475569;
            border-bottom: 2px solid #cbd5e1;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 12px;
            color: #94a3b8;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <div class="title">Plagiarism Inspection Report</div>
                <small style="color:#64748b;">Generated on {date_str} | Backend: {backend_name}</small>
            </div>
            <div class="badge">{result.verdict} ({result.combined_score * 100:.1f}%)</div>
        </div>

        <div style="margin-bottom: 24px;">
            <p><strong>Reference Document:</strong> {doc_a_name} ({len(result.sentences_a)} sentences)</p>
            <p><strong>Suspect Document:</strong> {doc_b_name} ({len(result.sentences_b)} sentences)</p>
        </div>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{result.combined_score * 100:.1f}%</div>
                <div class="metric-label">Combined Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.tfidf_score * 100:.1f}%</div>
                <div class="metric-label">Lexical (TF-IDF)</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.semantic_score * 100:.1f}%</div>
                <div class="metric-label">Semantic Similarity</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.bigram_containment * 100:.1f}%</div>
                <div class="metric-label">Bigram Containment</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{result.lcs_ratio * 100:.1f}%</div>
                <div class="metric-label">LCS Match Ratio</div>
            </div>
        </div>

        <h3 style="margin-bottom:8px;color:#0f172a;">Flagged Sentence Alignments ({len(result.sentence_matches)})</h3>
        <table>
            <thead>
                <tr>
                    <th style="width:40px;text-align:center;">#</th>
                    <th style="width:14%;text-align:center;">Source</th>
                    <th style="width:33%;">Document A (Reference)</th>
                    <th style="width:33%;">Document B (Suspect)</th>
                    <th style="width:8%;text-align:center;">Similarity</th>
                    <th style="width:12%;text-align:center;">Type</th>
                    <th style="width:10%;text-align:center;">Confidence</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>

        <div class="footer">
            NLP Plagiarism Detection Engine &bull; Academic Micro-Project &bull; Automated Analysis
        </div>
    </div>
</body>
</html>
"""
    return html
