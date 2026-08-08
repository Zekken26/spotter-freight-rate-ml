"""Build the employer-ready PDF assessment report with ReportLab."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#123047")
TEAL = colors.HexColor("#0B6472")
LIGHT = colors.HexColor("#EAF2F3")
PALE = colors.HexColor("#F5F8F9")
SLATE = colors.HexColor("#455A64")
LINE = colors.HexColor("#CBD8DC")


def find_repository_root() -> Path:
    """Locate the project without importing the ML runtime."""
    for start in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for directory in (start, *start.parents):
            if (directory / "train-test.csv").is_file() and (directory / "reports").is_dir():
                return directory
    raise FileNotFoundError("Could not locate the Spotter project root")


def page_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(0.65 * inch, 0.48 * inch, 7.85 * inch, 0.48 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SLATE)
    canvas.drawString(0.65 * inch, 0.28 * inch, "Spotter Freight-Rate ML Assessment")
    canvas.drawRightString(7.85 * inch, 0.28 * inch, f"Page {doc.page}")
    canvas.restoreState()


def styles():
    base = getSampleStyleSheet()
    return {
        "cover": ParagraphStyle(
            "Cover", parent=base["Title"], fontName="Helvetica-Bold", fontSize=30,
            leading=35, textColor=NAVY, alignment=TA_LEFT, spaceAfter=18,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", parent=base["Normal"], fontSize=15, leading=21,
            textColor=TEAL, spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=21,
            leading=25, textColor=NAVY, spaceAfter=14,
        ),
        "h2": ParagraphStyle(
            "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13,
            leading=17, textColor=TEAL, spaceBefore=10, spaceAfter=7,
        ),
        "body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.6,
            leading=14, textColor=colors.HexColor("#263238"), spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "Small", parent=base["BodyText"], fontName="Helvetica", fontSize=8.2,
            leading=11.5, textColor=SLATE, spaceAfter=5,
        ),
        "callout": ParagraphStyle(
            "Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=11,
            leading=15, textColor=NAVY, leftIndent=9, rightIndent=9, spaceBefore=6,
            spaceAfter=8, borderColor=TEAL, borderWidth=1, borderPadding=10,
            backColor=LIGHT,
        ),
        "center": ParagraphStyle(
            "Center", parent=base["BodyText"], fontSize=9, leading=12,
            textColor=SLATE, alignment=TA_CENTER,
        ),
    }


def p(text: str, style) -> Paragraph:
    return Paragraph(text, style)


def section_title(number: str, title: str, style) -> Paragraph:
    return p(f"{number}  {title}", style)


def report_table(data, widths, header=True, font_size=8.1) -> Table:
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold" if header else "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else NAVY),
        ("BACKGROUND", (0, 0), (-1, 0), NAVY if header else PALE),
        ("ROWBACKGROUNDS", (0, 1 if header else 0), (-1, -1), [colors.white, PALE]),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    table.setStyle(TableStyle(commands))
    return table


def build(root: Path) -> Path:
    output = root / "reports" / "spotter_ml_assessment_report.pdf"
    chart = root / "scorer_results" / "candidate_december.png"
    if not chart.is_file():
        raise FileNotFoundError("Run score.py before building the report")
    metadata = json.loads((root / "reports" / "final_model_metadata.json").read_text())
    summary = json.loads((root / "reports" / "final_prediction_summary.json").read_text())
    s = styles()
    doc = SimpleDocTemplate(
        str(output), pagesize=letter, rightMargin=0.65 * inch, leftMargin=0.65 * inch,
        topMargin=0.65 * inch, bottomMargin=0.65 * inch,
        title="Spotter Freight-Rate ML Assessment", author="Candidate",
    )
    story = []

    story += [Spacer(1, 0.55 * inch), p("SPOTTER", s["subtitle"]),
              p("Freight-Rate<br/>Machine Learning Assessment", s["cover"]),
              p("Final model selection, validation evidence, predictions, and reproducibility", s["subtitle"]),
              Spacer(1, 0.25 * inch),
              p("Main submission", s["h2"]),
              p("Full-feature direct-target Ridge (alpha=10) trained on all 48,000 labeled rows.", s["callout"]),
              p("December chart", s["h2"]),
              p("Separately validated December-compatible CatBoost: 100 trees, depth 8, learning rate 0.10, seed 42, CPU.", s["callout"]),
              Spacer(1, 0.25 * inch),
              p("Evidence-led selection | Chronological validation | Immutable source inputs | Scorer-ready outputs", s["center"]),
              PageBreak()]

    story += [section_title("01", "Executive decision", s["h1"]),
              p("The main submission uses Ridge because it led the declared four-fold temporal ranking: mean MAE $152.17, worst-fold MAE $174.05, mean RMSE $633.64, and MAE standard deviation $14.77. Advanced CatBoost candidates improved October and tail slices but were less stable across months.", s["body"]),
              p("The December output uses a separate compatible CatBoost model. It reduced rolling MAE from $203.59 for compatible Ridge to $167.72 while avoiding fabricated market, quote, coordinate, or route inputs.", s["body"]),
              p("Outcome", s["h2"]),
              report_table([
                  ["Artifact", "Contract", "Status"],
                  ["Main predictions", "12,000 template-aligned IDs", "PASS"],
                  ["December predictions", "31 dates; original 7 columns", "PASS"],
                  ["Repository tests", "19 contract and robustness tests", "PASS"],
                  ["Employer scorer", "Both CSV contracts and chart", "PASS"],
              ], [2.15*inch, 3.55*inch, 0.9*inch]),
              p("Final metrics are calculated by Spotter after submission; no hidden validation target was available or used.", s["small"]),
              p("Why two models?", s["h2"]),
              p("Assessment validation supplies market and quote fields, while the fixed December file does not. Separate role-specific models preserve the strongest evidence available for each exact prediction contract.", s["callout"]),
              PageBreak()]

    story += [section_title("02", "Data, quality, and leakage controls", s["h1"]),
              p("The labeled data contains 48,000 loads from January through October 2025. Assessment validation contains 12,000 November-December loads. The chart input contains one fixed Lexington-to-Fort Wayne Dry Van load for every December date.", s["body"]),
              report_table([
                  ["Dataset", "Rows", "Period", "Role"],
                  ["Labeled training", "48,000", "2025-01-01 to 2025-10-31", "Model development and final fit"],
                  ["Assessment validation", "12,000", "2025-11 to 2025-12", "Main predictions; target hidden"],
                  ["December chart inputs", "31", "2025-12-01 to 2025-12-31", "Fixed-lane daily predictions"],
              ], [1.45*inch, 0.75*inch, 1.8*inch, 2.6*inch]),
              p("Quality audit", s["h2"]),
              p("Schemas, row counts, identifier uniqueness, date ranges, missingness, duplicate rows, numeric coercion, infinite values, categorical cardinality, outliers, and train-to-validation shifts were checked. The seven employer files remain byte-identical to the recorded SHA-256 manifest.", s["body"]),
              p("Leakage controls", s["h2"]),
              p("load_id and posted_rate are prohibited model inputs. All preprocessing is fitted inside each temporal training fold. Date and domain features are deterministic and target-free. No validation target is fitted, inspected, or imputed.", s["body"]),
              p("Ridge applies fold-fit numeric medians, missing indicators, scaling, constant categorical imputation, and unknown-safe one-hot encoding. CatBoost handles native categoricals and missing values without dropping rows.", s["body"]),
              PageBreak()]

    story += [section_title("03", "Chronological validation", s["h1"]),
              p("Random splitting would leak future market regimes into training. Four expanding windows recreate the forward-looking prediction setting and allow stability to influence selection.", s["body"]),
              report_table([
                  ["Fold", "Training", "Validation", "Train rows", "Validation rows"],
                  ["July", "Jan-Jun", "July", "28,806", "4,912"],
                  ["August", "Jan-Jul", "August", "33,718", "4,759"],
                  ["September", "Jan-Aug", "September", "38,477", "4,670"],
                  ["October", "Jan-Sep", "October", "43,147", "4,853"],
              ], [1.0*inch, 1.25*inch, 1.25*inch, 1.15*inch, 1.25*inch]),
              p("Ranking criteria", s["h2"]),
              p("Selection prioritized rolling mean MAE, then worst-fold MAE and fold stability. October performance, RMSE, WMAPE, tail behavior, unseen categories, and operational feature availability were secondary but explicit checks.", s["body"]),
              p("Metrics", s["h2"]),
              report_table([
                  ["Metric", "Interpretation"],
                  ["MAE", "Typical absolute dollar error; primary ranking"],
                  ["RMSE", "Emphasizes rare large misses"],
                  ["WMAPE", "Portfolio absolute error relative to portfolio value"],
              ], [1.25*inch, 5.35*inch]),
              Spacer(1, 0.12*inch),
              p("The target is right-skewed. A low MAE alone can obscure severe underprediction of rare expensive loads, so tail and business slices are mandatory.", s["callout"]),
              PageBreak()]

    results = [
        ["Model / policy", "Mean MAE", "SD", "Worst", "Oct MAE", "RMSE", "WMAPE"],
        ["Ridge full", "152.17", "14.77", "174.05", "143.47", "633.64", "6.379%"],
        ["CatBoost market-only", "153.47", "32.21", "201.27", "141.45", "638.03", "6.429%"],
        ["CatBoost full", "158.44", "39.39", "215.15", "134.52", "640.78", "6.639%"],
        ["CatBoost no-signal", "161.16", "36.61", "213.63", "149.13", "638.30", "6.753%"],
        ["CatBoost Dec-compatible", "167.72", "42.96", "228.44", "144.68", "643.34", "7.029%"],
        ["Ridge Dec-compatible", "203.59", "37.13", "251.44", "170.63", "642.63", "8.549%"],
    ]
    story += [section_title("04", "Model comparison and selection", s["h1"]),
              report_table(results, [1.65*inch, .78*inch, .52*inch, .67*inch, .7*inch, .67*inch, .68*inch], font_size=7.1),
              p("Main-model decision", s["h2"]),
              p("Market-only CatBoost nearly tied mean MAE and slightly improved October, but its worst fold was $27.22 worse and its variability was more than twice Ridge's. Full CatBoost delivered the best October MAE but was weaker across the complete time series. Ridge therefore remains the defensible main model.", s["body"]),
              p("December decision", s["h2"]),
              p("Compatible CatBoost improved mean MAE by $35.87 (17.6%) and October MAE by $25.95 (15.2%) over compatible Ridge. It uses only fields available in the fixed December contract.", s["callout"]),
              p("Signal policy", s["h2"]),
              p("market_index is useful within labeled temporal folds but its provenance and future availability are unconfirmed. It remains in the main model because assessment validation supplies it. It is excluded from December because fabricating future values would introduce an unsupported dependency.", s["body"]),
              PageBreak()]

    story += [section_title("05", "Error analysis and robustness", s["h1"]),
              p("October error slices show the key remaining business risks.", s["body"]),
              report_table([
                  ["Slice", "Rows", "Ridge MAE", "Interpretation"],
                  ["All October", "4,853", "$143.47", "Strong typical-load accuracy"],
                  ["Top 10% rate", "505", "$595.98", "Expensive loads remain compressed"],
                  ["Distance >2,000", "769", "$284.09", "Higher long-haul dollar risk"],
                  ["Reefer", "1,190", "$173.13", "Higher error than other equipment"],
                  ["Dry Van", "2,745", "$133.24", "Best-supported equipment segment"],
              ], [1.45*inch, .7*inch, .9*inch, 3.55*inch]),
              p("Advanced-model evidence", s["h2"]),
              p("CatBoost improved expensive, long-distance, Reefer, and synthetic unseen-city slices, but it did not erase severe tail risk. Its top-10% RMSE remained about $1,900 in October.", s["body"]),
              p("Unseen-city stress", s["h2"]),
              p("A deterministic feature-only stress test removed five frequent endpoint cities entirely from training. CatBoost market-only achieved MAE $121.85, compared with $138.52 for Ridge. All pipelines accepted unknown categories without failure. This is robustness evidence, not an official temporal selection fold.", s["body"]),
              p("The final set contains eight new cities and 1,461 unseen routes - more exposure than the labeled time folds - so hidden-period error may differ from the backtest.", s["callout"]),
              PageBreak()]

    overall = summary["overall"]
    dec = summary["december"]
    story += [section_title("06", "Final fit and prediction sanity", s["h1"]),
              p(f"Both models were trained on all {metadata['training']['rows']:,} labeled rows after selection was locked. Model files, parameters, package versions, output hashes, and positive-floor diagnostics are recorded in final_model_metadata.json.", s["body"]),
              report_table([
                  ["Prediction set", "Rows", "Minimum", "Mean", "Median", "Maximum"],
                  ["Assessment validation", f"{overall['count']:,}", f"${overall['minimum']:,.2f}", f"${overall['mean']:,.2f}", f"${overall['median']:,.2f}", f"${overall['maximum']:,.2f}"],
                  ["Training target", "48,000", "$57.22", "$2,373.98", "$2,030.76", "$25,533.00"],
                  ["December fixed lane", f"{dec['count']}", f"${dec['minimum']:,.2f}", f"${dec['mean']:,.2f}", f"${dec['median']:,.2f}", f"${dec['maximum']:,.2f}"],
              ], [1.65*inch, .65*inch, .9*inch, .9*inch, .9*inch, 1.0*inch]),
              p("Sanity findings", s["h2"]),
              p("Validation predictions rise coherently across distance bands and are highest on average for Reefer. November and December assessment means are close ($2,418.76 and $2,425.06). No raw final predictions were nonpositive, so the $0.000001 safety floor changed zero values.", s["body"]),
              p("December behavior", s["h2"]),
              p("The first five fixed-lane values are $825.57, $841.42, $860.76, $861.71, and $864.73. The last three rise to $972.20, $980.19, and $987.35. That year-end step is a calendar extrapolation from the trained model, not causal proof of a market event.", s["callout"]),
              PageBreak()]

    story += [section_title("07", "December fixed-lane chart", s["h1"]),
              p("Lexington to Fort Wayne | 360 miles | Dry Van | 32,000 lb | only date changes", s["subtitle"]),
              Spacer(1, 0.15*inch),
              Image(str(chart), width=7.1*inch, height=3.57*inch),
              Spacer(1, 0.18*inch),
              p("The chart was generated by score.py after it validated the 12,000-row submission and the 31-row December contract. Rates mostly vary in a weekly pattern before the learned year-end step.", s["body"]),
              p("Interpretation guardrail", s["h2"]),
              p("The December model is intentionally signal-free and compatible with the supplied fields. The curve is a model output under fixed load attributes; it is not a confidence interval, external market forecast, or guarantee of transactable rates.", s["callout"]),
              PageBreak()]

    story += [section_title("08", "Reproducibility, limitations, and conclusion", s["h1"]),
              p("Reproducibility", s["h2"]),
              p("The finalization script locates the repository, validates schemas, fits both locked models, aligns IDs one-to-one, writes new outputs, serializes models, and records SHA-256 hashes. Tests rerun inference from the saved artifacts and compare results at tight numerical tolerance.", s["body"]),
              report_table([
                  ["Control", "Result"],
                  ["Employer inputs", "Unchanged; SHA-256 manifest matches"],
                  ["Feature exclusions", "load_id and posted_rate absent"],
                  ["Main alignment", "One-to-one merge in template order"],
                  ["Saved inference", "Deterministic rerun passes"],
                  ["Scorer", "Both files validated; chart created"],
              ], [2.0*inch, 4.6*inch]),
              p("Limitations", s["h2"]),
              p("The official hidden metric is unknown. Rare expensive loads remain underpredicted. The final city/route mix is not fully represented by labeled folds. Main-model market signals have operational uncertainty. CatBoost tuning was purposefully CPU-bounded, although a focused larger refinement was worse.", s["body"]),
              p("Conclusion", s["h2"]),
              p("The locked solution follows the strongest available evidence: stable full-feature Ridge for assessment validation and separately validated compatible CatBoost for the December chart. Both artifacts are positive, finite, schema-correct, scorer-valid, and reproducible.", s["callout"])]

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)
    return output


def main() -> None:
    output = build(find_repository_root())
    print(f"Created report: {output}")


if __name__ == "__main__":
    main()
