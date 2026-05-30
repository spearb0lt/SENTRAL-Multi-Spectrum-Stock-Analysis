"""Report generation: HTML summary + PDF via reportlab."""
import io
import base64
from datetime import datetime


def _fmt(val, pct=False, cur="₹", decimals=2):
    if val is None: return "N/A"
    try:
        v = float(val)
        if pct:   return f"{v:.{decimals}f}%"
        if cur:   return f"{cur}{v:,.{decimals}f}"
        return f"{v:.{decimals}f}"
    except Exception:
        return str(val)


def generate_html_report(results: dict) -> str:
    """Generate full HTML report from analysis results dict."""
    info         = results.get("info", {})
    ticker       = results.get("ticker", "N/A")
    company      = results.get("company_name", ticker)
    cur          = results.get("currency_sym", "₹")
    signal_data  = results.get("signal", {})
    dcf          = results.get("dcf", {})
    altman       = results.get("altman", {})
    piotroski    = results.get("piotroski", {})
    risk         = results.get("risk", {})
    sentiment_df = results.get("sentiment_df")
    thesis       = results.get("thesis", "")
    news_items   = results.get("news_items", [])
    bt           = results.get("backtest", {})

    signal      = signal_data.get("signal", "HOLD 🟡")
    sig_score   = signal_data.get("final_score", 0)
    pillars     = signal_data.get("pillar_scores", {})
    sig_color   = {"BUY 🟢": "#2e7d32", "SELL 🔴": "#c62828"}.get(signal, "#f57f17")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build news rows
    news_rows = ""
    for n in news_items[:20]:
        news_rows += f"""<tr>
          <td><a href="{n.get('url','#')}" target="_blank">{n.get('title','')[:120]}</a></td>
          <td>{n.get('source','')}</td>
          <td>{str(n.get('published',''))[:10]}</td>
        </tr>"""

    # Pillar table
    pillar_rows = ""
    for k, v in pillars.items():
        clr  = "#4caf50" if v >= 0 else "#f44336"
        bar  = "█" * int(abs(v) * 10)
        pillar_rows += f"<tr><td>{k}</td><td style='color:{clr};font-weight:bold'>{v:+.3f}</td><td style='color:{clr}'>{bar}</td></tr>"

    # Sentiment summary
    sent_rows = ""
    if sentiment_df is not None and not sentiment_df.empty:
        for idx, row in sentiment_df.iterrows():
            clr = "#4caf50" if row["avg_score"] >= 0 else "#f44336"
            sent_rows += f"""<tr>
              <td>{idx}</td>
              <td style='color:{clr}'>{row['avg_score']:+.3f}</td>
              <td>{row.get('positive%',0):.0f}%</td>
              <td>{row.get('neutral%',0):.0f}%</td>
              <td>{row.get('negative%',0):.0f}%</td>
              <td>{int(row.get('n_articles',0))}</td>
            </tr>"""

    # Backtest top 5
    bt_rows = ""
    if "df_results" in bt:
        for name, row in bt["df_results"].head(6).iterrows():
            clr = "#4caf50" if row["Total Return"] >= 0 else "#f44336"
            bt_rows += f"""<tr>
              <td>{"★ " if name==bt.get('best_name') else ""}{name}</td>
              <td style='color:{clr}'>{row['Total Return']:+.1%}</td>
              <td>{row['Sharpe']:.3f}</td>
              <td style='color:#f44336'>{row['Max Drawdown']:.1%}</td>
              <td>{row['Win Rate']:.1%}</td>
            </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>SENTRAL Report — {ticker}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #e6edf3; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 28px; color: #58a6ff; border-bottom: 2px solid #21262d; padding-bottom: 12px; }}
  h2 {{ font-size: 18px; color: #79c0ff; margin: 28px 0 12px; }}
  .signal-badge {{ display: inline-block; padding: 12px 32px; border-radius: 8px;
                   background: {sig_color}; color: white; font-size: 26px; font-weight: bold;
                   letter-spacing: 2px; margin: 16px 0; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 12px 0; }}
  .metric-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                  padding: 14px; text-align: center; }}
  .metric-card .val {{ font-size: 20px; font-weight: bold; color: #f0f6fc; }}
  .metric-card .lbl {{ font-size: 11px; color: #8b949e; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }}
  th {{ background: #21262d; color: #79c0ff; padding: 8px 12px; text-align: left; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #21262d; }}
  tr:hover td {{ background: #1c2128; }}
  .thesis {{ background: #161b22; border-left: 4px solid #58a6ff; padding: 16px;
             border-radius: 4px; white-space: pre-wrap; line-height: 1.7; }}
  .footer {{ text-align: center; color: #8b949e; font-size: 11px; margin-top: 40px; padding-top: 16px;
             border-top: 1px solid #21262d; }}
  a {{ color: #58a6ff; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body><div class="container">
<h1>📊 SENTRAL Analysis Report — {company} ({ticker})</h1>
<p style="color:#8b949e;margin:4px 0 16px">Generated: {now}</p>

<div class="signal-badge">{signal}</div>
<p>Composite Score: <b>{sig_score:+.4f}</b> &nbsp;|&nbsp; Confidence: <b>{signal_data.get('confidence_pct',0):.1f}%</b></p>

<h2>📐 Pillar Breakdown</h2>
<table>
  <tr><th>Pillar</th><th>Score</th><th>Bar</th></tr>
  {pillar_rows}
</table>

<h2>💰 Key Metrics</h2>
<div class="metric-grid">
  <div class="metric-card"><div class="val">{_fmt(info.get('currentPrice') or info.get('regularMarketPrice'), cur=cur)}</div><div class="lbl">Current Price</div></div>
  <div class="metric-card"><div class="val">{_fmt(info.get('trailingPE'), cur='', decimals=1)}x</div><div class="lbl">P/E Ratio</div></div>
  <div class="metric-card"><div class="val">{_fmt(info.get('returnOnEquity',0)*100 if info.get('returnOnEquity') else None, pct=True)}</div><div class="lbl">ROE</div></div>
  <div class="metric-card"><div class="val">{_fmt(risk.get('sharpe'), cur='')}</div><div class="lbl">Sharpe Ratio</div></div>
  <div class="metric-card"><div class="val">{_fmt(risk.get('max_drawdown'), pct=True)}</div><div class="lbl">Max Drawdown</div></div>
  <div class="metric-card"><div class="val">{altman.get('z_score','N/A')} ({altman.get('zone','N/A')})</div><div class="lbl">Altman Z-Score</div></div>
  <div class="metric-card"><div class="val">{piotroski.get('score','N/A')}/9 ({piotroski.get('signal','N/A')})</div><div class="lbl">Piotroski F-Score</div></div>
  <div class="metric-card"><div class="val">{_fmt(dcf.get('intrinsic'), cur=cur)}</div><div class="lbl">DCF Intrinsic Value</div></div>
</div>

<h2>🧠 AI Sentiment Analysis</h2>
<table>
  <tr><th>Model</th><th>Avg Score</th><th>Positive%</th><th>Neutral%</th><th>Negative%</th><th>Articles</th></tr>
  {sent_rows}
</table>

<h2>📝 Investment Thesis</h2>
<div class="thesis">{thesis}</div>

<h2>📈 Strategy Backtest (Top by Sharpe)</h2>
<table>
  <tr><th>Strategy</th><th>CAGR</th><th>Sharpe</th><th>Max DD</th><th>Win Rate</th></tr>
  {bt_rows}
</table>

<h2>📰 Recent News</h2>
<table>
  <tr><th>Headline</th><th>Source</th><th>Date</th></tr>
  {news_rows}
</table>

<div class="footer">
  SENTRAL Multi-Spectrum Stock Analysis &nbsp;|&nbsp; Generated {now}<br>
  ⚠ This report is for informational purposes only and does not constitute financial advice.
</div>
</div></body></html>"""
    return html


def generate_pdf_report(results: dict) -> bytes:
    """Generate a summary PDF using reportlab. Returns bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        buf      = io.BytesIO()
        doc      = SimpleDocTemplate(buf, pagesize=A4,
                                      leftMargin=20*mm, rightMargin=20*mm,
                                      topMargin=20*mm, bottomMargin=20*mm)
        styles   = getSampleStyleSheet()
        BG       = colors.HexColor("#0d1117")
        BLUE     = colors.HexColor("#58a6ff")
        GREEN    = colors.HexColor("#4caf50")
        RED      = colors.HexColor("#f44336")
        GOLD     = colors.HexColor("#f9a825")
        LTGRAY   = colors.HexColor("#e6edf3")
        DKGRAY   = colors.HexColor("#8b949e")
        ROWBG    = colors.HexColor("#161b22")

        title_style = ParagraphStyle("Title", parent=styles["Title"],
                                      textColor=BLUE, fontSize=18, spaceAfter=6)
        h2_style    = ParagraphStyle("H2", parent=styles["Heading2"],
                                      textColor=BLUE, fontSize=13, spaceBefore=14, spaceAfter=4)
        body_style  = ParagraphStyle("Body", parent=styles["Normal"],
                                      textColor=LTGRAY, fontSize=9, leading=14)
        small_style = ParagraphStyle("Small", parent=styles["Normal"],
                                      textColor=DKGRAY, fontSize=8, leading=12)

        ticker  = results.get("ticker", "N/A")
        company = results.get("company_name", ticker)
        signal  = results.get("signal", {}).get("signal", "HOLD 🟡")
        score   = results.get("signal", {}).get("final_score", 0)
        now     = datetime.now().strftime("%Y-%m-%d %H:%M")

        story = []
        story.append(Paragraph(f"SENTRAL Analysis — {company} ({ticker})", title_style))
        story.append(Paragraph(f"Generated: {now}", small_style))
        story.append(Spacer(1, 8*mm))

        # Signal
        sig_clr = GREEN if "BUY" in signal else (RED if "SELL" in signal else GOLD)
        sig_tbl = Table([[signal]], colWidths=[60*mm])
        sig_tbl.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,-1), sig_clr),
            ("TEXTCOLOR",  (0,0), (-1,-1), colors.white),
            ("FONTNAME",   (0,0), (-1,-1), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 16),
            ("ALIGN",      (0,0), (-1,-1), "CENTER"),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0,0), (-1,-1), [sig_clr]),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
        ]))
        story.append(sig_tbl)
        story.append(Paragraph(f"Composite Score: {score:+.4f}", body_style))
        story.append(Spacer(1, 6*mm))

        # Pillars
        story.append(Paragraph("Pillar Breakdown", h2_style))
        pillars = results.get("signal", {}).get("pillar_scores", {})
        p_data  = [["Pillar", "Score"]]
        for k, v in pillars.items():
            p_data.append([k, f"{v:+.3f}"])
        p_tbl = Table(p_data, colWidths=[80*mm, 40*mm])
        p_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#21262d")),
            ("TEXTCOLOR",    (0,0), (-1,0), BLUE),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("TEXTCOLOR",    (0,1), (-1,-1), LTGRAY),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [ROWBG, colors.HexColor("#1c2128")]),
            ("GRID",         (0,0), (-1,-1), 0.25, colors.HexColor("#30363d")),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(p_tbl)
        story.append(Spacer(1, 5*mm))

        # Key metrics
        story.append(Paragraph("Key Metrics", h2_style))
        info     = results.get("info", {})
        cur      = results.get("currency_sym", "₹")
        altman   = results.get("altman", {})
        piof     = results.get("piotroski", {})
        risk     = results.get("risk", {})
        dcf      = results.get("dcf", {})
        m_data   = [
            ["Metric", "Value", "Metric", "Value"],
            ["Current Price", _fmt(info.get("currentPrice") or info.get("regularMarketPrice"), cur=cur),
             "Sharpe Ratio", _fmt(risk.get("sharpe"), cur="")],
            ["P/E Ratio", _fmt(info.get("trailingPE"), cur="", decimals=1),
             "Max Drawdown", _fmt(risk.get("max_drawdown"), pct=True)],
            ["ROE", _fmt((info.get("returnOnEquity",0) or 0)*100, pct=True),
             "Altman Z", f"{altman.get('z_score','N/A')} ({altman.get('zone','N/A')})"],
            ["DCF Intrinsic", _fmt(dcf.get("intrinsic"), cur=cur),
             "Piotroski F", f"{piof.get('score','N/A')}/9 ({piof.get('signal','N/A')})"],
        ]
        m_tbl = Table(m_data, colWidths=[55*mm, 40*mm, 55*mm, 40*mm])
        m_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,0), colors.HexColor("#21262d")),
            ("TEXTCOLOR",    (0,0), (-1,0), BLUE),
            ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
            ("TEXTCOLOR",    (0,1), (-1,-1), LTGRAY),
            ("FONTSIZE",     (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[ROWBG, colors.HexColor("#1c2128")]),
            ("GRID",         (0,0), (-1,-1), 0.25, colors.HexColor("#30363d")),
            ("TOPPADDING",   (0,0), (-1,-1), 4),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ]))
        story.append(m_tbl)
        story.append(Spacer(1, 5*mm))

        # Thesis
        thesis = results.get("thesis", "")
        if thesis:
            story.append(Paragraph("Investment Thesis", h2_style))
            # Strip markdown
            clean = thesis.replace("**","").replace("*","").replace("#","").strip()
            for para in clean.split("\n\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), body_style))
                    story.append(Spacer(1, 3*mm))

        # Disclaimer
        story.append(Spacer(1, 6*mm))
        story.append(HRFlowable(width="100%", thickness=0.5, color=DKGRAY))
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "⚠ This report is for informational purposes only and does not constitute financial advice. "
            "Past performance is not indicative of future results.",
            small_style))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        return b""   # reportlab not installed
