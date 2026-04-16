from __future__ import annotations

import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def build_report_subject(vehicle_id: str, risk_label: str) -> str:
    return f"Fleet Maintenance Report | {vehicle_id} | {risk_label}"


def build_report_html(vehicle_data: dict, ml_result: dict, agent_result: dict) -> str:
    vehicle_id = vehicle_data.get("vehicle_id", "N/A")
    vehicle_type = vehicle_data.get("vehicle_type", "N/A")
    risk_label = ml_result.get("risk_label", "Unknown")
    risk_score = ml_result.get("risk_score", 0.0)
    contributing_factors = ml_result.get("contributing_factors", {})
    health_summary = agent_result.get("health_summary", "Not available.")
    action_plan = agent_result.get("action_plan", "Not available.")
    disclaimer = agent_result.get("disclaimer", "Please verify the assessment manually.")
    rag_context = agent_result.get("retrieved_guidelines", "Not available.")

    factor_items = "".join(
        f"<li><strong>{feature.replace('_', ' ').title()}:</strong> {assessment}</li>"
        for feature, assessment in contributing_factors.items()
    ) or "<li>No major contributing factors were identified.</li>"

    action_html = "<br>".join(action_plan.splitlines()) if action_plan else "Not available."
    rag_html = "<br>".join(rag_context.splitlines()) if rag_context else "Not available."

    return f"""
    <html>
      <body style="margin:0;padding:0;background:#f4f7fb;font-family:Arial,sans-serif;color:#132238;">
        <div style="max-width:760px;margin:24px auto;background:#ffffff;border:1px solid #dbe4f0;border-radius:16px;overflow:hidden;">
          <div style="padding:28px 32px;background:linear-gradient(135deg,#103b8c,#0e7490);color:#ffffff;">
            <div style="font-size:13px;letter-spacing:1.2px;text-transform:uppercase;opacity:0.85;">Fleet AI Maintenance Advisory</div>
            <h1 style="margin:10px 0 6px;font-size:28px;">Vehicle Health Report</h1>
            <div style="font-size:15px;">Vehicle <strong>{vehicle_id}</strong> | {vehicle_type}</div>
          </div>

          <div style="padding:28px 32px;">
            <table style="width:100%;border-collapse:collapse;margin-bottom:24px;">
              <tr>
                <td style="padding:14px;border:1px solid #e5edf7;border-radius:12px;background:#f8fbff;">
                  <div style="font-size:12px;text-transform:uppercase;color:#5f718a;letter-spacing:1px;">Risk Label</div>
                  <div style="margin-top:6px;font-size:24px;font-weight:700;color:#0f172a;">{risk_label}</div>
                </td>
                <td style="padding:14px;border:1px solid #e5edf7;border-radius:12px;background:#f8fbff;">
                  <div style="font-size:12px;text-transform:uppercase;color:#5f718a;letter-spacing:1px;">Risk Probability</div>
                  <div style="margin-top:6px;font-size:24px;font-weight:700;color:#0f172a;">{risk_score:.1%}</div>
                </td>
              </tr>
            </table>

            <h2 style="font-size:18px;color:#103b8c;margin:0 0 10px;">Executive Summary</h2>
            <p style="font-size:14px;line-height:1.7;margin:0 0 20px;">{health_summary}</p>

            <h2 style="font-size:18px;color:#103b8c;margin:0 0 10px;">Recommended Action Plan</h2>
            <div style="font-size:14px;line-height:1.7;margin:0 0 20px;">{action_html}</div>

            <h2 style="font-size:18px;color:#103b8c;margin:0 0 10px;">Contributing Factors</h2>
            <ul style="font-size:14px;line-height:1.7;margin:0 0 20px;padding-left:20px;">
              {factor_items}
            </ul>

            <h2 style="font-size:18px;color:#103b8c;margin:0 0 10px;">Retrieved Maintenance Guidance</h2>
            <div style="font-size:14px;line-height:1.7;margin:0 0 20px;background:#f8fbff;border:1px solid #e5edf7;border-radius:12px;padding:16px;">
              {rag_html}
            </div>

            <h2 style="font-size:18px;color:#103b8c;margin:0 0 10px;">Operational Disclaimer</h2>
            <p style="font-size:13px;line-height:1.7;margin:0;color:#5f718a;">{disclaimer}</p>
          </div>
        </div>
      </body>
    </html>
    """


def send_report_email(
    smtp_host: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    sender_email: str,
    recipient_email: str,
    subject: str,
    html_body: str,
) -> None:
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = sender_email
    message["To"] = recipient_email
    message.attach(MIMEText(html_body, "html"))

    context = ssl.create_default_context()
    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls(context=context)
        server.login(smtp_username, smtp_password)
        server.sendmail(sender_email, recipient_email, message.as_string())
