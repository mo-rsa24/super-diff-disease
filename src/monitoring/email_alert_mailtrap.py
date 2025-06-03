# utils/email_alert_mailtrap.py

import os
import json
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Environment Variables (from your `.env` or system)
# ──────────────────────────────────────────────────────────────────────────────
#
#   MAILTRAP_API_URL     := "https://sandbox.api.mailtrap.io/api/send/3759865"
#   MAILTRAP_API_TOKEN   := (your Mailtrap bearer token)
#   EMAIL_SENDER         := "monitor@superdiff.org"
#   ALERT_RECIPIENT      := (comma-separated list of recipients, or a single email)
#
# Example .env entries:
#   MAILTRAP_API_URL="https://sandbox.api.mailtrap.io/api/send/3759865"
#   MAILTRAP_API_TOKEN="9b74c5a4c855c25eafe75b471b9b203a"
#   EMAIL_SENDER="monitor@superdiff.org"
#   ALERT_RECIPIENT="dev-team@example.com"
#
# Make sure to load these before running training. E.g., using python-dotenv:
#
#   from dotenv import load_dotenv
#   load_dotenv() 
#   (then import this module)
#
# ──────────────────────────────────────────────────────────────────────────────

MAILTRAP_API_URL = os.getenv(
    "MAILTRAP_API_URL",
    "https://sandbox.api.mailtrap.io/api/send/3759865"
)
MAILTRAP_API_TOKEN = os.getenv(
    "MAILTRAP_API_TOKEN",
    "9b74c5a4c855c25eafe75b471b9b203a"
).strip()
SENDER_EMAIL = os.getenv(
    "EMAIL_SENDER",
    "monitor@superdiff.org"
).strip()
# You can specify multiple recipients, comma‐separated:
ALERT_RECIPIENTS = [
    r.strip() 
    for r in os.getenv("ALERT_RECIPIENT", "1858893@students.wits.ac.za").split(",")
]

def _send_mailtrap_email(subject: str, body: str, category: str = "Training Alert"):
    """
    Sends a single email via Mailtrap’s send endpoint.
    Raises an exception if Mailtrap returns an HTTP error.
    """

    # Build payload
    payload = {
        "from": {"email": SENDER_EMAIL, "name": "SuperDiff Monitor"},
        "to": [{"email": recipient} for recipient in ALERT_RECIPIENTS],
        "subject": subject,
        "text": body,
        "category": category
    }

    headers = {
        "Authorization": f"Bearer {MAILTRAP_API_TOKEN}",
        "Content-Type": "application/json"
    }

    response = requests.post(MAILTRAP_API_URL, 
                             headers=headers, 
                             data=json.dumps(payload))

    # Raise on any 4xx/5xx
    response.raise_for_status()
    # If you want to log the response JSON, uncomment below:
    # print("Mailtrap response:", response.json())
    return response.json()


def alert_on_failure(run_id: str, last_epoch: int, error_msg: str):
    """
    Send a training‐failure alert via Mailtrap.
    Args:
      - run_id: Unique run identifier (e.g., "run_2025_06_02_tb_v1")
      - last_epoch: The epoch at which training failed (0 if before any epoch).
      - error_msg: The exception or error message string.
    """
    subject = f"🚨 SuperDiff Training Failure (Run: {run_id})"
    body = (
        f"⚠️ Training Failure Alert\n\n"
        f"Run ID      : {run_id}\n"
        f"Last Epoch  : {last_epoch}\n"
        f"Error       :\n{error_msg}\n\n"
        "Please investigate logs and attempt to resume with the latest checkpoint."
    )
    category = "Training Failure Alert"

    try:
        _send_mailtrap_email(subject, body, category)
        print("📨 Failure alert email sent via Mailtrap API.")
    except Exception as e:
        # In production, replace print with your logger
        print(f"❌ Could not send failure alert: {e}")


def alert_on_success(run_id: str, total_epochs: int, duration: str):
    """
    Send a training‐success email via Mailtrap.
    Args:
      - run_id: Unique run identifier.
      - total_epochs: How many epochs the training loop ran.
      - duration: Human‐readable training duration (e.g., "2:15:30").
    """
    subject = f"✅ SuperDiff Training Completed (Run: {run_id})"
    body = (
        f"🏁 Training Completed Successfully\n\n"
        f"Run ID         : {run_id}\n"
        f"Total Epochs   : {total_epochs}\n"
        f"Total Duration : {duration}\n\n"
        "You may now proceed to evaluation or archive artifacts."
    )
    category = "Training Success"

    try:
        _send_mailtrap_email(subject, body, category)
        print("📨 Success email sent via Mailtrap API.")
    except Exception as e:
        print(f"❌ Could not send success email: {e}")
