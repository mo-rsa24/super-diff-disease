# src/monitoring/alert_notifier.py

import smtplib
from email.message import EmailMessage
import os

import os
import requests
import json

def send_failure_email(run_id, reason, epoch, checkpoint_path, recipient="1858893@students.wits.ac.za"):
    body = f"""
Training run ID: {run_id}
Status: FAILED
Reason: {reason}
Epoch: {epoch}
Last checkpoint: {checkpoint_path}

To resume:
1. Load checkpoint from above path
2. Ensure loss values are valid
3. Run: python train.py --resume {checkpoint_path}

Logs: {checkpoint_path}/../logs/
TensorBoard: tensorboard --logdir {checkpoint_path}/../tensorboard
"""

    url = "https://sandbox.api.mailtrap.io/api/send/3759865"
    api_token = os.getenv("MAILTRAP_API_TOKEN", "9b74c5a4c855c25eafe75b471b9b203a ")
    sender_email = os.getenv("EMAIL_SENDER", "monitor@superdiff.org")

    payload = {
        "from": {"email": sender_email, "name": "SuperDiff Monitor"},
        "to": [{"email": recipient}],
        "subject": f"🚨 SuperDiff Training Failure at Epoch {epoch}",
        "text": body,
        "category": "Training Failure Alert"
    }

    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print("📨 Failure alert email sent via Mailtrap API.")
    except Exception as e:
        print(f"⚠️ Failed to send email via Mailtrap API: {e}")