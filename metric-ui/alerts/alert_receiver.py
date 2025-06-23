import requests
import json
import os
import datetime

# --- CONFIG ---
ALERTMANAGER_URL = os.getenv("ALERTMANAGER_URL", "http://localhost:9093")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")
TIME_WINDOW = int(os.getenv("TIME_WINDOW", 60))

# pull active alerts from Alertmanager
def get_active_alerts():
    endpoint = f"{ALERTMANAGER_URL}/api/v2/alerts"
    try:
        response = requests.get(endpoint)
        response.raise_for_status()  
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error querying Alertmanager: {e}")
        return None
    
def send_slack_message(payload):
    headers = {"Content-Type": "application/json"}
    if SLACK_WEBHOOK_URL == "":
        print("No Slack URL found")
        return False
    try:
        response = requests.post(SLACK_WEBHOOK_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        print("Slack message sent successfully!")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error sending Slack message: {e}")
        return False
    
# formats slack message for a single alert
def format_slack_message(alert_data):
    alertname = alert_data['labels'].get('alertname', 'N/A')
    severity = alert_data['labels'].get('severity', 'info').upper()
    summary = alert_data['annotations'].get('summary', 'No summary provided.')
    description = alert_data['annotations'].get('description', 'No description provided.')
    generator_url = alert_data.get('generatorURL', 'No generator URL.')
    starts_at_iso = alert_data.get('startsAt')

    starts_at_formatted = "N/A"
    if starts_at_iso:
        try:
            dt_object_utc = datetime.datetime.fromisoformat(starts_at_iso.replace('Z', '+00:00'))
            starts_at_formatted = dt_object_utc.strftime('%Y-%m-%d %H:%M:%S UTC')
        except ValueError:
            pass

    color_emoji = ""
    if severity == "CRITICAL":
        color_emoji = ":red_circle: "
    elif severity == "WARNING":
        color_emoji = ":large_orange_circle: "
    elif severity == "INFO":
        color_emoji = ":large_blue_circle: "

    message_text = (
        f"{color_emoji}*ALERT: {alertname} [{severity}]*\n\n"
        f"*{summary}*\n\n"
        f"{description}\n"
        f"Started At: {starts_at_formatted}\n"
        f"<{generator_url}|View on console>"
    )

    payload = {
        "text": message_text,
        "mrkdwn": True
    }

    return payload

# filters Alertmanager alerts by name + time and sends a slack message for each resulting alert
def process_vllm_alerts_and_notify(alerts, time_window=TIME_WINDOW):
    if alerts:
        now_utc = datetime.datetime.now(datetime.timezone.utc) 
        time_threshold = now_utc - datetime.timedelta(seconds=time_window)

        for alert in alerts:
            alertname = alert['labels'].get('alertname', '')
            starts_at_iso = alert.get('startsAt')
            test_alert_label = alert['labels'].get('test_alert', 'false').lower()

            # filter by VLLM alerts
            if alertname.startswith("VLLM") and test_alert_label != 'true':
                
                if starts_at_iso:
                    try:
                        # filter by time, only notify on alerts that started within given time window
                        alert_start_time_utc = datetime.datetime.fromisoformat(starts_at_iso.replace('Z', '+00:00'))
                        if alert_start_time_utc >= time_threshold:
                            print(f"\n--- Found NEW relevant VLLM Alert: {alertname} ---")
                            slack_payload = format_slack_message(alert)
                            send_slack_message(slack_payload)
                        else:
                            print(f"VLLM alert '{alertname}' is active but started too long ago ({alert_start_time_utc}). Skipping.")
                    except ValueError:
                        print(f"Warning: Could not parse startsAt time for alert '{alertname}': {starts_at_iso}")

def main():
    alerts = get_active_alerts()
    process_vllm_alerts_and_notify(alerts)
    
if __name__ == "__main__":
    main()