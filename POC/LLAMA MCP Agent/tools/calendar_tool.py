from langchain_core.tools import tool
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import os, datetime

SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

@tool
def create_event(title: str, datetime_iso: str, duration_minutes: int = 60) -> str:
    """Create a Google Calendar event."""
    creds = None
    token_path = "credentials/token.json"
    creds_path = "credentials/google_credentials.json"

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    else:
        flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
        creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    service = build('calendar', 'v3', credentials=creds)
    start_time = datetime.datetime.fromisoformat(datetime_iso)
    end_time = start_time + datetime.timedelta(minutes=duration_minutes)

    event = {
        'summary': title,
        'start': {'dateTime': start_time.isoformat(), 'timeZone': 'UTC'},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': 'UTC'},
    }

    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {created_event.get('htmlLink')}"