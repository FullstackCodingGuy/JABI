from langchain_core.tools import tool
import smtplib
from email.mime.text import MIMEText

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email using SMTP."""
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # Use Gmail App Password

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = to

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, [to], msg.as_string())
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        return f"Error sending email: {e}"