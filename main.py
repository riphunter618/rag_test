import os
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import logging
from supabase import create_client

SUPABASE_URL = "https://jchomjeizcpetzglspun.supabase.co"
  # IMPORTANT: use service role key (server only)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# --------------------------------------------------
# App Configuration
# --------------------------------------------------
app = FastAPI(
    title="Tour Booking Calendar API",
    description="Single + Bulk tour booking confirmations with Google Calendar invites.",
    version="4.0.0"
)

# --------------------------------------------------
# Enable CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Google Calendar Setup
# --------------------------------------------------
SCOPES = ["https://www.googleapis.com/auth/calendar"]

LOCAL_TOKEN_PATH = "token.json"
LOCAL_CLIENT_SECRET_PATH = "client_secrets.json"

CLIENT_SECRET_PATH = (
    "/etc/secrets/client_secrets.json"
    if os.path.exists("/etc/secrets/client_secrets.json")
    else LOCAL_CLIENT_SECRET_PATH
)

TOKEN_PATH = (
    "/etc/secrets/token.json"
    if os.path.exists("/etc/secrets/token.json")
    else LOCAL_TOKEN_PATH
)

print("Using CLIENT_SECRET_PATH:", CLIENT_SECRET_PATH)
print("Using TOKEN_PATH:", TOKEN_PATH)

# --------------------------------------------------
# Google Auth
# --------------------------------------------------
def get_calendar_service():
    creds = None

    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds:
            if not os.path.exists(CLIENT_SECRET_PATH):
                raise Exception("client_secrets.json not found")

            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_PATH, SCOPES
            )
            creds = flow.run_local_server(
                port=0,
                access_type='offline',
                prompt='consent'
            )
        try:
            with open(TOKEN_PATH, "w") as token:
                token.write(creds.to_json())
        except OSError as e:
            logging.warning(f'could not write token file, {e}')

    return build("calendar", "v3", credentials=creds)

# --------------------------------------------------
# Models
# --------------------------------------------------
class CalendarEvent(BaseModel):
    title: str
    description: Optional[str] = None
    startDateTime: str
    endDateTime: str


class BookingPayload(BaseModel):
    customerEmail: str
    customerFirstName: str
    customerLastName: str
    customerPhone: Optional[str] = None
    tourType: str
    numberOfParticipants: int
    bookingDate: str
    bookingTime: str
    isParticipantAdult: bool
    hasAcceptedTerms: bool
    digitalSignature: Optional[str] = None
    paymentMethod: str
    paymentStatus: str
    tourPrice: float
    calendarEvent: CalendarEvent
    fulfillmentStatus: str
    orderTimestamp: str
    #terms_and_conditions:HttpUrl
    #cancellation_policy:HttpUrl
    #waiver_form:HttpUrl
    #approval_status:str


class BulkBookingPayload(BaseModel):
    bookings: List[BookingPayload]

# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/docs")

# --------------------------------------------------
# Helper Function
# --------------------------------------------------
def create_event(service, booking):
    summary = f"{booking.calendarEvent.title} - {booking.tourType}"

    description = f"""
Booking Confirmation

Customer: {booking.customerFirstName} {booking.customerLastName}
Email: {booking.customerEmail}
Phone: {booking.customerPhone or 'N/A'}

Tour Type: {booking.tourType}
Participants: {booking.numberOfParticipants}
Adult: {'Yes' if booking.isParticipantAdult else 'No'}

Booking Date: {booking.bookingDate}
Time: {booking.bookingTime}

Payment Method: {booking.paymentMethod}
Payment Status: {booking.paymentStatus}
Price: ₹{booking.tourPrice}

Fulfillment Status: {booking.fulfillmentStatus}
Order Timestamp: {booking.orderTimestamp}

Terms Accepted: {'Yes' if booking.hasAcceptedTerms else 'No'}

Terms and conditions: {booking.terms_and_conditions}
cancellation policy: {booking.cancellation_policy}
waiver form: {booking.waiver_form}
Signature: {booking.digitalSignature or 'N/A'}
    """.strip()

    event_body = {
        "summary": summary,
        "description": description,

        "start": {
            "dateTime": booking.calendarEvent.startDateTime,
            "timeZone": "Asia/Kolkata"
        },

        "end": {
            "dateTime": booking.calendarEvent.endDateTime,
            "timeZone": "Asia/Kolkata"
        },

        "attendees": [
            {"email": booking.customerEmail}
        ],

        "status": "confirmed",

        "guestsCanModify": False,
        "guestsCanInviteOthers": False,
        "guestsCanSeeOtherGuests": False,

        "reminders": {
            "useDefault": True
        }
    }

    created_event = service.events().insert(
        calendarId="primary",
        body=event_body,
        sendUpdates="all",
        conferenceDataVersion=0
    ).execute()

    return created_event

# --------------------------------------------------
# Single Booking API
# --------------------------------------------------
@app.post("/create-booking-event")
async def create_booking_event(booking: BookingPayload):
    try:
        data = booking.dict()
        #returdata.pop('calendarEvent')
        data["calendar_event"] = data.pop("calendarEvent")
        data["approval_status"] = "pending"
        response = supabase.table("bookings").insert(data).execute()
        return {
            "status": "pending",
            "message": "Booking stored and waiting for admin approval",
            "booking": response.data[0]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --------------------------------------------------
# Bulk Booking API
# --------------------------------------------------
@app.post("/bulk-create-bookings")
async def bulk_create_bookings(data: BulkBookingPayload):
    try:
        service = get_calendar_service()
        results = []

        for booking in data.bookings:
            created_event = create_event(service, booking)

            results.append({
                "customer": booking.customerFirstName,
                "email": booking.customerEmail,
                "eventId": created_event.get("id"),
                "eventLink": created_event.get("htmlLink")
            })

            time.sleep(2)   # delay for better email delivery

        return {
            "status": "success",
            "message": f"{len(results)} bookings created successfully.",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/webhook/booking-approved")
async def booking_approved_webhook(payload: dict):
    try:
        record = payload.get("record")
        return record
        '''if not record:
            return {"status": "ignored"}

        if record.get("approval_status") != "approved":
            return {"status": "ignored"}

        # Prevent duplicate processing
        if record.get("event_created"):
            return {"status": "already processed"}

        # Convert to your model
        record["calendarEvent"] = record.pop("calendar_event")
        booking = BookingPayload(**record)

        # Create event
        service = get_calendar_service()
        created_event = create_event(service, booking)

        # Mark as processed
        supabase.table("bookings").update({
            "event_created": True
        }).eq("id", record["id"]).execute()

        return {
            "status": "success",
            "eventId": created_event.get("id")
        }'''

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
