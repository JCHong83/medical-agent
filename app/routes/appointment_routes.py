from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from app.core.supabase_client import supabase
from app.services.booking_service import BookingService

from pydantic import BaseModel
from typing import Optional

# Request Model for Booking
class BookingRequest(BaseModel):
  doctor_id: str
  patient_id: str
  clinic_id: str
  service_id: str
  start_ts: str # ISO Format: 2026-05-20T10:30:00Z
  notes: Optional[str] = None

class ServiceCreateRequest(BaseModel):
  doctor_id: str
  name: str
  description: Optional[str] = None
  duration_minutes: int
  price: float

router = APIRouter(prefix="/appointments", tags=["Appointments"])

@router.get("/slots/{doctor_id}")
async def get_slots(doctor_id: str, service_id: str, date: str):
  """Fetches available slots for a doctor on a specific date."""
  try:
    # Get Service Duration
    service_res = supabase.table("doctor_services").select("duration_minutes").eq("id", service_id).single().execute()
    if not service_res.data:
      raise HTTPException(status_code=404, detail="Service not found")
    duration = service_res.data["duration_minutes"]

    # Check for Exceptions (Override)
    ext_res = supabase.table("availability_exceptions").select("*").eq("doctor_id", doctor_id).eq("date", date).execute()

    target_date = datetime.strptime(date, "%Y-%m-%d")
    
    if ext_res.data:
      exc = ext_res.data[0]
      if not exc["is_available"]:
        return {"slots": []} # Doctor is off (Vacation/Sick)
      # Use specific times from exception
      canvas_start = datetime.strptime(exc["start_time"], "%H:%M:%S").time()
      canvas_end = datetime.strptime(exc["end_time"], "%H:%M:%S").time()
    else:
      # Get Standard Availability
      day_of_week = target_date.weekday()
      avail_res = supabase.table("doctor_availabilities").select("start_time, end_time").eq("doctor_id", doctor_id).eq("day_of_week", day_of_week).execute()

      if not avail_res.data:
        return {"slots": []}
      
      canvas_start = datetime.strptime(avail_res.data[0]["start_time"], "%H:%M:%S").time()
      canvas_end = datetime.strptime(avail_res.data[0]["end_time"], "%H:%M:%S").time()

    # Get Existing Appointments
    appt_res = supabase.table("appointments").select("start_ts, end_ts").eq("doctor_id", doctor_id).neq("status", "cancelled").filter("start_ts", "gte", f"{date}T00:00:00Z").filter("start_ts", "lte", f"{date}T23:59:59Z").execute()

    processed_appts = [
      {"start": datetime.fromisoformat(a["start_ts"].replace('Z', '+00:00')),
       "end": datetime.fromisoformat(a["end_ts"].replace('Z', '+00:00'))}
       for a in appt_res.data
    ]

    # Calculate Gaps
    available_slots = BookingService.get_slots(canvas_start, canvas_end, processed_appts, duration, target_date)

    return {"status": "success", "slots": available_slots}
  
  except Exception as e:
    print(f"Error: {e}")
    raise HTTPException(status_code=500, detail=str(e))
  
# Appointment Creation
@router.post("/book")
async def book_appointment(req: BookingRequest):
  try:
    # Fetch Service Details (Duration & Price)
    service_res = supabase.table("doctor_services")\
      .select("duration_minutes, price")\
      .eq("id", req.service_id)\
      .single().execute()
    
    if not service_res.data:
      raise HTTPException(status_code=404, detail="Service not found")
    
    duration = service_res.data["duration_minutes"]
    price = service_res.data["price"]

    # Calculate end_ts
    start_dt = datetime.fromisoformat(req.start_ts.replace('Z', '+00:00'))
    end_dt = start_dt + timedelta(minutes=duration)

    # Create the appointment
    booking_data = {
      "doctor_id": req.doctor_id,
      "patient_id": req.patient_id,
      "clinic_id": req.clinic_id,
      "service_id": req.service_id,
      "start_ts": start_dt.isoformat(),
      "end_ts": end_dt.isoformat(),
      "status": "confirmed", # Confirmation step?
      "price": price,
      "notes": req.notes,
      "payment_status": "unpaid"
    }

    res = supabase.table("appointments").insert(booking_data).execute()

    return {
      "status": "success",
      "message": "Appointment booked successfully",
      "data": res.data[0]
    }
  
  except Exception as e:
    # If the DB constraint 'appointments_no_overlap' triggers, it will raise an error here
    print(f"Booking Error: {e}")
    if "overlap" in str(e).lower():
      raise HTTPException(status_code=409, detail="This time slot was just taken. Please choose another.")
    raise HTTPException(status_code=500, detail=str(e))
  

# Appointmenting logic
@router.get("/services/{doctor_id}")
async def get_doctor_services(doctor_id: str):
  res = supabase.table("doctor_services")\
    .select("*")\
    .eq("doctor_id", doctor_id)\
    .eq("is_active", True)\
    .execute()
  return {"services": res.data}


@router.post("/services")
async def create_doctor_service(req: ServiceCreateRequest):
  try:
    data = {
      "doctor_id": req.doctor_id,
      "name": req.name,
      "description": req.description,
      "duration_minutes": req.duration_minutes,
      "price": req.price,
      "is_active": True
    }
    res = supabase.table("doctor_services").insert(data).execute()
    return {"status": "success", "service": res.data[0]}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@router.delete("/services/{service_id}")
async def delete_doctor_service(service_id: str):
  try:
    supabase.table("doctor_services").delete().eq("id", service_id).execute()
    return {"status": "success", "message": "Service deleted"}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))