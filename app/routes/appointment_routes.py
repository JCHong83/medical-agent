from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import List
from app.core.supabase_client import supabase

router = APIRouter(prefix="/appointments", tags=["Appointments"])

@router.get("/slots/{doctor_id}")
async def get_available_slots(doctor_id: str, date: str):
  """Fetches available slots for a doctor on a specific date."""
  try:
    # Day of week math
    target_date = datetime.strptime(date, "%Y-%m-%d")
    day_of_week = target_date.weekday()

    # Query Doctor Availabilities
    avail_res = supabase.table("doctor_availabilities") \
      .select("*") \
      .eq("doctor_id", doctor_id) \
      .eq("day_of_week", day_of_week) \
      .execute()
    
    if not avail_res.data:
      return {"slots": []}
    
    # Query Existing Bookings
    appts_res = supabase.table("appointments") \
      .select("start_ts") \
      .eq("doctor_id", doctor_id) \
      .eq("status", "confirmed") \
      .filter("start_ts", "gte", f"{date}T00:00:00") \
      .filter("start_ts", "lte", f"{date}T23:59:59") \
      .execute()
    
    booked_slots = [
      datetime.fromisoformat(a["start_ts"]).strftime("%H:%M")
      for a in appts_res.data
    ]

    # Generate the 30min slots
    slots = []
    for period in avail_res.data:
      start = datetime.strptime(period["start_time"], "%H:%M:%S")
      end = datetime.strptime(period["end_time"], "%H:%M:%S")

      curr = start
      while curr + timedelta(minutes=30) <= end:
        s_str = curr.strftime("%H:%M")
        if s_str not in booked_slots:
          slots.append(s_str)
        curr += timedelta(minutes=30)

    return {"status": "success", "slots": slots}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))