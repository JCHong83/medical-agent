from datetime import datetime, timedelta, time

class BookingService:
  @staticmethod
  def get_slots(canvas_start: time, canvas_end: time, appointments: list, duration: int, target_date: datetime):
    """
    Generates available start times for a given duration.
    appointments: list of dicts with {'start': datetime, 'end': datetime}
    """
    slots = []

    # Combine date and time to create start/end datetimes for the canvas
    current_dt = datetime.combine(target_date.date(), canvas_start)
    end_dt = datetime.combine(target_date.date(), canvas_end)

    # Incremental step for start times (e.g., allow booking every 15 mins)
    step = timedelta(minutes=15)
    service_delta = timedelta(minutes=duration)

    while current_dt + service_delta <= end_dt:
      potential_end = current_dt + service_delta

      # Check for overlaps with existing appointments
      is_blocked = False
      for appt in appointments:
        # Overlap check: (StartA < EndB) and (EndA > StartB)
        if current_dt < appt['end'] and potential_end > appt['start']:
          is_blocked = True
          # Optimization: jump to the end of this appointment
          current_dt = appt['end']
          break

      if not is_blocked:
        # Only add if slot is in the future
        if current_dt > datetime.now():
          slots.append(current_dt.strftime("%H:%M"))
        current_dt += step

    return slots