import uuid
from app.core.supabase_client import supabase

def seed_system():
  print("🚀 Initializing mock doctor ecosystem data transaction...")
  
  # 1. Targets or constructs a doctor profile reference identity
  doctor_auth_id = str(uuid.uuid4()) # In real world, this maps directly to auth.users entries
  
  # Insert Profile Record
  profile = {
    "id": doctor_auth_id,
    "full_name": "Dr. Alessandro Bianchi",
    "avatar_url": "https://images.unsplash.com/photo-1622253692010-333f2da6031d?auto=format&fit=crop&w=200&q=80"
  }
  supabase.table("profiles").upsert(profile).execute()
  print("✅ Profile instantiated.")

  # 2. Map Profile into the specific Doctors entity entry
  doctor_entry = {
    "id": doctor_auth_id,
    "specialties": ["Dentist", "Odontoiatria"],
    "verification_status": "verified",
    "bio": "Expert specialist in root canal therapies and dynamic orthodontic alignment pipelines."
  }
  supabase.table("doctors").upsert(doctor_entry).execute()
  print("✅ Doctor specialized mapping verified.")

  # 3. Provision a Sample Physical Clinic Entity Space
  clinic_id = str(uuid.uuid4())
  clinic_data = {
    "id": clinic_id,
    "name": "Milano Central Health Hub",
    "address": "Via Vittor Pisani 14, Milano, Italy"
  }
  supabase.table("clinics").upsert(clinic_data).execute()
  print("✅ Clinic hub location instantiated.")

  # 4. Tie Doctor entry to the Clinic location via join tracking matrix
  join_data = {
    "doctor_id": doctor_auth_id,
    "clinic_id": clinic_id
  }
  supabase.table("doctor_clinics").upsert(join_data).execute()

  # 5. Build Base Recurring Availability Canvas Templates (e.g. Monday Working Window)
  availability = {
    "doctor_id": doctor_auth_id,
    "clinic_id": clinic_id,
    "is_recurring": True,
    "day_of_week": 0, # Monday representation matching backend weekday() math indices
    "start_time": "09:00:00",
    "end_time": "17:00:00"
  }
  supabase.table("doctor_availabilities").insert(availability).execute()
  print("✅ Base weekly timeline canvas deployed.")
  
  print(f"\n🎉 SEED COMPLETE! Copy this Doctor ID tracking string for local frontend overrides:\n👉 {doctor_auth_id}")

if __name__ == "__main__":
  seed_system()