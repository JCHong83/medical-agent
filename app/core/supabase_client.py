import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

if not url or not key:
  raise ValueError("Supabse URL or Key missing in environment variables.")

supabase: Client = create_client(url, key)