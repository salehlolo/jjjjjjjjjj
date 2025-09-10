import os
# --- .env loader (supports local development) ---
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(), override=True)
except Exception:
    pass


def load_settings():
    """
    Loads credentials from env, supporting both naming styles:
    - OKX_API_KEY / OKX_SECRET_KEY / OKX_PASSPHRASE
    - OKX_API_KEY / OKX_SECRET / OKX_PASSWORD
    """
    api_key = os.getenv("OKX_API_KEY")
    secret  = os.getenv("OKX_SECRET") or os.getenv("OKX_SECRET_KEY")
    passwd  = os.getenv("OKX_PASSWORD") or os.getenv("OKX_PASSPHRASE")

    missing = []
    if not api_key: missing.append("OKX_API_KEY")
    if not secret:  missing.append("OKX_SECRET or OKX_SECRET_KEY")
    if not passwd:  missing.append("OKX_PASSWORD or OKX_PASSPHRASE")
    if missing:
        raise EnvironmentError("Missing required environment variable(s): " + ", ".join(missing))

    return {
        "OKX_API_KEY": api_key,
        "OKX_SECRET":  secret,
        "OKX_PASSWORD": passwd,
        "ENABLE_SANDBOX": os.getenv("ENABLE_SANDBOX", "true").lower() == "true",
    }
