from datetime import datetime
import pytz

async def get_current_time(timezone: str = "Australia/Melbourne") -> dict:
    """Use this tool when you need to know the current time in a specific timezone.
    
    Args:
        timezone: Timezone to get the current time in (e.g. "America/New_York")
    Returns:
        dict: Dictionary containing the current time in the specified timezone in the format "YYYY-MM-DD HH:MM:SS"
    """
    return {
        "current_time": datetime.now(pytz.timezone(timezone)).strftime("%Y-%m-%d %H:%M:%S")
    }
