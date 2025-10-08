# sanity_check_config.py
from config import get_settings
if __name__ == "__main__":
    s = get_settings()
    print("Config OK. Interval:", s.interval, "TopN:", s.top_n)
