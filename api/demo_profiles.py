"""
GasSetu — Demo profiles for the dashboard
==========================================
Fictional consumers with made-up booking histories, so the dashboard can show each outcome without
real personal data and without calling the paid GSTN API. The GSTINs come from the fictional
DEMO_REGISTRY in core/gstn_check.py. Dates are generated relative to today.
"""

from datetime import date, timedelta

from core.features import features_from_refill_dates
from core.gstn_check import DEMO_GSTINS


def _every(gap_days: int, count: int, last_days_ago: int = 9) -> list:
    """`count` refills, one every `gap_days`, the latest `last_days_ago` days back."""
    return [last_days_ago + i * gap_days for i in range(count)]


def _profiles() -> list:
    restaurant = DEMO_GSTINS["Annapurna Family Restaurant"]
    return [
        dict(key="normal_household", label="Normal household", group="household",
             consumer_name="Priya Sharma", address="Flat 12B, Mahavir Nagar, Kandivali West, Mumbai", pincode="400067",
             cylinder_type="domestic", household_size=4, is_pmuy=False, is_rural=False,
             days_ago=_every(52, 14)),
        dict(key="ujjwala_low_usage", label="Ujjwala home, 4 refills a year", group="household",
             consumer_name="Sunita Devi", address="House 22, At Post Shirgaon, Tal. Palghar", pincode="401404",
             cylinder_type="domestic", household_size=5, is_pmuy=True, is_rural=True,
             days_ago=_every(88, 8)),
        dict(key="joint_family", label="Joint family of nine", group="household",
             consumer_name="Ramesh Patil", address="Patil Wada, Old Panvel", pincode="410206",
             cylinder_type="domestic", household_size=9, is_pmuy=False, is_rural=False,
             days_ago=_every(31, 23)),
        dict(key="captured_connection", label="Low-usage connection that suddenly books every 25 days", group="suspicious",
             consumer_name="Kamalabai Jadhav", address="House 14, Gram Panchayat Road, Wada", pincode="421303",
             cylinder_type="domestic", household_size=2, is_pmuy=True, is_rural=True, mobile_changes_12m=1,
             days_ago=_every(26, 8, last_days_ago=4) + _every(121, 4, last_days_ago=330)),
        dict(key="max_rate_booker", label="Two-person home booking the moment the lock ends, all year", group="suspicious",
             consumer_name="Meena Devi", address="Flat 2B, Nehru Nagar, Kurla East, Mumbai", pincode="400024",
             cylinder_type="domestic", household_size=2, is_pmuy=False, is_rural=False,
             days_ago=_every(26, 28, last_days_ago=3)),
        dict(key="restaurant_domestic", label="Restaurant books a domestic cylinder", group="registry",
             consumer_name="Annapurna Restaurant", address="Shop 4, Sector 17 Market, Vashi, Navi Mumbai", pincode="400703",
             cylinder_type="domestic", consumer_gstin=restaurant, household_size=4, days_ago=[]),
        dict(key="restaurant_commercial", label="Same restaurant, commercial cylinder", group="registry",
             consumer_name="Annapurna Restaurant", address="Shop 4, Sector 17 Market, Vashi, Navi Mumbai", pincode="400703",
             cylinder_type="commercial", consumer_gstin=restaurant, household_size=4, days_ago=[]),
        dict(key="borrowed_gstin", label="Commercial booking with someone else's GSTIN", group="registry",
             consumer_name="Rahul Verma", address="Shop 9, Sector 2, Airoli, Navi Mumbai", pincode="400708",
             cylinder_type="commercial", consumer_gstin=restaurant, household_size=4, days_ago=[]),
        dict(key="home_office", label="Consultant with a GST registration at home", group="registry",
             consumer_name="Priya Nair", address="Flat 12B, Sea Breeze CHS, Sector 9, Kharghar, Navi Mumbai", pincode="410210",
             cylinder_type="domestic", consumer_gstin=DEMO_GSTINS["Priya Nair"], household_size=3,
             is_pmuy=False, is_rural=False, days_ago=_every(60, 12)),
        dict(key="suspended_gstin", label="Commercial booking, GST registration suspended", group="registry",
             consumer_name="Konkan Sweets", address="Shop 11, Station Road, Nerul, Navi Mumbai", pincode="400706",
             cylinder_type="commercial", consumer_gstin=DEMO_GSTINS["Konkan Sweets and Farsan"], household_size=4, days_ago=[]),
    ]


def demo_profiles(today: date = None) -> list:
    """The profiles with refill dates and the history numbers the API would derive from them."""
    today = today or date.today()
    out = []
    for p in _profiles():
        dates = sorted(today - timedelta(days=d) for d in p.pop("days_ago") if d < 730)
        profile = {"consumer_gstin": "", "state_code": "27", "company": "indane", "is_pmuy": False,
                   "is_rural": False, "mobile_changes_12m": 0, **p}
        profile["refill_dates"] = [d.isoformat() for d in dates]
        profile["history"] = {k: int(v) for k, v in features_from_refill_dates(dates, today).items()} if dates else None
        out.append(profile)
    return out
