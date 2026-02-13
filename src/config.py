RAW_COLUMNS = [
    "Airline", "Source", "Source Name", "Destination", "Destination Name",
    "Departure Date & Time", "Arrival Date & Time", "Duration (hrs)",
    "Stopovers", "Aircraft Type", "Class", "Booking Source",
    "Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)",
    "Seasonality", "Days Before Departure"
]

TARGET = "Total Fare (BDT)"

DATE_COLS = ["Departure Date & Time", "Arrival Date & Time"]
NUMERIC_COLS = ["Duration (hrs)", "Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)", "Days Before Departure"]

CATEGORICAL_COLS = [
    "Airline", "Source", "Source Name", "Destination", "Destination Name",
    "Stopovers", "Aircraft Type", "Class", "Booking Source", "Seasonality"
]

# quality constraints
NON_NEGATIVE_COLS = ["Duration (hrs)", "Base Fare (BDT)", "Tax & Surcharge (BDT)", "Total Fare (BDT)", "Days Before Departure"]
