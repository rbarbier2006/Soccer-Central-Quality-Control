# profiles.py

from dataclasses import dataclass
from typing import Dict, List, Optional


def excel_col_to_index(col: str) -> int:
    col = col.strip().upper()
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


@dataclass(frozen=True)
class SurveyProfile:
    key: str

    # Where to group rows (team)
    group_col_index: int
    unassigned_label: str = "UNASSIGNED"

    # Who responded (player name or family name)
    respondent_name_index: int
    respondent_singular: str = "player"
    respondent_plural: str = "players"

    # Question columns
    rating_col_indices: List[int] = None
    yesno_col_indices: List[int] = None
    choice_col_index: Optional[int] = None

    # QQ settings
    qq_rating_col_index: Optional[int] = None  # which rating question defines "rating" for QQ
    denominator_map: Optional[Dict[str, int]] = None  # roster size / family count per team

    # Display / metadata
    team_coach_map: Dict[str, str] = None
    chart_labels: Optional[Dict[int, str]] = None  # optional pretty labels by chart number


# -----------------------------
# Team / coach metadata (shared)
# -----------------------------

TEAM_COACH_MAP: Dict[str, str] = {
    "MLS HG U19": "Jorge",
    "MLS HG U17": "Chris",
    "MLS HG U16": "David K",
    "MLS HG U15": "Jorge",
    "MLS HG U14": "David K",
    "MLS HG U13": "Chris M",

    "MLS AD U19": "Michael",
    "MLS AD U17": "Michael",
    "MLS AD U16": "Miguel",
    "MLS AD U15": "Miguel",
    "MLS AD U14": "Junro",
    "MLS AD U13": "Miguel",

    "TX2 U19": "Jesus",
    "TX2 U17": "Fernando",
    "TX2 U16": "Jesus",
    "TX2 U15": "Claudia",
    "TX2 U14": "Rene/Claudia",
    "TX2 U13": "Claudia/Rene",
    "TX2 U12": "Armando",
    "TX2 U11": "Armando",

    "Athenians U16": "Rumen",
    "Athenians U13": "Keeley",
    "Athenians WDDOA U12": "Keeley",
    "Athenians WDDOA U11": "Robert",
    "Athenians PDF U10": "Robert",
    "Athenians PDF U9": "Robert",

    "WDDOA U12": "Adam",
    "WDDOA U11": "Adam",

    "PDF U10 White": "Steven",
    "PDF U9 White": "Steven",
    "PDF U10 Red": "Pablo",
    "PDF U9 Red": "Pablo",
}

TEAM_ROSTER_SIZE: Dict[str, int] = {
    "MLS HG U19": 19,
    "MLS HG U17": 19,
    "MLS HG U16": 13,
    "MLS HG U15": 12,
    "MLS HG U14": 15,
    "MLS HG U13": 17,

    "MLS AD U19": 19,
    "MLS AD U17": 17,
    "MLS AD U16": 19,
    "MLS AD U15": 18,
    "MLS AD U14": 19,
    "MLS AD U13": 15,

    "TX2 U19": 14,
    "TX2 U17": 19,
    "TX2 U16": 22,
    "TX2 U15": 22,
    "TX2 U14": 17,
    "TX2 U13": 15,
    "TX2 U12": 13,
    "TX2 U11": 11,

    "Athenians U16": 15,
    "Athenians U13": 14,
    "Athenians WDDOA U12": 8,
    "Athenians WDDOA U11": 11,
    "Athenians PDF U10": 11,
    "Athenians PDF U9": 5,

    "WDDOA U12": 10,
    "WDDOA U11": 14,

    "PDF U10 White": 8,
    "PDF U9 White": 11,
    "PDF U10 Red": 9,
    "PDF U9 Red": 8,
}

# If you later get "number of families per team", put it here.
# For now, leave as None (percent will not show; QQ will use fraction=1.0).
TEAM_FAMILY_COUNT = None


# -----------------------------
# Players profile (uses your existing excel_processor constants)
# -----------------------------
from excel_processor import (
    PLAYER_NAME_INDEX,
    RATING_COL_INDICES,
    YESNO_COL_INDICES,
    CHOICE_COL_INDEX,
    GROUP_COL_INDEX,
)

PLAYERS_CHART_LABELS = {
    1: "(1)Safety and Support",
    2: "(2)Improvement",
    3: "(3)Instructions and Feedback",
    4: "(4)Coaches Listening",
    5: "(5)Effort and Discipline",
    6: "(6)SC Value Alignment",
    7: "(7)Overall Experience",
    8: "(8)Team Belonging",
    9: "(9)Cycle Enjoyment",
}

PLAYERS_PROFILE = SurveyProfile(
    key="players",
    group_col_index=GROUP_COL_INDEX,
    respondent_name_index=PLAYER_NAME_INDEX,
    respondent_singular="player",
    respondent_plural="players",
    rating_col_indices=RATING_COL_INDICES,
    yesno_col_indices=YESNO_COL_INDICES,
    choice_col_index=CHOICE_COL_INDEX,
    # Your QQ currently uses "Overall Experience" which is the 7th rating question
    # In your current code: overall_idx = rating_indices[6]
    # We keep that logic in pdf_report.py if qq_rating_col_index is None.
    qq_rating_col_index=None,
    denominator_map=TEAM_ROSTER_SIZE,
    team_coach_map=TEAM_COACH_MAP,
    chart_labels=PLAYERS_CHART_LABELS,
)


# -----------------------------
# Families profile (based on your sketch)
# Adjust if your real sheet differs.
# -----------------------------
FAMILIES_PROFILE = SurveyProfile(
    key="families",
    group_col_index=excel_col_to_index("H"),
    respondent_name_index=excel_col_to_index("G"),
    respondent_singular="response",
    respondent_plural="responses",

    # Ratings: I, J, and L through X
    rating_col_indices=(
        [excel_col_to_index("I"), excel_col_to_index("J")] +
        list(range(excel_col_to_index("L"), excel_col_to_index("X") + 1))
    ),

    # Yes/No: K
    yesno_col_indices=[excel_col_to_index("K")],

    # Choice: Y
    choice_col_index=excel_col_to_index("Y"),

    # Overall satisfaction: Q (used for QQ rating)
    qq_rating_col_index=excel_col_to_index("Q"),

    # If you later define TEAM_FAMILY_COUNT, swap this in:
    denominator_map=TEAM_FAMILY_COUNT,

    team_coach_map=TEAM_COACH_MAP,
    chart_labels=None,  # optional: you can add a mapping later like the players one
)

PROFILES = {
    "players": PLAYERS_PROFILE,
    "families": FAMILIES_PROFILE,
}
