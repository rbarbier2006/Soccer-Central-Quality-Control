# profiles.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SurveyProfile:
    # REQUIRED (no defaults) - MUST COME FIRST
    key: str
    respondent_singular: str
    respondent_plural: str

    respondent_name_index: int
    group_col_index: int

    rating_col_indices: List[int]
    yesno_col_indices: List[int]

    # OPTIONAL / DEFAULTS - MUST COME AFTER REQUIRED
    choice_col_index: Optional[int] = None
    qq_rating_col_index: Optional[int] = None
    unassigned_label: str = "UNASSIGNED"

    # Use default_factory for mutable types
    team_coach_map: Dict[str, str] = field(default_factory=dict)
    denominator_map: Dict[str, int] = field(default_factory=dict)
    chart_labels: Dict[int, str] = field(default_factory=dict)


# ----------------------------
# PLAYERS PROFILE (your current one)
# ----------------------------

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

PLAYERS_TEAM_COACH_MAP = {
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

PLAYERS_DENOMINATOR_MAP = {
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

PLAYERS_PROFILE = SurveyProfile(
    key="players",
    respondent_singular="player",
    respondent_plural="players",
    respondent_name_index=6,   # update if needed
    group_col_index=7,         # update if needed
    rating_col_indices=[0, 1, 2, 3, 4, 5, 6],   # update to your real rating columns
    yesno_col_indices=[7, 8],                  # update to your real yes/no columns
    choice_col_index=9,        # update or set None
    qq_rating_col_index=None,  # leave None to use 7th rating as fallback
    unassigned_label="UNASSIGNED",
    team_coach_map=PLAYERS_TEAM_COACH_MAP,
    denominator_map=PLAYERS_DENOMINATOR_MAP,
    chart_labels=PLAYERS_CHART_LABELS,
)


# ----------------------------
# FAMILIES PROFILE (placeholder, you will set real indices)
# ----------------------------

FAMILIES_PROFILE = SurveyProfile(
    key="families",
    respondent_singular="family",
    respondent_plural="families",
    respondent_name_index=0,      # TODO set correctly
    group_col_index=0,            # TODO set correctly
    rating_col_indices=[],        # TODO set correctly
    yesno_col_indices=[],         # TODO set correctly
    choice_col_index=None,        # TODO if you have one
    qq_rating_col_index=None,     # TODO if summary should use a specific rating column
    unassigned_label="UNASSIGNED",
    team_coach_map=PLAYERS_TEAM_COACH_MAP,  # or a separate map if needed
    denominator_map=PLAYERS_DENOMINATOR_MAP, # or separate if needed
    chart_labels={},              # optional
)

PROFILES = {
    "players": PLAYERS_PROFILE,
    "families": FAMILIES_PROFILE,
}
