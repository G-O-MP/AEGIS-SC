"""
8类语义映射: 18类 → 8类 (ground_combat_vehicle, infantry, air_platform,
weapon_system, air_defense_system, naval_platform, civilian, fortification)
"""
CLASS_MAP_8 = {
    "main_battle_tank": "ground_combat_vehicle",
    "armored_personnel_carrier": "ground_combat_vehicle",
    "military_truck": "ground_combat_vehicle",
    "military_vehicle": "ground_combat_vehicle",

    "soldier": "infantry",
    "camouflage_soldier": "infantry",

    "military_aircraft": "air_platform",
    "fighter_aircraft": "air_platform",
    "bomber_aircraft": "air_platform",

    "weapon": "weapon_system",
    "artillery": "weapon_system",
    "rocket_launcher": "weapon_system",

    "missile_system": "air_defense_system",
    "radar": "air_defense_system",

    "warship": "naval_platform",

    "civilian": "civilian",
    "civilian_vehicle": "civilian",

    "trench": "fortification",

    "undefined_vehicle": "unknown"
}

CLASSES_8 = sorted(set(CLASS_MAP_8.values()))

CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES_8)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES_8)}
