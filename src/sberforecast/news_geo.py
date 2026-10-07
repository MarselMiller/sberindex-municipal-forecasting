"""Conservative news geography from the supplied historical municipality table.

Names are matched as complete phrases after case/whitespace/Unicode
normalization. No fuzzy matching, inferred regions, web geography, or default
national broadcast is used. All recorded names of the same supplied ID are
retained; this lookup does not reconstruct boundaries or name validity at an
article's publication date.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

from .macro_features import normalize_region_id


# Explicit, inspectable spelling/case aliases. An alias is installed only when
# its canonical name exists in the supplied table. This is a limited rule
# catalog, not a morphological model or a current administrative dictionary.
AUDITED_REGION_ALIASES: dict[str, tuple[str, ...]] = {
    "Республика Адыгея": ("Адыгея", "Адыгее"),
    "Республика Алтай": ("Республике Алтай",),
    "Республика Башкортостан": ("Башкортостан", "Башкортостане", "Башкирия", "Башкирии"),
    "Республика Бурятия": ("Бурятия", "Бурятии"),
    "Республика Дагестан": ("Дагестан", "Дагестане"),
    "Республика Ингушетия": ("Ингушетия", "Ингушетии"),
    "Кабардино-Балкарская Республика": ("Кабардино-Балкария", "Кабардино-Балкарии"),
    "Республика Калмыкия": ("Калмыкия", "Калмыкии"),
    "Карачаево-Черкесская Республика": ("Карачаево-Черкесия", "Карачаево-Черкесии"),
    "Республика Карелия": ("Карелия", "Карелии"),
    "Республика Коми": ("Республике Коми",),
    "Республика Марий Эл": ("Марий Эл", "Республике Марий Эл"),
    "Республика Мордовия": ("Мордовия", "Мордовии"),
    "Республика Саха (Якутия)": ("Якутия", "Якутии", "Республика Саха", "Республике Саха"),
    "Республика Северная Осетия — Алания": ("Северная Осетия", "Северной Осетии"),
    "Республика Татарстан": ("Татарстан", "Татарстане"),
    "Республика Тыва": ("Тыва", "Тыве", "Тува", "Туве"),
    "Удмуртская Республика": ("Удмуртия", "Удмуртии"),
    "Республика Хакасия": ("Хакасия", "Хакасии"),
    "Чеченская Республика": ("Чечня", "Чечне"),
    "Чувашская Республика": ("Чувашия", "Чувашии"),
    "Алтайский край": ("Алтайском крае", "Алтайского края"),
    "Забайкальский край": ("Забайкальском крае", "Забайкальского края"),
    "Камчатский край": ("Камчатском крае", "Камчатского края"),
    "Красноярский край": ("Красноярском крае", "Красноярского края"),
    "Пермский край": ("Пермском крае", "Пермского края"),
    "Приморский край": ("Приморском крае", "Приморского края"),
    "Ставропольский край": ("Ставропольском крае", "Ставропольского края"),
    "Хабаровский край": ("Хабаровском крае", "Хабаровского края"),
    "Московская область": ("Московской области", "Подмосковье", "Подмосковья"),
    "Ленинградская область": ("Ленинградской области",),
    "Нижегородская область": ("Нижегородской области",),
    "Новосибирская область": ("Новосибирской области",),
    "Омская область": ("Омской области",),
    "Ростовская область": ("Ростовской области",),
    "Самарская область": ("Самарской области",),
    "Саратовская область": ("Саратовской области",),
    "Свердловская область": ("Свердловской области",),
    "Тюменская область": ("Тюменской области",),
    "Челябинская область": ("Челябинской области",),
    "Кемеровская область": ("Кемеровской области", "Кузбасс", "Кузбассе"),
    "Ненецкий автономный округ": ("Ненецком автономном округе", "НАО"),
    "Ханты-Мансийский автономный округ — Югра": ("Ханты-Мансийский автономный округ", "Югра", "Югре", "ХМАО"),
    "Ямало-Ненецкий автономный округ": ("Ямало-Ненецком автономном округе", "ЯНАО"),
    "Москва": ("Москве", "Москвы"),
    "Санкт-Петербург": ("Санкт-Петербурге", "Санкт-Петербурга"),
}

# Explicit personal-name collisions found among supplied dictionary short
# names. This limited catalog is not named-entity recognition. A colliding
# alias requires territorial wording at THAT occurrence; a region elsewhere
# in the document cannot turn a person's name into a municipality.
AUDITED_PERSON_COLLISION_ALIASES: dict[str, str] = {
    "Белинский": "surname",
    "Владимир": "given_name",
    "Дзержинский": "surname",
    "Киров": "surname",
    "Королёв": "surname",
    "Лермонтов": "surname",
    "Ломоносов": "surname",
    "Пушкин": "surname",
    "Чехов": "surname",
}
PERSON_COLLISION_QUALIFIER_PATTERNS = (
    r"(?<!\w)(?:город(?:а|е|ом)?|г\.|городской\s+округ|муниципальный\s+(?:округ|район)|район(?:а|е)?)\s+$",
    r"^\s+(?:муниципальный\s+(?:район|округ)|городской\s+округ|район)(?!\w)",
)

DICTIONARY_COLUMNS = (
    "municipality_id", "municipality_name", "municipality_name_short",
    "region_id", "region", "municipality_aliases", "region_aliases",
)


def _normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _aliases(values: object) -> tuple[str, ...]:
    """Stable normalized aliases, preserving one original spelling."""
    unique = {_normalize(value): str(value).strip() for value in sorted(set(values))}
    return tuple(unique[key] for key in sorted(unique))


def build_geography_dictionary(raw: pd.DataFrame) -> pd.DataFrame:
    """Build one row per supplied ID with all recorded historical name aliases.

    Project input fields are ``territory_id``, ``municipal_district_name``,
    ``municipal_district_name_short``, ``region_code``, ``region_name``.
    Explicit ``municipality_id``/``territory_name`` input is also accepted for
    prepared tables; names are never manufactured from ID or OKTMO prefixes.
    """
    id_column = "territory_id" if "territory_id" in raw else "municipality_id"
    name_column = "municipal_district_name" if "municipal_district_name" in raw else "territory_name"
    required = {id_column, name_column, "region_code", "region_name"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"Missing geography columns: {sorted(missing)}")
    columns = sorted(required | ({"municipal_district_name_short"} if "municipal_district_name_short" in raw else set()))
    table = raw[columns].copy()
    if table[list(required)].isna().any().any():
        raise ValueError("Missing municipality name/ID or region name/code.")
    for column in required:
        table[column] = table[column].astype(str).str.strip()
        if table[column].eq("").any():
            raise ValueError(f"Empty geography field: {column}")
    table["region_code"] = table["region_code"].map(normalize_region_id)
    if table["region_code"].eq("RU").any():
        raise ValueError("Municipalities require an explicit regional code, not RU.")
    if table.groupby(id_column)["region_code"].nunique().gt(1).any():
        raise ValueError("Ambiguous municipality-to-region correspondence.")
    if table.groupby("region_code")["region_name"].nunique().gt(1).any():
        raise ValueError("A region code has multiple names; supply an explicit canonical correspondence.")
    if table.groupby("region_name")["region_code"].nunique().gt(1).any():
        raise ValueError("A region name has multiple regional codes.")
    records = []
    for municipality_id, rows in table.groupby(id_column, sort=True):
        full_names = _aliases(rows[name_column].tolist())
        short_names = _aliases(rows["municipal_district_name_short"].dropna().astype(str).str.strip().loc[lambda values: values.ne("")].tolist()) if "municipal_district_name_short" in rows else ()
        region = rows["region_name"].iloc[0]
        records.append({
            "municipality_id": str(municipality_id),
            "municipality_name": full_names[0],
            "municipality_name_short": short_names[0] if short_names else "",
            "region_id": rows["region_code"].iloc[0],
            "region": region,
            "municipality_aliases": _aliases((*full_names, *short_names)),
            "region_aliases": _aliases((region, *AUDITED_REGION_ALIASES.get(region, ()))),
        })
    return pd.DataFrame(records, columns=DICTIONARY_COLUMNS)


@lru_cache(maxsize=16384)
def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(_normalize(alias)) + r"(?!\w)")


def _matches(text: str, alias: str) -> bool:
    return _alias_pattern(alias).search(text) is not None


def _collision_alias_is_qualified(text: str, span: tuple[int, int]) -> bool:
    before, after = text[:span[0]], text[span[1]:]
    return (re.search(PERSON_COLLISION_QUALIFIER_PATTERNS[0], before) is not None
            or re.search(PERSON_COLLISION_QUALIFIER_PATTERNS[1], after) is not None)


def _result(level: str, rule: str, *, region: str = "", region_id: str = "", municipality_id: str = "", confidence: float = 0.0, evidence: str = "") -> dict:
    return {
        "geography_level": level, "region": region, "region_id": region_id,
        "municipality_id": municipality_id, "geography_confidence": confidence,
        "geography_rule": rule, "geography_evidence": evidence,
    }


def resolve_geography(title: str, snippet: str, dictionary: pd.DataFrame, source: str = "") -> dict:
    """Assign an exact supplied territory, or leave ambiguity explicitly unknown.

    Confidence is fixed rule strength, not a calibrated probability. Local
    mentions take precedence over explicit national context. Multiple local
    territories, ambiguous unqualified names, and conflicting region/МО names
    are unknown and are never converted into national coverage.
    """
    missing = set(DICTIONARY_COLUMNS) - set(dictionary)
    if missing:
        raise ValueError(f"Missing geography dictionary columns: {sorted(missing)}")
    text = _normalize(f"{title or ''} {snippet or ''}")
    regions: dict[str, tuple[str, list[str]]] = {}
    municipalities: dict[str, tuple[pd.Series, list[str]]] = {}
    municipality_mentions: dict[tuple[int, int], dict[str, tuple[pd.Series, list[str]]]] = {}
    checked_regions: set[str] = set()
    collision_aliases = {_normalize(alias) for alias in AUDITED_PERSON_COLLISION_ALIASES}
    rejected_collisions: set[str] = set()
    for _, row in dictionary.iterrows():
        for alias in row["municipality_aliases"]:
            for match in _alias_pattern(alias).finditer(text):
                if _normalize(alias) in collision_aliases and not _collision_alias_is_qualified(text, match.span()):
                    rejected_collisions.add(str(alias))
                    continue
                mention = municipality_mentions.setdefault(match.span(), {})
                key = str(row["municipality_id"])
                if key not in mention:
                    mention[key] = (row, [])
                mention[key][1].append(alias)
        region_id = str(row["region_id"])
        if region_id not in checked_regions:
            checked_regions.add(region_id)
            region_matches = [alias for alias in row["region_aliases"] if _matches(text, alias)]
            if region_matches:
                regions[region_id] = (str(row["region"]), region_matches)
    # A full official name resolves its contained short-name occurrence; a
    # separate short-name occurrence elsewhere in the text stays ambiguous.
    maximal_mentions = {
        span: choices for span, choices in municipality_mentions.items()
        if not any(other != span and other[0] <= span[0] and other[1] >= span[1] for other in municipality_mentions)
    }
    for choices in maximal_mentions.values():
        for key, (row, aliases) in choices.items():
            if key not in municipalities:
                municipalities[key] = (row, [])
            municipalities[key][1].extend(aliases)
    evidence = ";".join(
        [f"region:{key}:{'|'.join(value[1])}" for key, value in sorted(regions.items())]
        + [f"municipality:{key}:{'|'.join(value[1])}" for key, value in sorted(municipalities.items())]
        + [f"rejected_unqualified_person_collision_alias_v1:{alias}" for alias in sorted(rejected_collisions)]
    )
    if len(regions) > 1:
        return _result("unknown", "ambiguous_multiple_regions", evidence=evidence)
    if municipalities:
        candidates = municipalities
        if regions:
            region_id = next(iter(regions))
            # Regional context may disambiguate a homonym, but it may not
            # erase a separately named municipality from another region.
            if any(not any(str(value[0]["region_id"]) == region_id for value in choices.values()) for choices in maximal_mentions.values()):
                return _result("unknown", "conflicting_region_municipality", evidence=evidence)
            candidates = {key: value for key, value in municipalities.items() if str(value[0]["region_id"]) == region_id}
            if not candidates:
                return _result("unknown", "conflicting_region_municipality", evidence=evidence)
        if len(candidates) != 1:
            return _result("unknown", "ambiguous_municipality_name", evidence=evidence)
        municipality_id, (row, aliases) = next(iter(candidates.items()))
        official_name = any(_normalize(alias) == _normalize(row["municipality_name"]) for alias in aliases)
        rule = "exact_historical_municipality_name" if official_name else "exact_historical_municipality_alias"
        if any(_normalize(alias) in collision_aliases for alias in aliases):
            rule += "_qualified_person_collision_v1"
        if regions:
            rule += "_with_explicit_region"
        return _result("municipality", rule, region=str(row["region"]), region_id=str(row["region_id"]), municipality_id=municipality_id, confidence=0.99 if regions else 0.95 if official_name else 0.90, evidence=evidence)
    if regions:
        region_id, (region, aliases) = next(iter(regions.items()))
        official = any(_normalize(alias) == _normalize(region) for alias in aliases)
        return _result("region", "exact_historical_region_name" if official else "audited_region_alias_v1", region=region, region_id=region_id, confidence=0.95 if official else 0.85, evidence=evidence)
    national = re.search(r"(?<!\w)(?:в россии|по россии|в российской федерации|по всей стране|по всей россии|в целом по стране|российская экономика|экономика россии|на федеральном уровне|общероссийск\w*)(?!\w)", text)
    if national:
        return _result("national", "explicit_national_context_v1", region="Россия", region_id="RU", confidence=0.90, evidence=";".join(part for part in (evidence, national.group()) if part))
    monetary = re.search(r"(?<!\w)(?:ключев\w*\s+ставк\w*|денежно-кредитн\w*\s+политик\w*)(?!\w)", text)
    if _normalize(source) == "cbr" and monetary:
        return _result("national", "cbr_explicit_national_monetary_policy_v1", region="Россия", region_id="RU", confidence=0.95, evidence=";".join(part for part in (evidence, f"source:cbr;{monetary.group()}") if part))
    if rejected_collisions:
        return _result("unknown", "unqualified_person_collision_alias_v1", evidence=evidence)
    return _result("unknown", "no_exact_geography_match_v1")
