"""Synthetic territory fixtures; these are not measured real news labels."""
import pandas as pd
import pytest

from sberforecast.news_geo import build_geography_dictionary, resolve_geography


@pytest.fixture
def raw_geography():
    return pd.DataFrame([
        {"territory_id": "m01", "municipal_district_name": "городской округ Майкоп", "municipal_district_name_short": "Майкоп", "region_code": "01", "region_name": "Республика Адыгея", "year_from": 2018, "year_to": 9999},
        {"territory_id": "m02", "municipal_district_name": "Красный муниципальный район", "municipal_district_name_short": "Красный", "region_code": "01", "region_name": "Республика Адыгея", "year_from": 2018, "year_to": 2024},
        {"territory_id": "m02", "municipal_district_name": "Красный муниципальный округ", "municipal_district_name_short": "Красный", "region_code": "01", "region_name": "Республика Адыгея", "year_from": 2024, "year_to": 9999},
        {"territory_id": "m03", "municipal_district_name": "Красный муниципальный район", "municipal_district_name_short": "Красный", "region_code": "50", "region_name": "Московская область", "year_from": 2018, "year_to": 9999},
        {"territory_id": "m04", "municipal_district_name": "городской округ Озёрный", "municipal_district_name_short": "Озёрный", "region_code": "50", "region_name": "Московская область", "year_from": 2018, "year_to": 9999},
        {"territory_id": "m05", "municipal_district_name": "Новая улица муниципальный округ", "municipal_district_name_short": "Новая улица", "region_code": "50", "region_name": "Московская область", "year_from": 2018, "year_to": 9999},
    ])


@pytest.fixture
def geography(raw_geography):
    return build_geography_dictionary(raw_geography)


def test_dictionary_deduplicates_ids_retains_historical_names(raw_geography):
    before = raw_geography.copy(deep=True)
    dictionary = build_geography_dictionary(pd.concat([raw_geography, raw_geography]))
    pd.testing.assert_frame_equal(raw_geography, before)
    assert len(dictionary) == 5
    row = dictionary.set_index("municipality_id").loc["m02"]
    assert row["region_id"] == "1"
    assert "Красный муниципальный район" in row["municipality_aliases"]
    assert "Красный муниципальный округ" in row["municipality_aliases"]
    assert isinstance(row["municipality_aliases"], tuple)


def test_territory_name_prepared_table_is_explicitly_supported():
    dictionary = build_geography_dictionary(pd.DataFrame({"territory_id": ["a"], "territory_name": ["Дальний округ"], "region_code": ["2"], "region_name": ["Регион А"]}))
    result = resolve_geography("Дальний округ", "", dictionary)
    assert result["municipality_id"] == "a"


@pytest.mark.parametrize("column", ["territory_id", "municipal_district_name", "region_code", "region_name"])
def test_missing_required_field_rejected(raw_geography, column):
    with pytest.raises(ValueError, match="Missing geography columns"):
        build_geography_dictionary(raw_geography.drop(columns=column))


def test_conflicting_region_per_id_rejected(raw_geography):
    raw_geography.loc[1, "region_code"] = "50"
    raw_geography.loc[1, "region_name"] = "Московская область"
    with pytest.raises(ValueError, match="Ambiguous municipality"):
        build_geography_dictionary(raw_geography)


def test_conflicting_name_per_region_code_rejected(raw_geography):
    raw_geography.loc[0, "region_name"] = "Другой регион"
    with pytest.raises(ValueError, match="multiple names"):
        build_geography_dictionary(raw_geography)


@pytest.mark.parametrize("bad", [None, "", "  "])
def test_empty_required_value_rejected(raw_geography, bad):
    raw_geography.loc[0, "municipal_district_name"] = bad
    with pytest.raises(ValueError, match="Missing|Empty"):
        build_geography_dictionary(raw_geography)


def test_unique_dictionary_city_resolves_exactly(geography):
    result = resolve_geography("Майкоп: открыли магазин", "", geography)
    assert result["geography_level"] == "municipality"
    assert result["municipality_id"] == "m01"
    assert result["region_id"] == "1"
    assert "Майкоп" in result["geography_evidence"]


def test_ambiguous_city_without_region_is_unknown(geography):
    result = resolve_geography("Красный", "Открыли магазин", geography)
    assert result["geography_level"] == "unknown"
    assert result["municipality_id"] == result["region_id"] == ""
    assert result["geography_rule"] == "ambiguous_municipality_name"


def test_ambiguous_city_with_explicit_region_resolves(geography):
    result = resolve_geography("Красный", "Московская область", geography)
    assert result["geography_level"] == "municipality"
    assert result["municipality_id"] == "m03"
    assert result["region_id"] == "50"
    assert "explicit_region" in result["geography_rule"]


def test_ambiguous_city_with_audited_case_alias_resolves(geography):
    result = resolve_geography("Красный", "Новость из Московской области", geography)
    assert result["municipality_id"] == "m03"
    assert "Московской области" in result["geography_evidence"]


def test_punctuation_case_yo_normalized_without_fuzzy_matching(geography):
    result = resolve_geography("ОЗЕРНЫЙ!", "", geography)
    assert result["municipality_id"] == "m04"
    typo = resolve_geography("Озернный", "", geography)
    assert typo["geography_level"] == "unknown"


def test_short_alias_never_matches_inside_larger_word(geography):
    result = resolve_geography("Майкопский", "Красныйборск", geography)
    assert result["geography_level"] == "unknown"


def test_names_are_full_phrases_whitespace_normalized(geography):
    result = resolve_geography("Новая    улица", "", geography)
    assert result["municipality_id"] == "m05"


def test_region_only_keeps_region_level(geography):
    result = resolve_geography("Обзор: Республика Адыгея", "", geography)
    assert result["geography_level"] == "region"
    assert result["municipality_id"] == ""


def test_multiple_regions_unknown(geography):
    result = resolve_geography("Республика Адыгея и Московская область", "", geography)
    assert result["geography_rule"] == "ambiguous_multiple_regions"


def test_conflicting_city_and_region_unknown(geography):
    result = resolve_geography("Майкоп", "Московская область", geography)
    assert result["geography_rule"] == "conflicting_region_municipality"


def test_multiple_municipalities_same_region_stay_unknown(geography):
    result = resolve_geography("Майкоп и Красный", "Республика Адыгея", geography)
    assert result["geography_level"] == "unknown"


def test_region_context_does_not_erase_separately_named_other_region(geography):
    result = resolve_geography("Майкоп и Озёрный", "Московская область", geography)
    assert result["geography_level"] == "unknown"
    assert result["geography_rule"] == "conflicting_region_municipality"


def test_longer_official_name_disambiguates_contained_short_alias(geography):
    result = resolve_geography("Красный муниципальный округ", "", geography)
    assert result["geography_level"] == "municipality"
    assert result["municipality_id"] == "m02"


def test_separate_ambiguous_mention_not_erased_by_longer_name(geography):
    result = resolve_geography("Красный муниципальный округ", "Другой Красный", geography)
    assert result["geography_level"] == "unknown"


def test_no_national_fallback_without_evidence(geography):
    result = resolve_geography("Данные о зарплатах", "", geography, source="federal_news")
    assert result["geography_level"] == "unknown"


def test_explicit_national_context(geography):
    result = resolve_geography("В России выросли зарплаты", "", geography)
    assert result["geography_level"] == "national"
    assert result["region_id"] == "RU"
    assert result["municipality_id"] == ""


def test_cbr_source_alone_does_not_imply_national(geography):
    result = resolve_geography("Объявление регионального управления", "", geography, "cbr")
    assert result["geography_level"] == "unknown"


def test_cbr_explicit_monetary_context_is_national(geography):
    result = resolve_geography("Совет директоров сохранил ключевую ставку", "", geography, "cbr")
    assert result["geography_level"] == "national"
    assert result["geography_rule"] == "cbr_explicit_national_monetary_policy_v1"


def test_cbr_regional_text_keeps_local_geography(geography):
    result = resolve_geography("Майкоп", "Обзор ключевой ставки региональным управлением", geography, "cbr")
    assert result["municipality_id"] == "m01"


def test_ambiguous_city_is_not_upgraded_by_national_context(geography):
    result = resolve_geography("Красный", "В России: ключевая ставка", geography, "cbr")
    assert result["geography_level"] == "unknown"


def test_row_order_does_not_change_dictionary_or_resolution(raw_geography):
    first = build_geography_dictionary(raw_geography)
    second = build_geography_dictionary(raw_geography.iloc[::-1])
    pd.testing.assert_frame_equal(first, second)
    assert resolve_geography("Красный", "Московской области", first) == resolve_geography("Красный", "Московской области", second)


@pytest.fixture
def person_collision_geography():
    return build_geography_dictionary(pd.DataFrame([
        {"territory_id": "synthetic-vladimir", "municipal_district_name": "городской округ город Владимир", "municipal_district_name_short": "Владимир", "region_code": "33", "region_name": "Владимирская область"},
        {"territory_id": "synthetic-kirov", "municipal_district_name": "городской округ город Киров", "municipal_district_name_short": "Киров", "region_code": "43", "region_name": "Кировская область"},
        {"territory_id": "synthetic-maykop", "municipal_district_name": "городской округ Майкоп", "municipal_district_name_short": "Майкоп", "region_code": "1", "region_name": "Республика Адыгея"},
    ]))


@pytest.mark.parametrize("title,snippet", [
    ("Синтетическое заявление о выборах", "Президент РФ Владимир Путин высказал мнение."),
    ("Синтетический рассказ об интервью", "Президент России Владимир Путин рассказал о беседе."),
    ("Синтетическое сообщение о внешней политике", "По словам президента РФ Владимира: Владимир Путин сделал заявление."),
    ("Синтетическое предложение о безопасности", "Президент России Владимир Путин предложил новый порядок."),
    ("Синтетический комментарий зарубежного обозревателя", "Обозреватель заявил, что Владимир Зеленский сделал заявление."),
])
def test_five_corpus_person_name_failure_shapes_never_assign_city(
    person_collision_geography, title, snippet,
):
    result = resolve_geography(title, snippet, person_collision_geography)
    assert result["geography_level"] == "unknown"
    assert result["municipality_id"] == result["region_id"] == ""
    assert result["geography_rule"] == "unqualified_person_collision_alias_v1"
    assert "rejected_unqualified_person_collision_alias_v1:Владимир" in result["geography_evidence"]


@pytest.mark.parametrize("title", [
    "город Владимир", "В городе Владимир состоялась встреча", "г. Владимир",
    "городской округ Владимир", "городской округ город Владимир",
])
def test_city_wording_qualifies_person_collision_at_its_occurrence(person_collision_geography, title):
    result = resolve_geography(title, "", person_collision_geography)
    assert result["geography_level"] == "municipality"
    assert result["municipality_id"] == "synthetic-vladimir"


@pytest.mark.parametrize("surname", ["Путин", "Зеленский"])
def test_own_region_does_not_turn_person_name_into_city(person_collision_geography, surname):
    result = resolve_geography(f"Владимир {surname}", "Владимирская область", person_collision_geography)
    assert result["geography_level"] == "region"
    assert result["region_id"] == "33"
    assert result["municipality_id"] == ""


def test_other_region_context_is_not_conflicting_with_rejected_person(person_collision_geography):
    result = resolve_geography("Владимир Путин", "Республика Адыгея", person_collision_geography)
    assert result["geography_level"] == "region"
    assert result["region_id"] == "1"
    assert result["municipality_id"] == ""


def test_city_and_person_occurrences_are_resolved_independently(person_collision_geography):
    result = resolve_geography("город Владимир", "Владимир Путин сделал заявление", person_collision_geography)
    assert result["municipality_id"] == "synthetic-vladimir"
    assert "municipality:synthetic-vladimir:Владимир" in result["geography_evidence"]
    assert "rejected_unqualified_person_collision_alias_v1:Владимир" in result["geography_evidence"]


def test_unrelated_city_word_never_qualifies_person_name(person_collision_geography):
    result = resolve_geography("Новый город", "Президент Владимир Путин", person_collision_geography)
    assert result["geography_level"] == "unknown"


def test_noncolliding_unique_city_preserved_alongside_person(person_collision_geography):
    result = resolve_geography("Майкоп", "Владимир Путин сделал заявление", person_collision_geography)
    assert result["municipality_id"] == "synthetic-maykop"


def test_surname_collision_requires_its_own_territorial_qualification(person_collision_geography):
    person = resolve_geography("Сергей Киров", "Кировская область", person_collision_geography)
    assert person["geography_level"] == "region"
    assert person["municipality_id"] == ""
    city = resolve_geography("город Киров", "", person_collision_geography)
    assert city["municipality_id"] == "synthetic-kirov"


def test_bare_collision_alias_stays_unqualified_even_in_its_region(person_collision_geography):
    result = resolve_geography("Владимир", "Владимирская область", person_collision_geography)
    assert result["geography_level"] == "region"
    assert result["municipality_id"] == ""


def test_rejected_person_alias_does_not_block_explicit_national_scope(person_collision_geography):
    result = resolve_geography("В России", "Владимир Путин сделал заявление", person_collision_geography)
    assert result["geography_level"] == "national"
    assert result["municipality_id"] == ""
    assert "rejected_unqualified_person_collision_alias_v1:Владимир" in result["geography_evidence"]
