"""Synthetic classification checks, not measured accuracy on real news."""
import pytest

from sberforecast.news_topics import DIRECTIONS, EVENT_TYPES, TOPICS, classify_document


@pytest.mark.parametrize("title,topic", [
    ("ЦБ принял решение повысить ключевую ставку", "macro_policy"),
    ("Выросла инфляция", "prices_inflation"),
    ("Зарплаты выросли", "income_wages"),
    ("Безработица снизилась", "employment"),
    ("Открытие магазина", "retail_consumption"),
    ("Выросло промышленное производство", "business_activity"),
    ("Введён режим ЧС после наводнения", "disaster_emergency"),
    ("Открытие нового моста", "transport_infrastructure"),
    ("Аномальная жара", "weather_extreme"),
    ("Принят закон о налогах", "regulation"),
    ("Введены санкции", "sanctions_external"),
    ("Мэр сообщил о заседании администрации", "local_government"),
    ("Открылась выставка искусства", "other"),
])
def test_fixed_topic_catalog(title, topic):
    result = classify_document(title)
    assert result["topic"] == topic
    assert 0 <= result["confidence"] <= 1


@pytest.mark.parametrize("title,event", [
    ("Объявлен прогноз", "announcement"),
    ("Принято решение", "decision"),
    ("Зарплаты выросли", "increase"),
    ("Цены снизились", "decrease"),
    ("Введено ограничение", "restriction"),
    ("Субсидии для предприятий", "support"),
    ("Закрытие магазина", "closure"),
    ("Открытие магазина", "opening"),
    ("Авария на производстве", "accident"),
    ("Введён режим ЧС", "emergency"),
    ("Забастовка работников", "strike"),
    ("Дефицит топлива", "shortage"),
    ("Перебои поставок", "disruption"),
    ("Культурная жизнь города", "other"),
])
def test_fixed_event_catalog(title, event):
    assert classify_document(title)["event_type"] == event


@pytest.mark.parametrize("title,direction", [
    ("Зарплаты выросли", "positive"),
    ("Зарплаты снизились", "negative"),
    ("Цены выросли", "negative"),
    ("Цены снизились", "positive"),
    ("Безработица выросла", "negative"),
    ("Безработица снизилась", "positive"),
    ("Количество рабочих мест увеличилось", "positive"),
    ("Количество рабочих мест сократилось", "negative"),
    ("Авария на предприятии", "negative"),
    ("Субсидии для предприятий", "positive"),
    ("Ставка без изменений", "neutral"),
    ("ЦБ сохранил ключевую ставку", "neutral"),
    ("ЦБ повысил ключевую ставку", "unknown"),
    ("ЦБ снизил ключевую ставку", "unknown"),
    ("Культурная жизнь", "unknown"),
])
def test_direction_is_explicit_rule_not_forecast_label(title, direction):
    assert classify_document(title)["direction"] == direction


def test_conflicting_signals_preserved_as_unknown():
    result = classify_document("Зарплаты выросли", "Но затем зарплаты снизились")
    assert result["direction"] == "unknown"
    assert "conflicting" in result["classification_rule"]


def test_title_and_snippet_are_used():
    result = classify_document("Экономический обзор", "Снизилась инфляция")
    assert result["topic"] == "prices_inflation"
    assert result["event_type"] == "decrease"


def test_source_never_supplies_topic_without_text():
    result = classify_document("", source="cbr")
    assert result["topic"] == "other"
    assert result["direction"] == "unknown"
    assert result["confidence"] == 0


def test_mixed_topics_all_audited_with_fixed_priority():
    result = classify_document("ЦБ решил повысить ключевую ставку", "Сохраняется инфляция")
    assert result["topic"] == "macro_policy"
    assert "topic_macro_policy_v1" in result["classification_rule"]
    assert "topic_prices_inflation_v1" in result["classification_rule"]
    assert result["confidence"] <= 0.65


@pytest.mark.parametrize("title", ["Зарплаты не повысили", "Не планируется повышение зарплат", "Рост зарплат отсутствует"])
def test_negated_actions_do_not_become_positive_observations(title):
    result = classify_document(title)
    assert result["direction"] != "positive"


def test_no_substring_keyword_matches_inside_larger_word():
    result = classify_document("Мэрияда и ростовщик")
    assert result["event_type"] == "other"


def test_taxonomy_and_audit_are_stable():
    first = classify_document("ВВЕДЕН РЕЖИМ ЧС", "Сильный снегопад")
    second = classify_document("ВВЕДЕН РЕЖИМ ЧС", "Сильный снегопад")
    assert first == second
    assert first["topic"] in TOPICS
    assert first["event_type"] in EVENT_TYPES
    assert first["direction"] in DIRECTIONS
    assert "v1" in first["classification_rule"]
