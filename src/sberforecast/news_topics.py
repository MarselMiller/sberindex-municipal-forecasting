"""Deterministic, auditable news topic/event/direction rules for E06b.

Rules use title and permitted snippet only. Rule scores are heuristic match
strengths, not accuracy estimates or calibrated probabilities. Direction is a
textual economic interpretation; it is neither a shock label nor evidence of
a future change in municipal consumption. No model, fitted vocabulary, or
future observation is involved.
"""
from __future__ import annotations

import re
import unicodedata

TOPICS = (
    "macro_policy", "prices_inflation", "income_wages", "employment",
    "retail_consumption", "business_activity", "disaster_emergency",
    "transport_infrastructure", "weather_extreme", "regulation",
    "sanctions_external", "local_government", "other",
)
EVENT_TYPES = (
    "announcement", "decision", "increase", "decrease", "restriction",
    "support", "closure", "opening", "accident", "emergency", "strike",
    "shortage", "disruption", "other",
)
DIRECTIONS = ("positive", "negative", "neutral", "unknown")

# Ordered priority resolves mixed-topic text reproducibly. The ordering is
# fixed before examining any forecast/shock results; every match is audited.
TOPIC_RULES: tuple[tuple[str, str, str], ...] = (
    ("macro_policy", "topic_macro_policy_v1", r"ключев\w*\s+ставк\w*|денежно[- ]кредитн\w*\s+политик\w*|монетарн\w*\s+политик\w*"),
    ("sanctions_external", "topic_sanctions_external_v1", r"санкци\w*|эмбарго\w*|внешнеторгов\w*|экспортн\w*\s+ограничени\w*|импортн\w*\s+ограничени\w*"),
    ("disaster_emergency", "topic_disaster_emergency_v1", r"чрезвычайн\w*\s+ситуаци\w*|режим\w*\s+чс|наводнен\w*|землетрясени\w*|лесн\w*\s+пожар\w*|техногенн\w*|эвакуаци\w*|авари\w*"),
    ("weather_extreme", "topic_weather_extreme_v1", r"аномальн\w*\s+(?:жар\w*|холод\w*|мороз\w*)|сильн\w*\s+(?:снегопад\w*|ливн\w*|ветер\w*)|снегопад\w*|ураган\w*|засух\w*|шторм\w*|ледян\w*\s+дожд\w*"),
    ("prices_inflation", "topic_prices_inflation_v1", r"инфляци\w*|потребительск\w*\s+цен\w*|(?:рост|снижен\w*|повышен\w*)\s+цен\w*|подорож\w*|подешев\w*|цен\w*\s+(?:вырос\w*|сниз\w*|повыс\w*|упал\w*)"),
    ("income_wages", "topic_income_wages_v1", r"зарплат\w*|заработн\w*\s+плат\w*|доход\w*\s+населени\w*|пенси\w*|мрот\w*|социальн\w*\s+выплат\w*"),
    ("employment", "topic_employment_v1", r"безработиц\w*|занятост\w*|рабоч\w*\s+мест\w*|увольнен\w*|ваканси\w*|забастовк\w*|рынок\s+труда"),
    ("retail_consumption", "topic_retail_consumption_v1", r"розничн\w*\s+(?:торговл\w*|продаж\w*|оборот\w*)|потребительск\w*\s+(?:расход\w*|спрос\w*)|потреблени\w*|магазин\w*|торгов\w*\s+центр\w*"),
    ("transport_infrastructure", "topic_transport_infrastructure_v1", r"транспорт\w*|железнодорож\w*|автодорог\w*|дорог\w*|мост\w*|аэропорт\w*|метро\w*|автобус\w*|инфраструктур\w*"),
    ("regulation", "topic_regulation_v1", r"закон\w*|постановлени\w*|регулировани\w*|лицензи\w*|налог\w*|запрет\w*|норматив\w*"),
    ("business_activity", "topic_business_activity_v1", r"предприяти\w*|производств\w*|промышлен\w*|делов\w*\s+активност\w*|инвестици\w*|банкротств\w*|бизнес\w*|завод\w*"),
    ("local_government", "topic_local_government_v1", r"муниципал\w*|администраци\w*|губернатор\w*|мэр\w*|местн\w*\s+власт\w*|городск\w*\s+совет\w*|региональн\w*\s+правительств\w*"),
)

EVENT_RULES: tuple[tuple[str, str, str], ...] = (
    ("emergency", "event_emergency_v1", r"чрезвычайн\w*\s+ситуаци\w*|режим\w*\s+чс|наводнен\w*|землетрясени\w*|эвакуаци\w*|лесн\w*\s+пожар\w*"),
    ("accident", "event_accident_v1", r"авари\w*|катастроф\w*|столкновени\w*|взрыв\w*"),
    ("strike", "event_strike_v1", r"забастовк\w*|басту\w*"),
    ("shortage", "event_shortage_v1", r"дефицит\w*|нехватк\w*|недостаток\w*"),
    ("disruption", "event_disruption_v1", r"сбо\w*|перебо\w*|нарушени\w*\s+(?:поставок|снабжени\w*|движени\w*)|отключени\w*|приостанов\w*"),
    ("restriction", "event_restriction_v1", r"ограничени\w*|запрет\w*|эмбарго\w*|санкци\w*"),
    ("support", "event_support_v1", r"поддержк\w*|субсиди\w*|дотаци\w*|льгот\w*|компенсаци\w*"),
    ("closure", "event_closure_v1", r"закрыти\w*|закрыл\w*|закро\w*|закрыва\w*|ликвидаци\w*|ликвидиров\w*"),
    ("opening", "event_opening_v1", r"открыти\w*|открыл\w*|откро\w*|открыва\w*|ввод\w*\s+в\s+эксплуатаци\w*|введен\w*\s+в\s+эксплуатаци\w*"),
    ("decision", "event_decision_v1", r"решени\w*|утверд\w*|принял\w*|принят\w*|постановлени\w*|одобр\w*"),
    ("announcement", "event_announcement_v1", r"объяв\w*|анонс\w*|сообщ\w*|заяв\w*|планир\w*|намерен\w*|прогноз\w*|ожида\w*"),
    ("increase", "event_increase_v1", r"повыс\w*|повышен\w*|увелич\w*|вырос\w*|возрос\w*|рост(?:а|у|ом|е)?|подорож\w*"),
    ("decrease", "event_decrease_v1", r"сниз\w*|снижен\w*|сократ\w*|сокращен\w*|уменьш\w*|упал\w*|падени\w*|подешев\w*"),
)

_UP = r"повыс\w*|повышен\w*|увелич\w*|вырос\w*|возрос\w*|рост(?:а|у|ом|е)?|подорож\w*"
_DOWN = r"сниз\w*|снижен\w*|сократ\w*|сокращен\w*|уменьш\w*|упал\w*|падени\w*|подешев\w*"
_NEGATIVE = r"авари\w*|катастроф\w*|наводнен\w*|землетрясени\w*|пожар\w*|режим\w*\s+чс|забастовк\w*|дефицит\w*|нехватк\w*|перебо\w*|сбо\w*|отключени\w*|увольнен\w*|банкротств\w*"
_POSITIVE = r"поддержк\w*|субсиди\w*|дотаци\w*|компенсаци\w*|открыти\w*|открыл\w*"
_NEUTRAL = r"без\s+изменени\w*|не\s+измен\w*|сохран\w*\s+(?:ключев\w*\s+)?ставк\w*|остав\w*\s+(?:ключев\w*\s+)?ставк\w*\s+без"


def _normalize(title: str, snippet: str) -> str:
    text = unicodedata.normalize("NFKC", f"{title or ''} {snippet or ''}").casefold().replace("ё", "е")
    text = re.sub(r"[‐‑‒–—−]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _match(pattern: str, text: str) -> re.Match | None:
    return re.search(r"(?<!\w)(?:" + pattern + r")(?!\w)", text)


def _unnegated_match(pattern: str, text: str) -> re.Match | None:
    for match in re.finditer(r"(?<!\w)(?:" + pattern + r")(?!\w)", text):
        # Local explicit negation blocks asserting the named action. This is
        # deliberately conservative, not general-purpose semantic parsing.
        before = text[:match.start()]
        after = text[match.end():]
        preceding_negation = re.search(r"(?<!\w)(?:не|нет|без|отсутстви\w*)\s+(?:\w+\s+){0,2}$", before)
        following_negation = re.search(r"^\s+(?:\w+\s+){0,2}(?:отсутств\w*|не\s+произош\w*)(?!\w)", after)
        if not preceding_negation and not following_negation:
            return match
    return None


def _direction(topic: str, event: str, text: str) -> tuple[str, str]:
    neutral = _match(_NEUTRAL, text)
    up = _unnegated_match(_UP, text)
    down = _unnegated_match(_DOWN, text)
    adverse = _unnegated_match(_NEGATIVE, text)
    beneficial = _unnegated_match(_POSITIVE, text)
    signals: set[str] = set()
    rules: list[str] = []
    if neutral:
        signals.add("neutral")
        rules.append("explicit_unchanged")
    # Monetary tightening/easing does not establish the sign of subsequent
    # spending. In particular, key-rate increases remain direction unknown.
    if topic == "prices_inflation":
        if up:
            signals.add("negative")
            rules.append("prices_increase")
        if down:
            signals.add("positive")
            rules.append("prices_decrease")
    elif topic in {"income_wages", "retail_consumption", "business_activity"}:
        if up:
            signals.add("positive")
            rules.append("income_or_activity_increase")
        if down:
            signals.add("negative")
            rules.append("income_or_activity_decrease")
    elif topic == "employment" and _match(r"безработиц\w*", text):
        if up:
            signals.add("negative")
            rules.append("unemployment_increase")
        if down:
            signals.add("positive")
            rules.append("unemployment_decrease")
    elif topic == "employment" and _match(r"рабоч\w*\s+мест\w*|занятост\w*", text):
        if up:
            signals.add("positive")
            rules.append("employment_increase")
        if down:
            signals.add("negative")
            rules.append("employment_decrease")
    if adverse:
        signals.add("negative")
        rules.append("explicit_adverse_event")
    if beneficial:
        signals.add("positive")
        rules.append("explicit_support_or_opening")
    if event == "restriction":
        # Removal of restrictions is not interpreted as an adverse event.
        if not _match(r"отмен\w*\s+(?:\w+\s+){0,2}(?:ограничени\w*|запрет\w*|санкци\w*)", text):
            signals.add("negative")
            rules.append("explicit_restriction")
    if len(signals) == 1:
        return next(iter(signals)), "direction_v1:" + "+".join(rules)
    if len(signals) > 1:
        return "unknown", "direction_conflicting_text_v1:" + "+".join(rules)
    return "unknown", "direction_no_unambiguous_signal_v1"


def classify_document(title: str, snippet: str = "", source: str = "") -> dict:
    """Return the fixed topic/event/direction schema and rule evidence.

    Source never supplies a topic by itself. Multiple topic/event matches use
    the documented fixed priority and remain visible in the audit string.
    Unmatched/negated actions are ``other``; conflicting signs are ``unknown``.
    """
    text = _normalize(title, snippet)
    topics = [(name, rule, match.group()) for name, rule, pattern in TOPIC_RULES if (match := _match(pattern, text))]
    events = [(name, rule, match.group()) for name, rule, pattern in EVENT_RULES if (match := _unnegated_match(pattern, text))]
    topic = topics[0][0] if topics else "other"
    event = events[0][0] if events else "other"
    direction, direction_rule = _direction(topic, event, text)
    audit = [f"{rule}[{evidence}]" for _, rule, evidence in topics + events]
    audit.append(direction_rule)
    confidence = 0.90 if len(topics) == 1 and events else 0.75 if topics else 0.50 if events else 0.0
    if len(topics) > 1:
        confidence = min(confidence, 0.65)
    return {
        "topic": topic, "event_type": event, "direction": direction,
        "confidence": confidence, "classification_rule": ";".join(audit),
    }
