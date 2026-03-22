import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from flair.data import Sentence
from flair.models import SequenceTagger
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    DateRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    UrlRecognizer,
)


POLICIES = {
    "exact_match": {
        "gold_dir": "datasets/gold_exact_match",
        "results_dir": "results/exact_match",
    },
    "application_oriented": {
        "gold_dir": "datasets/gold_application_oriented",
        "results_dir": "results/application_oriented",
    },
}

LABEL_PRIORITY = {
    "EMAIL_ADDRESS": 100,
    "IBAN_CODE": 95,
    "IP_ADDRESS": 90,
    "URL": 85,
    "DATE_TIME": 80,
    "PHONE_NUMBER": 70,
    "POSTAL_CODE": 60,
    "PERSON": 50,
    "ORGANIZATION": 40,
    "LOCATION": 30,
}


@dataclass(frozen=True)
class Treffer:
    start: int
    ende: int
    label: str
    source: str
    score: float = 0.0
    from_regex: bool = False
    from_ner: bool = False

    def überschneidet(self, other: "Treffer") -> bool:
        return not (self.ende <= other.start or other.ende <= self.start)

    def länge(self) -> int:
        return self.ende - self.start

    def text(self, original_text: str) -> str:
        return original_text[self.start:self.ende]

    def with_flags(
        self,
        *,
        regex: bool | None = None,
        ner: bool | None = None,
    ) -> "Treffer":
        return replace(
            self,
            from_regex=self.from_regex if regex is None else regex,
            from_ner=self.from_ner if ner is None else ner,
        )


@dataclass(frozen=True)
class GoldSpan:
    start: int
    ende: int
    text: str

    def überschneidet(self, pred: Treffer) -> bool:
        return not (self.ende <= pred.start or pred.ende <= self.start)


@dataclass(frozen=True)
class GoldEntity:
    label: str
    spans: list[GoldSpan]

    def hauptspan(self) -> GoldSpan:
        return self.spans[0]


@dataclass
class EvalCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def precision(self) -> float:
        d = self.tp + self.fp
        return (self.tp / d) if d else 0.0

    def recall(self) -> float:
        d = self.tp + self.fn
        return (self.tp / d) if d else 0.0

    def f1(self) -> float:
        p = self.precision()
        r = self.recall()
        d = p + r
        return (2.0 * p * r / d) if d else 0.0


@dataclass(frozen=True)
class DebugEntry:
    dataset_name: str
    kind: str
    label: str
    start: int
    ende: int
    text: str
    source: str
    pred_label: str | None = None
    pred_start: int | None = None
    pred_ende: int | None = None
    pred_text: str | None = None
    pred_source: str | None = None
    pred_score: float | None = None


class FlairGermanRecognizer(EntityRecognizer):
    ENTITIES = ["PERSON", "ORGANIZATION", "LOCATION"]

    CHECK_LABEL_GROUPS = [
        ({"LOCATION"}, {"LOC", "LOCATION", "GPE"}),
        ({"PERSON"}, {"PER", "PERSON"}),
        ({"ORGANIZATION"}, {"ORG", "ORGANIZATION"}),
    ]

    MODEL_LANGUAGES = {
        "de": "flair/ner-german-large",
    }

    def __init__(
        self,
        supported_language: str = "de",
        model_name: str | None = None,
        supported_entities: list[str] | None = None,
    ) -> None:
        if supported_entities is None:
            supported_entities = self.ENTITIES

        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name="FlairGermanRecognizer",
        )

        resolved_model_name = model_name or self.MODEL_LANGUAGES[supported_language]
        self.model_name = resolved_model_name
        self.tagger = SequenceTagger.load(resolved_model_name)

    def load(self) -> None:
        return None

    def _matches_entity(self, requested_entity: str, flair_label: str) -> bool:
        requested = str(requested_entity).strip().upper()
        predicted = str(flair_label).strip().upper()

        for requested_group, flair_group in self.CHECK_LABEL_GROUPS:
            if requested in requested_group and predicted in flair_group:
                return True

        return False

    def analyze(
        self,
        text: str,
        entities: list[str] | None,
        nlp_artifacts: Any = None,
    ) -> list[RecognizerResult]:
        sentence = Sentence(text)
        self.tagger.predict(sentence)

        requested_entities = entities if entities else self.supported_entities
        results: list[RecognizerResult] = []

        for requested_entity in requested_entities:
            if requested_entity not in self.supported_entities:
                continue

            for span in sentence.get_spans("ner"):
                flair_label = span.labels[0].value

                if not self._matches_entity(requested_entity, flair_label):
                    continue

                results.append(
                    RecognizerResult(
                        entity_type=requested_entity,
                        start=int(span.start_position),
                        end=int(span.end_position),
                        score=float(span.score),
                        analysis_explanation=None,
                        recognition_metadata={
                            "recognizer_name": "FlairGermanRecognizer",
                            "recognizer_identifier": self.id,
                        },
                    )
                )

        return results


def load_mapping(mapping_path: Path) -> dict[str, list[str]]:
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping-Datei nicht gefunden: {mapping_path}")

    payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mapping-Datei muss ein JSON-Objekt sein.")

    normalized_mapping: dict[str, list[str]] = {}

    for gold_label, values in payload.items():
        gold_label_norm = str(gold_label).strip().upper()
        if not gold_label_norm:
            continue

        if not isinstance(values, list):
            raise ValueError(f"Mapping für {gold_label_norm} muss eine Liste sein.")

        normalized_values: list[str] = []
        seen: set[str] = set()

        for value in values:
            norm = str(value).strip().upper()
            if not norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            normalized_values.append(norm)

        normalized_mapping[gold_label_norm] = normalized_values

    return normalized_mapping


def build_nlp_engine():
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {
                "lang_code": "de",
                "model_name": "de_core_news_lg",
            }
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    return provider.create_engine()


def build_registry() -> RecognizerRegistry:
    registry = RecognizerRegistry(supported_languages=["de"])

    registry.add_recognizer(
        FlairGermanRecognizer(
            supported_language="de",
            model_name="flair/ner-german-large",
            supported_entities=["PERSON", "ORGANIZATION", "LOCATION"],
        )
    )
    registry.add_recognizer(EmailRecognizer(supported_language="de"))
    registry.add_recognizer(PhoneRecognizer(supported_language="de"))
    registry.add_recognizer(UrlRecognizer(supported_language="de"))
    registry.add_recognizer(DateRecognizer(supported_language="de"))
    registry.add_recognizer(IbanRecognizer(supported_language="de"))

    return registry


def build_analyzer() -> AnalyzerEngine:
    nlp_engine = build_nlp_engine()
    registry = build_registry()

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["de"],
    )


def infer_source(result: RecognizerResult) -> tuple[str, bool, bool]:
    metadata = getattr(result, "recognition_metadata", {}) or {}
    recognizer_name = str(metadata.get("recognizer_name", "")).lower()
    recognizer_identifier = str(metadata.get("recognizer_identifier", "")).lower()
    combined = f"{recognizer_name} {recognizer_identifier}"

    if "flair" in combined:
        return "ner", False, True

    return "regex", True, False


def normalize_external_label(label: str) -> str:
    return str(label or "").strip().upper()


def normalize_gold_label(label: str) -> str:
    return str(label or "").strip().upper()


def recognizer_result_to_treffer(result: RecognizerResult) -> Treffer:
    source, from_regex, from_ner = infer_source(result)

    return Treffer(
        start=int(result.start),
        ende=int(result.end),
        label=normalize_external_label(str(result.entity_type)),
        source=source,
        score=float(result.score),
        from_regex=from_regex,
        from_ner=from_ner,
    )


def deduplicate_exact(results: list[Treffer]) -> list[Treffer]:
    seen: set[tuple[int, int, str, str]] = set()
    out: list[Treffer] = []

    for item in results:
        key = (item.start, item.ende, item.label, item.source)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)

    return out


def filter_to_mapped_labels(results: list[Treffer], mapping: dict[str, list[str]]) -> list[Treffer]:
    allowed_labels: set[str] = set()

    for pred_labels in mapping.values():
        for pred_label in pred_labels:
            allowed_labels.add(normalize_external_label(pred_label))

    return [item for item in results if item.label in allowed_labels]


def resolve_same_span_label_conflicts(results: list[Treffer]) -> list[Treffer]:
    grouped: dict[tuple[int, int], list[Treffer]] = {}

    for item in results:
        key = (item.start, item.ende)
        grouped.setdefault(key, []).append(item)

    resolved: list[Treffer] = []

    for items in grouped.values():
        best = sorted(
            items,
            key=lambda x: (
                -LABEL_PRIORITY.get(x.label, 0),
                -x.score,
                x.label,
                x.source,
            ),
        )[0]
        resolved.append(best)

    resolved.sort(key=lambda x: (x.start, x.ende, x.label, x.source))
    return resolved


def resolve_overlaps_largest_span_wins(results: list[Treffer]) -> list[Treffer]:
    if not results:
        return []

    candidates = sorted(
        results,
        key=lambda x: (
            -x.länge(),
            -x.score,
            -LABEL_PRIORITY.get(x.label, 0),
            x.start,
            x.ende,
            x.label,
            x.source,
        ),
    )

    selected: list[Treffer] = []

    for candidate in candidates:
        if any(candidate.überschneidet(existing) for existing in selected):
            continue
        selected.append(candidate)

    selected.sort(key=lambda x: (x.start, x.ende, x.label, x.source))
    return selected


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def read_gold(path: Path, mapping: dict[str, list[str]]) -> list[GoldEntity]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entities = payload.get("entities", [])

    out: list[GoldEntity] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        raw_label = entity.get("label")

        if isinstance(raw_label, list):
            if not raw_label:
                continue
            gold_label = normalize_gold_label(str(raw_label[0]))
        else:
            gold_label = normalize_gold_label(str(raw_label))

        if gold_label not in mapping:
            continue

        spans: list[GoldSpan] = []

        start = entity.get("start")
        end = entity.get("end")
        txt = str(entity.get("text", ""))

        if isinstance(start, int) and isinstance(end, int):
            spans.append(GoldSpan(start=int(start), ende=int(end), text=txt))

        alternatives = entity.get("alternatives")
        if isinstance(alternatives, list):
            for alt in alternatives:
                if not isinstance(alt, dict):
                    continue

                alt_start = alt.get("start")
                alt_end = alt.get("end")
                alt_text = str(alt.get("text", ""))

                if not isinstance(alt_start, int) or not isinstance(alt_end, int):
                    continue

                span = GoldSpan(start=int(alt_start), ende=int(alt_end), text=alt_text)
                if span not in spans:
                    spans.append(span)

        if not spans:
            continue

        spans.sort(key=lambda s: (s.start, s.ende))
        out.append(GoldEntity(label=gold_label, spans=spans))

    out.sort(key=lambda g: (g.hauptspan().start, g.hauptspan().ende, g.label))
    return out


def dataset_meta(dataset_name: str) -> tuple[str, str]:
    try:
        index = int(dataset_name.split("_")[-1])
    except Exception:
        return "Unknown", "unknown"

    domains = [
        "Supporttickets",
        "E-Mail",
        "HR-Dokumente",
        "Verträge",
        "Chats",
    ]

    domain_idx = (index - 1) // 6
    within = (index - 1) % 6

    domain = domains[domain_idx] if 0 <= domain_idx < len(domains) else "Unknown"

    if within in (0, 1):
        structure = "structured"
    elif within in (2, 3):
        structure = "regular"
    else:
        structure = "unstructured"

    return domain, structure


def detect_with_existing_analyzer_for_text(
    analyzer: AnalyzerEngine,
    text: str,
    mapping: dict[str, list[str]],
) -> list[Treffer]:
    raw_results = analyzer.analyze(
        text=text,
        language="de",
        entities=None,
    )

    raw_treffer = [recognizer_result_to_treffer(result) for result in raw_results]
    raw_treffer.sort(key=lambda item: (item.start, item.ende, item.label, item.source))

    filtered_treffer = filter_to_mapped_labels(raw_treffer, mapping)
    unique_treffer = deduplicate_exact(filtered_treffer)
    same_span_resolved = resolve_same_span_label_conflicts(unique_treffer)
    final_treffer = resolve_overlaps_largest_span_wins(same_span_resolved)

    return final_treffer


def label_matches(gold_label: str, pred_label: str, mapping: dict[str, list[str]]) -> bool:
    accepted = mapping.get(gold_label, [])
    pred_label_norm = normalize_external_label(pred_label)
    return pred_label_norm in accepted


def evaluate_predictions(
    dataset_name: str,
    preds: list[Treffer],
    golds: list[GoldEntity],
    text: str,
    mapping: dict[str, list[str]],
) -> tuple[EvalCounts, list[DebugEntry]]:
    counts = EvalCounts()
    debug_entries: list[DebugEntry] = []

    matched_pred_indices: set[int] = set()
    partial_pred_indices: set[int] = set()

    for gold in golds:
        exact_match_idx = None
        exact_match_span = None

        for gold_span in gold.spans:
            for pred_idx, pred in enumerate(preds):
                if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
                    continue
                if not label_matches(gold.label, pred.label, mapping):
                    continue
                if pred.start == gold_span.start and pred.ende == gold_span.ende:
                    exact_match_idx = pred_idx
                    exact_match_span = gold_span
                    break
            if exact_match_idx is not None:
                break

        if exact_match_idx is not None and exact_match_span is not None:
            counts.tp += 1
            matched_pred_indices.add(exact_match_idx)

            pred = preds[exact_match_idx]
            debug_entries.append(
                DebugEntry(
                    dataset_name=dataset_name,
                    kind="TP",
                    label=gold.label,
                    start=exact_match_span.start,
                    ende=exact_match_span.ende,
                    text=exact_match_span.text or text[exact_match_span.start:exact_match_span.ende],
                    source="gold",
                    pred_label=pred.label,
                    pred_start=pred.start,
                    pred_ende=pred.ende,
                    pred_text=pred.text(text),
                    pred_source=pred.source,
                    pred_score=pred.score,
                )
            )
            continue

        partial_match_idx = None
        partial_match_span = None

        for gold_span in gold.spans:
            for pred_idx, pred in enumerate(preds):
                if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
                    continue
                if not label_matches(gold.label, pred.label, mapping):
                    continue
                if not (pred.ende <= gold_span.start or gold_span.ende <= pred.start):
                    partial_match_idx = pred_idx
                    partial_match_span = gold_span
                    break
            if partial_match_idx is not None:
                break

        if partial_match_idx is not None and partial_match_span is not None:
            counts.fp += 1
            counts.fn += 1
            partial_pred_indices.add(partial_match_idx)

            pred = preds[partial_match_idx]
            debug_entries.append(
                DebugEntry(
                    dataset_name=dataset_name,
                    kind="FP/FN",
                    label=gold.label,
                    start=partial_match_span.start,
                    ende=partial_match_span.ende,
                    text=partial_match_span.text or text[partial_match_span.start:partial_match_span.ende],
                    source="gold",
                    pred_label=pred.label,
                    pred_start=pred.start,
                    pred_ende=pred.ende,
                    pred_text=pred.text(text),
                    pred_source=pred.source,
                    pred_score=pred.score,
                )
            )
            continue

        hauptspan = gold.hauptspan()
        counts.fn += 1
        debug_entries.append(
            DebugEntry(
                dataset_name=dataset_name,
                kind="FN",
                label=gold.label,
                start=hauptspan.start,
                ende=hauptspan.ende,
                text=hauptspan.text or text[hauptspan.start:hauptspan.ende],
                source="gold",
            )
        )

    for pred_idx, pred in enumerate(preds):
        if pred_idx in matched_pred_indices or pred_idx in partial_pred_indices:
            continue

        counts.fp += 1
        debug_entries.append(
            DebugEntry(
                dataset_name=dataset_name,
                kind="FP",
                label=pred.label,
                start=pred.start,
                ende=pred.ende,
                text=pred.text(text),
                source=pred.source,
                pred_label=pred.label,
                pred_start=pred.start,
                pred_ende=pred.ende,
                pred_text=pred.text(text),
                pred_source=pred.source,
                pred_score=pred.score,
            )
        )

    debug_entries.sort(key=lambda x: (x.start, x.ende, x.kind, x.label))
    return counts, debug_entries


def format_dataset_block(
    dataset_name: str,
    domain: str,
    policy_name: str,
    counts: EvalCounts,
) -> str:
    return (
        f"{dataset_name:<12} | "
        f"{domain:<14} | "
        f"{policy_name:<20} | "
        f"TP={counts.tp:>3} FP={counts.fp:>3} FN={counts.fn:>3} | "
        f"P={counts.precision():.3f} R={counts.recall():.3f} F1={counts.f1():.3f}"
    )


def format_debug_block(
    dataset_name: str,
    domain: str,
    policy_name: str,
    counts: EvalCounts,
    golds: list[GoldEntity],
    debug_entries: list[DebugEntry],
    text: str,
) -> str:
    lines: list[str] = []

    lines.append(f"DATASET: {dataset_name}")
    lines.append(f"DOMAIN: {domain}")
    lines.append("")
    lines.append(f"POLICY {policy_name}")
    lines.append(f"TP={counts.tp} FP={counts.fp} FN={counts.fn} | P={counts.precision():.3f} R={counts.recall():.3f} F1={counts.f1():.3f}")
    lines.append("")
    lines.append("TARGET LABELS")
    lines.append("----------------------------------------------------------------------")

    for g in golds:
        if len(g.spans) == 1:
            span = g.spans[0]
            shown_text = span.text or text[span.start:span.ende]
            lines.append(f"{g.label:10s} {span.start:4d}:{span.ende:<4d} '{shown_text}'")
        else:
            first = g.spans[0]
            first_text = first.text or text[first.start:first.ende]
            alt_parts: list[str] = []

            for alt in g.spans[1:]:
                alt_text = alt.text or text[alt.start:alt.ende]
                alt_parts.append(f"{alt.start}:{alt.ende} '{alt_text}'")

            lines.append(
                f"{g.label:10s} {first.start:4d}:{first.ende:<4d} '{first_text}' | ALT: " + " OR ".join(alt_parts)
            )

    lines.append("")
    lines.append("EVALUATION")
    lines.append("----------------------------------------------------------------------")

    if not debug_entries:
        lines.append("Keine.")
    else:
        for entry in debug_entries:
            if entry.kind == "TP":
                lines.append(
                    f"TP      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}' "
                    f"| pred {entry.pred_label:14s} {entry.pred_start:4d}:{entry.pred_ende:<4d} '{entry.pred_text}'"
                )
            elif entry.kind == "FN":
                lines.append(
                    f"FN      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}'"
                )
            elif entry.kind == "FP/FN":
                lines.append(
                    f"FP/FN   {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}' "
                    f"| pred {entry.pred_label:14s} {entry.pred_start:4d}:{entry.pred_ende:<4d} '{entry.pred_text}'"
                )
            elif entry.kind == "FP":
                lines.append(
                    f"FP      {entry.label:10s} {entry.start:4d}:{entry.ende:<4d} '{entry.text}'"
                )

    lines.append("")
    lines.append("======================================================================")
    lines.append("")

    return "\n".join(lines)


def format_label_report_summary(
    label_entries: dict[str, dict[str, list[DebugEntry]]],
    policy_name: str,
) -> str:
    lines: list[str] = []
    lines.append(f"LABEL REPORT | POLICY: {policy_name}")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("")
    lines.append("LABEL      | TP | PARTIAL | FN | FP")
    lines.append("------------------------------------")

    ordered_labels = sorted(label_entries.keys())

    for label in ordered_labels:
        tp_count = len(label_entries[label]["TP"])
        partial_count = len(label_entries[label]["FP/FN"])
        fn_count = len(label_entries[label]["FN"])
        fp_count = len(label_entries[label]["FP"])

        if tp_count == 0 and partial_count == 0 and fn_count == 0 and fp_count == 0:
            continue

        lines.append(
            f"{label:<10} | {tp_count:>2} | {partial_count:>7} | {fn_count:>2} | {fp_count:>2}"
        )

    lines.append("")
    return "\n".join(lines)


def format_label_report_debug(
    label_entries: dict[str, dict[str, list[DebugEntry]]],
    policy_name: str,
) -> str:
    lines: list[str] = []
    lines.append(f"LABEL REPORT | POLICY: {policy_name}")
    lines.append("--------------------------------------------------------------------------------")
    lines.append("")

    ordered_labels = sorted(label_entries.keys())

    for label in ordered_labels:
        tp_entries = sorted(label_entries[label]["TP"], key=lambda x: (x.dataset_name, x.start, x.ende))
        partial_entries = sorted(label_entries[label]["FP/FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fn_entries = sorted(label_entries[label]["FN"], key=lambda x: (x.dataset_name, x.start, x.ende))
        fp_entries = sorted(label_entries[label]["FP"], key=lambda x: (x.dataset_name, x.start, x.ende))

        tp_count = len(tp_entries)
        partial_count = len(partial_entries)
        fn_count = len(fn_entries)
        fp_count = len(fp_entries)

        if tp_count == 0 and partial_count == 0 and fn_count == 0 and fp_count == 0:
            continue

        lines.append(label)
        lines.append("--------------------------------------------------------------------------------")
        lines.append(f"TP={tp_count} | PARTIAL={partial_count} | FN={fn_count} | FP={fp_count}")
        lines.append("")

        if tp_count > 0:
            lines.append(f"ERKANNT (TP): {tp_count}")
            for entry in tp_entries:
                pred_source = entry.pred_source or "?"
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [{pred_source}] '{entry.text}'"
                )
            lines.append("")

        if partial_count > 0:
            lines.append(f"NICHT VOLLSTÄNDIG ERKANNT (PARTIAL): {partial_count}")
            for entry in partial_entries:
                pred_source = entry.pred_source or "?"
                pred_label = entry.pred_label or "?"
                pred_start = entry.pred_start if entry.pred_start is not None else -1
                pred_ende = entry.pred_ende if entry.pred_ende is not None else -1
                pred_text = entry.pred_text or ""
                lines.append(
                    f"  - {entry.dataset_name} gold {entry.start}:{entry.ende} [gold] '{entry.text}'"
                    f" | pred {pred_label} {pred_start}:{pred_ende} [{pred_source}] '{pred_text}'"
                )
            lines.append("")

        if fn_count > 0:
            lines.append(f"NICHT ERKANNT (FN): {fn_count}")
            for entry in fn_entries:
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [gold] '{entry.text}'"
                )
            lines.append("")

        if fp_count > 0:
            lines.append(f"UNERWARTET (FP): {fp_count}")
            for entry in fp_entries:
                lines.append(
                    f"  - {entry.dataset_name} {entry.start}:{entry.ende} [{entry.source}] '{entry.text}'"
                )
            lines.append("")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def warmup(analyzer: AnalyzerEngine, warmup_text: str, mapping: dict[str, list[str]]) -> None:
    _ = detect_with_existing_analyzer_for_text(analyzer, warmup_text, mapping)


def run_policy(
    policy_name: str,
    data_dir: Path,
    gold_dir: Path,
    results_dir: Path,
    analyzer: AnalyzerEngine,
    mapping: dict[str, list[str]],
) -> None:
    default_dir = results_dir / "default"
    debug_dir = results_dir / "debug"

    default_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    output_path = default_dir / "Results.txt"
    label_output_path = default_dir / "Label_Results.txt"
    debug_output_path = debug_dir / "Results_debug.txt"
    label_debug_output_path = debug_dir / "Label_Results_debug.txt"

    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold directory nicht gefunden: {gold_dir}")

    text_files = sorted(data_dir.glob("Dataset_*.txt"))
    if not text_files:
        raise FileNotFoundError(f"Keine Dataset-Textdateien gefunden in: {data_dir}")

    global_counts = EvalCounts()
    output_blocks: list[str] = []
    debug_blocks: list[str] = []

    label_entries: dict[str, dict[str, list[DebugEntry]]] = {
        label: {"TP": [], "FP/FN": [], "FN": [], "FP": []}
        for label in sorted(mapping.keys())
    }

    print(f"Starte Batch-Evaluation für Policy: {policy_name}")

    for idx, text_file in enumerate(text_files, start=1):
        dataset_name = text_file.stem
        gold_file = gold_dir / f"{dataset_name}.json"

        if not gold_file.exists():
            raise FileNotFoundError(f"Passende Gold-Datei fehlt: {gold_file}")

        text = read_text(text_file)
        gold_entities = read_gold(gold_file, mapping)
        preds = detect_with_existing_analyzer_for_text(analyzer, text, mapping)

        counts, debug_entries = evaluate_predictions(
            dataset_name=dataset_name,
            preds=preds,
            golds=gold_entities,
            text=text,
            mapping=mapping,
        )

        global_counts.tp += counts.tp
        global_counts.fp += counts.fp
        global_counts.fn += counts.fn

        for entry in debug_entries:
            if entry.label in label_entries and entry.kind in label_entries[entry.label]:
                label_entries[entry.label][entry.kind].append(entry)

        domain, _structure = dataset_meta(dataset_name)

        output_blocks.append(
            format_dataset_block(
                dataset_name=dataset_name,
                domain=domain,
                policy_name=policy_name,
                counts=counts,
            )
        )

        debug_blocks.append(
            format_debug_block(
                dataset_name=dataset_name,
                domain=domain,
                policy_name=policy_name,
                counts=counts,
                golds=gold_entities,
                debug_entries=debug_entries,
                text=text,
            )
        )

        print(f"[{idx:02d}/{len(text_files):02d}] {policy_name} | {dataset_name} fertig")

    summary_lines = [
        "",
        "GLOBAL SUMMARY (micro-averaged over all datasets)",
        "----------------------------------------------------------------------",
        f"POLICY {policy_name:<18} | TP={global_counts.tp:4d} FP={global_counts.fp:4d} FN={global_counts.fn:4d} | P={global_counts.precision():.3f} R={global_counts.recall():.3f} F1={global_counts.f1():.3f}",
        "",
    ]

    final_output_lines = [
        "DATASET      | DOMAIN         | POLICY               | COUNTS                          | METRICS",
        "---------------------------------------------------------------------------------------------",
        *output_blocks,
        *summary_lines,
    ]
    output_path.write_text("\n".join(final_output_lines).rstrip() + "\n", encoding="utf-8")

    final_debug_output = "".join(debug_blocks) + "\n".join(summary_lines)
    debug_output_path.write_text(final_debug_output, encoding="utf-8")

    label_report_summary = format_label_report_summary(label_entries, policy_name=policy_name)
    label_output_path.write_text(label_report_summary, encoding="utf-8")

    label_report_debug = format_label_report_debug(label_entries, policy_name=policy_name)
    label_debug_output_path.write_text(label_report_debug, encoding="utf-8")

    print(f"Ergebnisdatei geschrieben: {output_path}")
    print(f"Label-Datei geschrieben: {label_output_path}")
    print(f"Debug-Datei geschrieben: {debug_output_path}")
    print(f"Label-Debug-Datei geschrieben: {label_debug_output_path}")


def main() -> None:
    data_dir = Path("datasets/data")
    mapping_path = Path("mapping_presidio.json")

    mapping = load_mapping(mapping_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory nicht gefunden: {data_dir}")

    print("Initialisiere Presidio + Flair + spaCy einmalig ...")
    init_start = time.perf_counter()
    analyzer = build_analyzer()
    init_end = time.perf_counter()

    text_files = sorted(data_dir.glob("Dataset_*.txt"))
    if not text_files:
        raise FileNotFoundError(f"Keine Dataset-Textdateien gefunden in: {data_dir}")

    first_text = read_text(text_files[0])
    print("Führe Warmup-Run aus ...")
    warmup_start = time.perf_counter()
    warmup(analyzer, first_text, mapping)
    warmup_end = time.perf_counter()

    print(f"Initialisierung: {(init_end - init_start):.3f} s")
    print(f"Warmup: {(warmup_end - warmup_start):.3f} s")

    for policy_name, policy_cfg in POLICIES.items():
        gold_dir = Path(policy_cfg["gold_dir"])
        results_dir = Path(policy_cfg["results_dir"])

        run_policy(
            policy_name=policy_name,
            data_dir=data_dir,
            gold_dir=gold_dir,
            results_dir=results_dir,
            analyzer=analyzer,
            mapping=mapping,
        )


if __name__ == "__main__":
    main()