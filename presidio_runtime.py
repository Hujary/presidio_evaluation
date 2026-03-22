from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from flair.data import Sentence
from flair.models import SequenceTagger
from presidio_analyzer import (
    AnalyzerEngine,
    EntityRecognizer,
    RecognizerRegistry,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    DateRecognizer,
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    UrlRecognizer,
)


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


def filter_to_mapped_labels(
    results: list[Treffer],
    mapping: dict[str, list[str]],
) -> list[Treffer]:
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


def warmup(
    analyzer: AnalyzerEngine,
    warmup_text: str,
    mapping: dict[str, list[str]],
) -> None:
    _ = detect_with_existing_analyzer_for_text(analyzer, warmup_text, mapping)


def measure_detection_runtime_ms(
    analyzer: AnalyzerEngine,
    text: str,
    mapping: dict[str, list[str]],
    runs: int,
) -> tuple[list[float], int]:
    values: list[float] = []
    last_count = 0

    for _ in range(runs):
        t0 = time.perf_counter_ns()
        treffer = detect_with_existing_analyzer_for_text(analyzer, text, mapping)
        t1 = time.perf_counter_ns()

        values.append((t1 - t0) / 1_000_000.0)
        last_count = len(treffer)

    return values, last_count


def format_dataset_runtime_line(
    dataset_name: str,
    domain: str,
    structure: str,
    char_count: int,
    hit_count: int,
    values_ms: list[float],
) -> str:
    mean_ms = statistics.mean(values_ms) if values_ms else 0.0
    median_ms = statistics.median(values_ms) if values_ms else 0.0
    min_ms = min(values_ms) if values_ms else 0.0
    max_ms = max(values_ms) if values_ms else 0.0
    delta_ms = max_ms - min_ms
    stdev_ms = statistics.pstdev(values_ms) if len(values_ms) > 1 else 0.0
    mean_ms_per_label = (mean_ms / hit_count) if hit_count else 0.0

    return (
        f"{dataset_name:<12} | "
        f"{domain:<13} | "
        f"{structure:<12} | "
        f"chars={char_count:>5} | "
        f"labels={hit_count:>3} | "
        f"mean_ms={mean_ms:>8.3f} | "
        f"median_ms={median_ms:>8.3f} | "
        f"min_ms={min_ms:>8.3f} | "
        f"max_ms={max_ms:>8.3f} | "
        f"delta_ms={delta_ms:>8.3f} | "
        f"stdev_ms={stdev_ms:>8.3f} | "
        f"ms_per_label={mean_ms_per_label:>7.3f}"
    )


def format_global_runtime_summary(
    all_values_ms: list[float],
    per_dataset_means_ms: list[float],
    dataset_count: int,
    total_chars: int,
    total_hits: int,
    runs_per_dataset: int,
) -> str:
    mean_all = statistics.mean(all_values_ms) if all_values_ms else 0.0
    median_all = statistics.median(all_values_ms) if all_values_ms else 0.0
    min_all = min(all_values_ms) if all_values_ms else 0.0
    max_all = max(all_values_ms) if all_values_ms else 0.0
    delta_all = max_all - min_all
    stdev_all = statistics.pstdev(all_values_ms) if len(all_values_ms) > 1 else 0.0

    mean_dataset_mean = statistics.mean(per_dataset_means_ms) if per_dataset_means_ms else 0.0
    median_dataset_mean = statistics.median(per_dataset_means_ms) if per_dataset_means_ms else 0.0
    min_dataset_mean = min(per_dataset_means_ms) if per_dataset_means_ms else 0.0
    max_dataset_mean = max(per_dataset_means_ms) if per_dataset_means_ms else 0.0
    delta_dataset_mean = max_dataset_mean - min_dataset_mean
    stdev_dataset_mean = statistics.pstdev(per_dataset_means_ms) if len(per_dataset_means_ms) > 1 else 0.0

    total_runtime_ms = sum(all_values_ms)
    total_processed_labels_across_runs = total_hits * runs_per_dataset
    average_runtime_per_resolved_label_ms = (
        total_runtime_ms / total_processed_labels_across_runs
        if total_processed_labels_across_runs
        else 0.0
    )

    lines = [
        "",
        "GLOBAL RUNTIME SUMMARY",
        "======================================================================",
        f"datasets={dataset_count}",
        f"runs_per_dataset={runs_per_dataset}",
        f"total_runs={len(all_values_ms)}",
        f"total_chars={total_chars}",
        f"total_resolved_labels={total_hits}",
        "",
        "ALL RUNS",
        f"mean_ms={mean_all:.3f}",
        f"median_ms={median_all:.3f}",
        f"min_ms={min_all:.3f}",
        f"max_ms={max_all:.3f}",
        f"delta_ms={delta_all:.3f}",
        f"stdev_ms={stdev_all:.3f}",
        "",
        "PER-DATASET MEAN RUNTIMES",
        f"mean_ms={mean_dataset_mean:.3f}",
        f"median_ms={median_dataset_mean:.3f}",
        f"min_ms={min_dataset_mean:.3f}",
        f"max_ms={max_dataset_mean:.3f}",
        f"delta_ms={delta_dataset_mean:.3f}",
        f"stdev_ms={stdev_dataset_mean:.3f}",
        "",
        "LABEL-NORMALIZED RUNTIME",
        f"estimated_total_runtime_ms={total_runtime_ms:.3f}",
        f"resolved_labels_across_all_datasets={total_hits}",
        f"resolved_labels_across_all_runs={total_processed_labels_across_runs}",
        f"average_runtime_per_resolved_label_ms={average_runtime_per_resolved_label_ms:.3f}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    data_dir = Path("datasets/data")
    mapping_path = Path("mapping_presidio.json")
    results_dir = Path("results")
    output_path = results_dir / "Times.txt"

    results_dir.mkdir(parents=True, exist_ok=True)

    mapping = load_mapping(mapping_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory nicht gefunden: {data_dir}")

    text_files = sorted(data_dir.glob("Dataset_*.txt"))
    if not text_files:
        raise FileNotFoundError(f"Keine Dataset-Textdateien gefunden in: {data_dir}")

    if args.only:
        wanted = {item.strip() for item in args.only if item.strip()}
        text_files = [p for p in text_files if p.stem in wanted]

    if not text_files:
        raise FileNotFoundError("Nach --only sind keine Datasets mehr übrig.")

    print("Initialisiere Presidio + Flair + spaCy einmalig ...")
    init_start = time.perf_counter()
    analyzer = build_analyzer()
    init_end = time.perf_counter()

    first_text = read_text(text_files[0])
    print("Führe Warmup-Run aus ...")
    warmup_start = time.perf_counter()
    warmup(analyzer, first_text, mapping)
    warmup_end = time.perf_counter()

    print(f"Initialisierung: {(init_end - init_start):.3f} s")
    print(f"Warmup: {(warmup_end - warmup_start):.3f} s")
    print("Starte Runtime-Benchmark ...")

    lines: list[str] = [
        "PRESIDIO RUNTIME REPORT",
        "======================================================================",
        f"runs_per_dataset={max(1, int(args.runs))}",
        "measurement=full detection pipeline until resolved labels",
        "steps=analyze -> recognizer_result_to_treffer -> filter_to_mapped_labels -> deduplicate_exact -> resolve_same_span_label_conflicts -> resolve_overlaps_largest_span_wins",
        "",
        f"initialization_s={(init_end - init_start):.3f}",
        f"warmup_s={(warmup_end - warmup_start):.3f}",
        "",
        "DATASET      | DOMAIN        | STRUCTURE    | CHARS | LABELS |  MEAN_MS | MEDIAN_MS |   MIN_MS |   MAX_MS | DELTA_MS | STDEV_MS | MS_PER_LABEL",
        "-----------------------------------------------------------------------------------------------------------------------------------------------------",
    ]

    all_values_ms: list[float] = []
    per_dataset_means_ms: list[float] = []
    total_chars = 0
    total_hits = 0

    for idx, text_file in enumerate(text_files, start=1):
        dataset_name = text_file.stem
        text = read_text(text_file)
        domain, structure = dataset_meta(dataset_name)

        values_ms, hit_count = measure_detection_runtime_ms(
            analyzer=analyzer,
            text=text,
            mapping=mapping,
            runs=max(1, int(args.runs)),
        )

        total_chars += len(text)
        total_hits += hit_count
        all_values_ms.extend(values_ms)
        per_dataset_means_ms.append(statistics.mean(values_ms) if values_ms else 0.0)

        lines.append(
            format_dataset_runtime_line(
                dataset_name=dataset_name,
                domain=domain,
                structure=structure,
                char_count=len(text),
                hit_count=hit_count,
                values_ms=values_ms,
            )
        )

        print(f"[{idx:02d}/{len(text_files):02d}] {dataset_name} fertig")

    lines.append(
        format_global_runtime_summary(
            all_values_ms=all_values_ms,
            per_dataset_means_ms=per_dataset_means_ms,
            dataset_count=len(text_files),
            total_chars=total_chars,
            total_hits=total_hits,
            runs_per_dataset=max(1, int(args.runs)),
        )
    )

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"Runtime-Datei geschrieben: {output_path}")


if __name__ == "__main__":
    main()