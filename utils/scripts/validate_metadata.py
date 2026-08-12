#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate static-meta blocks in all content documents."""

import json
import logging
import re
from pathlib import Path

from metadata_enums import (
    CannFeature,
    CoverType,
    Framework,
    Hardware,
    LlmSpeculativeInference,
    MetadataField,
    MultimodalDitCache,
    Operator,
    Parallelism,
    Quantization,
    enum_values,
)
from metadata_labels import FIELD_LABELS, LABEL_LANGUAGES, VALUE_LABELS


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTENT_ROOTS = ("infer", "train", "embodied_ai", "cann_features")
META_PATTERN = re.compile(r"<!--\s*static-meta\s*\n([\s\S]*?)\n\s*-->")
ALLOWED_FIELDS = enum_values(MetadataField)
ENUM_FIELDS = {
    "quantization": enum_values(Quantization),
    "parallelism": enum_values(Parallelism),
    "operator": enum_values(Operator),
    "cannFeatures": enum_values(CannFeature),
    "hardware": enum_values(Hardware),
    "frameworks": enum_values(Framework),
}
SCOPED_ENUM_FIELDS = {
    "llmSpeculativeInference": {
        "roots": ("infer/llm/",),
        "allowed": enum_values(LlmSpeculativeInference),
    },
    "multimodalDitCache": {
        "roots": ("infer/multimodal/",),
        "allowed": enum_values(MultimodalDitCache),
    },
}
CANN_FEATURES_FIELDS = {"sidebarTitle", "cannFeatures"}
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_label_entry(name, labels):
    errors = []
    if not isinstance(labels, dict):
        return [f"{name} must define localized labels"]
    missing = [lang for lang in LABEL_LANGUAGES if not isinstance(labels.get(lang), str) or not labels[lang].strip()]
    if missing:
        errors.append(f"{name} is missing labels for languages: {missing}")
    return errors


def validate_metadata_labels():
    """Ensure every machine enum has front-end display labels."""

    errors = []
    unknown_field_labels = sorted(set(FIELD_LABELS) - ALLOWED_FIELDS)
    if unknown_field_labels:
        errors.append(f"FIELD_LABELS contains unknown metadata fields: {unknown_field_labels}")
    for field in sorted(ALLOWED_FIELDS):
        errors.extend(validate_label_entry(f"FIELD_LABELS[{field!r}]", FIELD_LABELS.get(field)))

    value_specs = {
        **ENUM_FIELDS,
        **{field: spec["allowed"] for field, spec in SCOPED_ENUM_FIELDS.items()},
        "cover.type": enum_values(CoverType),
    }
    unknown_value_fields = sorted(set(VALUE_LABELS) - set(value_specs))
    if unknown_value_fields:
        errors.append(f"VALUE_LABELS contains unknown enum fields: {unknown_value_fields}")
    for field, allowed_values in value_specs.items():
        labels = VALUE_LABELS.get(field)
        if not isinstance(labels, dict):
            errors.append(f"VALUE_LABELS[{field!r}] must define labels for enum values")
            continue
        missing = sorted(allowed_values - set(labels))
        if missing:
            errors.append(f"VALUE_LABELS[{field!r}] is missing enum values: {missing}")
        unknown = sorted(set(labels) - allowed_values)
        if unknown:
            errors.append(f"VALUE_LABELS[{field!r}] contains unknown enum values: {unknown}")
        for value in sorted(allowed_values & set(labels)):
            errors.extend(validate_label_entry(f"VALUE_LABELS[{field!r}][{value!r}]", labels[value]))
    return errors


def content_documents():
    return sorted(path for name in CONTENT_ROOTS for path in (REPO_ROOT / name).rglob("*.md"))


def paired_language_document(relative):
    """Return the en counterpart for a document under a zh directory."""

    parts = list(relative.parts)
    try:
        language_index = parts.index("zh")
    except ValueError:
        return None
    parts[language_index] = "en"
    return Path(*parts)


def validate_language_metadata(metadata_by_path):
    """Ensure paired zh/en documents share all metadata except sidebarTitle."""

    errors = []
    for zh_path, zh_metadata in sorted(metadata_by_path.items()):
        en_path = paired_language_document(zh_path)
        if en_path is None or en_path not in metadata_by_path:
            continue
        en_metadata = metadata_by_path[en_path]
        zh_comparable = {key: value for key, value in zh_metadata.items() if key != "sidebarTitle"}
        en_comparable = {key: value for key, value in en_metadata.items() if key != "sidebarTitle"}
        if zh_comparable == en_comparable:
            continue
        zh_only = sorted(set(zh_comparable) - set(en_comparable))
        en_only = sorted(set(en_comparable) - set(zh_comparable))
        changed = sorted(
            key
            for key in set(zh_comparable) & set(en_comparable)
            if zh_comparable[key] != en_comparable[key]
        )
        details = []
        if zh_only:
            details.append(f"zh-only fields: {zh_only}")
        if en_only:
            details.append(f"en-only fields: {en_only}")
        if changed:
            details.append(f"different values: {changed}")
        errors.append(
            f"{zh_path} and {en_path}: static-meta differs outside sidebarTitle ({'; '.join(details)})"
        )
    return errors


def validate_array(path, field, value, allowed=None):
    errors = []
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        return [f"{path}: {field} must be an array of non-empty strings"]
    if not value:
        return [f"{path}: {field} must be omitted instead of using an empty array"]
    if len(value) != len(set(value)):
        errors.append(f"{path}: {field} contains duplicate values")
    if allowed is not None:
        unknown = sorted(set(value) - allowed)
        if unknown:
            errors.append(f"{path}: {field} contains unknown values: {unknown}")
        if "none" in allowed and len(value) > 1 and "none" in value:
            errors.append(f"{path}: {field} cannot mix none with other values")
    return errors


def validate_cover(path, cover, markdown):
    if not isinstance(cover, dict):
        return [f"{path}: cover must be an object"]
    cover_type = cover.get("type")
    if cover_type not in enum_values(CoverType):
        return [f"{path}: invalid cover type: {cover_type!r}"]
    keys = set(cover)
    if cover_type == CoverType.IMAGE:
        source = cover.get("source")
        if keys != {"type", "source"} or not isinstance(source, str) or not source.strip():
            return [f"{path}: image cover requires only a non-empty source"]
        if source.startswith("http://"):
            return [f"{path}: remote cover image must use HTTPS: {source}"]
        if not source.startswith("https://"):
            resolved = (REPO_ROOT / path.parent / source.split("?", 1)[0]).resolve()
            if not resolved.is_file():
                return [f"{path}: cover image does not exist: {source}"]
    elif cover_type == CoverType.MERMAID:
        index = cover.get("index")
        diagrams = re.findall(r"^\s*```\s*mermaid\s*$", markdown, re.IGNORECASE | re.MULTILINE)
        if keys != {"type", "index"} or not isinstance(index, int) or not 1 <= index <= len(diagrams):
            return [f"{path}: Mermaid cover index is invalid"]
    elif keys != {"type"}:
        return [f"{path}: {cover_type} cover only accepts the type field"]
    return []


def main():
    documents = content_documents()
    errors = validate_metadata_labels()
    sidebar_titles = {}
    metadata_by_path = {}
    for path in documents:
        markdown = path.read_text(encoding="utf-8")
        matches = list(META_PATTERN.finditer(markdown))
        relative = path.relative_to(REPO_ROOT)
        if len(matches) != 1:
            errors.append(f"{relative}: expected exactly one static-meta block, found {len(matches)}")
            continue
        if not re.search(r"<!--\s*static-meta\s*\n[\s\S]*?\n\s*-->\s*$", markdown):
            errors.append(f"{relative}: static-meta block must be at the end of the document")
        try:
            metadata = json.loads(matches[0].group(1))
        except json.JSONDecodeError as error:
            errors.append(f"{relative}: invalid static-meta JSON: {error}")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"{relative}: static-meta must be an object")
            continue
        metadata_by_path[relative] = metadata
        relative_posix = relative.as_posix()
        if relative_posix.startswith("cann_features/"):
            missing = sorted(CANN_FEATURES_FIELDS - set(metadata))
            forbidden = sorted(set(metadata) - CANN_FEATURES_FIELDS)
            if missing:
                errors.append(f"{relative}: cann_features metadata is missing fields: {missing}")
            if forbidden:
                errors.append(
                    f"{relative}: cann_features only allows sidebarTitle and cannFeatures; found: {forbidden}")
            if "cannFeatures" in metadata and (
                not isinstance(metadata["cannFeatures"], list) or len(metadata["cannFeatures"]) != 1
            ):
                errors.append(f"{relative}: cann_features cannFeatures must contain exactly one own topic")
        unknown = sorted(set(metadata) - ALLOWED_FIELDS)
        if unknown:
            errors.append(f"{relative}: unknown metadata fields: {unknown}")
        sidebar_title = metadata.get("sidebarTitle")
        if not isinstance(sidebar_title, str) or not sidebar_title.strip():
            errors.append(f"{relative}: sidebarTitle is required")
        elif len(sidebar_title) > 24:
            errors.append(f"{relative}: sidebarTitle must not exceed 24 characters")
        else:
            title_key = (relative.parent, sidebar_title.strip())
            duplicate = sidebar_titles.get(title_key)
            if duplicate is not None:
                errors.append(f"{relative}: sidebarTitle duplicates {duplicate}: {sidebar_title.strip()!r}")
            else:
                sidebar_titles[title_key] = relative
        for field, allowed in ENUM_FIELDS.items():
            if field in metadata:
                errors.extend(validate_array(relative, field, metadata[field], allowed))
        for field, spec in SCOPED_ENUM_FIELDS.items():
            applies = relative_posix.startswith(spec["roots"])
            if field not in metadata:
                if applies:
                    errors.append(f"{relative}: {field} is required under {spec['roots']}")
                continue
            if not applies:
                errors.append(f"{relative}: {field} is only allowed under {spec['roots']}")
                continue
            errors.extend(validate_array(relative, field, metadata[field], spec["allowed"]))
        if "cover" in metadata:
            errors.extend(validate_cover(relative, metadata["cover"], markdown))
        if not re.search(r"^#\s+\S", markdown, re.MULTILINE):
            errors.append(f"{relative}: document is missing an H1")

    errors.extend(validate_language_metadata(metadata_by_path))
    if errors:
        logger.error("Metadata validation failed:")
        for error in errors:
            logger.error("  - %s", error)
        return 1
    logger.info("Validated metadata for %s documents", len(documents))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
