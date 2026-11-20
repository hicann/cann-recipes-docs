# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

"""Build the static site's local navigation and full-text search indexes."""

import html
import base64
import json
import logging
import os
import re
import sys
from operator import itemgetter

LOGGER = logging.getLogger(__name__)

NS = "cann"
DOCS_REPO = "cann-recipes-docs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
SCRIPTS_DIR = os.path.join(DOCS_ROOT, "utils", "scripts")
SEARCH_INDEX = []

REPO_CONFIG = {
    "Infer": {"repo": DOCS_REPO, "branch": "master", "basePath": "infer"},
    "Train": {"repo": DOCS_REPO, "branch": "master", "basePath": "train"},
    "Embodied_AI": {"repo": DOCS_REPO, "branch": "master", "basePath": "embodied_ai"},
    "CANN Features": {
        "repo": DOCS_REPO,
        "branch": "master",
        "basePath": "cann_features",
        "flat": True,
    },
}
SOURCE_REPO_URL = f"https://gitcode.com/{NS}/{DOCS_REPO}"
STATIC_METADATA_FIELDS = (
    "quantization", "parallelism", "operator", "cannFeatures", "hardware",
    "frameworks", "cover", "llmSpeculativeInference", "multimodalDitCache",
)


def metadata_enum_manifest():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    try:
        from metadata_enums import (CannFeature, Framework, Hardware,
                                    LlmSpeculativeInference, MultimodalDitCache,
                                    Operator, Parallelism, Quantization)
    except Exception as error:
        LOGGER.warning("加载元数据枚举失败：%s", error)
        return {}
    fields = {
        "quantization": Quantization, "parallelism": Parallelism,
        "operator": Operator, "cannFeatures": CannFeature, "hardware": Hardware,
        "frameworks": Framework, "llmSpeculativeInference": LlmSpeculativeInference,
        "multimodalDitCache": MultimodalDitCache,
    }
    return {field: [item.value for item in enum_class] for field, enum_class in fields.items()}


def metadata_label_manifest():
    if SCRIPTS_DIR not in sys.path:
        sys.path.insert(0, SCRIPTS_DIR)
    try:
        from metadata_labels import metadata_label_manifest as labels
        return labels()
    except Exception as error:
        LOGGER.warning("加载元数据标签失败：%s", error)
        return {"fields": {}, "values": {}}


def local_path(repo, relative_path, require_file=True):
    if repo != DOCS_REPO:
        return None
    root = os.path.realpath(DOCS_ROOT)
    candidate = os.path.realpath(os.path.join(root, *relative_path.replace("\\", "/").split("/")))
    if os.path.commonpath((root, candidate)) != root:
        raise RuntimeError(f"Unsafe local path: {relative_path}")
    if require_file and not os.path.isfile(candidate):
        return None
    return candidate


def list_tree(repo, path):
    directory = local_path(repo, path, require_file=False)
    if not directory or not os.path.isdir(directory):
        return []
    result = []
    for entry in os.scandir(directory):
        child_path = f"{path.rstrip('/')}/{entry.name}" if path else entry.name
        result.append({"name": entry.name, "path": child_path,
                       "type": "tree" if entry.is_dir() else "blob"})
    return sorted(result, key=itemgetter("name"))


def list_md_files(repo, dir_path):
    return sorted(item["name"] for item in list_tree(repo, dir_path)
                   if item["type"] == "blob" and item["name"].lower().endswith(".md"))


def walk_model_tree(repo, current_path, sub_name, relative_parts, models):
    reports = list_md_files(repo, f"{current_path}/zh")
    if reports:
        if relative_parts:
            model_name = relative_parts[-1]
            doc_path = f"{sub_name}/{'/'.join(relative_parts)}"
        else:
            # Support source-style subcategories whose documents live directly
            # under <subcategory>/zh, such as train/features/.
            model_name = sub_name
            doc_path = sub_name
        models.append({"name": model_name, "docPath": doc_path,
                       "reports": reports})
    for child in list_tree(repo, current_path):
        if child["type"] == "tree" and child["name"] not in {"zh", "en", "figures"}:
            walk_model_tree(repo, child["path"], sub_name,
                            relative_parts + [child["name"]], models)


def discover_subcategories(repo, base_path):
    subcategories = []
    for sub in list_tree(repo, base_path):
        if sub["type"] != "tree":
            continue
        sub_name = sub["name"]
        models = []

        walk_model_tree(repo, sub["path"], sub_name, [], models)
        duplicates = {m["name"] for m in models if sum(x["name"] == m["name"] for x in models) > 1}
        for model in models:
            if model["name"] in duplicates:
                model["name"] = model["docPath"][len(sub_name) + 1:]
        if models:
            subcategories.append({"name": sub_name, "models": models})
    return subcategories


def extract_title(markdown, fallback):
    content = re.sub(r"^---\s*\n[\s\S]*?\n---\s*\n?", "", markdown, count=1)
    heading = re.search(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", content, re.MULTILINE)
    if not heading:
        return re.sub(r"\.md$", "", fallback, flags=re.IGNORECASE)
    title = re.sub(r"<[^>]+>|[*_`~]+", "", heading.group(1)).strip()
    return title or re.sub(r"\.md$", "", fallback, flags=re.IGNORECASE)


def extract_metadata(markdown):
    match = re.search(r"<!--\s*static-meta\s*([\s\S]*?)-->", markdown, re.IGNORECASE)
    if not match:
        return {}
    try:
        metadata = json.loads(match.group(1).strip())
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid static-meta JSON: {error}") from error
    if not isinstance(metadata, dict):
        raise RuntimeError("static-meta must contain a JSON object")
    return metadata


def discover_cover_image(markdown, source_path):
    """Return a cover marker using image, Mermaid, or the site fallback."""
    source_dir = os.path.dirname(source_path)
    image_refs = \
        re.finditer(
            r"!\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)|<img\b[^>]*\bsrc=['\"]([^'\"]+)['\"]",
            markdown,
            re.IGNORECASE
        )
    for match in image_refs:
        image_ref = (match.group(1) or match.group(2)).strip().strip("<>")
        if re.match(r"^(?:https?:|data:|//)", image_ref, re.IGNORECASE):
            continue
        image_ref = image_ref.split("#", 1)[0].split("?", 1)[0]
        candidate_path = os.path.normpath(os.path.join(source_dir, image_ref.replace("\\", "/")))
        if not re.search(r"\.(?:png|jpe?g|gif|webp|svg)$", candidate_path, re.IGNORECASE):
            continue
        relative_path = os.path.relpath(candidate_path, DOCS_ROOT).replace(os.sep, "/")
        if local_path(DOCS_REPO, relative_path):
            return relative_path

    mermaid = re.search(r"```mermaid\s*\n([\s\S]*?)\n```", markdown, re.IGNORECASE)
    if mermaid:
        encoded = base64.b64encode(mermaid.group(1).strip().encode("utf-8")).decode("ascii")
        return f"mermaid:{encoded}"
    # The UI constructs the final fallback from the category theme, icon,
    # and model name when no image or Mermaid cover is available.
    return ""


def search_text(markdown):
    content = re.sub(r"^---\s*\n[\s\S]*?\n---\s*\n?", "", markdown, count=1)
    content = re.sub(r"<!--([\s\S]*?)-->|<[^>]+>", " ", content)
    content = re.sub(r"!\[([^\]]*)\]\([^)]+\)|\[([^\]]+)\]\([^)]+\)", r" \1\2 ", content)
    content = re.sub(r"^\s{0,3}#{1,6}\s*|^\s*[-+*>]\s*|[`*_~|]+", "", content, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", html.unescape(content)).strip()


def report_entry(category, model, doc_dir, report_file):
    zh_path = f"{doc_dir}/zh/{report_file}" if doc_dir else f"zh/{report_file}"
    source = local_path(DOCS_REPO, zh_path)
    if not source:
        raise RuntimeError(f"Local report file not found: {zh_path}")
    markdown = open(source, encoding="utf-8", errors="replace").read()
    metadata = extract_metadata(markdown)
    title = extract_title(markdown, report_file)
    sidebar = metadata.get("sidebarTitle") or title
    cover_image = discover_cover_image(markdown, zh_path)
    SEARCH_INDEX.append({"category": category, "model": model, "file": report_file,
                         "language": "zh", "sourcePath": zh_path,
                         "text": search_text(markdown)})
    translations = {}
    en_path = f"{doc_dir}/en/{report_file}" if doc_dir else f"en/{report_file}"
    en_source = local_path(DOCS_REPO, en_path)
    if en_source:
        english = open(en_source, encoding="utf-8", errors="replace").read()
        en_meta = extract_metadata(english)
        translations["en"] = {"title": extract_title(english, report_file),
                               "sidebarTitle": en_meta.get("sidebarTitle") or extract_title(english, report_file),
                               "sourcePath": en_path}
        SEARCH_INDEX.append({"category": category, "model": model, "file": report_file,
                             "language": "en", "sourcePath": en_path,
                             "text": search_text(english)})
    entry = {"file": report_file, "title": title, "sidebarTitle": str(sidebar).strip(),
             "sourceRepo": DOCS_REPO, "sourceRepoUrl": SOURCE_REPO_URL,
             "sourcePath": zh_path, "translations": translations}
    if cover_image:
        entry["coverImage"] = cover_image
    entry.update({field: metadata[field] for field in STATIC_METADATA_FIELDS if field in metadata})
    return entry


def main():
    global SEARCH_INDEX
    SEARCH_INDEX = []
    manifest = {"_metadataEnums": metadata_enum_manifest(),
                "_metadataLabels": metadata_label_manifest()}
    for category, config in REPO_CONFIG.items():
        base = config["basePath"]
        LOGGER.info("扫描 %s（目录：%s）", category, base)
        entry = {"ns": NS, "repo": DOCS_REPO, "branch": config["branch"], "basePath": base}
        if config.get("flat"):
            entry["flat"] = True
            entry["reports"] = [report_entry(category, category, base, file)
                                 for file in list_md_files(DOCS_REPO, f"{base}/zh")]
        else:
            entry["subcategories"] = []
            for sub in discover_subcategories(DOCS_REPO, base):
                models = []
                for model in sub["models"]:
                    models.append({"name": model["name"], "docPath": model["docPath"],
                                   "reports": [report_entry(category, model["name"],
                                                             f"{base}/{model['docPath']}", file)
                                                for file in model["reports"]]})
                if models:
                    entry["subcategories"].append({"name": sub["name"], "models": models})
        manifest[category] = entry
    content_dir = os.path.join(SCRIPT_DIR, "content")
    os.makedirs(content_dir, exist_ok=True)
    with open(os.path.join(content_dir, "index.json"), "w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")
    with open(os.path.join(content_dir, "search-index.json"), "w", encoding="utf-8") as output:
        json.dump(SEARCH_INDEX, output, ensure_ascii=False, separators=(",", ":"))
        output.write("\n")
    LOGGER.info("完成：生成索引（%s 条语言版本记录）", len(SEARCH_INDEX))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    main()
