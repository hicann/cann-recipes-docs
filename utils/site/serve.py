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

"""Serve the site locally with index.html fallback for document routes."""

import argparse
import functools
import http.server
from pathlib import Path
from urllib.parse import unquote, urlsplit


class SpaHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, repo_root, site_root, **kwargs):
        self.repo_root = Path(repo_root)
        self.site_root = Path(site_root)
        super().__init__(*args, directory=str(self.site_root), **kwargs)

    def _candidate_paths(self, request_path):
        relative_parts = [
            part for part in request_path.split("/") if part not in ("", ".", "..")
        ]
        return (
            self.site_root.joinpath(*relative_parts),
            self.repo_root.joinpath(*relative_parts),
        )

    def translate_path(self, path):
        request_path = unquote(urlsplit(path).path)
        site_path, repo_path = self._candidate_paths(request_path)
        if site_path.exists():
            return str(site_path)
        if repo_path.exists():
            return str(repo_path)
        return str(site_path)

    def send_head(self):
        request_path = unquote(urlsplit(self.path).path)
        site_path, repo_path = self._candidate_paths(request_path)
        is_document_route = not Path(request_path).suffix
        if (
            not site_path.exists()
            and not repo_path.exists()
            and is_document_route
            and "text/html" in self.headers.get("Accept", "text/html")
        ):
            self.path = "/index.html"
        return super().send_head()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    site_root = Path(__file__).resolve().parent
    handler = functools.partial(SpaHandler, repo_root=repo_root, site_root=site_root)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"CANN Recipes Blog: http://127.0.0.1:{args.port}")
    server.serve_forever()
