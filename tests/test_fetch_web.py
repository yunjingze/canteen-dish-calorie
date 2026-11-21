# -*- coding: utf-8 -*-
"""Tests for the Bing-Images collector (scripts/fetch_web.py).

The collector only has to do two things reliably: read the original image URLs
out of a Bing result page, and never keep a file whose bytes are not an image.
Both are testable without touching the network.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fetch():
    spec = importlib.util.spec_from_file_location("fetch_web", ROOT / "scripts/fetch_web.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_web"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parse_bing_pulls_original_urls(fetch):
    page = (
        '<a class="iusc" m="{&quot;murl&quot;:&quot;https://a.example/1.jpg&quot;,'
        '&quot;turl&quot;:&quot;https://t.example/1.jpg&quot;}"></a>'
        '<a class="iusc" m="{&quot;murl&quot;:&quot;//b.example/2.jpg&quot;}"></a>'
    )
    urls = fetch.parse_bing(page)
    assert urls == ["https://a.example/1.jpg", "https://b.example/2.jpg"]


def test_parse_bing_dedups_and_skips_non_http(fetch):
    page = (
        'murl&quot;:&quot;https://a.example/1.jpg&quot;'
        'murl&quot;:&quot;https://a.example/1.jpg&quot;'
        'murl&quot;:&quot;data:image/png;base64,AAAA&quot;'
        'murl&quot;:&quot;ftp://c.example/3.jpg&quot;'
    )
    assert fetch.parse_bing(page) == ["https://a.example/1.jpg"]


def test_parse_bing_on_empty_page(fetch):
    assert fetch.parse_bing("") == []


def test_looks_like_image_accepts_real_headers(fetch):
    assert fetch.looks_like_image(b"\xff\xd8\xff\xe0" + b"0" * 40)
    assert fetch.looks_like_image(b"\x89PNG\r\n\x1a\n" + b"0" * 40)
    assert fetch.looks_like_image(b"RIFF" + b"\x00" * 4 + b"WEBP" + b"0" * 40)


def test_looks_like_image_rejects_html_and_empty(fetch):
    assert not fetch.looks_like_image(b"<html><body>404</body></html>")
    assert not fetch.looks_like_image(b"")


def test_sources_roundtrip(fetch, tmp_path):
    path = tmp_path / "sources.csv"
    rows = [("0001.jpg", "https://a.example/1.jpg"), ("0002.png", "https://b.example/2.png")]
    fetch.write_sources(path, rows)
    assert fetch.read_sources(path) == rows


def test_sources_missing_file_is_empty(fetch, tmp_path):
    assert fetch.read_sources(tmp_path / "nope.csv") == []


def test_next_index_continues_after_existing_numbers(fetch, tmp_path):
    (tmp_path / "0001.jpg").write_bytes(b"x")
    (tmp_path / "0007.jpg").write_bytes(b"x")
    (tmp_path / "sources.csv").write_bytes(b"file,url\n")
    assert fetch.next_index(tmp_path) == 8
    assert fetch.next_index(tmp_path / "missing") == 1


def test_existing_hashes_reads_only_images(fetch, tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"aaa")
    (tmp_path / "sources.csv").write_bytes(b"file,url\n")
    hashes = fetch.existing_hashes(tmp_path)
    assert len(hashes) == 1


def test_query_templates_cover_every_dish(fetch):
    from foodcls.classes_cn import CN_CLASS_NAMES, CN_ZH

    assert len(CN_CLASS_NAMES) == 20
    for slug in CN_CLASS_NAMES:
        assert slug in CN_ZH
        # 查询只能用菜名本身：加「食堂 快餐」这类语境词会让 Bing 返回无关图
        assert fetch.QUERY_TEMPLATE.format(zh=CN_ZH[slug]) == CN_ZH[slug]
