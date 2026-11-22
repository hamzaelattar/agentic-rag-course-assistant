"""Recherche Web avec une source d'actualités de secours."""

import html
import re
import xml.etree.ElementTree as ET

import requests
from ddgs import DDGS
from ddgs.exceptions import DDGSException


def _news_query(question: str) -> str:
    query = re.sub(r"^recherche(?:r)?(?: sur internet)?\s+", "", question, flags=re.I).strip()
    query = re.sub(
        r"^(?:les?\s+)?(?:dernières?\s+)?(?:nouveautés|actualités|nouvelles|publications)\s+(?:de|du|des|sur)\s+",
        "",
        query,
        flags=re.I,
    ).strip()
    return query or question


def _news_search(question: str, limit: int, recent: bool):
    query = _news_query(question)
    if recent:
        query += " when:30d"

    response = requests.get(
        "https://news.google.com/rss/search",
        params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
        timeout=10,
    )
    response.raise_for_status()
    items = ET.fromstring(response.content).findall("./channel/item")

    return [
        {
            "title": item.findtext("title") or "",
            "url": item.findtext("link") or "",
            "snippet": (
                (item.findtext("pubDate") or "") + " — "
                + html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
            ).strip(),
        }
        for item in items[:limit]
        if item.findtext("link")
    ]


def web_search(question: str, limit: int = 5):
    """Chercher sur le Web avec une source de secours pour l'actualité."""
    print("[TOOL] Recherche Web...")
    recent = bool(re.search(r"derni[eè]res?|nouveaut[eé]s|actualit[eé]s|r[eé]centes?", question, re.I))

    if recent:
        try:
            news = _news_search(question, limit, recent=True)
            if news:
                return news
        except (requests.RequestException, ET.ParseError):
            pass

    try:
        results = DDGS().text(question, max_results=limit)
    except DDGSException:
        results = []

    if results:
        return [
            {"title": item.get("title", ""), "url": item["href"], "snippet": item.get("body", "")}
            for item in results
            if item.get("href")
        ]

    return _news_search(question, limit, recent=False)
