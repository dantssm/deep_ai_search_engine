# src/utils/html.py
"""HTML formatting utilities for research output."""

import html
import re
from typing import List, Dict, Set


def format_answer_html(text: str, sources: List[Dict]) -> str:
    """Format AI answer into HTML with clickable source links."""
    url_map = {i: s.get("url", "#") for i, s in enumerate(sources, 1)}
    source_titles = {i: s.get("title", f"Source {i}") for i, s in enumerate(sources, 1)}
    used_sources: Set[int] = set()

    text = re.sub(r'\.\s+\*\s+', '.\n* ', text)
    text = re.sub(r'\s+\*\s+(?=[A-Z])', '\n* ', text)

    text = re.sub(r'^#{4}\s+(.+)$', r'<h4>\1</h4>', text, flags=re.MULTILINE)
    text = re.sub(r'^#{3}\s+(.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^#{2}\s+(.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^#\s+(.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)

    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'(?<!\n)\*([^\*\n]+?)\*(?!\*)', r'<em>\1</em>', text)

    text = re.sub(r'^\*\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^-\s+(.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'((?:<li>.*?</li>\s*)+)', r'<ul>\1</ul>', text, flags=re.DOTALL)

    source_pattern = re.compile(r'Source\s+(\d+)', re.IGNORECASE)

    def replace_source(match):
        num = int(match.group(1))
        if num in url_map:
            used_sources.add(num)
            url = html.escape(url_map[num])
            return f'<a href="{url}" target="_blank" class="source-link">Source {num}</a>'
        return match.group(0)

    text = source_pattern.sub(replace_source, text)

    result = '<div class="chat-block">'
    for para in text.split('\n\n'):
        para = para.strip()
        if not para:
            continue
        if re.match(r'^<(h[1-6]|ul|ol|li|div|p)', para):
            result += para
        else:
            result += f'<p>{para.replace(chr(10), "<br>")}</p>'

    if used_sources:
        result += '<div class="sources-section"><div class="sources-header">Sources:</div>'
        for num in sorted(used_sources):
            title = source_titles.get(num, f"Source {num}")[:80]
            url = html.escape(url_map[num])
            escaped_title = html.escape(title)
            result += f'<div class="source-item"><span class="source-number">[{num}]</span> '
            result += f'<a href="{url}" target="_blank" class="source-link">{escaped_title}</a></div>'
        result += '</div>'

    return result + '</div>'