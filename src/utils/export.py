"""Export research reports from structured JSON to various formats."""

import re
import os
from typing import Dict, List, Set
from datetime import datetime
import json


def export_to_json(result: Dict, output_path: str) -> bool:
    """Export research result as JSON.
    
    Args:
        result: Research result dictionary
        output_path: Where to save JSON file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"❌ JSON export failed: {e}")
        return False


def extract_citations_from_text(text: str) -> Set[int]:
    """Extract all citation IDs from text, handling both [1] and [1, 2, 3] formats.
    
    Examples:
        "[1]" -> {1}
        "[1, 2, 3]" -> {1, 2, 3}
        "[1][2][3]" -> {1, 2, 3}
        "text [1, 2] more [3]" -> {1, 2, 3}
    
    Args:
        text: Text containing citations
    
    Returns:
        Set of unique citation IDs
    """
    cited_ids = set()
    
    citation_patterns = re.findall(r'\[([0-9,\s]+)\]', text)
    
    for pattern in citation_patterns:
        ids = pattern.split(',')
        for id_str in ids:
            id_str = id_str.strip()
            if id_str.isdigit():
                cited_ids.add(int(id_str))
    
    return cited_ids


def export_to_markdown_from_json(result: Dict, output_path: str) -> bool:
    """Export research result as Markdown from structured JSON.
    
    Args:
        result: Research result dictionary with:
            - query: str
            - report_text: str
            - sources: List[Dict] (with 'id', 'title', 'url')
            - citations: List[int]
            - quality_metrics: Dict
            - timestamp: str
        output_path: Where to save Markdown file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        query = result.get("query", "Research Report")
        report_text = result.get("report_text", "")
        sources = result.get("sources", [])
        quality = result.get("quality_metrics", {})
        timestamp = result.get("timestamp", datetime.now().isoformat())
        
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        cited_ids = extract_citations_from_text(report_text)

        md = f"""# Research Report

**Query:** {query}  
**Generated:** {formatted_time}  
**Confidence:** {quality.get('confidence', 0):.1%}  
**Sources Found:** {quality.get('source_count', 0)}  
**Sources Cited:** {len(cited_ids)}  

---

"""
        
        md += report_text.strip()
        
        md += "\n\n---\n\n## References\n\n"
        
        source_map = {s.get('id'): s for s in sources}
        
        for source_id in sorted(cited_ids):
            if source_id in source_map:
                source = source_map[source_id]
                title = source.get('title', 'Untitled')
                url = source.get('url', '#')
                md += f"[{source_id}] **{title}**  \n    {url}\n\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md)
        
        return True
    
    except Exception as e:
        print(f"Markdown export failed: {e}")
        return False


def export_to_html_from_json(result: Dict, output_path: str) -> bool:
    """Export research result as HTML from structured JSON.
    
    Args:
        result: Research result dictionary
        output_path: Where to save HTML file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        query = result.get("query", "Research Report")
        report_text = result.get("report_text", "")
        sources = result.get("sources", [])
        quality = result.get("quality_metrics", {})
        timestamp = result.get("timestamp", datetime.now().isoformat())
        
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_time = timestamp
        
        html_report = _markdown_to_html(report_text)
        
        cited_ids = extract_citations_from_text(report_text)
        
        source_map = {s.get('id'): s for s in sources}
        
        def expand_citations(match):
            ids_str = match.group(1)
            ids = [id.strip() for id in ids_str.split(',') if id.strip().isdigit()]
            
            if len(ids) == 1:
                return match.group(0)
            
            return ''.join(f'[{id}]' for id in ids)
        
        html_report = re.sub(r'\[([0-9,\s]+)\]', expand_citations, html_report)
        
        def replace_citation(match):
            source_id = int(match.group(1))
            if source_id in source_map:
                source = source_map[source_id]
                url = source.get("url", "#")
                title = source.get("title", "Source")
                title = title.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                return f'<a href="{url}" target="_blank" class="citation" title="{title}">[{source_id}]</a>'
            return match.group(0)
        
        html_report = re.sub(r'\[(\d+)\]', replace_citation, html_report)
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Research Report - {query}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 40px 20px;
            color: #1f2937;
            background: #ffffff;
        }}
        
        h1 {{
            font-size: 2.5rem;
            color: #111827;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 3px solid #6366f1;
        }}
        
        h2 {{
            font-size: 1.75rem;
            color: #1f2937;
            margin-top: 32px;
            margin-bottom: 16px;
        }}
        
        h3 {{
            font-size: 1.35rem;
            color: #374151;
            margin-top: 24px;
            margin-bottom: 12px;
        }}
        
        .meta {{
            margin-bottom: 32px;
            padding-bottom: 16px;
            border-bottom: 1px solid #e5e7eb;
            font-size: 0.95rem;
            line-height: 1.8;
            color: #4b5563;
        }}
        
        .meta-item {{
            margin-bottom: 8px;
        }}
        
        .meta-label {{
            font-weight: 600;
            color: #4b5563;
        }}
        
        .meta-value {{
            color: #1f2937;
        }}
        
        .report {{
            margin: 32px 0;
        }}
        
        p {{
            margin: 16px 0;
            text-align: justify;
        }}
        
        .citation {{
            color: #6366f1;
            text-decoration: none;
            font-weight: 600;
            padding: 2px 4px;
            border-radius: 3px;
            transition: background-color 0.2s;
            white-space: nowrap;
        }}
        
        .citation:hover {{
            background-color: #eef2ff;
            text-decoration: underline;
        }}
        
        .sources {{
            margin-top: 48px;
            padding-top: 32px;
            border-top: 2px solid #e5e7eb;
        }}
        
        .sources h2 {{
            font-size: 1.5rem;
            margin-bottom: 24px;
            color: #111827;
        }}
        
        .source-item {{
            margin-bottom: 16px;
            padding: 8px 0;
            transition: all 0.2s;
            display: flex;
            align-items: baseline;
            gap: 8px;
        }}
        
        .source-item:hover {{
            transform: translateX(4px);
        }}
        
        .source-id {{
            font-weight: 700;
            color: #6366f1;
            min-width: 32px;
            flex-shrink: 0;
        }}
        
        .source-title {{
            font-weight: 500;
            color: #1f2937;
        }}
        
        .source-item a {{
            color: #1f2937;
            text-decoration: none;
            transition: color 0.2s;
        }}
        
        .source-item a:hover {{
            color: #6366f1;
            text-decoration: underline;
        }}
        
        strong {{
            color: #111827;
            font-weight: 600;
        }}
        
        em {{
            font-style: italic;
            color: #4b5563;
        }}
        
        ul, ol {{
            margin: 16px 0;
            padding-left: 32px;
        }}
        
        li {{
            margin: 8px 0;
        }}
        
        @media print {{
            body {{
                max-width: 100%;
                padding: 20px;
            }}
            
            .citation {{
                color: #000;
            }}
            
            a {{
                text-decoration: none;
                color: #000;
            }}
        }}
        
        @media (max-width: 768px) {{
            body {{
                padding: 20px 16px;
            }}
            
            h1 {{
                font-size: 2rem;
            }}
            
            h2 {{
                font-size: 1.5rem;
            }}
            
            .meta {{
                padding: 16px;
            }}
            
            .source-item {{
                flex-direction: column;
                align-items: flex-start;
                gap: 4px;
            }}
        }}
    </style>
</head>
<body>
    <h1>Research Report</h1>
    
    <div class="meta">
        <div class="meta-item">
            <span class="meta-label">Query:</span>
            <span class="meta-value">{query}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Generated:</span>
            <span class="meta-value">{formatted_time}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Confidence:</span>
            <span class="meta-value">{quality.get('confidence', 0):.1%}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Sources Found:</span>
            <span class="meta-value">{quality.get('source_count', 0)}</span>
        </div>
        <div class="meta-item">
            <span class="meta-label">Sources Cited:</span>
            <span class="meta-value">{len(cited_ids)}</span>
        </div>
    </div>
    
    <div class="report">
        {html_report}
    </div>
    
    <div class="sources">
        <h2>References</h2>
"""
        
        for source_id in sorted(cited_ids):
            if source_id in source_map:
                source = source_map[source_id]
                title = source.get('title', 'Untitled')
                url = source.get('url', '#')
                
                html += f"""        <div class="source-item">
            <span class="source-id">[{source_id}]</span>
            <span class="source-title"><a href="{url}" target="_blank">{title}</a></span>
        </div>
"""
        
        html += """    </div>
</body>
</html>"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return True
    
    except Exception as e:
        print(f"HTML export failed: {e}")
        return False


def _markdown_to_html(text: str) -> str:
    """Convert basic markdown formatting to HTML.
    
    Supports:
    - Headers (## Header)
    - Bold (**text**)
    - Italic (*text*)
    - Paragraphs
    - Lists (- item)
    """
    lines = text.split('\n')
    html_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        
        if not line:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append('')
            continue
        
        if line.startswith('###'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h3>{line[3:].strip()}</h3>')
        elif line.startswith('##'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h2>{line[2:].strip()}</h2>')
        elif line.startswith('#'):
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<h1>{line[1:].strip()}</h1>')
        
        elif line.startswith('- ') or line.startswith('* '):
            if not in_list:
                html_lines.append('<ul>')
                in_list = True
            html_lines.append(f'<li>{line[2:].strip()}</li>')
        
        else:
            if in_list:
                html_lines.append('</ul>')
                in_list = False
            html_lines.append(f'<p>{line}</p>')
    
    if in_list:
        html_lines.append('</ul>')
    
    html = '\n'.join(html_lines)
    
    html = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', html)
    html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'(?<!\*)\*([^\*]+?)\*(?!\*)', r'<em>\1</em>', html)
    
    return html


def generate_markdown(query: str, report: str, sources: List[Dict]) -> str:
    """Legacy function - converts old format to new format and exports.
    
    DEPRECATED: Use export_to_markdown_from_json instead.
    """
    result = {
        "query": query,
        "report_text": report,
        "sources": [{"id": i, **s} for i, s in enumerate(sources, 1)],
        "citations": list(range(1, len(sources) + 1)),
        "quality_metrics": {},
        "timestamp": datetime.now().isoformat()
    }
    
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as f:
        tmp_path = f.name
    
    export_to_markdown_from_json(result, tmp_path)
    
    with open(tmp_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    os.unlink(tmp_path)
    return content