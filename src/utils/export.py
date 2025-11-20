"""Export research reports to Markdown"""

import re
from typing import Dict, Set
from datetime import datetime


def extract_citations_from_text(text: str) -> Set[int]:
    """Extract citation IDs from text like [1], [2, 3], etc."""
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
    """Export research result as Markdown file."""
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