"""Prompt templates for research pipeline."""

from typing import List, Dict


def get_topic_breakdown_prompt(query: str, num_topics: int) -> str:
    """Generate prompt for breaking query into sub-topics"""
    return f"""Break this research question into {num_topics} specific sub-topics:

Question: "{query}"

RULES:
1. Each sub-topic MUST include the main subject from the query.
2. Be SPECIFIC and SEARCHABLE (good for Google Search).
3. Cover different aspects of the question.
4. Generate EXACTLY {num_topics} topics, one per line
5. NO numbering, NO markdown formatting

Example Input: "How does quantum computing work?"
Example Output:
Quantum computing basic principles and qubits
Quantum algorithms and quantum gates explained
Current quantum computers and their applications

Now generate {num_topics} sub-topics for: "{query}"
"""


def get_reasoning_prompt(query: str) -> str:
    """Generate prompt for explaining research strategy."""
    return f"""In 1-2 sentences, explain the research strategy for: "{query}"

Focus on what angles we'll explore and why."""


def get_refinement_prompt(query: str, current_topics: List[str], feedback: str, num_topics: int) -> str:
    """Generate prompt for refining research plan based on feedback"""
    topics_list = '\n'.join(f"- {t}" for t in current_topics)
    
    return f"""Refine this research plan based on user feedback.

Original Query: "{query}"

Current Sub-topics:
{topics_list}

User Feedback: "{feedback}"

Generate {num_topics} improved sub-topics that address the feedback.
- One per line
- NO numbering
- NO markdown formatting
- Be specific and searchable"""


def get_reflection_prompt(topic: str, parent_query: str, context: str, searches: List[str], num_chunks: int) -> str:
    """Generate prompt for research reflection and decision-making"""
    return f"""Analyze research progress on: "{topic}"

Parent Question: "{parent_query}"

Retrieved Content ({num_chunks} chunks):
{context}

Previous Searches: {', '.join(searches)}

Evaluate the quality of retrieved information and decide next steps.

Return valid JSON with this structure:
{{
    "facts_learned": ["fact1", "fact2"],
    "gaps": ["missing info 1", "missing info 2"],
    "confidence": 0.0-1.0,
    "continue_research": true|false,
    "next_query": "specific search query to fill gaps"
}}

Rules:
- confidence should reflect content quality and completeness
- If chunks are insufficient or low quality, continue_research should be true
- next_query should target identified gaps
- If no new info in last 2 searches, stop"""


def get_summarization_prompt(topic: str, parent_query: str, facts: List[str], sources: List[str]) -> str:
    """Generate prompt for final summary of research findings"""
    facts_text = '\n'.join(f"- {f}" for f in facts[:20])
    sources_text = '\n'.join(sources)
    
    return f"""Write a focused research summary about: "{topic}"

This is part of a larger report on: "{parent_query}"

Key Information Found:
{facts_text}

Sources Available for Citation:
{sources_text}

Write a clear, factual summary (3-5 sentences) that:
1. States WHAT was found (specific facts, numbers, names)
2. CITES sources using [1], [2], [3] format for ALL factual claims
3. Presents information DIRECTLY - no meta-commentary
4. Acknowledges different viewpoints if they exist
5. Uses CONCRETE language - no "unclear" or "under investigation"

CRITICAL: Write in a UNIFIED voice. This will be integrated into a larger report.

CITATION REQUIREMENTS:
- Every factual claim MUST have a citation: "MC Lyte was a pioneer [1]"
- Use ONLY the numbered sources above [1], [2], [3]...
- Multiple sources: "This is supported by research [1, 2, 3]"
- DO NOT write without citations

WRONG (no citations):
- "MC Lyte was a pioneer in hip hop"

RIGHT (with citations):
- "MC Lyte [1] was a pioneer in hip hop [1, 2]"
- "Queen Latifah [3] and MC Lyte [1] were pioneers [1, 3]"

Write the summary now (3-5 sentences with citations):"""


def get_followup_topics_prompt(query: str, completed: List[Dict], gaps: List[str], avg_confidence: float) -> str:
    """Generate prompt for identifying research gaps and new topics"""
    completed_text = '\n'.join(f"- {r['topic']}" for r in completed)
    gaps_text = '\n'.join(f"- {g}" for g in gaps[:5])
    
    return f"""Research on "{query}" is incomplete (Confidence: {avg_confidence:.2f}).
    
Identified Gaps:
{gaps_text}

Current Topics Covered:
{completed_text}

Generate 3 NEW, specific search topics to fill these gaps.
Do not repeat covered topics.
Output exactly 3 lines, no numbering."""


def get_synthesis_prompt(query: str, findings_by_topic: str, source_list: str) -> str:
    """Generate prompt for final report synthesis."""
    num_sources = len([line for line in source_list.split('\n') if line.strip().startswith('[')])
    
    return f"""You are writing a comprehensive research report on: "{query}"

Below are research findings from multiple topics. Each finding already has citations in [N] format.

RESEARCH FINDINGS (already cited):

{findings_by_topic}

SOURCES LIST:

{source_list}

YOUR JOB:
1. Synthesize these findings into ONE cohesive, well-structured report
2. PRESERVE all existing citations [1], [2], [3] from the findings
3. You can reorganize, rewrite, and combine findings for better flow
4. You can add citations to additional claims if needed
5. Remove redundant information but keep all important facts

CITATION RULES:
- Keep all existing citations from findings: if findings say "MC Lyte [1]", preserve [1]
- Use ONLY numbered sources [1] through [{num_sources}]
- Format: [1], [2], [3] or [1, 2, 3] for multiple
- Every factual claim should have a citation
- NEVER use formats like [R1], [Topic 1], or [Research Area]

REPORT STRUCTURE:

1. Executive Summary (2-3 paragraphs)
   - Direct answer to the query
   - Key findings with citations

2. Detailed Analysis (5-8 paragraphs)
   - Synthesize findings by theme (not by topic)
   - Every claim cited

3. Key Findings (3-5 bullet points)
   - Each with citations

4. Different Perspectives (if sources disagree)
   - Present conflicting viewpoints with citations

5. Conclusion (1-2 paragraphs)
   - Summary with citations

WRITING STYLE:
- Write in ONE unified voice
- State facts confidently: "Nicki Minaj holds the record [1]"
- Use specific numbers, names, dates from sources
- Be direct - no "the research shows" or "according to findings"
- NO meta-commentary about the research process

Now write the complete research report (1500-2000 words):"""


def format_sources_for_synthesis(sources: List[Dict], max_sources: int = 40) -> str:
    """Format sources for synthesis prompt with clear numbering"""
    formatted = []
    for s in sources[:max_sources]:
        source_id = s.get('id', 0)
        title = s.get('title', 'Untitled')[:80]
        url = s.get('url', '')
        formatted.append(f"[{source_id}] {title}\n    URL: {url}")
    
    return '\n\n'.join(formatted)


def format_context_chunks(chunks: List[Dict]) -> str:
    """Format retrieved chunks for reflection prompt"""
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        relevance = chunk.get("score", 0)
        content = chunk.get("content", "")[:200]
        context_parts.append(f"[{i}] (relevance: {relevance:.2f})\n{content}")
    
    return '\n\n'.join(context_parts)