"""Centralized prompt templates for the research system.

This module separates prompt engineering from Python logic,
making prompts easier to:
- Review and modify
- Version control
- A/B test
- Translate
"""

from typing import List, Dict

def get_topic_breakdown_prompt(query: str, num_topics: int) -> str:
    """Generate prompt for breaking query into sub-topics."""
    return f"""Break this research question into {num_topics} specific sub-topics:

Question: "{query}"

RULES:
- Each sub-topic MUST include the main subject from the query
- Be SPECIFIC and SEARCHABLE (good for Google Search)
- Cover different aspects of the question
- Generate EXACTLY {num_topics} topics, one per line
- NO numbering, NO markdown formatting

Example for "How does quantum computing work?":
Quantum computing basic principles and qubits
Quantum algorithms and quantum gates explained
Current quantum computers and their applications

Now generate {num_topics} sub-topics for: "{query}"
"""


def get_reasoning_prompt(query: str) -> str:
    """Generate prompt for explaining research strategy."""
    return f"""In 1-2 sentences, explain the research strategy for: "{query}"

Focus on what angles we'll explore and why."""


def get_refinement_prompt(query: str, current_topics: List[str], 
                         feedback: str, num_topics: int) -> str:
    """Generate prompt for refining research plan based on feedback."""
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


def get_reflection_prompt(topic: str, parent_query: str, context: str, 
                          coverage: Dict, searches: List[str]) -> str:
    """Generate prompt for research reflection and decision-making.
    
    This prompt uses JSON mode for reliable structured output.
    """
    return f"""Analyze research progress on: "{topic}"

Parent Question: "{parent_query}"

Retrieved Content:
{context}

Coverage Analysis:
- Coverage Score: {coverage.get('coverage_score', 0):.1%}
- Answered: {', '.join(coverage.get('answered_aspects', [])) if coverage.get('answered_aspects') else 'None'}
- Missing: {', '.join(coverage.get('missing_aspects', [])) if coverage.get('missing_aspects') else 'None'}

Previous Searches: {', '.join(searches)}

Evaluate quality and decide next steps.

Return valid JSON with this structure:
{{
    "facts_learned": ["fact1", "fact2"],
    "gaps": ["gap1", "gap2"],
    "confidence": 0.0-1.0,
    "content_quality": "high|medium|low",
    "continue_research": true|false,
    "next_query": "better search query"
}}

Rules:
- confidence should reflect coverage and content quality
- If coverage < 50%, continue_research should be true
- next_query should target missing aspects
- If no new info in last 2 searches, stop"""


def get_summarization_prompt(topic: str, parent_query: str, facts: List[str]) -> str:
    """Generate prompt for final summary of research findings."""
    facts_text = '\n'.join(f"- {f}" for f in facts[:20])
    
    return f"""Write a focused research summary about: "{topic}"

This is part of a larger report on: "{parent_query}"

Key Information Found:
{facts_text}

Write a clear, factual summary (3-5 sentences) that:
1. States WHAT was found (specific facts, numbers, names)
2. Presents information DIRECTLY - no meta-commentary
3. Acknowledges different viewpoints if they exist
4. Uses CONCRETE language - no "unclear" or "under investigation"
5. Writes as if YOU discovered this information

CRITICAL: Write in a UNIFIED voice. This will be integrated into a larger report.

WRONG EXAMPLES (never do this):
- "The research summary indicates that vaccines work by..."
- "According to the summary, studies show..."
- "Research Summary: Vaccines work by..."
- "Summary of findings: ..."

RIGHT EXAMPLES (do this):
- "Vaccines work by introducing antigens that trigger..."
- "Studies demonstrate that B cells respond to..."
- "The immune system recognizes pathogens through..."

Write the summary now (3-5 sentences, direct voice):"""

def get_followup_topics_prompt(query: str, completed: List[Dict], 
                              gaps: List[str], avg_confidence: float) -> str:
    """Generate prompt for identifying research gaps and new topics."""
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
    """Generate prompt for final report synthesis with CLEAR citation instructions."""
    
    return f"""You are writing a comprehensive research report on: "{query}"

Below are research findings from multiple sub-topics. Your job is to synthesize this into ONE cohesive report.

═══════════════════════════════════════════════════════════════
RESEARCH FINDINGS BY SUB-TOPIC:
═══════════════════════════════════════════════════════════════

{findings_by_topic}

═══════════════════════════════════════════════════════════════
SOURCES AVAILABLE TO CITE:
═══════════════════════════════════════════════════════════════

{source_list}

═══════════════════════════════════════════════════════════════
CRITICAL CITATION RULES - READ CAREFULLY:
═══════════════════════════════════════════════════════════════

YOU MUST CITE SOURCES USING ONLY THE NUMBERED FORMAT: [1], [2], [3], etc.

CORRECT CITATIONS:
- "Nicki Minaj has four number-one albums [1]."
- "Early pioneers like MC Lyte [2] and Roxanne Shante [3] established..."
- "Studies show conflicting results [1, 2, 3]."

WRONG - NEVER USE THESE:
- "[Topic: Commercial success]" ← NEVER cite "topics"
- "[Source: Website]" ← NEVER cite this way
- "According to research [Topic: ...]" ← NEVER do this
- No citation at all ← ALWAYS cite claims

CITATION REQUIREMENTS:
1. EVERY factual claim MUST have a citation [1], [2], etc.
2. Use ONLY the numbered sources from the list above
3. NEVER invent citation formats - ONLY use [1], [2], [3]...
4. You can cite multiple sources: [1, 2, 3]
5. When sources disagree: "Source [1] claims X, while [2] argues Y"
6. Cite specific sources, not "topics" or "research"

═══════════════════════════════════════════════════════════════
REPORT STRUCTURE:
═══════════════════════════════════════════════════════════════

Write your report with these sections:

1. **Executive Summary** (2-3 paragraphs)
   - Direct answer to the query with citations
   - Key findings with source references

2. **Detailed Analysis** (5-8 paragraphs)
   - In-depth exploration with HEAVY citations
   - Every claim needs [1], [2], etc.
   - Present different perspectives when sources disagree

3. **Key Findings** (3-5 bullet points)
   - Specific, cited facts
   - Each bullet MUST have citations

4. **Different Perspectives** (if sources disagree)
   - Present conflicting viewpoints
   - Cite which source says what

5. **Conclusion** (1-2 paragraphs)
   - Summary with citations
   - Synthesis of findings

═══════════════════════════════════════════════════════════════
WRITING STYLE RULES:
═══════════════════════════════════════════════════════════════

DO THIS:
- Write in ONE unified voice (this is YOUR report)
- State facts confidently: "Minaj holds the record [1]..."
- Use specific numbers, names, dates from sources
- Integrate citations naturally into sentences
- Be ASSERTIVE with proper citations

NEVER DO THIS:
- "The research summary says..." ← Write directly
- "According to findings..." ← Just state the fact with citation
- "Topic: Commercial success shows..." ← NEVER cite topics
- Vague language like "unclear" or "under investigation"
- Missing citations on factual claims

═══════════════════════════════════════════════════════════════

Now write the complete research report (1500-2000 words) with PROPER numbered source citations [1], [2], [3], etc.:"""


def format_sources_for_synthesis(sources: List[Dict], max_sources: int = 40) -> str:
    """Format sources for synthesis prompt with clear numbering."""
    formatted = []
    for s in sources[:max_sources]:
        source_id = s.get('id', 0)
        title = s.get('title', 'Untitled')[:80]
        url = s.get('url', '')
        formatted.append(f"[{source_id}] {title}\n    URL: {url}")
    
    return '\n\n'.join(formatted)


def format_findings_by_topic(completed: List[Dict]) -> str:
    """Format research findings by topic for synthesis.
    
    NOTE: We use 'Sub-topic:' to avoid confusion with citations.
    """
    formatted = []
    for r in completed:
        topic = r['topic']
        findings = r['findings']
        formatted.append(f"Sub-topic: {topic}\n{findings}")
    
    return '\n\n───────────────────────────────────────\n\n'.join(formatted)


def format_context_chunks(chunks: List[Dict]) -> str:
    """Format retrieved chunks for reflection prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks[:5], 1):
        relevance = chunk.get("score", 0)
        content = chunk.get("content", "")[:200]
        context_parts.append(f"[{i}] (relevance: {relevance:.2f})\n{content}")
    
    return '\n\n'.join(context_parts)