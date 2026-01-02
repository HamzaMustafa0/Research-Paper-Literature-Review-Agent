import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import requests
from tqdm import tqdm

SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
FIELDS = "title,authors,year,abstract,url,citationCount,venue"

MEMORY_PATH = "../memory.json"


@dataclass
class PaperNote:
    paperId: str
    title: str
    year: Optional[int]
    url: Optional[str]
    citationCount: Optional[int]
    venue: Optional[str]
    authors: List[str]
    abstract: Optional[str]
    tags: List[str]
    notes: Dict[str, Any]


def load_memory(path: str = MEMORY_PATH) -> List[PaperNote]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [PaperNote(**x) for x in data]
    except FileNotFoundError:
        return []


def save_memory(memory: List[PaperNote], path: str = MEMORY_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in memory], f, ensure_ascii=False, indent=2)


def semantic_scholar_search(query: str, limit: int = 10, year_from: Optional[int] = None) -> List[Dict[str, Any]]:
    params = {
        "query": query,
        "limit": limit,
        "fields": FIELDS
    }
    if year_from is not None:
        # Semantic Scholar supports year filters via 'year' like "2021-2026"
        params["year"] = f"{year_from}-"

    r = requests.get(SEMANTIC_SCHOLAR_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("data", [])


def score_paper(p: Dict[str, Any]) -> float:
    """
    Simple ranking:
    - citations help
    - recency helps a bit
    - having an abstract helps
    """
    citations = p.get("citationCount") or 0
    year = p.get("year") or 0
    abstract_bonus = 1.0 if p.get("abstract") else 0.0
    # Recency bonus: newer years slightly higher
    recency_bonus = max(0, year - 2018) * 0.2
    return (citations ** 0.5) + recency_bonus + abstract_bonus


def make_notes_from_abstract(p: Dict[str, Any], user_question: str) -> Dict[str, Any]:
    """
    v1: we do "analysis" without an LLM, using a simple template.
    In v2, you’d replace this with an LLM summarizer.
    """
    abstract = (p.get("abstract") or "").strip()
    title = p.get("title") or "Untitled"

    # Very simple heuristics for "tags"
    tags = []
    lower = (title + " " + abstract).lower()
    for kw in ["lstm", "transformer", "neural", "bayesian", "kalman", "physics", "hybrid", "gaussian", "xgboost", "random forest"]:
        if kw in lower:
            tags.append(kw)

    notes = {
        "relevance_to_question": f"Candidate paper for: {user_question}",
        "key_idea_guess": abstract[:400] + ("..." if len(abstract) > 400 else ""),
        "why_it_matters_guess": "Read the full paper for details; abstract suggests relevance.",
        "limitations_guess": "Unknown from abstract alone.",
    }
    return {"tags": tags, "notes": notes}


def add_to_memory(memory: List[PaperNote], paper: Dict[str, Any], user_question: str) -> None:
    paper_id = paper.get("paperId")
    if not paper_id:
        return
    if any(x.paperId == paper_id for x in memory):
        return  # already saved

    authors = [a.get("name", "").strip() for a in (paper.get("authors") or []) if a.get("name")]
    analysis = make_notes_from_abstract(paper, user_question)

    memory.append(
        PaperNote(
            paperId=paper_id,
            title=paper.get("title") or "",
            year=paper.get("year"),
            url=paper.get("url"),
            citationCount=paper.get("citationCount"),
            venue=paper.get("venue"),
            authors=authors,
            abstract=paper.get("abstract"),
            tags=analysis["tags"],
            notes=analysis["notes"],
        )
    )


def synthesize_answer(memory: List[PaperNote], user_question: str, top_n: int = 6) -> str:
    """
    v1 synthesis: uses stored notes + titles/abstract snippets.
    In v2, you’d use an LLM to produce a much better synthesis.
    """
    if not memory:
        return "No papers in memory yet."

    # pick best matches: if question keywords appear in title/abstract
    q = user_question.lower().split()
    def match_score(n: PaperNote) -> int:
        text = (n.title + " " + (n.abstract or "")).lower()
        return sum(1 for w in q if len(w) > 3 and w in text)

    ranked = sorted(memory, key=lambda n: (match_score(n), (n.citationCount or 0), (n.year or 0)), reverse=True)
    picks = ranked[:top_n]

    lines = []
    lines.append(f"Question: {user_question}\n")
    lines.append("What I found (based on abstracts + saved notes):")
    lines.append("")

    for i, p in enumerate(picks, 1):
        author_str = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        cite = f"{author_str} ({p.year})" if p.year else author_str
        lines.append(f"{i}. {p.title}")
        lines.append(f"   - {cite} | citations: {p.citationCount or 0}")
        if p.tags:
            lines.append(f"   - tags: {', '.join(p.tags)}")
        if p.url:
            lines.append(f"   - link: {p.url}")
        idea = (p.notes.get("key_idea_guess") or "").strip()
        if idea:
            lines.append(f"   - abstract snippet: {idea}")
        lines.append("")

    lines.append("Next best upgrade (v2): add an LLM to extract structured info (method, dataset, results, limitations) from PDFs.")
    return "\n".join(lines)


def run_agent(user_question: str, search_limit: int = 12, year_from: Optional[int] = 2021) -> None:
    memory = load_memory()

    print("Searching papers…")
    papers = semantic_scholar_search(user_question, limit=search_limit, year_from=year_from)

    if not papers:
        print("No results found.")
        return

    ranked = sorted(papers, key=score_paper, reverse=True)

    print("Saving top papers to memory…")
    for p in tqdm(ranked[:min(10, len(ranked))]):
        add_to_memory(memory, p, user_question)
        time.sleep(0.8)  # be polite to API

    save_memory(memory)

    print("\n---\n")
    print(synthesize_answer(memory, user_question))


if __name__ == "__main__":
    q = input("Enter your research question: ").strip()
    run_agent(q)
