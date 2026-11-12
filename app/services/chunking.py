import re
from uuid import uuid4


def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 160) -> list[dict[str, str]]:
    if max_chars < 100:
        raise ValueError("max_chars must be at least 100")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be between 0 and max_chars")
    # Preserve line boundaries so Markdown headings and paragraph structure remain visible to extractors.
    clean = "\n".join(re.sub(r"[\t ]+", " ", line).strip() for line in text.splitlines() if line.strip()).strip()
    if not clean:
        return []
    chunks = []
    start = 0
    while start < len(clean):
        end = min(start + max_chars, len(clean))
        if end < len(clean):
            boundary = clean.rfind(". ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary + 1
        piece = clean[start:end].strip()
        if piece:
            chunks.append({"id": str(uuid4()), "content": piece})
        if end >= len(clean):
            break
        next_start = max(end - overlap_chars, start + 1)
        start = next_start
    return chunks
