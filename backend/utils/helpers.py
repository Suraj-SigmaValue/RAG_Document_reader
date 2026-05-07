import re

import tiktoken


def count_tokens(text: str, model: str = "text-embedding-ada-002") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def preprocess_for_bm25(text: str):
    return text.lower().split()


def is_table_like(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    if len(lines) < 3:
        return False

    markdown_table = False
    for index, line in enumerate(lines):
        if "|" in line and index + 1 < len(lines):
            next_line = lines[index + 1]
            if "|" in next_line and "---" in next_line:
                markdown_table = True
                break

    table_keywords = [
        "table", "schedule", "statement", "area", "rate", "cost", "amount",
        "fsi", "carpet", "built-up", "premium", "charges", "sr no",
        "description", "occupancy", "regulation"
    ]

    keyword_hit = any(keyword in text.lower() for keyword in table_keywords)
    numeric_lines = sum(1 for line in lines if re.search(r"\d", line))
    numeric_ratio = numeric_lines / len(lines)
    column_like_lines = sum(
        1 for line in lines
        if len(re.split(r"\s{2,}|\t", line)) >= 2
    )

    return markdown_table or keyword_hit or numeric_ratio > 0.35 or column_like_lines >= 3


def is_toc_like(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    toc_line_count = 0
    for line in lines:
        has_section = re.match(r"^\d+(?:\.\d+)*\s+", line)
        ends_with_page = re.search(r"(?:\.{2,}\s*|\s+)\d{1,4}$", line)
        if has_section and ends_with_page:
            toc_line_count += 1

    return (toc_line_count / len(lines)) >= 0.5
