import re
import sys
import time
from dataclasses import dataclass, asdict

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

PERIODIC_TABLE_URL = "https://en.wikipedia.org/wiki/Periodic_table"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MPI Lab3 Script; +https://github.com)"
}

EXCLUDE_SECTION_TITLES = {
    "See also",
    "References",
    "Notes",
    "External links",
    "Further reading",
    "Ссылки",
    "Примечания"
}

SYMBOL_PATTERN = re.compile(r"^[A-Z][a-z]$")
HEADING_TAGS = {"h2", "h3", "h4", "h5", "h6"}


@dataclass
class ElementPageResult:
    symbol: str
    url: str
    self_count: int
    all_counts: dict


def fetch_html(url: str, session: requests.Session) -> str:
    response = session.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.text


def parse_periodic_table(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="mw-content-text")
    if content is None:
        raise RuntimeError("Unable to find periodic table main content.")

    symbols = {}
    for a in content.find_all("a", href=True):
        text = a.get_text(strip=True)
        if SYMBOL_PATTERN.match(text):
            href = a["href"]
            if href.startswith("/wiki/") and ":" not in href:
                symbols[text] = "https://en.wikipedia.org" + href
    result = sorted(symbols.items(), key=lambda item: item[0])
    return result


def has_class(tag, class_name: str) -> bool:
    classes = tag.get("class") or []
    return class_name in classes


def normalize_heading(text: str) -> str:
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def find_main_content(soup: BeautifulSoup):
    content = soup.find(id="mw-content-text")
    if content is not None:
        candidates = content.find_all("div", class_="mw-parser-output")
        if not candidates and content.get_text(strip=True):
            return content
    else:
        candidates = soup.find_all("div", class_="mw-parser-output")

    candidates = [tag for tag in candidates if tag.get_text(strip=True)]
    if candidates:
        return max(candidates, key=lambda tag: len(tag.get_text(" ", strip=True)))
    if content is not None:
        return content
    raise RuntimeError("Unable to find main page text.")


def extract_heading_text(tag) -> str | None:
    if tag.name in HEADING_TAGS:
        return tag.get_text(separator=" ", strip=True)
    if has_class(tag, "mw-heading"):
        heading = tag.find(HEADING_TAGS, recursive=False)
        if heading is not None:
            return heading.get_text(separator=" ", strip=True)
    if tag.name == "section":
        heading = tag.find(HEADING_TAGS, recursive=False)
        if heading is not None:
            return heading.get_text(separator=" ", strip=True)
    return None


def is_excluded_section_heading(tag) -> bool:
    headline = extract_heading_text(tag)
    if headline is None:
        return False

    normalized = normalize_heading(headline)
    excluded_titles = {normalize_heading(title) for title in EXCLUDE_SECTION_TITLES}
    return any(
        normalized == title or normalized.startswith(f"{title} ")
        for title in excluded_titles
    )


def count_symbol_occurrences(text: str, symbol: str) -> int:
    pattern = re.compile(
        rf"(?<![A-Za-z]){re.escape(symbol)}(?=[A-Z0-9\s,.;:()\[\]{{}}+\-–—/]|$)"
    )
    return len(pattern.findall(text))


def extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    parser_output = find_main_content(soup)

    for wrapper in list(parser_output.find_all(["meta", "link"])):
        wrapper.unwrap()

    for unwanted in parser_output.select(
        "style, script, noscript, #toc, .toc, .vector-toc, .mw-editsection"
    ):
        unwanted.decompose()

    visible_parts = []
    for child in parser_output.children:
        if getattr(child, "name", None) and is_excluded_section_heading(child):
            break
        visible_parts.append(child)

    text = " ".join(part.get_text(separator=" ", strip=True) for part in visible_parts)
    return text


def count_symbols(text: str, symbols: list[str]) -> dict[str, int]:
    counts = {}
    for symbol in symbols:
        counts[symbol] = count_symbol_occurrences(text, symbol)
    return counts


def split_work(items: list, parts: int) -> list[list]:
    return [items[i::parts] for i in range(parts)]


def compute_results(symbol_url_pairs: list[tuple[str, str]], all_symbols: list[str]) -> list[ElementPageResult]:
    session = requests.Session()
    results: list[ElementPageResult] = []
    for symbol, url in symbol_url_pairs:
        html = fetch_html(url, session)
        text = extract_main_text(html)
        symbol_count = count_symbol_occurrences(text, symbol)
        counts = count_symbols(text, all_symbols)
        results.append(ElementPageResult(symbol=symbol, url=url, self_count=symbol_count, all_counts=counts))
    return results


def merge_results(gathered: list[list[ElementPageResult]]) -> list[ElementPageResult]:
    merged: list[ElementPageResult] = []
    for sublist in gathered:
        merged.extend(sublist)
    return merged


def format_top_results(results: list[tuple[str, int]]) -> str:
    lines = []
    for rank, (symbol, value) in enumerate(results, start=1):
        lines.append(f"{rank}. {symbol}: {value}")
    return "\n".join(lines)


def run_mpi() -> None:
    if MPI is None:
        raise RuntimeError("mpi4py is not installed. Install mpi4py and run with mpiexec.")
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("Required libraries requests and beautifulsoup4 are not installed.")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = time.perf_counter()
    symbol_url_pairs = None

    if rank == 0:
        html = fetch_html(PERIODIC_TABLE_URL, requests.Session())
        symbol_url_pairs = parse_periodic_table(html)
        if not symbol_url_pairs:
            raise RuntimeError("Не найдены ссылки на элементы на странице периодической таблицы.")

    symbol_url_pairs = comm.bcast(symbol_url_pairs, root=0)
    all_symbols = [symbol for symbol, _ in symbol_url_pairs]
    split = split_work(symbol_url_pairs, size)
    local_pairs = split[rank]

    local_results = compute_results(local_pairs, all_symbols)
    gathered = comm.gather(local_results, root=0)

    if rank == 0:
        merged = merge_results(gathered)
        elapsed = time.perf_counter() - start_time

        self_top5 = sorted(merged, key=lambda item: item.self_count, reverse=True)[:5]
        total_counts: dict[str, int] = {symbol: 0 for symbol in all_symbols}
        for item in merged:
            for symbol, count in item.all_counts.items():
                total_counts[symbol] += count

        global_top5 = sorted(total_counts.items(), key=lambda item: item[1], reverse=True)[:5]

        print("MPI variant 5: two-letter element symbol page analysis")
        print(f"Processes: {size}")
        print(f"Element pages: {len(merged)}")
        print(f"Total elapsed time: {elapsed:.3f} seconds")
        print("\nTop-5 pages by self symbol occurrences:")
        for rank, item in enumerate(self_top5, start=1):
            print(f"{rank}. {item.symbol} ({item.url}) - {item.self_count}")

        print("\nTop-5 element symbols by total occurrences across all pages:")
        print(format_top_results(global_top5))


if __name__ == "__main__":
    try:
        run_mpi()
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        sys.exit(1)
