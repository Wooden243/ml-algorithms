"""Documentation checks: prevent known GitHub math errors and broken local links."""

from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parents[2]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def test_no_unsupported_github_math_macros():
    """The GitHub renderer rejects some LaTeX commands even when TeX accepts them."""
    for md_file in REPO_ROOT.rglob("*.md"):
        contents = md_file.read_text(encoding="utf-8")
        assert "\\operatorname" not in contents, f"Unsupported math macro: {md_file}"


def test_local_markdown_links_resolve():
    """Check references to files and directories in README and notes."""
    for md_file in REPO_ROOT.rglob("*.md"):
        text = md_file.read_text(encoding="utf-8")
        for target in LINK_RE.findall(text):
            target = target.split("#", 1)[0]
            if not target or target.startswith(("https://", "http://", "mailto:")):
                continue
            assert (md_file.parent / target).exists(), (
                f"Broken local Markdown link in {md_file}: {target}"
            )
