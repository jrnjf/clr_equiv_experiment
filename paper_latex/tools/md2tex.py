#!/usr/bin/env python3
"""Convert docs/PAPER_DRAFT_REV_B.md into paper_latex/paper.tex.

One-off build helper; the canonical artefact is paper.tex. Run from anywhere:
    python3 paper_latex/tools/md2tex.py
"""
import re, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MD = os.path.join(ROOT, "docs", "PAPER_DRAFT_REV_B.md")
OUT = os.path.join(ROOT, "paper_latex", "paper.tex")

raw = open(MD, encoding="utf-8").read().split("\n")
start = next(i for i, l in enumerate(raw) if l.strip() == "## Abstract")
lines = raw[start:]


def esc_text(t):
    t = t.replace("&", "\\&").replace("%", "\\%").replace("#", "\\#")
    t = t.replace("_", "\\_")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")
    t = t.replace("\u2014", "---").replace("\u00a7", "\\S{}")
    t = t.replace("\u0119", "\\k{e}").replace("\u00e4", '\\"a').replace("\u2020", "\\dag")
    t = t.replace("->", "$\\to$")
    t = re.sub(r"\[(\d+(?:,\d+)*)\]",
               lambda m: "\\cite{" + ",".join("ref" + x for x in m.group(1).split(",")) + "}", t)
    return t


def esc_code(c):
    c = c.replace("\\", "\\textbackslash{}")
    for a, b in [("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("_", "\\_"),
                 ("{", "\\{"), ("}", "\\}"), ("$", "\\$"),
                 ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
        c = c.replace(a, b)
    return c


def smart_quotes(t):
    out, openq, prev = [], True, ""
    for ch in t:
        if ch == '"' and prev != "\\":
            out.append("``" if openq else "''")
            openq = not openq
        else:
            out.append(ch)
        prev = ch
    return "".join(out)


def inline(t):
    t = t.replace("\n", " ")
    store = []
    def stash(prefix, raw):
        store.append((prefix, raw))
        return "\x00%s%d\x00" % (prefix, len(store) - 1)
    # Stash inline code first, then math, so emphasis may span either.
    t = re.sub(r"`([^`]*)`", lambda m: stash("C", m.group(1)), t)
    t = re.sub(r"<(https?://[^>]+)>", lambda m: stash("U", m.group(1)), t)
    t = re.sub(r"\$([^$]*)\$", lambda m: stash("M", m.group(1)), t)
    parts, last = [], 0
    for m in re.finditer(r"\*\*(.+?)\*\*|\*(.+?)\*", t):
        parts.append(smart_quotes(esc_text(t[last:m.start()])))
        if m.group(1) is not None:
            parts.append("\\textbf{" + smart_quotes(esc_text(m.group(1))) + "}")
        else:
            parts.append("\\textit{" + smart_quotes(esc_text(m.group(2))) + "}")
        last = m.end()
    parts.append(smart_quotes(esc_text(t[last:])))
    out = "".join(parts)
    def restore(m):
        prefix, raw = store[int(m.group(1))]
        if prefix == "M":
            return "$" + raw + "$"
        if prefix == "U":
            return "\\url{" + raw + "}"
        return "\\texttt{" + esc_code(raw) + "}"
    return re.sub(r"\x00[MCU](\d+)\x00", restore, out)


def parse_table(rows):
    cells_rows = []
    for r in rows:
        r = r.strip()
        if r.startswith("|"):
            r = r[1:]
        if r.endswith("|"):
            r = r[:-1]
        cells_rows.append([c.strip() for c in r.split("|")])
    header = cells_rows[0]
    body = cells_rows[2:]
    n = len(header)
    maxlen = max((len(c) for row in body for c in row), default=0)
    texty = maxlen > 40
    if texty:
        h0 = header[0].lower() if header else ""
        if h0 == "result":
            first = 0.26
        elif h0 in ("dimension", "arm", "comparison"):
            first = 0.18
        else:
            first = 0.14
        each = (0.91 - first) / (n - 1) if n > 1 else (0.91 - first)
        spec = "@{}>{\\raggedright\\arraybackslash}p{%.3f\\linewidth}" % first
        spec += "".join(">{\\raggedright\\arraybackslash}p{%.3f\\linewidth}" % each for _ in range(n - 1))
        spec += "@{}"
    else:
        spec = "@{}l" + "c" * (n - 1) + "@{}"
    cmp_tab = bool(header) and header[0].lower() == "comparison"
    font = "\\footnotesize" if cmp_tab else "\\small"
    out = ["\\begin{center}", font]
    if cmp_tab:
        out.append("\\setlength{\\tabcolsep}{3pt}")
    out += ["\\begin{tabular}{" + spec + "}", "\\toprule",
            " & ".join("\\textbf{" + inline(c) + "}" for c in header) + " \\\\",
            "\\midrule"]
    for row in body:
        out.append(" & ".join(inline(c) for c in row) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(out)


# ---- block parser ----
blocks, buf = [], []
def flush():
    global buf
    if buf:
        blocks.append(("para", "\n".join(buf)))
        buf = []

j = 0
while j < len(lines):
    s = lines[j].rstrip()
    st = s.strip()
    if st == "":
        flush(); j += 1; continue
    if st == "---":
        flush(); j += 1; continue
    if s.startswith("## "):
        flush(); blocks.append(("h2", s[3:].strip())); j += 1; continue
    if s.startswith("### "):
        flush(); blocks.append(("h3", s[4:].strip())); j += 1; continue
    if s.startswith("!["):
        flush()
        m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", s)
        path = m.group(1)
        k = j + 1
        while k < len(lines) and lines[k].strip() == "":
            k += 1
        cap = []
        while k < len(lines) and lines[k].strip() != "":
            cap.append(lines[k].strip()); k += 1
        blocks.append(("figure", (path, " ".join(cap))))
        j = k; continue
    if s.startswith("```"):
        flush()
        k = j + 1; code = []
        while k < len(lines) and not lines[k].startswith("```"):
            code.append(lines[k]); k += 1
        blocks.append(("code", "\n".join(code)))
        j = k + 1; continue
    if s.startswith("> "):
        flush()
        q = []
        while j < len(lines) and lines[j].startswith("> "):
            q.append(lines[j][2:]); j += 1
        blocks.append(("quote", " ".join(q)))
        continue
    if st.startswith("|"):
        flush()
        rows = []
        while j < len(lines) and lines[j].strip().startswith("|"):
            rows.append(lines[j].strip()); j += 1
        blocks.append(("table", rows))
        continue
    if re.match(r"^\s*\* ", s):
        flush(); items = []
        while j < len(lines):
            cur = lines[j]
            if re.match(r"^\s*\* ", cur):
                items.append([re.sub(r"^\s*\* ", "", cur).strip()]); j += 1
            elif cur.strip() == "":
                break
            elif cur.startswith("  ") and items:
                items[-1].append(cur.strip()); j += 1
            else:
                break
        blocks.append(("ul", items)); continue
    if re.match(r"^\s*\d+\. ", s):
        flush(); items = []
        while j < len(lines):
            cur = lines[j]
            m = re.match(r"^\s*\d+\. (.*)$", cur)
            if m:
                items.append([m.group(1).strip()]); j += 1
            elif cur.strip() == "":
                break
            elif cur.startswith("   ") and items:
                items[-1].append(cur.strip()); j += 1
            else:
                break
        blocks.append(("ol", items)); continue
    buf.append(s); j += 1
flush()

# ---- emit ----
body = []
i = 0
while i < len(blocks):
    kind, val = blocks[i]
    if kind == "h2":
        if val == "Abstract":
            body.append("\\begin{abstract}")
            i += 1
            while i < len(blocks) and blocks[i][0] == "para":
                body.append(inline(blocks[i][1])); body.append("")
                i += 1
            body.append("\\end{abstract}"); body.append("")
            continue
        if val == "References":
            body.append("\\begin{thebibliography}{14}")
            i += 1
            while i < len(blocks) and blocks[i][0] == "para":
                txt = blocks[i][1].strip()
                m = re.match(r"\[(\d+)\]\s+(.*)", txt, re.S)
                num, rest = m.group(1), m.group(2)
                body.append("\\bibitem{ref%s} %s" % (num, inline(rest)))
                body.append("")
                i += 1
            body.append("\\end{thebibliography}"); body.append("")
            continue
        if val.startswith("Appendix"):
            title = re.sub(r"^Appendix\s+[A-Z]\.\s*", "", val)
            body += ["\\appendix", "\\section{" + inline(title) + "}"]
            i += 1; continue
        title = re.sub(r"^\d+\.\s*", "", val)
        body.append("\\section{" + inline(title) + "}")
        i += 1; continue
    if kind == "h3":
        title = re.sub(r"^\d+\.\d+\s*", "", val)
        body.append("\\subsection{" + inline(title) + "}")
        i += 1; continue
    if kind == "para":
        body += [inline(val), ""]; i += 1; continue
    if kind == "figure":
        path, cap = val
        fn = path.split("/")[-1]
        cap = re.sub(r"^\*\*Figure[^*]*\*\*\s*", "", cap)
        cap = re.sub(r"^\*\*[^*]*\*\*\s*", "", cap)
        body += ["\\begin{figure}[htbp]", "\\centering",
                 "\\includegraphics[width=0.92\\linewidth]{figures/" + fn + "}",
                 "\\caption{" + inline(cap) + "}", "\\end{figure}", ""]
        i += 1; continue
    if kind == "table":
        body += [parse_table(val), ""]; i += 1; continue
    if kind == "code":
        body += ["\\begin{verbatim}", val, "\\end{verbatim}", ""]; i += 1; continue
    if kind == "ul":
        body.append("\\begin{itemize}")
        for item in val:
            body.append("\\item " + inline(" ".join(item)))
        body += ["\\end{itemize}", ""]; i += 1; continue
    if kind == "ol":
        body.append("\\begin{enumerate}")
        for item in val:
            body.append("\\item " + inline(" ".join(item)))
        body += ["\\end{enumerate}", ""]; i += 1; continue
    if kind == "quote":
        body += ["\\begin{quote}", inline(val), "\\end{quote}", ""]; i += 1; continue
    i += 1

preamble = r"""\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[margin=1in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{array}
\usepackage{amsmath,amssymb}
\usepackage[font=small,labelfont=bf]{caption}
\usepackage{microtype}
\usepackage[hidelinks]{hyperref}
\usepackage{xcolor}
\hypersetup{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}

\graphicspath{{figures/}}
\setlength{\parskip}{2pt}
\setlength{\parindent}{0pt}
\renewcommand{\arraystretch}{1.12}

\title{Faster, not better: mapping the anytime efficiency\\[2pt]
and boundaries of short-period learning-rate cycles}
\author{Anonymous Authors}
\date{Draft --- Revision B}

\begin{document}
\maketitle

"""
open(OUT, "w", encoding="utf-8").write(preamble + "\n".join(body) + "\n\\end{document}\n")
print("wrote", OUT, "lines:", len(open(OUT).read().split("\n")))
