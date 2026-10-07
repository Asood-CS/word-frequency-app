"""How long until random letters spell a word?

An interactive experiment: keep drawing random letter strings until one is a real
English word, count the failed attempts, and see how that count grows with word length.

Run with:  streamlit run app.py
"""

import csv
import gzip
import json
import math
import re
import string
import urllib.request
from collections import Counter
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
SOURCE_REPO = "https://github.com/CloudBytes-Academy/English-Dictionary-Open-Source"
RAW_BASE = "https://raw.githubusercontent.com/CloudBytes-Academy/English-Dictionary-Open-Source/main/"
# Where each local file lives in the source repository, for downloading it if missing.
SOURCE_PATHS = {
    "dictionary.jsonl.gz": "v2/dictionary.jsonl.gz",
    "dictionary-1913.csv": "csv/dictionary.csv",
    "LICENSE-DATA.md": "v2/LICENSE-DATA.md",
}
DICTIONARIES = {
    "Modern (v2, 2026)": "dictionary.jsonl.gz",
    "1913 Webster's (v1)": "dictionary-1913.csv",
}
DICTIONARY_NOTES = {
    "Modern (v2, 2026)": "the modern dictionary: the 1913 Webster's Unabridged combined with "
    "Open English WordNet",
    "1913 Webster's (v1)": "the 1913 Webster's Unabridged on its own",
}
LETTERS = string.ascii_lowercase

# Chart colours: categorical slot 1 of the validated reference palette, plus neutral ink.
DATA_COLOR = "#2a78d6"
FIT_COLOR = "#52514e"  # a fitted model, not a data series

CSS = """
<style>
.block-container { max-width: 860px; padding-top: 2.5rem; }
.kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .78rem;
          font-weight: 600; color: #2a78d6; margin-bottom: -.4rem; }
.lede { font-size: 1.18rem; line-height: 1.6; color: #3d3c39; }
.abstract { border-left: 3px solid #2a78d6; background: #f3f2ee; padding: 1rem 1.25rem;
            border-radius: 0 .6rem .6rem 0; line-height: 1.6; margin: 1rem 0 .5rem; }
.abstract b { font-weight: 600; }
.step { border: 1px solid #e4e2dc; border-radius: .6rem; padding: 1rem; background: #ffffff;
        margin-bottom: .75rem; }
.step .num { display: inline-block; width: 1.7rem; height: 1.7rem; border-radius: 50%;
             background: #2a78d6; color: #fff; text-align: center; line-height: 1.7rem;
             font-weight: 600; font-size: .9rem; margin-bottom: .5rem; }
.step h4 { margin: 0 0 .35rem; font-size: 1.02rem; padding: 0; }
.step p { margin: 0; font-size: .92rem; color: #52514e; line-height: 1.5; }
.param { padding: .9rem 0; border-bottom: 1px solid #e4e2dc; }
.param:last-child { border-bottom: none; }
.param .name { font-weight: 600; }
.param .default { color: #52514e; font-size: .85rem; margin-left: .4rem; }
.param p { margin: .3rem 0 0; color: #3d3c39; line-height: 1.55; }
.chip { display: inline-block; font-family: monospace; padding: .15rem .5rem;
        margin: .15rem; border-radius: .4rem; font-size: .92rem; }
.chip.hit { background: #e3f1ea; color: #0d5c3c; font-weight: 600; }
.chip.miss { background: #f0efeb; color: #6b6a66; }
.chip.more { background: transparent; color: #6b6a66; font-family: inherit; }
.finding { font-size: 1.05rem; line-height: 1.65; }
[data-testid="stMarkdownContainer"] code { color: #3d3c39; background: #efeee9; }
.footer { color: #6b6a66; font-size: .85rem; line-height: 1.6; }
</style>
"""


# ---------------------------------------------------------------- dictionary


def dictionary_file(name: str) -> str:
    """Path to a dictionary file, downloading it into data/ on first use if it's missing.

    Looks in data/ next to this file, then in a copy of the source repository cloned
    alongside this app, and otherwise downloads it from the source repository.
    """
    local = DATA_DIR / name
    if local.is_file():
        return str(local)
    sibling = APP_DIR.parent / "English-Dictionary-Open-Source" / SOURCE_PATHS[name]
    if sibling.is_file():
        return str(sibling)
    DATA_DIR.mkdir(exist_ok=True)
    partial = local.with_suffix(local.suffix + ".part")
    with st.spinner(f"Downloading the dictionary ({name}) for first use..."):
        urllib.request.urlretrieve(RAW_BASE + SOURCE_PATHS[name], partial)
        if name == "dictionary.jsonl.gz":
            # Its data license must travel with it.
            urllib.request.urlretrieve(RAW_BASE + SOURCE_PATHS["LICENSE-DATA.md"],
                                       DATA_DIR / "LICENSE-DATA.md")
    partial.replace(local)  # only a complete download gets the real name
    return str(local)


@st.cache_data(show_spinner="Loading the dictionary...")
def load_words(path: str, include_capitalised: bool) -> frozenset[str]:
    """Return the set of lowercased, purely alphabetic words in a dictionary file.

    Reads the v1 CSV or the v2 gzipped JSON Lines file, both with the standard library.
    v2 keeps capitals for names and abbreviations (Paris, DNA); those are skipped unless
    include_capitalised is set. v1 is all lowercase, so the flag changes nothing there.
    """
    if path.endswith(".jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            entries = [json.loads(line)["word"] for line in f]
    else:
        with open(path, encoding="utf-8", newline="") as f:
            entries = [row["word"] for row in csv.DictReader(f)]
    words = set()
    for w in entries:
        w = w.strip()
        if w.isascii() and w.isalpha() and (include_capitalised or w == w.lower()):
            words.add(w.lower())
    return frozenset(words)


@st.cache_data
def word_codes(words: frozenset[str]) -> dict[int, np.ndarray]:
    """Each word as a base-26 integer (a=0 ... z=25), sorted, grouped by length.

    Integers let millions of random strings be checked against the dictionary in one
    vectorised step.
    """
    by_len: dict[int, list[int]] = {}
    for w in words:
        if len(w) <= 13:  # 26**13 still fits in a 64-bit integer
            code = 0
            for c in w:
                code = code * 26 + ord(c) - 97
            by_len.setdefault(len(w), []).append(code)
    return {n: np.array(sorted(v), dtype=np.int64) for n, v in by_len.items()}


def decode(code: int, n: int) -> str:
    chars = []
    for _ in range(n):
        code, r = divmod(int(code), 26)
        chars.append(LETTERS[r])
    return "".join(reversed(chars))


def is_word(codes: np.ndarray, dictionary: np.ndarray) -> np.ndarray:
    if dictionary.size == 0:
        return np.zeros(codes.shape, dtype=bool)
    idx = np.minimum(np.searchsorted(dictionary, codes), dictionary.size - 1)
    return dictionary[idx] == codes


# ---------------------------------------------------------------- the experiment


def runs_until_word(dictionary, n, trials, cap, rng):
    """Run `trials` trials of "draw random strings until one is a word" at length n.

    Each trial records how many failed draws came before a real word, or `cap` if it
    gave up. Every letter is equally likely. Also returns the total number of strings
    drawn, which the probability estimate needs. Draws are generated in large batches and consumed in order, which gives the same
    result as drawing them one at a time.
    """
    runs, found, draws, cur = [], [], 0, 0
    while len(runs) < trials:
        size = min(500_000, (trials - len(runs)) * cap)
        batch = rng.integers(0, 26**n, size=size, dtype=np.int64)
        hit_pos = np.flatnonzero(is_word(batch, dictionary))
        p = k = 0
        while len(runs) < trials and p < size:
            while k < len(hit_pos) and hit_pos[k] < p:
                k += 1
            gap = int(hit_pos[k]) - p if k < len(hit_pos) else None
            need = cap - cur  # failures left before this trial gives up
            if gap is not None and gap < need:
                runs.append(cur + gap)
                found.append(decode(batch[hit_pos[k]], n))
                draws, p, cur = draws + gap + 1, p + gap + 1, 0
            else:
                avail = size - p if gap is None else gap
                if avail >= need:
                    runs.append(cap)
                    found.append(None)
                    draws, p, cur = draws + need, p + need, 0
                else:
                    draws, p, cur = draws + avail, p + avail, cur + avail
    return np.array(runs), found, draws


def sample_trial(dictionary, n, rng, limit=200_000):
    """One trial, keeping every string drawn, for the walk-through on the page."""
    batch = rng.integers(0, 26**n, size=limit, dtype=np.int64)
    hits = np.flatnonzero(is_word(batch, dictionary))
    end = int(hits[0]) + 1 if hits.size else limit
    return [decode(c, n) for c in batch[:end]], bool(hits.size)


def run_experiment(words, min_len, max_len, trials, cap, seed):
    rng = np.random.default_rng(seed or None)
    codes = word_codes(words)
    empty = np.array([], dtype=np.int64)
    rows, found_words = [], Counter()
    for n in range(min_len, max_len + 1):
        runs, found, draws = runs_until_word(codes.get(n, empty), n, trials, cap, rng)
        hits = sum(w is not None for w in found)
        found_words.update((w, n) for w in found if w is not None)
        rows.append(
            {
                "Length": n,
                "Trials": trials,
                "Found a word": hits,
                "Gave up": trials - hits,
                "Total draws": draws,
                "Average attempts": float(runs.mean()),
                # Every string drawn is an independent try with the same chance p of being a
                # word, so the best estimate of p is words found ÷ strings drawn. This stays
                # correct when trials give up: their draws count, and they add no word.
                "Probability (%)": 100 * hits / draws,
                "Low (%)": 100 * wilson(hits, draws)[0],
                "High (%)": 100 * wilson(hits, draws)[1],
            }
        )
    demo_len = min(max(3, min_len), max_len)
    demo = sample_trial(codes.get(demo_len, empty), demo_len, rng)
    return pd.DataFrame(rows), found_words, (demo_len, *demo)


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion: a plausible range for the true p."""
    if n == 0:
        return 0.0, 1.0
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------- fitting


def fit_exponential(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit y = a·e^(b·x) by linear regression on ln(y).

    R² is measured on ln(y), the scale the regression is done on.
    """
    ly = np.log(y)
    b, ln_a = np.polyfit(x, ly, 1)
    pred = ln_a + b * x
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return math.exp(ln_a), b, r2


def fit_log_quadratic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit log10(y) = c0 + c1·x + c2·x². Returns coefficients (highest power first) and R²."""
    ly = np.log10(y)
    coefs = np.polyfit(x, ly, 2)
    pred = np.polyval(coefs, x)
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return coefs, r2


# ---------------------------------------------------------------- formatting


def pct(x: float) -> str:
    """Readable percentage across many orders of magnitude (x is a fraction)."""
    if x == 0:
        return "0%"
    v = 100 * x
    if v >= 1:
        return f"{v:.1f}%"
    if v >= 0.001:
        return f"{v:.3g}%"
    return f"{v:.{1 - math.floor(math.log10(v))}f}%"


def big(n: float) -> str:
    for size, name in ((1e12, "trillion"), (1e9, "billion"), (1e6, "million")):
        if n >= size:
            return f"{n / size:,.1f} {name}"
    return f"{n:,.0f}"


def tex_num(v: float) -> str:
    """A number for LaTeX: 3 significant figures, scientific when very large or small."""
    if v == 0:
        return "0"
    if 100 <= abs(v) < 100_000:
        return f"{v:.0f}"
    if 0.01 <= abs(v) < 100:
        return f"{v:.3g}"
    e = math.floor(math.log10(abs(v)))
    return rf"{v / 10**e:.3g} \times 10^{{{e}}}"


def md_to_html(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    return re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)


# ---------------------------------------------------------------- charts


def styled(chart: alt.Chart, height: int = 320) -> alt.Chart:
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#52514e", titleColor="#52514e", domainColor="#c9c7c0",
                        gridColor="#ecebe6")
    )


def fit_chart(df: pd.DataFrame, value: str, title: str, fmt: str, curve_fn, log_y: bool,
              data_label: str) -> alt.Chart:
    """Scatter of one value per length, with a fitted curve drawn through it."""
    pts = df[["Length", value]].rename(columns={value: "Value"})
    if log_y:
        pts = pts[pts["Value"] > 0]
    pts["Series"] = data_label

    lengths = df["Length"].tolist()
    curve = pd.DataFrame({"Length": np.linspace(min(lengths), max(lengths), 200)})
    if curve_fn is not None:
        curve["Value"] = curve_fn(curve["Length"].to_numpy())
        curve["Series"] = "Best fit"

    if log_y:
        lowest = min(pts["Value"].min(), curve["Value"].min() if curve_fn else np.inf)
        highest = max(pts["Value"].max(), curve["Value"].max() if curve_fn else 0)
        ticks = [10.0**k for k in range(math.floor(math.log10(lowest)), math.ceil(math.log10(highest)) + 1)]
        # A little headroom above the top tick so points at the edge aren't cut off.
        scale = alt.Scale(type="log", domain=[ticks[0], ticks[-1] * 1.6], nice=False)
        axis = alt.Axis(values=ticks, format="~g")
    else:
        # Cap the axis just above the data so a steep fitted curve can't squash the points.
        scale = alt.Scale(domain=[0, 1.15 * pts["Value"].max()], nice=False)
        axis = alt.Axis(format=fmt)

    x = alt.X(
        "Length:Q",
        title="Word length (letters)",
        scale=alt.Scale(domain=[min(lengths) - 0.3, max(lengths) + 0.3], nice=False),
        axis=alt.Axis(values=lengths, format="d", grid=False),
    )
    y = alt.Y("Value:Q", title=title, scale=scale, axis=axis)
    color = alt.Color(
        "Series:N",
        scale=alt.Scale(domain=[data_label, "Best fit"], range=[DATA_COLOR, FIT_COLOR]),
        legend=alt.Legend(orient="top", title=None, labelFontSize=13, symbolStrokeWidth=2),
    )
    layers = []
    if curve_fn is not None:
        layers.append(alt.Chart(curve).mark_line(strokeWidth=2, clip=True).encode(x=x, y=y, color=color))
    layers.append(
        alt.Chart(pts).mark_point(filled=True, size=80, opacity=1).encode(
            x=x, y=y, color=color,
            tooltip=[alt.Tooltip("Length:Q", title="Length"),
                     alt.Tooltip("Value:Q", title=title, format=",.4~g")],
        )
    )
    return styled(alt.layer(*layers))


# ---------------------------------------------------------------- page

st.set_page_config(page_title="Random Letters to Words", page_icon="🔤", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Contents")
    st.markdown(
        "1. [The question](#the-question)\n"
        "2. [How the experiment works](#how-the-experiment-works)\n"
        "3. [Fitting a curve](#fitting-a-curve)\n"
        "4. [The parameters](#the-parameters)\n"
        "5. [Results](#results)\n"
        "6. [What to keep in mind](#what-to-keep-in-mind)"
    )

# The dictionary settings live in the form further down. Form values only change when the
# form is submitted, so reading them here keeps the whole page on one dictionary.
dict_name = st.session_state.get("dict_name", next(iter(DICTIONARIES)))
include_caps = st.session_state.get("include_caps", False)
try:
    dict_path = dictionary_file(DICTIONARIES[dict_name])
except OSError:
    st.error(
        f"The dictionary couldn't be found or downloaded. Download "
        f"[{SOURCE_PATHS[DICTIONARIES[dict_name]]}]({SOURCE_REPO}) from the "
        f"English-Dictionary-Open-Source repository and save it as "
        f"`data/{DICTIONARIES[dict_name]}` next to app.py."
    )
    st.stop()

words = load_words(dict_path, include_caps)
by_len = Counter(len(w) for w in words)

# ---- Title + summary
st.markdown('<p class="kicker">An experiment in probability</p>', unsafe_allow_html=True)
st.title("How long until random letters spell a word?")
st.markdown(
    '<p class="lede">Imagine typing random letters, a few at a time, until you happen to type '
    "a real English word. How many tries would it take? This app runs that experiment and "
    "shows how the wait grows as the words get longer.</p>",
    unsafe_allow_html=True,
)

# ---- 1. The question
st.header("The question", anchor="the-question")
st.markdown(
    "Real words are a tiny fraction of all the letter combinations that could exist. "
    "With 26 letters, a string of *n* letters can be arranged in 26ⁿ ways, but only a few "
    "of those arrangements are words. The longer the string, the wider that gap becomes:"
)
cols = st.columns(3)
for col, n in zip(cols, (3, 5, 8)):
    with col.container(border=True):
        st.markdown(f"**{n} letters**")
        st.metric("Possible strings", big(26**n))
        st.caption(f"Of these, {by_len[n]:,} are words in this dictionary.")
st.markdown(
    "So the question is: **if you keep drawing random strings of a given length, how many "
    "attempts does it take to hit a real word, and how does that change as the length grows?**"
)

# ---- 2. How it works
st.header("How the experiment works", anchor="how-the-experiment-works")
st.markdown("The experiment runs separately for every word length you choose, in six steps.")
steps = [
    ("Load the dictionary",
     "Read every entry and keep only plain a–z words, lowercased. Entries with spaces, "
     "hyphens or apostrophes can never come from random letters, so they're dropped."),
    ("Group the words by length",
     "For a length of, say, 4, set aside only the 4-letter words, so checking whether a "
     "random string is a word takes a single lookup."),
    ("Draw until a word appears",
     "Build a string by picking each letter at random, every letter equally likely. If it's "
     "a real word, the trial ends. If not, count one <b>failed attempt</b> and draw again."),
    ("Give up if it takes too long",
     "If a trial reaches the limit (100,000 failed attempts by default) without finding a "
     "word, it stops and records the limit. Without this, long lengths could run for hours."),
    ("Repeat and average",
     "Run many trials for each length and average their failed attempts. One trial is mostly "
     "luck; the average shows the typical wait."),
    ("Estimate the probability",
     "Every string drawn is a separate try with the same small chance of being a word. So the "
     "best estimate of that chance is simply <b>words found ÷ strings drawn</b>, counting the "
     "draws from every trial, including those that gave up."),
]
for i, (title, body) in enumerate(steps, 1):
    st.markdown(
        f'<div class="step"><div class="num">{i}</div><h4>{title}</h4><p>{body}</p></div>',
        unsafe_allow_html=True,
    )

with st.expander("Show the core of the experiment in code"):
    st.code(
        '''words_n = {w for w in words if len(w) == n}      # step 2

def failed_attempts(n, limit=100_000):            # step 3
    count = 0
    while count < limit:
        s = "".join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(n))
        if s in words_n:
            return count, True                    # found a word
        count += 1
    return limit, False                           # step 4: gave up

trials = [failed_attempts(n) for _ in range(50)]                    # step 5
average = sum(c for c, _ in trials) / len(trials)
found = sum(ok for _, ok in trials)
drawn = sum(c + ok for c, ok in trials)
probability = found / drawn                                         # step 6''',
        language="python",
    )
    st.caption(
        "The app runs exactly this procedure, but draws strings in large batches so that long "
        "lengths finish in seconds. Using batched draws in order gives the same results as "
        "drawing one at a time."
    )

# ---- 3. Fitting
st.header("Fitting a curve", anchor="fitting-a-curve")
st.markdown(
    "Each extra letter makes words rarer by roughly the same factor, so the average wait "
    "should grow, and the probability shrink, roughly exponentially. So both are fitted "
    "with an exponential curve:"
)
st.latex(r"y = a \cdot e^{\,b\,n}")
st.markdown(
    "Here *n* is the word length and *y* is either the average number of failed attempts or "
    "the probability. "
    "Taking the logarithm of both sides turns the curve into a straight line, "
    "ln *y* = ln *a* + *b n*, so *a* and *b* come from an ordinary linear regression on ln *y*. "
    "A positive *b* means growth and a negative *b* means decay. Each extra letter multiplies "
    "*y* by *e*<sup>*b*</sup>. R² says how well the line fits, where 1 is a perfect fit.",
    unsafe_allow_html=True,
)
st.markdown(
    "The fit isn't perfect, and the reason is interesting. The number of dictionary words grows "
    "up to about 8 letters and then falls, so the odds drop slowly for short words and faster "
    "for long ones. On a log scale this shows up as a bend rather than a straight line. The "
    "results include a third chart that fits that bend with one more term: "
    "log₁₀ *P* = *c*₀ + *c*₁ *n* + *c*₂ *n*²."
)

# ---- 4. Parameters
st.header("The parameters", anchor="the-parameters")
st.markdown("Each setting changes one part of the experiment. Here is what each one does.")
params = [
    ("Word lengths", "1 to 6 letters",
     "Which word lengths to test. Each length is run separately. Past about 7 letters, most "
     "trials hit the give-up limit, so those lengths take longer and say less."),
    ("Trials per length", "50",
     "How many times to run “draw until a word appears” for each length. More trials give a "
     "steadier average and a narrower probability range, but take longer."),
    ("Give up after", "100,000 failed attempts",
     "A trial stops after this many failed attempts and records the limit. "
     "Once most trials at a length give up, that length's average is pinned near the limit and "
     "understates the true wait. Raise it to explore longer lengths."),
    ("Dictionary", "Modern (v2, 2026)",
     "Which word list counts as “real words”. <b>Modern (v2)</b> combines the 1913 Webster's "
     "with Open English WordNet, so it includes words like <code>smartphone</code> and "
     "<code>email</code>. <b>1913 Webster's (v1)</b> is the older list on its own. A bigger "
     "dictionary means shorter waits."),
    ("Include names and abbreviations", "off",
     "Modern dictionary only. Its capitalised entries are mostly names and abbreviations, such "
     "as <code>Paris</code>, <code>AC</code> or <code>DNA</code>. Left off, they don't count as "
     "words. Turned on, they do, which sharply shortens the wait at 2 and 3 letters, where "
     "almost any pair of letters is some abbreviation."),
    ("Random seed", "0 (new each run)",
     "The starting number for the random generator. 0 gives fresh draws every run. Any other "
     "number repeats exactly the same draws, which helps when comparing settings fairly."),
]
with st.container(border=True):
    st.markdown(
        "".join(
            f'<div class="param"><span class="name">{n}</span><span class="default">default: {d}</span>'
            f"<p>{t}</p></div>"
            for n, d, t in params
        ),
        unsafe_allow_html=True,
    )

# ---- Controls
st.subheader("Run the experiment")
with st.form("controls", border=True):
    min_len, max_len = st.slider("Word lengths", 1, 12, (1, 6))
    c1, c2 = st.columns(2)
    trials = c1.number_input("Trials per length", min_value=5, max_value=500, value=50, step=5)
    cap = c2.number_input("Give up after", min_value=1_000, max_value=1_000_000, value=100_000,
                          step=10_000, help="Failed attempts before a trial gives up.")
    c3, c4 = st.columns(2)
    c3.selectbox("Dictionary", list(DICTIONARIES), key="dict_name")
    c4.toggle("Include names and abbreviations", key="include_caps",
              help="Modern dictionary only: count capitalised entries such as Paris or DNA.")
    seed = st.number_input("Random seed", min_value=0, value=0, step=1, help="0 = new draws every run.")
    submitted = st.form_submit_button("Run experiment", type="primary", width="stretch")

params_now = (min_len, max_len, int(trials), int(cap), int(seed))
if submitted or "results" not in st.session_state:
    with st.spinner("Drawing random strings..."):
        st.session_state.results = run_experiment(words, *params_now)
        st.session_state.params = params_now

results, found_words, (demo_len, demo_draws, demo_hit) = st.session_state.results
min_len, max_len, trials, cap, seed = st.session_state.params

# ---- 5. Results
st.header("Results", anchor="results")
caps_label = " with names and abbreviations" if include_caps and dict_name.startswith("Modern") else ""
st.caption(
    f"{dict_name} dictionary{caps_label}, {len(words):,} words · lengths {min_len}–{max_len} · "
    f"{trials:,} trials per length · give up after {cap:,} · seed {seed if seed else 'random'}"
)

total_draws = int(results["Total draws"].sum())
total_found = int(results["Found a word"].sum())
total_gave_up = int(results["Gave up"].sum())
m1, m2, m3 = st.columns(3)
m1.metric("Strings drawn", big(total_draws))
m2.metric("Trials that found a word", f"{total_found:,}")
m3.metric("Trials that gave up", f"{total_gave_up:,}")

st.subheader("One trial, step by step")
st.markdown(
    f"Here is a single trial at {demo_len} letters, showing every string it drew. "
    "The green one is the real word that ended it."
)
shown = demo_draws if len(demo_draws) <= 60 else demo_draws[:30] + [None] + demo_draws[-29:]
chips = []
for s in shown:
    if s is None:
        chips.append(f'<span class="chip more">… {len(demo_draws) - 59:,} more …</span>')
    else:
        hit = demo_hit and s is demo_draws[-1]
        chips.append(f'<span class="chip {"hit" if hit else "miss"}">{s}</span>')
st.markdown("".join(chips), unsafe_allow_html=True)
if demo_hit:
    st.caption(f"{len(demo_draws) - 1:,} failed attempts before “{demo_draws[-1]}”.")
else:
    st.caption(f"No word in {len(demo_draws):,} draws.")

log_y = st.toggle("Show the first two charts on a log scale", value=False)


def chart_block(value, title, fmt, var, growth, data_label):
    ok = results[results[value] > 0]
    fit = None
    if len(ok) >= 2:
        fit = fit_exponential(ok["Length"].to_numpy(float), ok[value].to_numpy(float))
    with st.container(border=True):
        st.markdown(f"**{title} by word length**")
        if fit:
            a, b, r2 = fit
            st.latex(rf"{var} = {tex_num(a)} \cdot e^{{{b:.3f}\,n}}")
            factor = math.exp(abs(b))
            change = (f"multiplies the wait by about {factor:.1f}" if growth
                      else f"divides the chance by about {factor:.1f}")
            st.caption(
                f"R² = {r2:.3f}. Each extra letter {change}, on average. "
                f"Fitted on the {len(ok)} lengths above zero."
            )
            curve = lambda x: a * np.exp(b * x)  # noqa: E731
        else:
            st.caption("Not enough lengths above zero to fit a curve.")
            curve = None
        st.altair_chart(fit_chart(results, value, title, fmt, curve, log_y, data_label), width="stretch")


st.subheader("How the wait grows")
chart_block("Average attempts", "Average failed attempts", ",.0f", r"\text{attempts}", True, "Average of trials")
st.subheader("How the probability shrinks")
chart_block("Probability (%)", "Probability (%)", ",.2~f", r"P\,(\%)", False, "Measured probability")

st.subheader("The bend on a log scale")
st.markdown(
    "Plotting the probability on a log scale makes every length visible. A pure exponential "
    "would be a straight line here; the points bend instead, so this chart fits a curve with "
    "an *n*² term."
)
ok = results[results["Probability (%)"] > 0]
if len(ok) >= 4:
    xs, ys = ok["Length"].to_numpy(float), ok["Probability (%)"].to_numpy(float)
    coefs, r2 = fit_log_quadratic(xs, ys)
    _, _, r2_line = fit_exponential(xs, ys)
    c2, c1, c0 = coefs
    with st.container(border=True):
        st.latex(rf"\log_{{10}} P(\%) = {c0:.3f} {c1:+.3f}\,n {c2:+.4f}\,n^2")
        st.caption(
            f"R² = {r2:.4f}, against {r2_line:.4f} for the straight-line (exponential) fit above. "
            "The negative n² term is the bend: each extra letter costs more than the one before."
        )
        st.altair_chart(
            fit_chart(results, "Probability (%)", "Probability (%, log scale)", "",
                      lambda x: 10 ** np.polyval(coefs, x), True, "Measured probability"),
            width="stretch",
        )
else:
    st.info("Test at least 4 lengths where a word was found to fit the bend.")

# Plain-language reading of this run.
notes = []
if total_gave_up:
    capped = results.loc[results["Gave up"] > 0, "Length"].tolist()
    notes.append(
        f"**{total_gave_up:,}** {'trial' if total_gave_up == 1 else 'trials'} gave up before "
        f"finding a word (at {'length' if len(capped) == 1 else 'lengths'} "
        f"{', '.join(map(str, capped))}). Each is recorded as {cap:,} failed attempts, so the average "
        "there is lower than the true wait and the growth curve flattens. Raise "
        "**Give up after** to see the full climb."
    )
first_zero = results.loc[results["Found a word"] == 0, "Length"]
if not first_zero.empty:
    notes.append(
        f"From **{int(first_zero.min())} letters**, no trial found a word, so the measured "
        "probability is 0 and those lengths are left out of the fits."
    )
if notes:
    st.markdown(f'<div class="finding">{md_to_html("<br><br>".join(notes))}</div>', unsafe_allow_html=True)
    st.write("")

tab_table, tab_words = st.tabs(["Numbers by length", f"Words found ({len(found_words):,} distinct)"])
with tab_table:
    st.dataframe(
        pd.DataFrame(
            {
                "Length": results["Length"],
                "Words in dictionary": results["Length"].map(lambda n: f"{by_len[n]:,}"),
                "Trials": results["Trials"],
                "Found a word": results["Found a word"],
                "Gave up": results["Gave up"],
                "Strings drawn": results["Total draws"].map("{:,}".format),
                "Average attempts": results["Average attempts"].map("{:,.1f}".format),
                "Probability": (results["Probability (%)"] / 100).map(pct),
                "95% range": [f"{pct(lo / 100)} to {pct(hi / 100)}"
                              for lo, hi in zip(results["Low (%)"], results["High (%)"])],
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Probability = words found ÷ strings drawn at that length. The 95% range (a Wilson "
        "interval) shows how far the true probability could plausibly be from that estimate; "
        "more trials narrow it."
    )
with tab_words:
    if found_words:
        st.dataframe(
            pd.DataFrame([(w, n, c) for (w, n), c in found_words.items()],
                         columns=["Word", "Length", "Times found"])
            .sort_values(["Length", "Times found", "Word"], ascending=[False, False, True]),
            hide_index=True, width="stretch", height=320,
        )
    else:
        st.write("No trial found a word. Try shorter lengths or a higher give-up limit.")

# ---- 6. Caveats
st.header("What to keep in mind", anchor="what-to-keep-in-mind")
st.markdown(
    "- **The give-up limit bends the results.** Trials that give up are recorded at the limit, "
    "not at their true wait, so averages at long lengths are too low.\n"
    "- **The probability is measured, not calculated.** It counts every string drawn, so it "
    "stays accurate even when trials give up. It can only be as precise as the number of words "
    "found: the 95% range in the table shows how precise each length is.\n"
    f"- **The dictionary shapes the answer.** This run uses {DICTIONARY_NOTES[dict_name]}, with "
    f"{len(words):,} distinct words. Both lists keep rare and archaic Webster's words like "
    "*lepry* and *roset*, which shorten the wait.\n"
    + (
        "- **Single letters count.** Every letter has its own entry in this dictionary, so at "
        "1 letter the first try always works and the number of failed attempts is 0. That "
        "length is left out of the attempts fit, since 0 can't be fitted on a log scale.\n"
        if by_len[1] == 26 else
        f"- **Single letters count.** {by_len[1]} of the 26 single letters have their own "
        "entries in this dictionary, so at 1 letter a word usually turns up on the first try.\n"
    )
    + "- **Fits describe, they don't explain.** The equations summarise the lengths you tested. "
    "Outside that range they can drift.\n"
    "- **Results wobble.** Each run gives slightly different averages. More trials per length "
    "shrink the wobble."
)

st.divider()
st.markdown(
    '<p class="footer">Data: <a href="https://github.com/CloudBytes-Academy/English-Dictionary-Open-Source">'
    "English-Dictionary-Open-Source</a>. v2: 1913 Webster's Unabridged (public domain) plus "
    "Open English WordNet (CC BY 4.0); see data/LICENSE-DATA.md for attribution. "
    "v1: 1913 Webster's Unabridged only.</p>",
    unsafe_allow_html=True,
)
