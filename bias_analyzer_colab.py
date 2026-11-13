"""
Geopolitical Media Bias ANALYZER (v2) -- run in Colab AFTER bias_demo_colab.py.

What it adds over the simple demo:
  1. WHY: highlights the exact words that push a sentence toward "biased" (leave-one-word-out test)
  2. WHOLE ARTICLES: paste text or a link; it scores every sentence and the whole article
  3. LIVE NEWS: pick an outlet, load its latest headlines, analyze with one click
  4. COMPARE: two outlets / two articles side by side (the "framing divergence" idea)

It reuses the trained model already in your notebook (variables: model, tok, device).
If the session was restarted, it loads the saved folder "bias_model" instead.
"""
import re, html, subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio", "trafilatura", "feedparser"], check=True)
import numpy as np, torch

try:
    model, tok, device
except NameError:
    import os
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    try:
        from google.colab import drive
        drive.mount("/content/drive")
    except Exception:
        pass
    path = "/content/drive/MyDrive/bias_model" if os.path.exists("/content/drive/MyDrive/bias_model") else "bias_model"
    device = "cuda" if torch.cuda.is_available() else "cpu"   # CPU is fine for analyzing
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(device)
model.eval()

# ------------------------------------------------------------------ model helpers
def probs(sentences, bs=32):
    """P(biased) for each sentence."""
    out = []
    for i in range(0, len(sentences), bs):
        b = tok(sentences[i:i + bs], truncation=True, padding=True, max_length=96, return_tensors="pt").to(device)
        with torch.no_grad():
            p = torch.softmax(model(**b).logits, dim=1)[:, 1].cpu().numpy()
        out.extend(p.tolist())
    return out

def word_impacts(sentence):
    """Leave-one-word-out: how much does P(biased) DROP when each word is removed?
    Big positive number = that word pushes the model toward 'biased'."""
    words = sentence.split()
    if len(words) < 2:
        return words, [0.0] * len(words), probs([sentence])[0]
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    p = probs([sentence] + variants)
    base = p[0]
    return words, [base - q for q in p[1:]], base

def clean(w):
    return re.sub(r"^\W+|\W+$", "", w).lower()

def highlight(words, impacts):
    parts = []
    for w, imp in zip(words, impacts):
        if imp > 0.04:
            a = min(0.85, 0.2 + imp * 2.5)
            parts.append(f'<span style="background:rgba(255,120,0,{a:.2f});padding:1px 3px;border-radius:3px">{html.escape(w)}</span>')
        elif imp < -0.04:
            parts.append(f'<span style="background:rgba(80,150,255,0.30);padding:1px 3px;border-radius:3px">{html.escape(w)}</span>')
        else:
            parts.append(html.escape(w))
    return " ".join(parts)

# ------------------------------------------------------------------ article analysis
def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z“"\'])', text)
    return [s for s in sents if 6 <= len(s.split()) <= 80][:40]

def analyze_core(text):
    sents = split_sentences(text)
    if not sents:
        return None
    p = probs(sents)
    order = np.argsort(p)[::-1]
    flagged = [i for i in order if p[i] > 0.5]
    top = flagged[:5]
    cards, trigger = [], {}
    for i in top:
        words, imps, base = word_impacts(sents[i])
        cards.append(
            f'<div style="margin:10px 0;padding:10px;border:1px solid #8884;border-radius:8px">'
            f'<b>{base:.0%} biased</b><br>{highlight(words, imps)}</div>')
        for w, im in zip(words, imps):
            c = clean(w)
            if im > 0.04 and len(c) > 2:
                trigger[c] = max(trigger.get(c, 0), im)
    top_words = sorted(trigger.items(), key=lambda kv: -kv[1])[:8]
    return dict(n=len(sents), flagged=len(flagged), share=len(flagged) / len(sents),
                avg=float(np.mean(p)), cards="".join(cards), words=top_words)

LEGEND = ('<div style="font-size:13px;opacity:.8">'
          '<span style="background:rgba(255,120,0,0.6);padding:1px 4px;border-radius:3px">orange</span> = word pushes toward BIASED &nbsp; '
          '<span style="background:rgba(80,150,255,0.3);padding:1px 4px;border-radius:3px">blue</span> = word pushes toward NEUTRAL</div>')

def label_for(share):
    return "Mostly neutral" if share < 0.2 else ("Some loaded wording" if share < 0.45 else "Heavily loaded wording")

def render(res, name=""):
    if res is None:
        return "Could not find enough sentences to analyze. Paste more text or try another link.", ""
    head = (f"### {name}{label_for(res['share'])}\n"
            f"**{res['flagged']} of {res['n']} sentences** ({res['share']:.0%}) use loaded wording. "
            f"Average bias score: **{res['avg']:.0%}**.\n\n")
    if res["words"]:
        head += "**Trigger words (strongest first):** " + ", ".join(f"`{w}`" for w, _ in res["words"])
    body = LEGEND + (res["cards"] or "<p>No sentence crossed the 'biased' threshold.</p>")
    return head, body

# ------------------------------------------------------------------ getting text
UA = "Mozilla/5.0 (compatible; BiasAnalyzerStudentProject/1.0)"

def get_text(src):
    src = (src or "").strip()
    if src.startswith("http"):
        import trafilatura
        raw = trafilatura.fetch_url(src)
        txt = trafilatura.extract(raw) if raw else None
        return txt
    return src

FEEDS = {
    "BBC World (UK)": "http://feeds.bbci.co.uk/news/world/rss.xml",
    "Al Jazeera (Qatar)": "https://www.aljazeera.com/xml/rss/all.xml",
    "The Hindu - International (India)": "https://www.thehindu.com/news/international/feeder/default.rss",
    "Times of India - World (India)": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
    "DW World (Germany)": "https://rss.dw.com/rdf/rss-en-world",
    "CNN World (USA)": "http://rss.cnn.com/rss/edition_world.rss",
}
ENTRIES = {}

def load_feed(name):
    import feedparser
    try:
        d = feedparser.parse(FEEDS[name], agent=UA)
        items = [(e.get("title", "").strip(), e.get("link", ""), e.get("summary", "")) for e in d.entries[:15]]
    except Exception:
        items = []
    ENTRIES[name] = {t: (l, s) for t, l, s in items if t}
    titles = list(ENTRIES[name])
    import gradio as gr
    return gr.Dropdown(choices=titles, value=titles[0] if titles else None,
                       label=f"Latest headlines ({len(titles)})" if titles else "Feed unavailable, try another outlet")

# ------------------------------------------------------------------ app actions
def analyze_input(src):
    txt = get_text(src)
    if not txt:
        return "Could not read that link. Paste the article text instead.", ""
    return render(analyze_core(txt))

def analyze_headline(feed, title):
    if not title or feed not in ENTRIES or title not in ENTRIES[feed]:
        return "Load headlines first, then pick one.", ""
    link, summ = ENTRIES[feed][title]
    txt = None
    try:
        txt = get_text(link)
    except Exception:
        pass
    note = ""
    if not txt or len(txt.split()) < 60:
        txt = title + ". " + re.sub(r"<[^>]+>", " ", summ or "")
        note = "*(Full article could not be fetched, so only the headline and summary were analyzed.)*\n\n"
    head, body = render(analyze_core(txt), name=f"{feed.split(' (')[0]}: ")
    return note + head, body

def compare(a, b):
    ta, tb = get_text(a), get_text(b)
    if not ta or not tb:
        return "Could not read one of the inputs. Paste the text instead.", "", ""
    ra, rb = analyze_core(ta), analyze_core(tb)
    if ra is None or rb is None:
        return "Not enough sentences in one of the inputs.", "", ""
    gap = abs(ra["share"] - rb["share"])
    summary = (f"### Framing divergence: {gap:.0%}\n"
               f"Source A: **{ra['share']:.0%}** of sentences loaded ({label_for(ra['share'])}). "
               f"Source B: **{rb['share']:.0%}** ({label_for(rb['share'])}).\n\n"
               f"A trigger words: {', '.join('`'+w+'`' for w,_ in ra['words']) or 'none'}  \n"
               f"B trigger words: {', '.join('`'+w+'`' for w,_ in rb['words']) or 'none'}")
    return summary, LEGEND + (ra["cards"] or "<p>No loaded sentences.</p>"), LEGEND + (rb["cards"] or "<p>No loaded sentences.</p>")

# ===== UI =====
import gradio as gr
HOW = """
### How it finds bias (no magic)
1. **Split** the article into sentences.
2. **Score** each sentence with our fine-tuned RoBERTa (trained on 3,674 expert-labeled sentences from BABE). It returns a probability that the wording is biased.
3. **Explain** each flagged sentence with a *leave-one-word-out test*: remove one word at a time and see how much the bias score drops. Words whose removal drops the score the most are the **trigger words**, shown in orange.
4. **Article score** = share of sentences above 50% bias.

### Honest limits
- It measures **loaded wording**, not whether facts are true.
- Trained on US political news, so geopolitics from other regions is a stretch.
- It catches accusing language better than flattering spin.
- Results are a signal for a human reader, not a verdict.
"""

with gr.Blocks(title="Geopolitical Media Bias Analyzer") as app:
    gr.Markdown("# Geopolitical Media Bias Analyzer\nFind loaded wording in news, and see exactly which words trigger it.")
    with gr.Tab("Paste text or link"):
        src = gr.Textbox(lines=6, label="Paste article text, or a link to an article")
        go = gr.Button("Analyze", variant="primary")
        v1, d1 = gr.Markdown(), gr.HTML()
        go.click(analyze_input, src, [v1, d1])
    with gr.Tab("Live news"):
        feed = gr.Dropdown(list(FEEDS), value=list(FEEDS)[0], label="Choose an outlet")
        load = gr.Button("Load latest headlines")
        heads = gr.Dropdown([], label="Headlines")
        go2 = gr.Button("Analyze this article", variant="primary")
        v2, d2 = gr.Markdown(), gr.HTML()
        load.click(load_feed, feed, heads)
        go2.click(analyze_headline, [feed, heads], [v2, d2])
    with gr.Tab("Compare two sources"):
        with gr.Row():
            ca = gr.Textbox(lines=5, label="Source A (text or link)")
            cb = gr.Textbox(lines=5, label="Source B (text or link)")
        go3 = gr.Button("Compare framing", variant="primary")
        v3 = gr.Markdown()
        with gr.Row():
            da, db = gr.HTML(), gr.HTML()
        go3.click(compare, [ca, cb], [v3, da, db])
    with gr.Tab("How it works"):
        gr.Markdown(HOW)

app.launch(share=True)
