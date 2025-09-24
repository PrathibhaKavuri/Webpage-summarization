import os, re, json, argparse, requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai

def load_key() -> str:
    load_dotenv()
    k = os.getenv("GEMINI_API_KEY")
    if not k:
        raise SystemExit("set GEMINI_API_KEY in .env")
    return k

def fetch_clean(url: str, timeout: int = 20) -> tuple[str, str]:
    r = requests.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (Summarizer)",
        "Accept-Language": "en",
    })
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    title = (soup.title.string or "").strip() if soup.title else ""
    for tag in soup(["script","style","noscript","header","footer","nav","aside","form","svg","img","video","audio","iframe","canvas"]):
        tag.decompose()
    parts = []
    for t in soup.find_all(["h1","h2","h3","p","li","blockquote"]):
        txt = t.get_text(" ", strip=True)
        if txt and len(txt.split()) >= 3: parts.append(txt)
    text = re.sub(r"\s+"," ", "\n".join(parts)).strip()
    if not text:
        raise SystemExit("Could not extract readable text from the page.")
    return title, text

def chunks(s: str, n: int = 12000, overlap: int = 400):
    if len(s) <= n: yield s; return
    i = 0
    while i < len(s):
        seg = s[i:i+n]
        j = seg.rfind("\n")
        if j > n*0.6: seg = seg[:j]
        yield seg
        i += max(1, len(seg) - overlap)

def make_model(model_name="gemini-2.5-flash"):
    genai.configure(api_key=load_key())

    try:
        return genai.GenerativeModel(model_name,
            generation_config={"response_mime_type":"application/json"})
    except Exception:
        return genai.GenerativeModel(model_name)

def parse_json(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S) or re.search(r"(\{.*\})", s, re.S)
        if m:
            try: return json.loads(m.group(1))
            except Exception: pass
    return {"abstract": s.strip(), "bullets": []}

def summarize_block(model, text: str, bullets=6, max_words=180) -> dict:
    prompt = {
        "role": "user",
        "parts": [(
            "You are a precise, neutral summarizer. "
            "Return STRICT JSON only with keys: abstract, bullets.\n"
            f'Format: {{"abstract":"<= {max_words} words","bullets":["up to {bullets} key points"]}}\n\n'
            f"CONTENT:\n{text[:16000]}"
        )]
    }
    resp = model.generate_content(prompt)
    out = getattr(resp, "text", None)
    if out is None and getattr(resp, "candidates", None):

        parts = resp.candidates[0].content.parts
        out = "".join(getattr(p, "text", "") for p in parts)
    if not out: out = ""
    return parse_json(out)

def summarize_url(url: str, model_name="gemini-2.5-flash", bullets=6, max_words=180) -> dict:
    title, text = fetch_clean(url)
    model = make_model(model_name)
    sect = [summarize_block(model, c, bullets=min(7, bullets), max_words=min(220, max_words)) for c in chunks(text)]
    merged_abs = " ".join(s.get("abstract","") for s in sect)[:20000]
    merged_bul = sum((s.get("bullets",[]) for s in sect), [])[:bullets]
    final = summarize_block(model, merged_abs, bullets=bullets, max_words=max_words)
    final.setdefault("abstract", merged_abs[:max_words*8])
    final["bullets"] = final.get("bullets", []) or merged_bul
    final["title"], final["url"] = title, url
    return final

def main():
    p = argparse.ArgumentParser(description="AI Web Page Summarizer (Gemini + bs4)")
    p.add_argument("--url", required=True)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--bullets", type=int, default=6)
    p.add_argument("--max-words", type=int, default=180)
    a = p.parse_args()
    try:
        res = summarize_url(a.url, a.model, a.bullets, a.max_words)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    except requests.HTTPError as e:
        raise SystemExit(f" HTTP {e.response.status_code if e.response else ''}: {e}")
    except Exception as e:
        raise SystemExit(f" {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()
