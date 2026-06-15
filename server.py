"""
PDF Translator Pro — Flask Server
Chạy: python3 server.py
Mở: http://localhost:5000
"""

import os, sys, time, base64, json, re, io, threading
import fitz
import pdfplumber
import pikepdf
import requests
from flask import Flask, request, jsonify, send_file, send_from_directory
from reportlab.lib.units import inch, cm, mm
pt = 1  # 1 point = 1 PDF unit
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image

app = Flask(__name__, static_folder='static')

# ── Progress store ─────────────────────────────────────────────────────────────
progress_store = {}  # job_id → {status, current, total, log, done, output_path, error}

# ── Font ───────────────────────────────────────────────────────────────────────
FONT_PATHS = [
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial Unicode MS.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

def register_font():
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                pdfmetrics.registerFont(TTFont("UniFont", path))
                return "UniFont"
            except: continue
    return "Helvetica"

FONT_NAME = register_font()

# Register NotoSans Bold nếu có
FONT_BOLD_NAME = FONT_NAME
_noto_bold = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSans-Bold.ttf')
if os.path.exists(_noto_bold):
    try:
        pdfmetrics.registerFont(TTFont("NotoSans-Bold", _noto_bold))
        FONT_BOLD_NAME = "NotoSans-Bold"
    except: pass
_noto_reg = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSans-Regular.ttf')
if os.path.exists(_noto_reg):
    try:
        pdfmetrics.registerFont(TTFont("NotoSans", _noto_reg))
        FONT_NAME = "NotoSans"
    except: pass

# ── Gemini ─────────────────────────────────────────────────────────────────────
STYLE_MAP = {
    'literal': 'Dịch sát nghĩa từng chữ, trung thành tối đa với bản gốc.',
    'natural': 'Dịch tự nhiên, trôi chảy như người Việt có học thức viết.',
    'academic': 'Dịch theo văn phong học thuật, chuyên ngành, trang trọng.',
    'simple': 'Dịch đơn giản, dễ hiểu, phù hợp với mọi đối tượng.',
}

def gemini_translate_text(text, api_key, model, style):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"Dịch đoạn văn bản sau từ tiếng Anh sang tiếng Việt. {STYLE_MAP.get(style, '')} Giữ nguyên cấu trúc đoạn văn. Chỉ trả về bản dịch:\n\n{text}"}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096}
    }
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()['candidates'][0]['content']['parts'][0]['text'].strip()

def gemini_vision(img_bytes, api_key, model, style):
    style_en = {'literal':'Translate word-for-word faithfully.','natural':'Translate naturally as a fluent Vietnamese speaker.','academic':'Translate in formal academic style.','simple':'Translate simply and clearly.'}.get(style,'')
    prompt = f"""This is a scanned book page image, possibly containing 2 pages side by side.
Analyze each page (left and right) separately:
- Plain text page → type: "text", translate to Vietnamese. {style_en}
- Diagram/chart/image → type: "diagram", only translate caption if present.
Return ONLY valid JSON, no extra text:
{{"left":{{"type":"text","blocks":[{{"text":"translated paragraph"}}]}},"right":{{"type":"diagram","blocks":[{{"text":"caption"}}]}}}}
If only 1 page: left=full content, right={{"type":"text","blocks":[]}}"""

    b64 = base64.b64encode(img_bytes).decode()
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": b64}}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 8192}
    }
    r = requests.post(url, json=payload, timeout=90)
    r.raise_for_status()
    raw = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    clean = re.sub(r'```json\n?|```\n?', '', raw).strip()
    try:
        p = json.loads(clean)
        return {
            'left': {'type': p.get('left',{}).get('type','text'), 'blocks': p.get('left',{}).get('blocks',[])},
            'right': {'type': p.get('right',{}).get('type','text'), 'blocks': p.get('right',{}).get('blocks',[])}
        }
    except:
        return {'left': {'type':'text','blocks':[{'text':raw}]}, 'right': {'type':'text','blocks':[]}}

# ── Text PDF layout extraction ─────────────────────────────────────────────────
def extract_blocks(page):
    words = page.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3, extra_attrs=['fontname','size'])
    if not words: return []
    lines = {}
    for w in words:
        y = round(w['top'])
        if y not in lines: lines[y] = []
        lines[y].append(w)

    blocks, prev_y, cur_para, cur_x0, cur_size, cur_font = [], None, [], None, None, None
    for y in sorted(lines.keys()):
        ws = sorted(lines[y], key=lambda w: w['x0'])
        line_text = ' '.join(w['text'] for w in ws)
        lx0 = ws[0]['x0']
        lsize = ws[0].get('size', 10) or 10
        lfont = ws[0].get('fontname','')
        if prev_y is not None and (y - prev_y) > lsize * 1.8:
            if cur_para:
                blocks.append({'text': ' '.join(cur_para), 'x0': cur_x0, 'y0': prev_y, 'size': cur_size, 'bold': 'Bold' in (cur_font or ''), 'page_h': page.height, 'page_w': page.width})
            cur_para, cur_x0, cur_size, cur_font = [line_text], lx0, lsize, lfont
        else:
            cur_para.append(line_text)
            if cur_x0 is None: cur_x0, cur_size, cur_font = lx0, lsize, lfont
        prev_y = y
    if cur_para:
        blocks.append({'text': ' '.join(cur_para), 'x0': cur_x0, 'y0': prev_y, 'size': cur_size, 'bold': 'Bold' in (cur_font or ''), 'page_h': page.height, 'page_w': page.width})
    return blocks

def render_text_page(c, page_w, page_h, blocks, translations):
    """1 trang gốc = 1 trang dịch. Heading: bold 11pt, Body: regular 9pt."""
    W = page_w
    H = page_h
    margin_x = W * 0.1
    margin_y = H * 0.07
    max_w = W - margin_x * 2

    c.setPageSize((W, H))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColorRGB(0, 0, 0)

    sizes = [b['size'] for b in blocks if b.get('size')]
    base_size = sorted(sizes)[len(sizes)//2] if sizes else 10

    y = H - margin_y
    for block, trans in zip(blocks, translations):
        if not trans.strip():
            y -= 9 * 0.5
            continue

        is_heading = block.get('bold') and (block.get('size') or base_size) > base_size
        FS = 11 if is_heading else 9
        LH = FS * 1.6

        try: c.setFont(FONT_BOLD_NAME if is_heading else FONT_NAME, FS)
        except: c.setFont('Helvetica-Bold' if is_heading else 'Helvetica', FS)

        y -= LH * 0.4
        words = trans.split()
        line = ''
        x = margin_x
        for word in words:
            test = (line + ' ' + word).strip()
            try: tw = c.stringWidth(test, FONT_NAME, FS)
            except: tw = len(test) * FS * 0.5
            if tw > max_w and line:
                if y >= margin_y:
                    c.drawString(x, y, line)
                y -= LH
                line = word
                x = margin_x
            else:
                line = test
        if line and y >= margin_y:
            c.drawString(x, y, line)
            y -= LH


def render_scan_page(c, blocks, page_w=595, page_h=842):
    c.setPageSize((page_w, page_h))
    c.setFillColorRGB(1,1,1)
    c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
    margin, x, y, fs = 40, 40, page_h-50, 10
    max_w = page_w - margin*2
    lh = fs*1.55
    c.setFillColorRGB(0,0,0)
    for block in blocks:
        txt = block.get('text','').strip()
        if not txt: continue
        is_h = len(txt) < 100 and txt.upper()==txt and len(txt)>2
        fsize = fs*1.1 if is_h else fs
        try: c.setFont(FONT_NAME, fsize)
        except: c.setFont('Helvetica', fsize)
        c.setFillColorRGB(0.05,0.1,0.3 if is_h else 0)
        words = txt.split(); line = ''
        for word in words:
            test = (line+' '+word).strip()
            try: w = c.stringWidth(test, FONT_NAME, fsize)
            except: w = len(test)*fsize*0.5
            if w > max_w and line:
                c.drawString(x, y, line); y -= lh; line = word
                if y < margin+20: break
            else: line = test
        if line and y > margin+20: c.drawString(x, y, line); y -= lh
        y -= lh*0.4

# ── PDF Text Replace helpers ───────────────────────────────────────────────────
def extract_blocks_with_pos(page_plumber):
    words = page_plumber.extract_words(keep_blank_chars=False, x_tolerance=3, y_tolerance=3, extra_attrs=['fontname','size'])
    if not words: return []
    lines = {}
    for w in words:
        y = round(w['top'], 1)
        if y not in lines: lines[y] = []
        lines[y].append(w)
    blocks, prev_y, cur_words = [], None, []
    for y in sorted(lines.keys()):
        ws = sorted(lines[y], key=lambda w: w['x0'])
        lsize = ws[0].get('size', 10) or 10
        if prev_y is not None and (y - prev_y) > lsize * 1.6:
            if cur_words:
                all_ws = sorted(cur_words, key=lambda w: (round(w['top']), w['x0']))
                blocks.append({'text': ' '.join(w['text'] for w in all_ws), 'x0': min(w['x0'] for w in all_ws), 'y0': min(w['top'] for w in all_ws), 'x1': max(w['x1'] for w in all_ws), 'y1': max(w['bottom'] for w in all_ws), 'size': all_ws[0].get('size',10) or 10, 'fontname': all_ws[0].get('fontname','')})
            cur_words = list(ws)
        else:
            cur_words.extend(ws)
        prev_y = y
    if cur_words:
        all_ws = sorted(cur_words, key=lambda w: (round(w['top']), w['x0']))
        blocks.append({'text': ' '.join(w['text'] for w in all_ws), 'x0': min(w['x0'] for w in all_ws), 'y0': min(w['top'] for w in all_ws), 'x1': max(w['x1'] for w in all_ws), 'y1': max(w['bottom'] for w in all_ws), 'size': all_ws[0].get('size',10) or 10, 'fontname': all_ws[0].get('fontname','')})
    return blocks

def translate_blocks_batch(texts, api_key, model, style):
    if not texts: return []
    combined = '\n|||BLOCK|||\n'.join(texts)
    prompt = f"Dịch từng đoạn sau từ tiếng Anh sang tiếng Việt. {STYLE_MAP.get(style,'')} Giữ nguyên separator |||BLOCK||| giữa các đoạn. Chỉ trả về bản dịch:\n\n{combined}"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    r = requests.post(url, json={"contents":[{"parts":[{"text":prompt}]}],"generationConfig":{"temperature":0.1,"maxOutputTokens":8192}}, timeout=60)
    r.raise_for_status()
    result = r.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    parts = result.split('|||BLOCK|||')
    while len(parts) < len(texts): parts.append('')
    return [p.strip() for p in parts[:len(texts)]]

def ensure_page_fonts(pdf, page):
    resources = page.obj.get('/Resources')
    if resources is None:
        page.obj['/Resources'] = pdf.make_indirect(pikepdf.Dictionary())
        resources = page.obj['/Resources']
    fonts = resources.get('/Font')
    if fonts is None:
        resources['/Font'] = pikepdf.Dictionary()
        fonts = resources['/Font']
    # Use standard PDF fonts with Identity-H encoding for Unicode support
    if '/F1' not in fonts:
        fonts['/F1'] = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name('/Font'),
            Subtype=pikepdf.Name('/Type1'),
            BaseFont=pikepdf.Name('/Helvetica'),
            Encoding=pikepdf.Name('/WinAnsiEncoding'),
        ))
    if '/F2' not in fonts:
        fonts['/F2'] = pdf.make_indirect(pikepdf.Dictionary(
            Type=pikepdf.Name('/Font'),
            Subtype=pikepdf.Name('/Type1'),
            BaseFont=pikepdf.Name('/Helvetica-Bold'),
            Encoding=pikepdf.Name('/WinAnsiEncoding'),
        ))

def encode_pdf_string(text):
    """Encode text safely for PDF - strip diacritics for latin-1 compatibility."""
    import unicodedata
    # Normalize and remove combining marks for fonts without Unicode support
    normalized = unicodedata.normalize('NFD', text)
    ascii_text = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    # Escape PDF special chars
    safe = ascii_text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
    try:
        return b'(' + safe.encode('latin-1', errors='replace') + b')'
    except:
        return b'(' + safe.encode('ascii', errors='replace') + b')'

def build_translated_stream(blocks, translations, page_h):
    lines = [b'BT']
    for block, trans in zip(blocks, translations):
        if not trans.strip(): continue
        x0 = block['x0']
        y = page_h - block['y1']
        fs = max(6, block['size'])
        is_bold = 'Bold' in block.get('fontname','')
        font_ref = b'/F2' if is_bold else b'/F1'
        avail_w = max(100, block['x1'] - x0)
        chars_per_line = max(10, int(avail_w / (fs * 0.52)))
        words = trans.split()
        text_lines = []
        cur = ''
        for word in words:
            test = (cur + ' ' + word).strip()
            if len(test) > chars_per_line and cur:
                text_lines.append(cur); cur = word
            else: cur = test
        if cur: text_lines.append(cur)
        lh = fs * 1.4
        lines.append(font_ref + f' {fs:.1f} Tf'.encode())
        lines.append(b'0 0 0 rg')
        lines.append(f'{x0:.2f} {y:.2f} Td'.encode())
        for i, tl in enumerate(text_lines):
            hex_str = encode_pdf_string(tl)
            if i == 0: lines.append(hex_str + b' Tj')
            else:
                lines.append(f'0 {-lh:.2f} Td'.encode())
                lines.append(hex_str + b' Tj')
    lines.append(b'ET')
    return b'\n'.join(lines)

def strip_text_from_stream(content_bytes):
    text = content_bytes.decode('latin-1', errors='replace')
    cleaned = re.sub(r'BT.*?ET', '', text, flags=re.DOTALL)
    return cleaned.encode('latin-1', errors='replace')

# ── Translation jobs ───────────────────────────────────────────────────────────
def run_job(job_id, filepath, api_key, model, style, mode, page_from, page_to):
    p = progress_store[job_id]
    output_path = filepath.replace('.pdf', f'_VI_{mode}.pdf')

    def log(msg): p['log'].append(msg); print(msg)

    try:
        if mode == 'replace':
            # ── REPLACE MODE: fitz redact + TextWriter with NotoSans Unicode ──
            FONT_REGULAR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSans-Regular.ttf')
            FONT_BOLD = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'NotoSans-Bold.ttf')

            # Download fonts if missing
            import urllib.request, ssl
            ssl_ctx = ssl.create_default_context()
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            if not os.path.exists(FONT_REGULAR):
                log("Đang tải font NotoSans...")
                try:
                    urllib.request.urlretrieve('https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf', FONT_REGULAR)
                except:
                    import urllib.request as ur
                    req = ur.Request('https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Regular.ttf')
                    with ur.urlopen(req, context=ssl_ctx) as r, open(FONT_REGULAR, 'wb') as f:
                        f.write(r.read())
            if not os.path.exists(FONT_BOLD):
                try:
                    urllib.request.urlretrieve('https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf', FONT_BOLD)
                except:
                    import urllib.request as ur
                    req = ur.Request('https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoSans/NotoSans-Bold.ttf')
                    with ur.urlopen(req, context=ssl_ctx) as r, open(FONT_BOLD, 'wb') as f:
                        f.write(r.read())

            with fitz.open(filepath) as doc:
                with pdfplumber.open(filepath) as plumber:
                    total = len(doc)
                    pf = max(1, page_from) - 1
                    pt_e = min(total, page_to or total) - 1
                    p['total'] = pt_e - pf + 1

                    font_reg = fitz.Font(fontfile=FONT_REGULAR)
                    font_bold = fitz.Font(fontfile=FONT_BOLD)

                    for i in range(pf, pt_e + 1):
                        p['current'] = i - pf + 1
                        log(f"[{p['current']}/{p['total']}] Trang {i+1}...")

                        page = doc[i]
                        plumber_page = plumber.pages[i]
                        pw, ph = page.rect.width, page.rect.height

                        # Get text blocks with positions from fitz
                        fitz_blocks = page.get_text("dict")["blocks"]
                        text_blocks = [b for b in fitz_blocks if b.get("type") == 0]

                        if not text_blocks:
                            log("  (trang trống)"); continue

                        # Extract text per block
                        block_texts = []
                        for b in text_blocks:
                            txt = ' '.join(
                                s["text"] for line in b.get("lines", [])
                                for s in line.get("spans", [])
                            ).strip()
                            block_texts.append(txt)

                        # Translate
                        try:
                            translations = translate_blocks_batch(block_texts, api_key, model, style)
                            log(f"  ✓ dịch xong ({len(text_blocks)} blocks)")
                        except Exception as e:
                            log(f"  ✗ lỗi dịch: {e}")
                            translations = block_texts

                        # Redact original text (white out)
                        # Xóa từng bbox text gốc, giữ nguyên hình ảnh/diagram
                        for b in text_blocks:
                            page.add_redact_annot(fitz.Rect(b["bbox"]), fill=(1, 1, 1))
                        page.apply_redactions()

                        tw = fitz.TextWriter(page.rect)

                        def wrap_lines(font, fontsize, avail_w, words):
                            lines = []
                            cur = ''
                            for word in words:
                                test = (cur + ' ' + word).strip()
                                if font.text_length(test, fontsize=fontsize) > avail_w and cur:
                                    lines.append(cur); cur = word
                                else: cur = test
                            if cur: lines.append(cur)
                            return lines

                        sizes = [s.get("size", 10)
                                 for b in text_blocks
                                 for line in b.get("lines", [])
                                 for s in line.get("spans", [])]
                        base_size = sorted(sizes)[len(sizes)//2] if sizes else 10

                        for b, trans in zip(text_blocks, translations):
                            if not trans.strip(): continue
                            bbox = fitz.Rect(b["bbox"])
                            x0, y0 = bbox.x0, bbox.y0
                            avail_w = bbox.width
                            avail_h = bbox.height

                            spans = [s for line in b.get("lines", []) for s in line.get("spans", [])]
                            is_bold = any('Bold' in s.get("font", "") for s in spans)
                            b_size = spans[0].get("size", base_size) if spans else base_size
                            is_heading = is_bold and b_size > base_size

                            chosen_font = font_bold if is_heading else font_reg
                            target_fs = 11 if is_heading else 9

                            # Shrink font nếu cần để vừa bbox, tối thiểu 7pt
                            fs_use = target_fs
                            for fs_try in [target_fs, target_fs - 1, target_fs - 2, 7]:
                                lines_out = wrap_lines(chosen_font, fs_try, avail_w, trans.split())
                                total_h = len(lines_out) * fs_try * 1.5
                                if total_h <= avail_h * 1.0:
                                    fs_use = fs_try
                                    break
                            else:
                                fs_use = 7
                                lines_out = wrap_lines(chosen_font, fs_use, avail_w, trans.split())

                            lh = fs_use * 1.5
                            y = y0 + fs_use
                            for line in lines_out:
                                if y > bbox.y1 + fs_use: break
                                tw.append((x0, y), line, font=chosen_font, fontsize=fs_use)
                                y += lh

                        tw.write_text(page)

                        if i < pt_e: time.sleep(6)

                # Extract selected pages only
                out = fitz.open()
                out.insert_pdf(doc, from_page=pf, to_page=pt_e)
                out.save(output_path)
                out.close()

        elif mode == 'text':
            with pdfplumber.open(filepath) as pdf:
                total = len(pdf.pages)
                pf = max(1, page_from) - 1
                pt_e = min(total, page_to or total) - 1
                p['total'] = pt_e - pf + 1
                c = rl_canvas.Canvas(output_path)
                for i in range(pf, pt_e + 1):
                    p['current'] = i - pf + 1
                    log(f"[{p['current']}/{p['total']}] Trang {i+1}...")
                    page = pdf.pages[i]
                    blocks = extract_blocks(page)
                    if not blocks:
                        log("  (trang trống)")
                        c.setPageSize((page.width, page.height))
                        c.showPage(); continue
                    full_text = "\n\n".join(b['text'] for b in blocks)
                    try:
                        translated = gemini_translate_text(full_text, api_key, model, style)
                        trans_blocks = [t.strip() for t in re.split(r'\n{2,}', translated)]
                        while len(trans_blocks) < len(blocks): trans_blocks.append('')
                        log(f"  ✓ dịch xong")
                    except Exception as e:
                        log(f"  ✗ lỗi: {e}")
                        trans_blocks = [b['text'] for b in blocks]
                    render_text_page(c, page.width, page.height, blocks, trans_blocks)
                    c.showPage()
                    if i < pt_e: time.sleep(6)
                c.save()

        else:  # scan
            pdf_doc = fitz.open(filepath)
            total = len(pdf_doc)
            pf = max(1, page_from) - 1
            pt_e = min(total, page_to or total) - 1
            p['total'] = pt_e - pf + 1
            c = rl_canvas.Canvas(output_path)
            for i in range(pf, pt_e + 1):
                p['current'] = i - pf + 1
                log(f"[{p['current']}/{p['total']}] Trang {i+1}...")
                page = pdf_doc[i]
                mat = fitz.Matrix(150/72, 150/72)
                pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
                img_bytes = pix.tobytes("jpeg")
                try:
                    result = gemini_vision(img_bytes, api_key, model, style)
                    log(f"  ✓ vision xong")
                except Exception as e:
                    log(f"  ✗ lỗi: {e}")
                    result = {'left':{'type':'text','blocks':[{'text':f'[Lỗi: {e}]'}]},'right':{'type':'text','blocks':[]}}

                for side in ['left', 'right']:
                    data = result[side]
                    if not data['blocks']: continue
                    if data['type'] == 'diagram':
                        half = Image.open(io.BytesIO(img_bytes))
                        w, h = half.size
                        box = (0,0,w//2,h) if side=='left' else (w//2,0,w,h)
                        cropped = half.crop(box)
                        buf = io.BytesIO(); cropped.save(buf, format='JPEG', quality=85); buf.seek(0)
                        c.setPageSize((595, 842))
                        c.drawImage(buf, 0, 0, 595, 842)
                        c.showPage()
                    else:
                        render_scan_page(c, data['blocks'])
                        c.showPage()
                if i < pt_e: time.sleep(13)
            c.save()
            pdf_doc.close()

        p['done'] = True
        p['output_path'] = output_path
        p['status'] = 'done'
        log(f"\n✅ Xong → {output_path}")

    except Exception as e:
        p['done'] = True
        p['error'] = str(e)
        p['status'] = 'error'
        log(f"\n✗ Lỗi: {e}")

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/translate', methods=['POST'])
def translate():
    file = request.files.get('file')
    if not file: return jsonify({'error': 'Thiếu file'}), 400

    upload_dir = '/tmp/pdf_translator'
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)

    job_id = str(int(time.time() * 1000))
    progress_store[job_id] = {
        'status': 'running', 'current': 0, 'total': 0,
        'log': [], 'done': False, 'output_path': None, 'error': None
    }

    t = threading.Thread(target=run_job, args=(
        job_id, filepath,
        request.form.get('api_key'),
        request.form.get('model', 'gemini-1.5-flash'),
        request.form.get('style', 'natural'),
        request.form.get('mode', 'text'),
        int(request.form.get('page_from', 1)),
        int(request.form.get('page_to', 999))
    ))
    t.daemon = True
    t.start()

    return jsonify({'job_id': job_id})

@app.route('/progress/<job_id>')
def get_progress(job_id):
    p = progress_store.get(job_id)
    if not p: return jsonify({'error': 'Job không tồn tại'}), 404
    return jsonify(p)

@app.route('/download/<job_id>')
def download(job_id):
    p = progress_store.get(job_id)
    if not p or not p.get('output_path'): return jsonify({'error': 'Chưa có file'}), 404
    return send_file(p['output_path'], as_attachment=True, download_name=os.path.basename(p['output_path']))

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("🚀 PDF Translator Pro đang chạy tại http://localhost:5000")
    app.run(debug=False, port=5000)
