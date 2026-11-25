"""PII scrubbing pipeline: rasterize -> face blur -> targeted redaction -> clean PDF."""
import sys, os, re, json
import numpy as np, cv2, pymupdf
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

ZOOM = 2.5           # ~180 DPI
PAD = 0.15           # face bbox padding
BASE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE, '..', 'models', 'blaze_face_short_range.tflite')

# ---------- manual regions, in PDF points: (page, kind, x0, y0, x1, y1) ----------
# kind: 'black' = solid fill, 'obscure' = pixelate + heavy blur, 'face' = face blur fallback
MANUAL = {
 'file1': [
  (2,'black', 53,103,279,190),      # "BAIN & COMPANY" wall lettering
  (2,'face',  234,286,281,381),     # colleague face, bottom-right of wall photo
  (2,'black', 680,153,765,163),     # byline card 1 (Geoffrey Weston / Rostislav Khomenko)
  (2,'black', 688,256,770,266),     # byline card 2
  (3,'black', 682,150,783,237),     # PEDA honor board + portraits + emblem
  (4,'black', 167,107,226,144),     # schedule board naming SUN Foundation
  (4,'black', 92,253,134,266),      # building name sign
  (4,'black', 28,300,68,315),       # placards with helpline text
  (4,'black', 265,258,292,292),     # phone-number sign + "Sec-66 Mohali" note
  (7,'black', 795,227,836,250),     # "Raghav 11/07/2x" signature on sketch
  (6,'face',  606,201,638,242),     # far-right man, horse photo (low-res, missed by detector)
  (6,'face',  571,220,596,252),     # woman in cap, horse photo
 ],
 'file2': [
  (3,'black', 732,186,762,205),     # sketch signature
  (4,'black', 253,311,381,418),     # PEDA honor board + portraits
  (4,'black', 191,115,239,169),     # DC office glass name board
  (5,'black', 603,185,648,205),     # instagram handle on gym photo
  (6,'black', 560,190,632,205),     # "Civil Surgeon / Deputy Commissioner SAS Nagar" strip
  (8,'black', 217,303,483,349),     # Hindustan Times masthead + URLs
  (10,'black',364,134,478,149),     # BAIN & COMPANY sign, lobby photo
  (10,'black',528,130,702,149),     # BAIN & COMPANY sign, reception photo
  (10,'black',668,118,687,136),     # Bain logo, top-right corner
  (10,'black',285,126,319,208),     # Bain logo fragment, left edge
  (10,'black',190,341,289,348),     # byline (Michael Sion, John Wenzel, ...)
  (10,'black',188,407,261,414),     # byline
  (10,'black',169,454,223,461),     # byline
 ],
 'file3': [
  (3,'black', 76,76,269,99),        # "#IFCT DI- Color Connect" header
  (3,'black', 442,254,471,275),     # AMEX logo
  (4,'black', 156,351,191,380),     # marathon bib number
  (4,'black', 71,412,150,422),      # sponsor text on backdrop
  (6,'black', 90,206,297,249),      # "Media Club IITM" sign
 ],
 'file5': [
  (2,'black', 150,158,292,184),     # "AMARANTH" store sign (mother's business)
  (2,'black', 696,165,826,320),     # "BKC Consultants / XBorder Tax" glass sign (father's firm)
  (3,'black', 292,131,440,212),     # "SMU International Connections" screen + flag + camp name
  (3,'black', 600,131,645,148),     # "Bandhan Mutual Fund" logo
  (4,'black', 132,136,165,179),     # banner lettering "TH"
  (4,'black', 95,211,185,244),      # banner lettering "LAU"
  (4,'black', 265,209,300,244),     # banner lettering "CH"
  (4,'black', 105,379,300,410),     # banner tagline / "Follow ..." handle
  (4,'black', 326,81,364,127),      # "Anpita" building name + Kolkata address
  (4,'black', 483,161,529,204),     # Lions Club banner with chapter name
  (4,'black', 304,214,342,327),     # second "The Launch Pad" banner
  (4,'obscure', 595,292,743,410),   # Hindi news clipping (names, school, photo)
 ],
 'file4': [
  (5,'obscure', 18,66,346,294),     # local news clipping (text + photo)
  (6,'obscure', 223,126,461,247),   # training screenshot with names / phone numbers
  (7,'black', 496,55,618,85),       # "Bitka IBEX" climbing wall logos
  (5,'face',  374,123,417,183),     # police officer, news photo (missed by detector)
 ],
}
# words to redact from the text layer (matched case-insensitively, punctuation stripped)
WORDS = {
 'file1': {'bain','bain.com','peda'},
 'file2': {'raghav','bain','peda'},
 'file3': {'akash','stern','iit'},
 'file4': {'amex','prayaas','nyu','stern'},
 'file5': {'launch','pad','tlp','smu','amaranth','bkc'},
}
# lines whose de-spaced text contains any of these are redacted whole (spaced-letter footers)
LINE_PATTERNS = {'file2': ['NYUSTERN'], 'file1': [], 'file3': [], 'file4': [], 'file5': []}

# ---------- face detection ----------
cascade_f = cv2.CascadeClassifier(os.path.join(BASE, '..', 'models', 'haarcascade_frontalface_default.xml'))
cascade_a = cv2.CascadeClassifier(os.path.join(BASE, '..', 'models', 'haarcascade_frontalface_alt2.xml'))
cascade_p = cv2.CascadeClassifier(os.path.join(BASE, '..', 'models', 'haarcascade_profileface.xml'))
_mp_opts = vision.FaceDetectorOptions(base_options=mpp.BaseOptions(model_asset_path=MODEL),
                                      min_detection_confidence=0.4)
mp_det = vision.FaceDetector.create_from_options(_mp_opts)

def mp_faces(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    res = mp_det.detect(img)
    out = []
    for d in res.detections:
        b = d.bounding_box
        out.append((b.origin_x, b.origin_y, b.width, b.height, float(d.categories[0].score)))
    return out

def haar_faces(gray, strict=False):
    out = []
    mn = 7 if strict else 5
    for c, m in ((cascade_f, mn), (cascade_a, mn - 1)):
        r = c.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=m, minSize=(20, 20))
        if len(r): out += [tuple(int(v) for v in x) for x in r]
    for g, flip in ((gray, False), (cv2.flip(gray, 1), True)):
        r = cascade_p.detectMultiScale(g, scaleFactor=1.08, minNeighbors=mn + 2, minSize=(20, 20))
        W = gray.shape[1]
        if len(r):
            out += [((int(W - x - w) if flip else int(x)), int(y), int(w), int(h)) for x, y, w, h in r]
    return out

def tiles(h, w, size, step):
    ys = list(range(0, max(h - size, 0) + 1, step)); xs = list(range(0, max(w - size, 0) + 1, step))
    if ys[-1] + size < h: ys.append(max(0, h - size))
    if xs[-1] + size < w: xs.append(max(0, w - size))
    for y in ys:
        for x in xs:
            yield x, y, min(size, w - x), min(size, h - y)

def skin_ratio(bgr):
    if bgr.size == 0: return 0.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m = cv2.inRange(ycc, (0, 133, 77), (255, 180, 135))
    return float(cv2.countNonZero(m)) / (bgr.shape[0] * bgr.shape[1])

def detect_faces(bgr, img_regions):
    """Faces live inside photos: detect only within embedded image regions, at several scales.
    Returns list of (x,y,w,h, region) in page pixels."""
    H, W = bgr.shape[:2]
    results = []
    for (x0, y0, x1, y1) in img_regions:
        x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(W, int(x1)), min(H, int(y1))
        rw, rh = x1 - x0, y1 - y0
        if rw < 24 or rh < 24: continue
        crop = bgr[y0:y1, x0:x1]
        if crop.std() < 6: continue
        region = (x0, y0, x1, y1)
        mp_c, haar_c = [], []
        # MediaPipe on whole region at 1x/2x
        for scale in (1.0, 2.0):
            up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else crop
            if max(up.shape[:2]) > 5000: continue
            for bx, by, bw, bh, sc in mp_faces(up):
                if sc >= 0.55: mp_c.append((bx/scale, by/scale, bw/scale, bh/scale, sc))
        # MediaPipe on tiles (upscaled to 512 px) for small faces in group photos; stricter score
        for size in (120, 200, 320):
            if rw < size and rh < size: continue
            for tx, ty, tw, th in tiles(rh, rw, size, size // 2):
                t = crop[ty:ty+th, tx:tx+tw]
                if t.size == 0 or t.std() < 6: continue
                scale = 512.0 / max(tw, th)
                up = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                for bx, by, bw, bh, sc in mp_faces(up):
                    fx, fy, fw, fh = tx + bx/scale, ty + by/scale, bw/scale, bh/scale
                    if sc >= 0.72:
                        mp_c.append((fx, fy, fw, fh, sc))
                    elif sc >= 0.45 and fw <= 70:
                        # small, lower-confidence hit: accept only if clearly skin-toned
                        box = crop[max(0,int(fy)):int(fy+fh), max(0,int(fx)):int(fx+fw)]
                        if box.size and skin_ratio(box) >= 0.55:
                            mp_c.append((fx, fy, fw, fh, sc))
        # Haar frontal/profile on the region at 1x and 2x
        for scale in (1.0, 2.0):
            up = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC) if scale != 1 else crop
            if max(up.shape[:2]) > 4000: continue
            g = cv2.equalizeHist(cv2.cvtColor(up, cv2.COLOR_BGR2GRAY))
            for bx, by, bw, bh in haar_faces(g, strict=True):
                haar_c.append((bx/scale, by/scale, bw/scale, bh/scale))
        keep = []
        def torso_under_face(a):
            # a large box directly below a smaller, confident box is a body, not a face
            ax, ay, aw, ah, _ = a
            if aw * ah < 0.08 * rw * rh: return False
            for (bx, by, bw, bh, bsc) in mp_c:
                if bsc < 0.6 or bw * bh > 0.4 * aw * ah: continue
                bcx, bcy = bx + bw / 2, by + bh / 2
                if ax - 0.2 * aw <= bcx <= ax + 1.2 * aw and ay - 1.5 * bh <= bcy <= ay + 0.3 * ah:
                    return True
            return False
        for (x, y, w, h, sc) in mp_c:
            if torso_under_face((x, y, w, h, sc)): continue
            box = crop[max(0,int(y)):int(y+h), max(0,int(x)):int(x+w)]
            if box.size == 0: continue
            sr = skin_ratio(box)
            frac = (w * h) / float(rw * rh)
            need = 0.06 if w < 60 else (0.15 if w < 120 else 0.22)
            if frac > 0.12: need = max(need, 0.35)   # a real face this large is mostly skin
            if sr < need: continue
            if w > 0.95 * rw or h > 0.95 * rh: continue
            keep.append((x, y, w, h))
        for (x, y, w, h) in haar_c:
            box = crop[max(0,int(y)):int(y+h), max(0,int(x)):int(x+w)]
            if box.size == 0: continue
            def similar(m):   # a confirming DNN box must be roughly the same size
                r = (m[2] * m[3]) / float(w * h)
                return 0.4 <= r <= 2.5 and iou((x, y, w, h), m[:4]) > 0.3
            confirmed = any(similar(m) for m in mp_c)
            sr = skin_ratio(box)
            small = w <= 0.35 * min(rw, rh) + 1
            if torso_under_face((x, y, w, h, 1.0)): continue
            if confirmed or (sr >= 0.5 and small):
                keep.append((x, y, w, h))
        for (x, y, w, h) in merge_boxes(keep):
            results.append((x + x0, y + y0, w, h, region))
    return results

def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0]+a[2], a[1]+a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0]+b[2], b[1]+b[3]
    iw = max(0, min(ax1, bx1) - max(ax0, bx0)); ih = max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter == 0: return 0.0
    union = a[2]*a[3] + b[2]*b[3] - inter
    return max(inter / union, 0.6 * inter / min(a[2]*a[3], b[2]*b[3]))

def merge_boxes(boxes, thr=0.35):
    boxes = [tuple(float(v) for v in b) for b in boxes if b[2] > 6 and b[3] > 6]
    merged = True
    while merged:
        merged = False
        out = []
        while boxes:
            a = boxes.pop()
            keep = []
            for b in boxes:
                if iou(a, b) > thr:
                    x0 = min(a[0], b[0]); y0 = min(a[1], b[1])
                    x1 = max(a[0]+a[2], b[0]+b[2]); y1 = max(a[1]+a[3], b[1]+b[3])
                    a = (x0, y0, x1-x0, y1-y0); merged = True
                else: keep.append(b)
            boxes = keep; out.append(a)
        boxes = out
    return boxes

# ---------- effects ----------
def blur_region(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(W, int(x1)), min(H, int(y1))
    if x1 - x0 < 2 or y1 - y0 < 2: return
    roi = img[y0:y1, x0:x1]
    k = int(max(x1-x0, y1-y0) * 0.9) | 1; k = max(k, 31)
    b = cv2.GaussianBlur(roi, (k, k), 0)
    b = cv2.GaussianBlur(b, (k, k), 0)
    img[y0:y1, x0:x1] = b

def blur_face(img, x, y, w, h, region=None):
    px, py = w * PAD, h * PAD
    x0, y0, x1, y1 = x - px, y - py, x + w + px, y + h + py
    if region is not None:   # keep the blur inside the photo it belongs to
        x0, y0 = max(x0, region[0]), max(y0, region[1]); x1, y1 = min(x1, region[2]), min(y1, region[3])
    blur_region(img, x0, y0, x1, y1)

def obscure_region(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0)); x1, y1 = min(W, int(x1)), min(H, int(y1))
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, (x1-x0)//24), max(1, (y1-y0)//24)), interpolation=cv2.INTER_AREA)
    pix = cv2.resize(small, (x1-x0, y1-y0), interpolation=cv2.INTER_LINEAR)
    k = (max(x1-x0, y1-y0) // 3) | 1
    img[y0:y1, x0:x1] = cv2.GaussianBlur(pix, (k, k), 0)

def black_region(img, x0, y0, x1, y1):
    H, W = img.shape[:2]
    cv2.rectangle(img, (max(0,int(x0)), max(0,int(y0))), (min(W,int(x1)), min(H,int(y1))), (0, 0, 0), -1)

def norm(w): return re.sub(r'[^a-z0-9.]', '', w.lower()).strip('.')

def text_targets(page, key):
    rects = []
    words = WORDS.get(key, set())
    for x0, y0, x1, y1, w, *_ in page.get_text('words'):
        if norm(w) in words: rects.append((x0, y0, x1, y1))
    pats = LINE_PATTERNS.get(key, [])
    if pats:
        for b in page.get_text('dict')['blocks']:
            for ln in b.get('lines', []):
                t = re.sub(r'\s', '', ''.join(s['text'] for s in ln['spans'])).upper()
                if any(p in t for p in pats): rects.append(tuple(ln['bbox']))
    return rects

def process(src, dst, key, preview_dir=None):
    doc = pymupdf.open(src)
    out = pymupdf.open()
    log = []
    for pno, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=pymupdf.Matrix(ZOOM, ZOOM), alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
        s = ZOOM
        # visible (clipped) extents of each placed image; fall back to raw image bboxes
        img_regions = [tuple(v*s for v in b['bbox']) for b in page.get_text('dict')['blocks'] if b['type'] == 1]
        if not img_regions:
            img_regions = [tuple(v*s for v in info['bbox']) for info in page.get_image_info()]
        faces = detect_faces(img, img_regions)
        for (x, y, w, h, region) in faces: blur_face(img, x, y, w, h, region)
        n_text = 0
        for (x0, y0, x1, y1) in text_targets(page, key):
            black_region(img, x0*s - 2, y0*s - 2, x1*s + 2, y1*s + 2); n_text += 1
        n_manual = 0
        for (p, kind, x0, y0, x1, y1) in MANUAL.get(key, []):
            if p != pno: continue
            n_manual += 1
            if kind == 'black': black_region(img, x0*s, y0*s, x1*s, y1*s)
            elif kind == 'obscure': obscure_region(img, x0*s, y0*s, x1*s, y1*s)
            elif kind == 'face': blur_region(img, x0*s, y0*s, x1*s, y1*s)
        log.append(dict(page=pno, faces=len(faces), text_redactions=n_text, manual=n_manual))
        ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        newpage = out.new_page(width=page.rect.width, height=page.rect.height)
        newpage.insert_image(newpage.rect, stream=jpg.tobytes())
        if preview_dir:
            cv2.imwrite(os.path.join(preview_dir, f'{key}_p{pno}.png'), cv2.resize(img, None, fx=1.4/ZOOM, fy=1.4/ZOOM, interpolation=cv2.INTER_AREA))
    out.set_metadata({})
    out.del_xml_metadata()
    out.save(dst, garbage=4, deflate=True, clean=True)
    out.close(); doc.close()
    return log

if __name__ == '__main__':
    jobs = json.loads(sys.argv[1]); prev = sys.argv[2] if len(sys.argv) > 2 else None
    for key, src, dst in jobs:
        log = process(src, dst, key, prev)
        print(key, '->', dst)
        for l in log: print('  ', l)
