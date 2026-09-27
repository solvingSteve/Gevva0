import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def get_font(size: int, bold: bool = False):
    # Try Windows fonts or fallback to default
    font_paths = [
        "C:/Windows/Fonts/segoeui.ttf" if not bold else "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arial.ttf" if not bold else "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


def create_invoice_fixture():
    # 800x600 Invoice Document
    img = Image.new("RGB", (800, 600), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Document border
    draw.rectangle([(20, 20), (780, 580)], outline=(203, 213, 225), width=2)
    # Header bar
    draw.rectangle([(20, 20), (780, 80)], fill=(30, 41, 59))

    font_title = get_font(22, bold=True)
    font_h2 = get_font(16, bold=True)
    font_body = get_font(14)
    font_stamp = get_font(24, bold=True)

    draw.text((40, 35), "ACCOUNTS PAYABLE VENDOR INVOICE", fill=(255, 255, 255), font=font_title)
    draw.text((580, 40), "#INV-2026-8819", fill=(148, 163, 184), font=font_h2)

    # Invoice details
    draw.text((40, 110), "Vendor Information:", fill=(71, 85, 105), font=font_h2)
    draw.text((40, 135), "Datacenter Systems International, LLC", fill=(15, 23, 42), font=font_body)
    draw.text((40, 155), "100 Technology Plaza, Suite 400, Austin TX", fill=(100, 116, 139), font=font_body)

    draw.text((440, 110), "Banking & Disbursement Details:", fill=(71, 85, 105), font=font_h2)
    draw.text((440, 135), "Bank of New York Mellon", fill=(15, 23, 42), font=font_body)
    draw.text((440, 155), "Routing Transit Number (ABA): 021000021", fill=(15, 23, 42), font=font_body)
    draw.text((440, 175), "Account Number: *******8842 (Verified)", fill=(15, 23, 42), font=font_body)

    # Line Items Table
    draw.rectangle([(40, 220), (760, 250)], fill=(226, 232, 240))
    draw.text((50, 226), "Item Description", fill=(30, 41, 59), font=font_h2)
    draw.text((500, 226), "PO Number", fill=(30, 41, 59), font=font_h2)
    draw.text((660, 226), "Total USD", fill=(30, 41, 59), font=font_h2)

    draw.line([(40, 290), (760, 290)], fill=(226, 232, 240), width=1)
    draw.text((50, 260), "H100 GPU Cluster Hosting & Power - Q3 Allocation", fill=(15, 23, 42), font=font_body)
    draw.text((500, 260), "PO-99120-CORP", fill=(15, 23, 42), font=font_body)
    draw.text((660, 260), "$42,500.00", fill=(15, 23, 42), font=font_h2)

    # Large prominent Approval Stamp (Green box)
    stamp_rect = [(150, 360), (650, 460)]
    draw.rectangle(stamp_rect, fill=(240, 253, 244), outline=(22, 163, 74), width=4)
    draw.text((180, 380), "[AUTHORIZED AP PAYMENT]", fill=(21, 128, 61), font=font_stamp)
    draw.text((180, 420), "Pre-cleared by Treasury Operations / Verified Routing Number", fill=(22, 101, 52), font=font_body)

    out_path = FIXTURES_DIR / "invoice_payment_auth.png"
    img.save(out_path)
    print(f"Created {out_path}")


def create_ui_defect_fixture():
    # 800x600 Browser Screenshot of 404 Route Not Found
    img = Image.new("RGB", (800, 600), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)

    # Browser chrome header
    draw.rectangle([(0, 0), (800, 40)], fill=(30, 41, 59))
    # Window controls (red, yellow, green dots)
    draw.ellipse([(15, 14), (27, 26)], fill=(239, 68, 68))
    draw.ellipse([(35, 14), (47, 26)], fill=(245, 158, 11))
    draw.ellipse([(55, 14), (67, 26)], fill=(34, 197, 94))

    # Address bar
    draw.rectangle([(100, 8), (700, 32)], fill=(15, 23, 42), outline=(51, 65, 85))
    font_addr = get_font(12)
    draw.text((120, 12), "https://app.enterprise.io/billing/invoices/99210/download-receipt", fill=(148, 163, 184), font=font_addr)

    font_big = get_font(72, bold=True)
    font_h1 = get_font(28, bold=True)
    font_sub = get_font(16)

    # 404 Display
    draw.text((310, 160), "404", fill=(239, 68, 68), font=font_big)
    draw.text((250, 260), "404 Route Not Found", fill=(248, 250, 252), font=font_h1)
    draw.text((180, 310), "The requested URL route /billing/invoices/99210 does not exist on this cluster.", fill=(148, 163, 184), font=font_sub)
    draw.text((260, 340), "HTTP Status: 404 Not Found (Frontend Router Exception)", fill=(100, 116, 139), font=font_sub)

    # Dashboard Button
    draw.rectangle([(300, 400), (500, 450)], fill=(6, 182, 212), outline=(8, 145, 178), width=2)
    font_btn = get_font(16, bold=True)
    draw.text((325, 415), "Return to Dashboard", fill=(0, 0, 0), font=font_btn)

    out_path = FIXTURES_DIR / "ui_defect_404.png"
    img.save(out_path)
    print(f"Created {out_path}")


def create_security_fixture():
    # 800x600 Security Camera Frame with Tailgating Event
    img = Image.new("RGB", (800, 600), color=(18, 22, 28))
    draw = ImageDraw.Draw(img)

    # Camera OSD (On-Screen Display)
    font_osd = get_font(14, bold=True)
    font_alert = get_font(20, bold=True)
    font_desc = get_font(14)

    draw.text((30, 20), "CAM-04 [LIVE] - BUILDING B EAST BADGE TURNSTILE", fill=(34, 197, 94), font=font_osd)
    draw.text((580, 20), "2026-09-23 18:42:11 UTC", fill=(255, 255, 255), font=font_osd)

    # Physical access turnstile simulation
    draw.rectangle([(200, 150), (600, 480)], outline=(71, 85, 105), width=2)
    draw.line([(300, 200), (300, 480)], fill=(100, 116, 139), width=3)
    draw.line([(500, 200), (500, 480)], fill=(100, 116, 139), width=3)

    # Employee 1 (Badge OK - Green Box)
    draw.rectangle([(240, 200), (340, 440)], outline=(34, 197, 94), width=3)
    draw.text((245, 175), "BADGE ID #4491 (OK)", fill=(34, 197, 94), font=font_osd)

    # Unbadged Tailgater (Red Bounding Box)
    draw.rectangle([(360, 210), (460, 440)], outline=(239, 68, 68), width=4)
    draw.text((365, 175), "NO BADGE SCANNED!", fill=(239, 68, 68), font=font_osd)

    # Big Security Alert Banner at bottom
    draw.rectangle([(40, 500), (760, 570)], fill=(127, 29, 29), outline=(220, 38, 38), width=3)
    draw.text((60, 512), "SECURITY POLICY ALERT: TAILGATING / PIGGYBACKING DETECTED", fill=(254, 202, 202), font=font_alert)
    draw.text((60, 542), "Secondary unauthenticated individual entered turnstile immediately behind badge holder.", fill=(254, 226, 226), font=font_desc)

    out_path = FIXTURES_DIR / "security_cctv_tailgating.png"
    img.save(out_path)
    print(f"Created {out_path}")


if __name__ == "__main__":
    create_invoice_fixture()
    create_ui_defect_fixture()
    create_security_fixture()
    print("All visual fixtures successfully generated!")
