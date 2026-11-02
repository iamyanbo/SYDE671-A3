from pathlib import Path
from playwright.sync_api import sync_playwright

root = Path(__file__).resolve().parent

print_style = """
@page { size: A4; }
@media print {
    body { background: white; font-size: 10.5pt; line-height: 1.4; }
    .wrap { padding: 0; margin: 0; max-width: none; }
    h1 { font-size: 36pt; margin: 10px 0; }
    h2 { font-size: 20pt; margin: 24px 0 10px; }
    h3 { font-size: 13pt; margin: 20px 0 8px; }
    p { margin: 0 0 9px; }
    pre { font-size: 8pt; line-height: 1.35; padding: 10px; white-space: pre-wrap; overflow-wrap: anywhere; }
    pre, table, figure, .equation { break-inside: avoid; page-break-inside: avoid; }
    h2, h3, summary, .section-label { break-after: avoid; page-break-after: avoid; }
    figure { margin: 9px 0 14px; }
    figure img { max-height: 235mm; object-fit: contain; }
    .inputs img, .gallery.sources img { height: 185px; }
    .inputs { max-width: 100%; grid-template-columns: repeat(2, 1fr); }
    .gallery.sources { grid-template-columns: repeat(3, 1fr); }
    details { display: block !important; margin: 10px 0; break-inside: avoid; }
    .print-lead { break-inside: avoid; }
    .print-section { break-inside: avoid; }
    h2:not(:first-of-type) { break-before: page; }
    summary { padding: 8px 10px; list-style: none; }
    summary::-webkit-details-marker { display: none; }
    .table-wrap { overflow: visible; }
    table { font-size: 9pt; margin: 9px 0 14px; }
    th, td { padding: 5px 7px; }
    .equation { margin: 9px 0; }
    .section-label { margin-top: 6px; }
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1040, "height": 900})
    page.goto((root / "index.html").as_uri())
    page.wait_for_function("Array.from(document.images).every(i => i.complete && i.naturalWidth > 0)")
    page.evaluate("document.fonts.ready")
    page.locator("details").evaluate_all("elements => elements.forEach(e => e.open = true)")
    page.evaluate("""() => {
        for (const heading of Array.from(document.querySelectorAll('h3'))) {
            let end = heading.nextElementSibling;
            while (end && end.tagName !== 'PRE' && !['H2', 'H3'].includes(end.tagName))
                end = end.nextElementSibling;
            if (!end || end.tagName !== 'PRE') continue;
            const group = document.createElement('div');
            group.className = 'print-lead';
            heading.before(group);
            let element = heading;
            while (element) {
                const next = element.nextElementSibling;
                group.append(element);
                if (element === end) break;
                element = next;
            }
        }
        const manual = document.querySelector('#manual-mosaic').closest('.print-lead');
        const result = manual.nextElementSibling;
        const section = document.createElement('div');
        section.className = 'print-section';
        manual.before(section);
        section.append(manual, result);
    }""")
    page.add_style_tag(content=print_style)
    page.pdf(path=str(root / "A3_Writeup.pdf"), format="A4", print_background=True,
             margin={"top": "14mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
             display_header_footer=True, header_template="<div></div>",
             footer_template='<div style="width:100%;text-align:center;font-size:9px;color:#66716c"><span class="pageNumber"></span> / <span class="totalPages"></span></div>')
    browser.close()

print("Created A3_Writeup.pdf.")
