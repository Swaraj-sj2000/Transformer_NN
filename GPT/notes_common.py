from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import Paragraph, Spacer, HRFlowable, Table, TableStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

W, H = A4
PW = W - 4.4 * cm

C_NAVY   = colors.HexColor("#1A1A2E")
C_BLUE   = colors.HexColor("#0F3460")
C_TEAL   = colors.HexColor("#00897B")
C_PURPLE = colors.HexColor("#5E35B1")
C_ORANGE = colors.HexColor("#E65100")
C_RED    = colors.HexColor("#C62828")
C_GREEN  = colors.HexColor("#2E7D32")
C_LGRAY  = colors.HexColor("#F5F5F5")
C_TEXT   = colors.HexColor("#1C1C1C")
C_MUTED  = colors.HexColor("#555555")
C_CODEBG = colors.HexColor("#EEF2FF")

SS = getSampleStyleSheet()

def mk(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=SS[parent], **kw)

sTitle = mk("sTitle","Title",fontSize=26,textColor=C_NAVY,alignment=TA_CENTER,spaceAfter=6,fontName="Helvetica-Bold")
sSubT  = mk("sSubT","Normal",fontSize=12,textColor=C_MUTED,alignment=TA_CENTER,spaceAfter=18,fontName="Helvetica")
sH2    = mk("sH2","Heading2",fontSize=12,textColor=C_BLUE,spaceBefore=12,spaceAfter=4,fontName="Helvetica-Bold")
sH3    = mk("sH3","Heading3",fontSize=10.5,textColor=C_PURPLE,spaceBefore=7,spaceAfter=3,fontName="Helvetica-Bold")
sBody  = mk("sBody","Normal",fontSize=9.5,textColor=C_TEXT,leading=14,spaceAfter=5,alignment=TA_JUSTIFY,fontName="Helvetica")
sBul   = mk("sBul","Normal",fontSize=9.5,textColor=C_TEXT,leading=13,spaceAfter=3,leftIndent=14,fontName="Helvetica")
sBul2  = mk("sBul2","Normal",fontSize=9,textColor=C_TEXT,leading=12,spaceAfter=2,leftIndent=28,fontName="Helvetica")
sCode  = mk("sCode","Normal",fontSize=8,textColor=colors.HexColor("#1A237E"),leading=11,fontName="Courier",backColor=C_CODEBG,borderPad=6,spaceAfter=7)
sMath  = mk("sMath","Normal",fontSize=10,textColor=C_ORANGE,leading=15,fontName="Courier-Bold",spaceAfter=4,alignment=TA_CENTER)
sNote  = mk("sNote","Normal",fontSize=9,textColor=C_MUTED,leading=12,fontName="Helvetica-Oblique",leftIndent=10,spaceAfter=4)

def HR(): return HRFlowable(width="100%",thickness=0.5,color=colors.HexColor("#BDBDBD"),spaceAfter=5,spaceBefore=5)
def SP(n=6): return Spacer(1,n)
def h2(t): return Paragraph(t,sH2)
def h3(t): return Paragraph(t,sH3)
def body(t): return Paragraph(t,sBody)
def bul(t): return Paragraph(f"&#8226;&nbsp;{t}",sBul)
def bul2(t): return Paragraph(f"&#8211;&nbsp;{t}",sBul2)
def math(t):
    t2 = t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return Paragraph(t2,sMath)
def note(t): return Paragraph(f"<i>Note: {t}</i>",sNote)

def code(t):
    t2 = t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
    return Paragraph(t2.replace("\n","<br/>").replace(" ","&nbsp;"),sCode)

def B(t): return f"<b>{t}</b>"
def I(t): return f"<i>{t}</i>"
def warn(t): return Paragraph(f"<b>Common mistake:</b> {t}",mk("w","Normal",fontSize=9.5,textColor=C_RED,leading=13,fontName="Helvetica",leftIndent=8,spaceAfter=4))
def good(t): return Paragraph(f"<b>Key insight:</b> {t}",mk("g","Normal",fontSize=9.5,textColor=C_GREEN,leading=13,fontName="Helvetica",leftIndent=8,spaceAfter=4))

def keybox(title,items,col=None):
    if col is None: col=C_BLUE
    hdr=[[Paragraph(f"<b>{title}</b>",mk("kh","Normal",fontSize=9.5,textColor=colors.white,fontName="Helvetica-Bold"))]]
    rows=[[Paragraph(f"&#8226; {x}",mk("kb","Normal",fontSize=9,leading=12,textColor=C_TEXT,fontName="Helvetica"))] for x in items]
    t=Table(hdr+rows,colWidths=[PW])
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(0,0),col),("BACKGROUND",(0,1),(-1,-1),C_LGRAY),("BOX",(0,0),(-1,-1),0.6,col),("INNERGRID",(0,1),(-1,-1),0.3,colors.HexColor("#BDBDBD")),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),8)]))
    return t

def warnbox(title,items): return keybox(title,items,col=C_RED)
def greenbox(title,items): return keybox(title,items,col=C_GREEN)

def cmp_table(headers,rows,col_ratios=None):
    if col_ratios is None:
        w=PW/len(headers); col_widths=[w]*len(headers)
    else:
        col_widths=[PW*r for r in col_ratios]
    hrow=[Paragraph(f"<b>{h}</b>",mk(f"th{i}","Normal",fontSize=9,textColor=colors.white,fontName="Helvetica-Bold")) for i,h in enumerate(headers)]
    brows=[[Paragraph(str(c),mk(f"td{i}{j}","Normal",fontSize=8.5,leading=12,textColor=C_TEXT,fontName="Helvetica")) for j,c in enumerate(r)] for i,r in enumerate(rows)]
    t=Table([hrow]+brows,colWidths=col_widths)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),C_NAVY),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,C_LGRAY]),("GRID",(0,0),(-1,-1),0.4,colors.HexColor("#BDBDBD")),("TOPPADDING",(0,0),(-1,-1),4),("BOTTOMPADDING",(0,0),(-1,-1),4),("LEFTPADDING",(0,0),(-1,-1),5),("VALIGN",(0,0),(-1,-1),"TOP")]))
    return t

def qa_table(qa_pairs):
    rows=[]
    for i,(q,a) in enumerate(qa_pairs):
        rows.append([Paragraph(f"<b>Q{i+1}.</b> {q}",mk(f"qq{i}","Normal",fontSize=9,leading=13,textColor=C_NAVY,fontName="Helvetica-Bold")),Paragraph(a,mk(f"aa{i}","Normal",fontSize=8.5,leading=12,textColor=C_TEXT,fontName="Helvetica",alignment=TA_JUSTIFY))])
    t=Table(rows,colWidths=[PW*0.35,PW*0.65])
    t.setStyle(TableStyle([("ROWBACKGROUNDS",(0,0),(-1,-1),[colors.white,C_LGRAY]),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#BDBDBD")),("VALIGN",(0,0),(-1,-1),"TOP"),("TOPPADDING",(0,0),(-1,-1),5),("BOTTOMPADDING",(0,0),(-1,-1),5),("LEFTPADDING",(0,0),(-1,-1),6)]))
    return t

def make_footer(label):
    def footer(canvas,doc):
        canvas.saveState()
        canvas.setFont("Helvetica",7.5)
        canvas.setFillColor(C_MUTED)
        canvas.drawString(2.2*cm,1.2*cm,label)
        canvas.drawRightString(W-2.2*cm,1.2*cm,f"Page {doc.page}")
        canvas.restoreState()
    return footer

def chapter_header(story,number,title,subtitle="",accent=C_TEAL):
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    story.append(Paragraph(f"Chapter {number}",mk(f"cn{number}","Normal",fontSize=11,textColor=accent,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=2)))
    story.append(Paragraph(title,mk(f"ct{number}","Normal",fontSize=19,textColor=C_NAVY,fontName="Helvetica-Bold",alignment=TA_CENTER,spaceAfter=4)))
    if subtitle:
        story.append(Paragraph(subtitle,mk(f"cs{number}","Normal",fontSize=10,textColor=C_MUTED,fontName="Helvetica-Oblique",alignment=TA_CENTER,spaceAfter=6)))
    story.append(HRFlowable(width="100%",thickness=2,color=accent,spaceAfter=12))
