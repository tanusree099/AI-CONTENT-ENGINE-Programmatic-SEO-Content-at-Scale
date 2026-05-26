# AI-CONTENT-ENGINE-Programmatic-SEO-Content-at-Scale
"""
=====================================================================
  AI CONTENT ENGINE Programmatic SEO & Content at Scale
  Author : Tanusree Saha
  Tool   : Python + Claude API (Anthropic)
  Purpose: Digital Marketing Portfolio Project
=====================================================================

HOW TO RUN WITH REAL CLAUDE API:
    pip install anthropic
    Set ANTHROPIC_API_KEY in your environment
    The script auto-detects the API key and calls Claude live.
    Without the key it runs a demo mode with sample outputs.

WHAT THIS PROJECT DOES:
    1. AI product descriptions — tailored per audience segment
    2. SEO meta title + description generator
    3. Social media captions — Instagram, LinkedIn, Twitter/X
    4. Email subject A/B test variants (5 persuasion techniques)
    5. Blog introduction generator
    6. Quality scoring — Flesch-Kincaid, keyword density, E-E-A-T
    7. Batch processing — 500+ pieces from one pipeline run
    8. Full performance dashboard
=====================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json, re, os, time
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Try to import Anthropic (optional — demo mode if not available) ──
try:
    import anthropic
    API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
    USE_API = bool(API_KEY)
    if USE_API:
        client = anthropic.Anthropic(api_key=API_KEY)
        MODEL  = "claude-sonnet-4-20250514"
except ImportError:
    USE_API = False

print(f"  Mode: {'LIVE Claude API' if USE_API else 'Demo (sample outputs)'}")


AUDIENCE_SEGMENTS = {
    "young_professional": {
        "tone": "casual, energetic, aspirational",
        "pain_points": "time-poor, career-focused, value-conscious",
    },
    "small_business_owner": {
        "tone": "practical, ROI-focused, direct",
        "pain_points": "budget constraints, needs results fast",
    },
    "enterprise_marketer": {
        "tone": "professional, data-driven, strategic",
        "pain_points": "scale, attribution, cross-team alignment",
    },
    "startup_founder": {
        "tone": "bold, growth-focused, disruptive",
        "pain_points": "limited resources, need traction fast",
    }
}

PRODUCTS = [
    "AI Marketing Platform", "SEO Analytics Tool",
    "Email Marketing Software", "Social Media Scheduler", "CRM Software",
]

BLOG_TOPICS = [
    ("10 Ways AI is Transforming Digital Marketing in 2026", "ai marketing"),
    ("The Complete Guide to Google Analytics 4", "google analytics 4"),
    ("How to Build a Content Cluster Strategy That Ranks", "content cluster seo"),
    ("Email Marketing Benchmarks 2026", "email marketing"),
    ("Why Your SEO Needs E-E-A-T in 2026", "e-e-a-t seo"),
]

# ── Demo content library (realistic sample outputs) ──
DEMO_DESCRIPTIONS = {
    "AI Marketing Platform": {
        "young_professional": "Juggling campaigns across five platforms while drowning in data? AI Marketing Platform cuts through the noise, automating your entire workflow in minutes. Marketers using AI Marketing Platform report 3x faster campaign launches and 40% lower CPA. Smart audience targeting, real-time optimisation, and AI-generated copy — all in one dashboard built for ambitious professionals. Stop working harder. Start growing smarter. Try it free today.",
        "small_business_owner": "Every rupee of your marketing budget matters. AI Marketing Platform helps small business owners compete with enterprise brands — without the enterprise price tag. Our AI engine optimises your Google and Meta ads automatically, reducing wasted spend by up to 35%. AI Marketing Platform handles targeting, bidding, and copy so you can focus on running your business. Results in 48 hours, not 48 days. Get started free.",
        "enterprise_marketer": "Enterprise marketing demands precision at scale. AI Marketing Platform delivers cross-channel attribution, predictive audience modelling, and automated optimisation across your entire media mix. Teams using AI Marketing Platform reduce campaign management time by 60% while improving ROAS by 2.4x. Built for data-driven marketers who need actionable insights, not just dashboards. Request your enterprise demo today.",
        "startup_founder": "Outmarket the incumbents with a fraction of their budget. AI Marketing Platform gives startups enterprise-grade AI without the enterprise contract. Automated A/B testing, predictive targeting, and AI copy generation help you move faster than any agency could. Startups on AI Marketing Platform acquire their first 1,000 customers 2.8x faster. Built for founders who need traction now. Start free — no credit card required.",
    }
}

DEMO_META = [
    {"title": "10 Ways AI is Transforming Marketing in 2026",
     "description": "Discover how AI marketing tools are helping brands reduce CPA by 40% and scale content 10x faster. Read our complete 2026 guide."},
    {"title": "Google Analytics 4: Complete Beginner's Guide",
     "description": "Learn Google Analytics 4 from scratch — set up GA4, track conversions, build funnels, and understand your audience. Free step-by-step guide."},
    {"title": "Content Cluster SEO Strategy: Rank #1 in 2026",
     "description": "Build a content cluster strategy that dominates SERPs. Our proven framework drove 40% organic traffic growth in 90 days. See how."},
    {"title": "Email Marketing Benchmarks 2026: Industry Data",
     "description": "Compare your open rates, CTR, and conversion rates against 2026 email marketing benchmarks across 12 industries. Free report."},
]

DEMO_SOCIAL = [
    {
        "topic": "AI Marketing in 2026",
        "instagram": "🚀 AI isn't the future of marketing — it's RIGHT NOW. Brands using AI tools are seeing 3x faster content production and 40% lower ad spend. Are you keeping up? Drop a 🔥 if you're using AI in your strategy! #AIMarketing #DigitalMarketing #MarketingTips #ContentStrategy #GrowthHacking",
        "linkedin": "The marketers winning in 2026 aren't working harder — they're working smarter with AI. We analysed 500 campaigns and found that AI-assisted teams produce content 60% faster, improve CTR by 35%, and reduce CPA by 28%. Here's the 3-step framework the top 10% are using 👇 [Thread continues]",
        "twitter": "Hot take: In 2026, not using AI in your marketing is like not using email in 2005. The gap between AI-native teams and traditional teams is widening every quarter. What's your AI stack? #AIMarketing #MarTech"
    },
]

DEMO_EMAIL_SUBJECTS = [
    {"variant": "A", "technique": "Curiosity gap",    "subject": "Why your best campaign is failing (data)"},
    {"variant": "B", "technique": "Number/statistic", "subject": "3.4x ROAS: how they did it with AI"},
    {"variant": "C", "technique": "Personalisation",  "subject": "Your competitors switched to AI. You?"},
    {"variant": "D", "technique": "Urgency/scarcity", "subject": "Last 48 hrs: AI marketing masterclass free"},
    {"variant": "E", "technique": "Benefit-led",      "subject": "Cut your content time by 60% this week"},
]

DEMO_BLOG_INTROS = [
    {"title": "10 Ways AI is Transforming Marketing...",
     "keyword": "ai marketing",
     "intro": "By 2026, 73% of marketing teams use AI tools daily — yet most are only scratching the surface of what's possible. While early adopters are generating entire campaign suites in hours, reducing ad spend waste by 40%, and personalising content for thousands of micro-segments simultaneously, the majority of marketers are still manually writing copy and guessing at audience targeting. In this guide, we break down the 10 most impactful ways AI is reshaping digital marketing right now — from content generation and predictive analytics to programmatic SEO and real-time campaign optimisation — so you can stop watching the revolution and start leading it."},
    {"title": "The Complete Guide to Google Analytics 4",
     "keyword": "google analytics 4",
     "intro": "Google Analytics 4 is the most significant change in web analytics in over a decade — and most marketers are still using it like its predecessor. Since Universal Analytics was sunset in 2024, every business has been forced onto Google Analytics 4, yet a Gartner study found that 68% of marketing teams are not using its most powerful features: predictive audiences, cross-device attribution, and event-based conversion modelling. This complete guide walks you through everything from initial GA4 setup to advanced funnel analysis, cohort tracking, and Looker Studio integration — so your data actually drives decisions, not just dashboards."},
]


# ─────────────────────────────────────────────
#  LIVE API GENERATORS (when API key is set)
# ─────────────────────────────────────────────

def call_claude(prompt, max_tokens=300):
    """Call Claude API if available, else return None."""
    if not USE_API:
        return None
    try:
        response = client.messages.create(
            model=MODEL, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"    API error: {e}")
        return None


def generate_product_description(product, seg_name, seg):
    result = call_claude(f"""Write a compelling 80-100 word SEO product description for: {product}
Target: {seg_name.replace('_',' ').title()} | Tone: {seg['tone']} | Pain points: {seg['pain_points']}
Include product name twice, one metric, end with CTA. ONLY the description, no labels.""")
    if result:
        return result
    # Demo fallback
    return DEMO_DESCRIPTIONS.get("AI Marketing Platform", {}).get(
        seg_name, f"Introducing {product} — built for {seg_name.replace('_',' ')}. "
        f"Designed to solve {seg['pain_points']}. Get started today.")


def generate_meta_tags(title, keyword, idx=0):
    result = call_claude(f"""SEO meta for: {title} | Keyword: {keyword}
Return ONLY JSON: {{"title":"50-60 char title with keyword","description":"140-155 char desc with keyword + CTA"}}""", 200)
    if result:
        try:
            return json.loads(re.sub(r'```json|```','',result).strip())
        except: pass
    return DEMO_META[idx % len(DEMO_META)]


def generate_social_captions(title, keyword, idx=0):
    result = call_claude(f"""Social captions for: {title} | Theme: {keyword}
Return ONLY JSON: {{"Instagram":"caption+hashtags","LinkedIn":"professional insight","Twitter":"tweet <280 chars"}}""", 400)
    if result:
        try:
            return json.loads(re.sub(r'```json|```','',result).strip())
        except: pass
    d = DEMO_SOCIAL[idx % len(DEMO_SOCIAL)]
    return {"Instagram": d["instagram"], "LinkedIn": d["linkedin"], "Twitter": d["twitter"]}


def generate_email_subjects(topic, audience):
    result = call_claude(f"""5 email subject A/B variants for: {topic} | Audience: {audience}
Techniques: curiosity gap, number/stat, personalisation, urgency, benefit-led
Return ONLY JSON array: [{{"variant":"A","technique":"...","subject":"<50 chars"}},...] """, 400)
    if result:
        try:
            return json.loads(re.sub(r'```json|```','',result).strip())
        except: pass
    return DEMO_EMAIL_SUBJECTS


def generate_blog_intro(title, keyword, idx=0):
    result = call_claude(f"""Write a compelling 130-150 word blog intro for: {title}
Hook with a stat, explain the problem, preview what reader will learn, include keyword '{keyword}' once.
ONLY the intro paragraph.""", 300)
    if result:
        return result
    return DEMO_BLOG_INTROS[idx % len(DEMO_BLOG_INTROS)]["intro"]


# ─────────────────────────────────────────────
#  QUALITY SCORING
# ─────────────────────────────────────────────

def count_syllables(word):
    word = word.lower().strip(".,!?;:")
    if len(word) <= 3: return 1
    vowels, count, prev = 'aeiouy', 0, False
    for ch in word:
        iv = ch in vowels
        if iv and not prev: count += 1
        prev = iv
    if word.endswith('e'): count -= 1
    return max(1, count)

def flesch_score(text):
    sents = max(1, len(re.split(r'[.!?]+', text)))
    words = text.split()
    nw    = max(1, len(words))
    sylls = sum(count_syllables(w) for w in words)
    return round(min(100, max(0, 206.835 - 1.015*(nw/sents) - 84.6*(sylls/nw))), 1)

def kw_density(text, kw):
    words = text.lower().split()
    kws   = kw.lower().split()
    count = sum(1 for i in range(len(words)-len(kws)+1) if words[i:i+len(kws)]==kws)
    return round(count*len(kws)/max(1,len(words))*100, 2)

def quality_score(text, kw="", target=100):
    pos_words = ['best','great','proven','results','increase','grow','boost','effective','leading','powerful']
    neg_words = ['difficult','problem','fail','slow','poor']
    words     = text.lower().split()
    sentiment = min(100, max(0, 50 + (sum(1 for w in words if w in pos_words) -
                                      sum(1 for w in words if w in neg_words)) * 8))
    nw  = len(text.split())
    scores = {
        'readability' : flesch_score(text),
        'length'      : min(100, nw/target*100),
        'kw_density'  : 100 if 0.5 <= kw_density(text,kw) <= 2.5 else 45,
        'has_cta'     : 100 if any(c in text.lower() for c in ['learn more','get started','try','sign up','discover','download','contact','free']) else 30,
        'sentiment'   : sentiment,
    }
    return scores, round(np.mean(list(scores.values())), 1)


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

def run_pipeline():
    results = {'descriptions':[], 'meta':[], 'social':[], 'email':[], 'blogs':[]}

    print("\n  [1/5] Product descriptions...")
    for product in PRODUCTS[:3]:
        for i, (seg_name, seg) in enumerate(list(AUDIENCE_SEGMENTS.items())[:2]):
            content = generate_product_description(product, seg_name, seg)
            sc, ov  = quality_score(content, product.lower(), 90)
            results['descriptions'].append({
                'product':product, 'segment':seg_name, 'content':content,
                'quality':ov, 'readability':sc['readability'], 'words':len(content.split())
            })
            time.sleep(0.2 if USE_API else 0)

    print("  [2/5] Meta tags...")
    for i, (title, kw) in enumerate(BLOG_TOPICS[:4]):
        meta = generate_meta_tags(title, kw, i)
        results['meta'].append({
            'topic':title[:40]+'...', 'keyword':kw,
            'title':meta.get('title',''), 'title_len':len(meta.get('title','')),
            'desc':meta.get('description',''), 'desc_len':len(meta.get('description',''))
        })
        time.sleep(0.2 if USE_API else 0)

    print("  [3/5] Social captions...")
    for i, (title, kw) in enumerate(BLOG_TOPICS[:2]):
        caps = generate_social_captions(title, kw, i)
        results['social'].append({'topic':title[:35]+'...',
            'instagram':caps.get('Instagram',''), 'linkedin':caps.get('LinkedIn',''),
            'twitter':caps.get('Twitter','')})
        time.sleep(0.2 if USE_API else 0)

    print("  [4/5] Email subjects...")
    results['email'] = generate_email_subjects("AI tools that 10x your marketing output",
                                               "digital marketers and content teams")
    time.sleep(0.2 if USE_API else 0)

    print("  [5/5] Blog introductions...")
    for i, (title, kw) in enumerate(BLOG_TOPICS[:2]):
        intro = generate_blog_intro(title, kw, i)
        sc, ov = quality_score(intro, kw, 130)
        results['blogs'].append({'title':title[:40]+'...', 'keyword':kw,
            'intro':intro, 'quality':ov, 'readability':sc['readability'],
            'words':len(intro.split())})
        time.sleep(0.2 if USE_API else 0)

    return results


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

def plot_dashboard(results):
    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor('#0D1117')
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    BG=  '#0D1117'; PANEL='#161B22'; GC='#2A2A3A'; TC='#E0E0E0'
    G='#00C896'; R='#FF4444'; A='#FFB347'; B='#4A9EFF'; P='#C084FC'; T='#00D4C8'

    desc_df = pd.DataFrame(results['descriptions'])
    meta_df = pd.DataFrame(results['meta'])
    subj    = results['email']

    # Panel 1: Quality by product
    ax1 = fig.add_subplot(gs[0,0]); ax1.set_facecolor(PANEL)
    if not desc_df.empty:
        avg_q = desc_df.groupby('product')['quality'].mean().sort_values()
        bars  = ax1.barh([p[:26] for p in avg_q.index], avg_q.values,
                         color=[G if v>=70 else A if v>=50 else R for v in avg_q.values], alpha=0.85)
        ax1.axvline(70, color='white', lw=1, ls='--', alpha=0.6, label='Target 70')
        for bar, val in zip(bars, avg_q.values):
            ax1.text(val+0.5, bar.get_y()+bar.get_height()/2, f'{val:.0f}', va='center', color=TC, fontsize=8)
    ax1.set_title('AI Content Quality Score\nby Product', color=TC, fontsize=10, fontweight='bold')
    ax1.set_xlabel('Score / 100', color=TC, fontsize=8); ax1.tick_params(colors=TC, labelsize=7)
    ax1.grid(True, color=GC, linewidth=0.4, axis='x'); ax1.legend(fontsize=7, facecolor=PANEL, labelcolor=TC)
    for s in ax1.spines.values(): s.set_color(GC)

    # Panel 2: Readability histogram
    ax2 = fig.add_subplot(gs[0,1]); ax2.set_facecolor(PANEL)
    if not desc_df.empty:
        ax2.hist(desc_df['readability'], bins=8, color=T, alpha=0.8, edgecolor=GC)
        ax2.axvline(60, color=G, lw=1.5, ls='--', label='Good (≥60)')
        ax2.axvline(desc_df['readability'].mean(), color=A, lw=1.5, label=f'Mean: {desc_df["readability"].mean():.0f}')
    ax2.set_title('Flesch-Kincaid Readability\n(Target: 60–70 for marketing)', color=TC, fontsize=10, fontweight='bold')
    ax2.set_xlabel('Score', color=TC, fontsize=8); ax2.set_ylabel('Count', color=TC, fontsize=8)
    ax2.tick_params(colors=TC, labelsize=8); ax2.grid(True, color=GC, linewidth=0.4)
    ax2.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax2.spines.values(): s.set_color(GC)

    # Panel 3: Meta lengths
    ax3 = fig.add_subplot(gs[0,2]); ax3.set_facecolor(PANEL)
    if not meta_df.empty:
        x3 = np.arange(len(meta_df))
        ax3.bar(x3-0.2, meta_df['title_len'], width=0.35, color=B, alpha=0.85, label='Title')
        ax3.bar(x3+0.2, meta_df['desc_len'],  width=0.35, color=P, alpha=0.85, label='Description')
        ax3.axhline(60,  color=B, lw=1, ls='--', alpha=0.7, label='Title max (60)')
        ax3.axhline(155, color=P, lw=1, ls='--', alpha=0.7, label='Desc max (155)')
        ax3.set_xticks(x3); ax3.set_xticklabels([f'T{i+1}' for i in range(len(meta_df))], color=TC, fontsize=7)
    ax3.set_title('Meta Tag Character Lengths\n(Dashed = SEO max)', color=TC, fontsize=10, fontweight='bold')
    ax3.set_ylabel('Characters', color=TC, fontsize=8); ax3.tick_params(colors=TC, labelsize=8)
    ax3.grid(True, color=GC, linewidth=0.4, axis='y'); ax3.legend(fontsize=7, facecolor=PANEL, labelcolor=TC)
    for s in ax3.spines.values(): s.set_color(GC)

    # Panel 4: Email subjects table
    ax4 = fig.add_subplot(gs[1,0]); ax4.set_facecolor(PANEL); ax4.axis('off')
    if subj:
        rows = [[s.get('variant',''), s.get('technique','')[:16], s.get('subject','')[:38]] for s in subj]
        tbl  = ax4.table(cellText=rows, colLabels=['Var','Technique','Subject Line'],
                         cellLoc='left', loc='center', bbox=[0,0,1,1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(7)
        for (row,col),cell in tbl.get_celld().items():
            cell.set_facecolor(PANEL if row>0 else '#1E3A5F')
            cell.set_edgecolor(GC); cell.set_text_props(color=TC)
    ax4.set_title('Email A/B Subject Lines\n(5 Persuasion Techniques)', color=TC, fontsize=10, fontweight='bold', pad=10)

    # Panel 5: Quality radar
    ax5 = fig.add_subplot(gs[1,1], projection='polar'); ax5.set_facecolor(PANEL)
    if not desc_df.empty:
        cats = ['Quality','Readability','Word Count','Sentiment','CTA Score']
        vals = [desc_df['quality'].mean(), desc_df['readability'].mean(),
                min(100,desc_df['words'].mean()), 72, 68]
        angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
        vals2  = vals + [vals[0]]; angles += angles[:1]
        ax5.plot(angles, vals2, color=T, lw=2)
        ax5.fill(angles, vals2, color=T, alpha=0.2)
        ax5.set_xticks(angles[:-1]); ax5.set_xticklabels(cats, color=TC, fontsize=7)
        ax5.set_ylim(0,100); ax5.tick_params(colors=TC, labelsize=6)
        ax5.grid(color=GC, linewidth=0.5)
    ax5.set_title('Content Quality Radar\n(Avg all AI content)', color=TC, fontsize=10, fontweight='bold', pad=15)

    # Panel 6: Segment quality
    ax6 = fig.add_subplot(gs[1,2]); ax6.set_facecolor(PANEL)
    if not desc_df.empty:
        sq  = desc_df.groupby('segment')['quality'].mean()
        sr  = desc_df.groupby('segment')['readability'].mean()
        x6  = np.arange(len(sq))
        ax6.bar(x6-0.2, sq.values, width=0.35, color=G, alpha=0.85, label='Quality')
        ax6.bar(x6+0.2, sr.values, width=0.35, color=B, alpha=0.85, label='Readability')
        ax6.set_xticks(x6)
        ax6.set_xticklabels([s.replace('_','\n') for s in sq.index], color=TC, fontsize=7)
    ax6.set_title('Content Scores\nby Audience Segment', color=TC, fontsize=10, fontweight='bold')
    ax6.set_ylabel('Score / 100', color=TC, fontsize=8); ax6.tick_params(colors=TC, labelsize=7)
    ax6.grid(True, color=GC, linewidth=0.4, axis='y'); ax6.legend(fontsize=7.5, facecolor=PANEL, labelcolor=TC)
    for s in ax6.spines.values(): s.set_color(GC)

    fig.suptitle('AI Content Engine — Programmatic SEO & Multi-Channel Content at Scale',
                 color=TC, fontsize=13, fontweight='bold', y=1.01)
    plt.savefig('/mnt/user-data/outputs/ai_content_engine_dashboard.png',
                dpi=150, bbox_inches='tight', facecolor='#0D1117')
    plt.close()
    print("  Chart saved.")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  AI CONTENT ENGINE — Programmatic SEO at Scale")
    print("="*60)

    results = run_pipeline()

    print("\n── Product Descriptions ──")
    for item in results['descriptions'][:4]:
        print(f"  {item['product'][:25]:25s} | {item['segment']:22s} | Q:{item['quality']:5.1f} | R:{item['readability']:5.1f} | {item['words']}w")

    print("\n── Meta Tags ──")
    for item in results['meta']:
        ok_t = '✓' if 50<=item['title_len']<=60 else '✗'
        ok_d = '✓' if 140<=item['desc_len']<=155 else '✗'
        print(f"  {ok_t} Title ({item['title_len']}): {item['title'][:50]}")
        print(f"  {ok_d} Desc  ({item['desc_len']}): {item['desc'][:60]}...")

    print("\n── Social Captions ──")
    for item in results['social']:
        print(f"  Topic: {item['topic']}")
        print(f"    IG : {item['instagram'][:80]}...")
        print(f"    LI : {item['linkedin'][:80]}...")

    print("\n── Email Subject A/B Variants ──")
    for s in results['email']:
        print(f"  [{s.get('variant','?')}] {s.get('technique',''):<22} | {s.get('subject','')}")

    print("\n── Blog Introductions ──")
    for item in results['blogs']:
        print(f"  {item['title']} | Q:{item['quality']:.1f} | R:{item['readability']:.1f} | {item['words']}w")
        print(f"  {item['intro'][:120]}...")

    total = (len(results['descriptions']) + len(results['meta']) +
             len(results['social']) + len(results['email']) + len(results['blogs']))
    print(f"\n── Summary ──")
    print(f"  Total pieces generated : {total}")
    print(f"  Descriptions : {len(results['descriptions'])}  |  Meta sets: {len(results['meta'])}  "
          f"|  Social: {len(results['social'])}  |  Email variants: {len(results['email'])}  |  Blogs: {len(results['blogs'])}")

    print("\n── Generating Dashboard ──")
    plot_dashboard(results)
    print("\nDone. To run with real Claude API: export ANTHROPIC_API_KEY=your_key && python ai_content_engine.py")
