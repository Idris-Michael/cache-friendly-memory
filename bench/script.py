"""A 40-turn ops chat with eight facts planted early, and a quiz that checks they survived.

Fictional client and people. The facts sit in turns 1-8, so by turn 40 every strategy except full
history has had to compress them.
"""

from __future__ import annotations

_ROSTER = "\n".join(
    f"- {name}: {sector}, retainer tier {tier}, reporting day {day}"
    for name, sector, tier, day in [
        ("Northgate Dental", "dental group", "B", "Monday"),
        ("Halden Outdoor", "outdoor retail", "A", "Tuesday"),
        ("Kestrel Legal", "law firm", "B", "Wednesday"),
        ("Brightwater Spa", "day spa", "C", "Thursday"),
        ("Morrow Bikes", "bike retail", "A", "Friday"),
        ("Pellam Recruitment", "recruitment", "B", "Monday"),
        ("Ashby Kitchens", "kitchen fitting", "C", "Tuesday"),
        ("Corran Labs", "B2B software", "A", "Wednesday"),
        ("Fenwick Pets", "pet supplies", "B", "Thursday"),
        ("Lyle & Hart", "fashion", "A", "Friday"),
    ]
    * 5
)

SYSTEM = f"""You are the operations assistant for a small UK analytics agency.

How you work:
- Answer in two to four plain sentences. British English. No emojis, no filler.
- Never invent numbers, IDs, names or dates. If something was not said, say it was not said.
- You cannot send email, change ad accounts or publish anything; you only draft and advise.
- When a task has an owner and a deadline, repeat both back.
- Prefer GA4 DebugView evidence over assumptions when diagnosing tracking.
- Smart Bidding needs at least 15 conversions a month before it is switched on.
- Broad match only with Smart Bidding active.
- Negative keywords come from the search terms report, never from guesses.

Services: GA4 audits and rebuilds, Google Tag Manager clean-ups, Consent Mode v2, server-side
tagging, Meta Pixel and Conversions API, Google Ads and Shopping restructures, weekly reporting.

Client roster (for reference; the current chat is about a new client):
{_ROSTER}
"""

TURNS = [
    "New client today: Tarn Valley Cycles, an online bike shop in Kendal.",
    "Their GA4 property ID is G-7Q2K9XW4. Please keep that handy.",
    "Main contact is Priya Shah, she is head of ecommerce.",
    "Agreed budget for the tracking rebuild is £2,400, fixed fee.",
    "The deadline for the rebuild is 12 December.",
    "They use Cookiebot as their consent platform.",
    "Their Shopify store is on the Dawn theme, version 15.",
    "Purchases are double counting in GA4: roughly 30% of orders appear twice.",
    "What would you check first for the double counting?",
    "OK. Can you draft a short plan for day one?",
    "Fine. What evidence should I capture in DebugView?",
    "Their developer says the purchase tag fires on the thank-you page and in a Shopify app.",
    "That sounds like the duplicate. How do we confirm it without touching live config?",
    "Good. What should the Consent Mode check look like for them?",
    "They also run Meta ads. Do we need to look at the Pixel now or later?",
    "Let's park Meta. Draft the message to their developer asking for read access to GTM.",
    "Shorter please, and less formal.",
    "Thanks. What could go wrong when we remove the app's purchase tag?",
    "How would we spot a drop in revenue reporting after the change?",
    "Write a two-line summary I can paste into our CRM.",
    "Now a different topic: our own weekly report template needs a tidy-up. Any ideas?",
    "Which three metrics would you put first?",
    "Agreed. Should ROAS be blended or per channel?",
    "Per channel it is. How do we show data gaps honestly?",
    "Good. Back to work items: what's my to-do list for tomorrow?",
    "Add a reminder to ask about their returns process; refunds may need a refund event.",
    "What does a GA4 refund event need?",
    "Would server-side tagging help this client?",
    "Not now then. What's the simplest version of the fix?",
    "Draft a status update for the client after day one.",
    "Make it friendlier.",
    "What should I log in our run history for this job?",
    "Is there any risk to the Ads account from fixing GA4?",
    "How long should we watch the numbers after the fix?",
    "Draft the closing checklist for the rebuild.",
    "What would a good handover note include?",
    "OK, last thing for today: anything we've forgotten?",
    "Thanks. Summarise what we decided today in three bullets.",
    "Remind me what the invoice should say.",
    "Great, that's all for now.",
]

# (question, accepted answers). Asked after turn 40 and never added to the history.
QUIZ = [
    ("What is the client's GA4 property ID?", ["G-7Q2K9XW4"]),
    ("Who is the main contact at the client?", ["Priya Shah"]),
    ("What is the agreed fee for the rebuild?", ["2,400", "2400"]),
    ("What is the rebuild deadline?", ["12 December", "December 12", "12th December"]),
    ("Which consent platform does the client use?", ["Cookiebot"]),
    ("Which Shopify theme is the store on?", ["Dawn"]),
    ("Roughly what share of orders is double counting?", ["30%", "30 per cent", "30 percent"]),
    ("What is the client's name and where are they based?", ["Tarn Valley"]),
]
