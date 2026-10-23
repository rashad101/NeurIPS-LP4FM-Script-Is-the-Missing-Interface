"""Condition 5 — code-mixing: matrix Indic language, English insertions.

Insertion is subject to the **free-morpheme constraint**: only a bare,
un-suffixed token may be replaced. ``समयसीमा`` ("deadline") contains the lexicon
key ``समय`` ("time"), but it is a single word — replacing inside it would
produce ungrammatical output and the resulting drop would measure our own
corruption rather than a register anyone speaks. Matching is therefore
whole-token, never substring.

Two insertion types are counted separately, because they are different
linguistic phenomena and the paper reports them apart:

``lexical``
    A content word swapped for its English equivalent (``समय`` -> ``time``).

``discourse``
    An English discourse marker spliced in (``actually``, ``basically``) without
    displacing anything — the commonest form of Hinglish in the wild.

A cell where nothing fired is a byte-identical copy of the native item, so it is
marked ``transform_applied=False`` and dropped by the metrics stage rather than
scored as an unlabelled duplicate of the baseline.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from functools import lru_cache

from .. import textnorm

#: Native word runs. Whole-token matching is what enforces the free-morpheme
#: constraint, so this must never be a substring search. The shared tokeniser
#: keeps matras attached, without which `राजधानी` would not match its own
#: lexicon key.
_TOKEN = textnorm.WORD_TOKEN

#: English discourse markers speakers splice into an Indic matrix clause.
DISCOURSE_MARKERS: tuple[str, ...] = (
    "actually",
    "basically",
    "obviously",
    "seriously",
    "anyway",
    "you know",
    "I mean",
    "by the way",
    "of course",
    "at least",
)

#: Content-word lexicons, native -> English. Deliberately content words only:
#: function words (``है``, ``बहुत``, ``कम``) are excluded because swapping them
#: changes the grammar rather than the register.
#:
#: Hindi is roughly 1.5x the size of the others, matching the spec's
#: matrix-Hindi framing. The per-language counts land in ``suite_manifest.json``
#: so non-Hindi code-mix results can be read with that in mind.
_LEXICONS: dict[str, dict[str, str]] = {
    "hi": {
        "समय": "time", "दिन": "day", "साल": "year", "महीना": "month",
        "सप्ताह": "week", "घंटा": "hour", "पानी": "water", "खाना": "food",
        "घर": "house", "स्कूल": "school", "कॉलेज": "college", "किताब": "book",
        "कंप्यूटर": "computer", "फोन": "phone", "गाड़ी": "car", "सड़क": "road",
        "शहर": "city", "गाँव": "village", "देश": "country", "दुनिया": "world",
        "सरकार": "government", "कंपनी": "company", "दफ्तर": "office",
        "नौकरी": "job", "पैसा": "money", "बाजार": "market", "दुकान": "shop",
        "अस्पताल": "hospital", "डॉक्टर": "doctor", "दवा": "medicine",
        "बीमारी": "disease", "परिवार": "family", "दोस्त": "friend",
        "बच्चा": "child", "शिक्षक": "teacher", "छात्र": "student",
        "परीक्षा": "exam", "कक्षा": "class", "सवाल": "question",
        "जवाब": "answer", "समस्या": "problem", "समाधान": "solution",
        "योजना": "plan", "बैठक": "meeting", "रिपोर्ट": "report",
        "खबर": "news", "तस्वीर": "picture", "संगीत": "music", "खेल": "game",
        "टीम": "team", "मैच": "match", "रेल": "train", "टिकट": "ticket",
        "होटल": "hotel", "कमरा": "room", "दरवाजा": "door", "खिड़की": "window",
        "रसोई": "kitchen", "बगीचा": "garden", "पेड़": "tree", "फूल": "flower",
        "नदी": "river", "पहाड़": "mountain", "मौसम": "weather", "बारिश": "rain",
        "सूरज": "sun", "चाँद": "moon", "रात": "night", "सुबह": "morning",
        "भाषा": "language", "कहानी": "story", "फिल्म": "movie",
        "बैंक": "bank", "पुलिस": "police", "अदालत": "court", "कानून": "law",
    },
    "bn": {
        "সময়": "time", "দিন": "day", "বছর": "year", "মাস": "month",
        "সপ্তাহ": "week", "ঘণ্টা": "hour", "জল": "water", "খাবার": "food",
        "বাড়ি": "house", "স্কুল": "school", "কলেজ": "college", "বই": "book",
        "কম্পিউটার": "computer", "ফোন": "phone", "গাড়ি": "car", "রাস্তা": "road",
        "শহর": "city", "গ্রাম": "village", "দেশ": "country", "পৃথিবী": "world",
        "সরকার": "government", "কোম্পানি": "company", "অফিস": "office",
        "চাকরি": "job", "টাকা": "money", "বাজার": "market", "দোকান": "shop",
        "হাসপাতাল": "hospital", "ডাক্তার": "doctor", "ওষুধ": "medicine",
        "রোগ": "disease", "পরিবার": "family", "বন্ধু": "friend",
        "শিশু": "child", "শিক্ষক": "teacher", "ছাত্র": "student",
        "পরীক্ষা": "exam", "প্রশ্ন": "question", "উত্তর": "answer",
        "সমস্যা": "problem", "সমাধান": "solution", "পরিকল্পনা": "plan",
        "সভা": "meeting", "খবর": "news", "ছবি": "picture", "গান": "music",
        "খেলা": "game", "দল": "team", "ট্রেন": "train", "টিকিট": "ticket",
        "হোটেল": "hotel", "ঘর": "room", "দরজা": "door", "গাছ": "tree",
        "ফুল": "flower", "নদী": "river", "পাহাড়": "mountain", "বৃষ্টি": "rain",
        "সূর্য": "sun", "রাত": "night", "সকাল": "morning", "ভাষা": "language",
    },
    "ta": {
        "நேரம்": "time", "நாள்": "day", "ஆண்டு": "year", "மாதம்": "month",
        "வாரம்": "week", "மணி": "hour", "தண்ணீர்": "water", "உணவு": "food",
        "வீடு": "house", "பள்ளி": "school", "கல்லூரி": "college", "புத்தகம்": "book",
        "கணினி": "computer", "தொலைபேசி": "phone", "கார்": "car", "சாலை": "road",
        "நகரம்": "city", "கிராமம்": "village", "நாடு": "country", "உலகம்": "world",
        "அரசு": "government", "நிறுவனம்": "company", "அலுவலகம்": "office",
        "வேலை": "job", "பணம்": "money", "சந்தை": "market", "கடை": "shop",
        "மருத்துவமனை": "hospital", "மருத்துவர்": "doctor", "மருந்து": "medicine",
        "நோய்": "disease", "குடும்பம்": "family", "நண்பன்": "friend",
        "குழந்தை": "child", "ஆசிரியர்": "teacher", "மாணவர்": "student",
        "தேர்வு": "exam", "கேள்வி": "question", "பதில்": "answer",
        "பிரச்சனை": "problem", "திட்டம்": "plan", "கூட்டம்": "meeting",
        "செய்தி": "news", "படம்": "picture", "இசை": "music", "விளையாட்டு": "game",
        "அணி": "team", "ரயில்": "train", "சீட்டு": "ticket", "அறை": "room",
        "கதவு": "door", "மரம்": "tree", "மலர்": "flower", "ஆறு": "river",
        "மலை": "mountain", "மழை": "rain", "சூரியன்": "sun", "இரவு": "night",
        "காலை": "morning", "மொழி": "language",
    },
    "te": {
        "సమయం": "time", "రోజు": "day", "సంవత్సరం": "year", "నెల": "month",
        "వారం": "week", "గంట": "hour", "నీరు": "water", "ఆహారం": "food",
        "ఇల్లు": "house", "పాఠశాల": "school", "కళాశాల": "college", "పుస్తకం": "book",
        "కంప్యూటర్": "computer", "ఫోన్": "phone", "కారు": "car", "రహదారి": "road",
        "నగరం": "city", "గ్రామం": "village", "దేశం": "country", "ప్రపంచం": "world",
        "ప్రభుత్వం": "government", "సంస్థ": "company", "కార్యాలయం": "office",
        "ఉద్యోగం": "job", "డబ్బు": "money", "మార్కెట్": "market", "దుకాణం": "shop",
        "ఆసుపత్రి": "hospital", "వైద్యుడు": "doctor", "మందు": "medicine",
        "వ్యాధి": "disease", "కుటుంబం": "family", "స్నేహితుడు": "friend",
        "పిల్లవాడు": "child", "ఉపాధ్యాయుడు": "teacher", "విద్యార్థి": "student",
        "పరీక్ష": "exam", "ప్రశ్న": "question", "సమాధానం": "answer",
        "సమస్య": "problem", "ప్రణాళిక": "plan", "సమావేశం": "meeting",
        "వార్త": "news", "చిత్రం": "picture", "సంగీతం": "music", "ఆట": "game",
        "జట్టు": "team", "రైలు": "train", "టికెట్": "ticket", "గది": "room",
        "తలుపు": "door", "చెట్టు": "tree", "పువ్వు": "flower", "నది": "river",
        "కొండ": "mountain", "వర్షం": "rain", "సూర్యుడు": "sun", "రాత్రి": "night",
        "ఉదయం": "morning", "భాష": "language",
    },
    "mr": {
        "वेळ": "time", "दिवस": "day", "वर्ष": "year", "महिना": "month",
        "आठवडा": "week", "तास": "hour", "पाणी": "water", "अन्न": "food",
        "घर": "house", "शाळा": "school", "महाविद्यालय": "college", "पुस्तक": "book",
        "संगणक": "computer", "दूरध्वनी": "phone", "गाडी": "car", "रस्ता": "road",
        "शहर": "city", "खेडे": "village", "देश": "country", "जग": "world",
        "सरकार": "government", "कंपनी": "company", "कार्यालय": "office",
        "नोकरी": "job", "पैसा": "money", "बाजार": "market", "दुकान": "shop",
        "रुग्णालय": "hospital", "डॉक्टर": "doctor", "औषध": "medicine",
        "आजार": "disease", "कुटुंब": "family", "मित्र": "friend",
        "मूल": "child", "शिक्षक": "teacher", "विद्यार्थी": "student",
        "परीक्षा": "exam", "प्रश्न": "question", "उत्तर": "answer",
        "समस्या": "problem", "योजना": "plan", "सभा": "meeting",
        "बातमी": "news", "चित्र": "picture", "संगीत": "music", "खेळ": "game",
        "संघ": "team", "रेल्वे": "train", "तिकीट": "ticket", "खोली": "room",
        "दरवाजा": "door", "झाड": "tree", "फूल": "flower", "नदी": "river",
        "डोंगर": "mountain", "पाऊस": "rain", "सूर्य": "sun", "रात्र": "night",
        "सकाळ": "morning", "भाषा": "language",
    },
}


@lru_cache(maxsize=None)
def lexicon_for(language: str) -> dict[str, str]:
    """Native -> English content-word lexicon for a language (may be empty)."""
    return dict(_LEXICONS.get((language or "").strip().lower(), {}))


@dataclass
class CodeMixReport:
    """Insertion accounting for one cell."""

    #: Tokens that were legal insertion sites (bare lexicon matches).
    candidates: int = 0
    #: Content words actually swapped for English.
    lexical: int = 0
    #: Discourse markers spliced in.
    discourse: int = 0

    @property
    def insertions(self) -> int:
        return self.lexical + self.discourse

    @property
    def applied(self) -> bool:
        return self.insertions > 0


def code_mix(
    text: str,
    language: str,
    rng: random.Random,
    insertion_rate: float,
    max_insertions: int,
    discourse_rate: float = 0.25,
) -> tuple[str, CodeMixReport]:
    """Insert English into an Indic matrix. Returns ``(text, report)``."""
    lexicon = lexicon_for(language)
    report = CodeMixReport()
    if not lexicon and discourse_rate <= 0.0:
        return text, report

    # Pass 1: find the legal sites. Whole-token only — this is the
    # free-morpheme constraint, and it is why `समयसीमा` is not a site.
    sites = [m for m in _TOKEN.finditer(text) if m.group(0) in lexicon]
    report.candidates = len(sites)

    chosen: set[int] = set()
    for match in sites:
        if report.lexical >= max_insertions:
            break
        if rng.random() < insertion_rate:
            chosen.add(match.start())
            report.lexical += 1

    def swap(match: re.Match[str]) -> str:
        if match.start() in chosen:
            return lexicon[match.group(0)]
        return match.group(0)

    out = _TOKEN.sub(swap, text)

    # Pass 2: at most one discourse marker, spliced between words rather than
    # displacing one — that is what makes it a different phenomenon from a
    # lexical swap, and why the two are counted apart.
    if discourse_rate > 0.0 and report.insertions < max_insertions:
        if rng.random() < discourse_rate:
            marker = rng.choice(DISCOURSE_MARKERS)
            words = out.split(" ")
            if len(words) > 1:
                position = rng.randrange(1, len(words))
                words.insert(position, marker)
                out = " ".join(words)
            else:
                out = f"{marker} {out}"
            report.discourse = 1

    return out, report
