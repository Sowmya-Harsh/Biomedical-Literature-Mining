import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize

# Run these once in your environment (e.g., at top of notebook)
nltk.download("punkt")
nltk.download("punkt_tab")   # if your NLTK version needs it
nltk.download("stopwords")


def clean_markup(text: str) -> str:
    """
    Clean simple markup/bullets from text (e.g., from HTML or PDFs).
    - Remove <sup>...</sup> citation markers
    - Remove any other simple HTML tags
    - Remove leading bullet characters like "•"
    """
    # Remove <sup>...</sup> blocks
    text = re.sub(r"<sup>.*?</sup>", "", text)
    # Remove any remaining HTML-like tags
    text = re.sub(r"<[^>]+>", "", text)
    # Remove leading bullets on each line
    text = re.sub(r"^\s*•\s*", "", text, flags=re.MULTILINE)
    return text


class TextSummarizer:
    """Extract key sentences from text using NLP (adaptive to query context + title analysis)"""

    def __init__(self):
        # General stop words from NLTK
        self.stop_words = set(stopwords.words("english"))

        # Biomedical action/importance words (universal across domains)
        self.universal_keywords = {
            "study", "research", "investigate", "examine", "analyze", "assess",
            "find", "found", "shows", "demonstrate", "reveal", "indicate",
            "suggest", "conclude", "propose", "determine", "establish",
            "associated", "linked", "relationship", "mechanism", "pathway",
            "important", "significant", "novel", "critical", "essential",
            "role", "function", "process", "method", "approach", "technique",
            "result", "outcome", "finding", "conclusion", "implication",
            "treatment", "therapy", "intervention", "prevention", "disease",
        }

    def extract_query_keywords(self, query: str) -> set:
        """
        Extract important keywords from user's search query.
        These become the focus for summarization.
        """
        words = query.lower().split()
        query_keywords = set()

        for word in words:
            # Remove common punctuation
            word = re.sub(r'[,.\'"()-]', "", word)

            # Skip stop words and very short words
            if word not in self.stop_words and len(word) > 2:
                query_keywords.add(word)

                # Also add word stems for matching variations
                if word.endswith("ies"):
                    query_keywords.add(word[:-3] + "y")
                elif word.endswith("es"):
                    query_keywords.add(word[:-2])
                elif word.endswith("s") and not word.endswith("ss"):
                    query_keywords.add(word[:-1])

        return query_keywords

    def analyze_title_focus(self, title: str, query_keywords: set) -> float:
        """
        Analyze how much the title focuses on the query keywords.
        Returns a focus score (0.0 to 1.0).
        """
        if not title or not query_keywords:
            return 0.0

        title_lower = title.lower()
        title_words = title_lower.split()

        # Remove stop words and strip punctuation
        title_words = [
            w.strip(".,;:!?()[]{}")
            for w in title_words
            if w.lower() not in self.stop_words
        ]

        # Count keyword matches
        keyword_matches = 0.0
        for word in title_words:
            if word in query_keywords:
                keyword_matches += 1.0
            else:
                # Substring match
                for keyword in query_keywords:
                    if keyword in word or word in keyword:
                        keyword_matches += 0.5
                        break

        if len(title_words) > 0:
            focus_score = min(keyword_matches / len(title_words), 1.0)
        else:
            focus_score = 0.0

        return focus_score

    def _get_sentences(self, text: str) -> list[str]:
        """Split text into sentences using NLTK, after cleaning markup."""
        cleaned = clean_markup(text)
        return [s.strip() for s in sent_tokenize(cleaned) if s.strip()]

    def _calculate_sentence_score(
        self,
        sentence: str,
        query_keywords: set = None,
        title_focus: float = 0.5,
    ) -> float:
        """
        Score a sentence based on:
        1. Presence of query keywords (what user searched for)
        2. Presence of universal importance keywords
        3. Sentence length (prefer medium-length sentences)
        4. Title focus (boost if paper strongly focuses on query)
        """
        if query_keywords is None:
            query_keywords = set()

        words = sentence.lower().split()

        # Filter out stop words
        important_words = [
            w.strip(".,;:!?()[]{}")
            for w in words
            if w.lower() not in self.stop_words and len(w) > 2
        ]

        if not important_words:
            return 0.0

        # Query keyword matches
        query_matches = 0.0
        for word in important_words:
            if word in query_keywords:
                query_matches += 1.0
            else:
                for keyword in query_keywords:
                    if keyword in word or word in keyword:
                        query_matches += 0.5
                        break

        # Universal importance keywords
        universal_matches = sum(
            1 for word in important_words if word in self.universal_keywords
        )

        # Length score: prefer 8–25 words
        length_score = 1.0 if 8 <= len(words) <= 25 else 0.5

        query_score = query_matches / len(important_words)
        universal_score = universal_matches / len(important_words)

        base_score = (
            query_score * 0.60
            + universal_score * 0.25
            + length_score * 0.15
        )

        # Boost if title is strongly focused on query
        title_boost = 1.0 + (title_focus * 0.3)

        return base_score * title_boost

    def _extract_key_phrases(self, text: str, num_phrases: int = 5) -> list[str]:
        """Extract key noun phrases from text (simple capitalized-phrase regex)."""
        cleaned = clean_markup(text)
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

        phrases: list[str] = []
        for match in re.finditer(pattern, cleaned):
            phrase = match.group(0).strip()
            if (
                len(phrase.split()) >= 1
                and phrase not in phrases
                and len(phrase) > 3
            ):
                phrases.append(phrase)

        return phrases[:num_phrases]

    def summarize(
        self,
        text: str,
        query: str = "",
        title: str = "",
        num_sentences: int = 3,
        return_phrases: bool = True,
    ) -> dict:
        """
        Summarize text extractively based on user's query and paper title.
        """
        if not text or len(text.strip()) < 30:
            return {
                "summary": text,
                "key_phrases": [],
                "importance_scores": [],
                "num_sentences": 0,
                "title_focus": 0.0,
            }

        # Extract keywords from user's query
        query_keywords = self.extract_query_keywords(query)

        # Analyze title focus on query
        title_focus = self.analyze_title_focus(title, query_keywords)

        # Get sentences
        sentences = self._get_sentences(text)

        if len(sentences) <= num_sentences:
            return {
                "summary": " ".join(sentences),
                "key_phrases": self._extract_key_phrases(text)
                if return_phrases
                else [],
                "importance_scores": [1.0] * len(sentences),
                "num_sentences": len(sentences),
                "title_focus": title_focus,
            }

        # Score each sentence
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._calculate_sentence_score(
                sentence, query_keywords, title_focus=title_focus
            )
            scored_sentences.append(
                {
                    "text": sentence,
                    "score": score,
                    "original_index": i,
                }
            )

        # Top N by score
        top_sentences = sorted(
            scored_sentences, key=lambda x: x["score"], reverse=True
        )[:num_sentences]

        # Preserve original order
        top_sentences = sorted(
            top_sentences, key=lambda x: x["original_index"]
        )

        summary_text = " ".join(s["text"] for s in top_sentences)
        scores = [s["score"] for s in top_sentences]

        key_phrases = (
            self._extract_key_phrases(text) if return_phrases else []
        )

        return {
            "summary": summary_text,
            "key_phrases": key_phrases,
            "importance_scores": scores,
            "num_sentences": len(top_sentences),
            "title_focus": title_focus,
        }

    def get_summary_bullets(
        self,
        text: str,
        query: str = "",
        title: str = "",
        num_bullets: int = 4,
    ) -> list[str]:
        """
        Extract key sentences as bullet points (context-aware + title-aware).
        """
        result = self.summarize(
            text, query=query, title=title, num_sentences=num_bullets
        )

        sentences = self._get_sentences(result["summary"])
        bullets = [f"• {s.strip()}" for s in sentences if s.strip()]

        return bullets if bullets else ["• No summary available"]

    def extract_findings(
        self, text: str, query: str = "", title: str = ""
    ) -> list[str]:
        """Extract specific research findings/conclusions (context-aware + title-aware)."""
        action_keywords = [
            "find",
            "found",
            "shows",
            "demonstrate",
            "reveal",
            "indicate",
            "suggest",
            "conclude",
            "propose",
            "determine",
            "establish",
            "associated",
            "linked",
            "relationship",
            "mechanism",
            "role",
            "probing",
            "responds",
            "preserve",
            "activation",
            "marking",
            "selectively",
            "quality",
            "critical",
            "important",
        ]

        query_keywords = self.extract_query_keywords(query)
        title_focus = self.analyze_title_focus(title, query_keywords)
        sentences = self._get_sentences(text)
        findings: list[str] = []

        for sentence in sentences:
            sentence_lower = sentence.lower()

            has_action = any(
                keyword in sentence_lower for keyword in action_keywords
            )
            has_query_match = any(
                keyword in sentence_lower for keyword in query_keywords
            )

            if title_focus > 0.6:
                if has_query_match:
                    findings.append(sentence.strip())
            elif has_action or has_query_match:
                findings.append(sentence.strip())

        return findings[:3] if findings else ["No specific findings extracted"]

    def extract_methodology(
        self, text: str, query: str = "", title: str = ""
    ) -> str:
        """Extract methodology/approach sentences (context-aware + title-aware)."""
        methodology_keywords = [
            "study",
            "design",
            "method",
            "technique",
            "approach",
            "prospective",
            "longitudinal",
            "retrospective",
            "analyze",
            "examine",
            "investigate",
            "evaluate",
            "assess",
            "measure",
            "compare",
            "review",
            "pathway",
        ]

        sentences = self._get_sentences(text)
        query_keywords = self.extract_query_keywords(query)
        title_focus = self.analyze_title_focus(title, query_keywords)

        # Prefer sentences with methodology + query context
        for sentence in sentences:
            sentence_lower = sentence.lower()
            has_method = any(
                keyword in sentence_lower for keyword in methodology_keywords
            )
            has_query = any(
                keyword in sentence_lower for keyword in query_keywords
            )

            if has_method and has_query:
                return sentence.strip()

        # Fallback: just methodology keywords
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(
                keyword in sentence_lower for keyword in methodology_keywords
            ):
                return sentence.strip()

        return sentences[0].strip() if sentences else "Not available"

    def extract_implications(
        self, text: str, query: str = "", title: str = ""
    ) -> str:
        """Extract clinical/research implications (context-aware + title-aware)."""
        implication_keywords = [
            "implication",
            "clinical",
            "therapeutic",
            "treatment",
            "prevention",
            "important",
            "significant",
            "novel",
            "critical",
            "especially",
            "relevance",
            "disease",
            "advance",
            "opportunity",
            "future",
        ]

        sentences = self._get_sentences(text)
        query_keywords = self.extract_query_keywords(query)
        title_focus = self.analyze_title_focus(title, query_keywords)

        # Look in final sentences first
        for sentence in reversed(sentences[-3:]):
            sentence_lower = sentence.lower()
            has_implication = any(
                keyword in sentence_lower for keyword in implication_keywords
            )
            has_query = any(
                keyword in sentence_lower for keyword in query_keywords
            )

            if has_implication or (has_query and len(sentence) > 20):
                return sentence.strip()

        # Fallback: search all sentences
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(
                keyword in sentence_lower for keyword in implication_keywords
            ):
                return sentence.strip()

        return sentences[-1].strip() if sentences else "Not available"
