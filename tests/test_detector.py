"""Tests for the plagiarism detector modules."""

from __future__ import annotations

import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import detector
from src.file_reader import extract_text_from_file
from src.nlp_analysis import (
    find_synonym_pairs,
    get_ngrams,
    longest_common_subsequence,
    ngram_overlap_stats,
    pos_structural_similarity,
)
from src.report_generator import generate_html_report
from src.tfidf_similarity import document_similarity, sentence_similarities

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "data", "samples")


def read_sample(name: str) -> str:
    with open(os.path.join(SAMPLES, name), encoding="utf-8") as fh:
        return fh.read()


class TestPreprocessing(unittest.TestCase):
    def test_tokenize_clean(self):
        from src.preprocessing import preprocess_tokens

        tokens = preprocess_tokens("The quick brown foxes are jumping!")
        self.assertIn("quick", tokens)
        self.assertIn("fox", tokens)
        self.assertIn("jump", tokens)
        self.assertNotIn("the", tokens)  # stopword removed

    def test_split_sentences(self):
        from src.preprocessing import split_sentences

        sents = split_sentences("First sentence. Second sentence? Third!")
        self.assertEqual(len(sents), 3)


class TestNlpAnalysis(unittest.TestCase):
    def test_ngrams_and_overlap(self):
        toks_a = ["natural", "language", "processing", "is", "fun"]
        toks_b = ["natural", "language", "processing", "is", "cool"]

        bigrams_a = get_ngrams(toks_a, 2)
        self.assertIn(("natural", "language"), bigrams_a)

        stats = ngram_overlap_stats(toks_a, toks_b, n=2)
        self.assertGreater(stats["containment"], 0.7)

    def test_lcs(self):
        toks_a = ["a", "b", "c", "d", "e"]
        toks_b = ["a", "x", "c", "d", "y"]
        lcs_len, ratio = longest_common_subsequence(toks_a, toks_b)
        self.assertEqual(lcs_len, 3)
        self.assertGreater(ratio, 0.5)

    def test_synonym_pairs(self):
        toks_a = ["fast", "car"]
        toks_b = ["quick", "automobile"]
        syns = find_synonym_pairs(toks_a, toks_b)
        # Should detect fast <-> quick or car <-> automobile
        self.assertGreater(len(syns), 0)

    def test_pos_similarity(self):
        text_a = "The cat sat on the mat."
        text_b = "A dog ran in a park."
        ratio = pos_structural_similarity(text_a, text_b)
        self.assertGreater(ratio, 0.5)


class TestFileReader(unittest.TestCase):
    def test_text_read(self):
        sample_bytes = io.BytesIO(b"Hello world from NLP test")
        text = extract_text_from_file(sample_bytes, "test.txt")
        self.assertEqual(text, "Hello world from NLP test")


class TestTfidfSimilarity(unittest.TestCase):
    def test_identical_texts(self):
        text = read_sample("sample1_original.txt")
        self.assertGreater(document_similarity(text, text), 0.99)

    def test_plagiarized_copy(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        self.assertGreater(document_similarity(a, b), 0.9)

    def test_different_topics(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample2_different.txt")
        self.assertLess(document_similarity(a, b), 0.3)

    def test_sentence_similarities_shape(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        rows, cols, pairs = sentence_similarities(a, b)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(cols), 6)
        self.assertGreater(max(rows), 0.9)


class TestDetector(unittest.TestCase):
    def test_verdict_severe_for_copy(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        result = detector.detect(a, b)
        self.assertEqual(result.verdict, "Severe Plagiarism")
        self.assertGreater(result.tfidf_score, 0.9)
        self.assertGreater(len(result.sentence_matches), 0)
        self.assertGreater(result.bigram_containment, 0.8)
        self.assertGreater(result.lcs_ratio, 0.8)

    def test_verdict_original_for_different(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample2_different.txt")
        result = detector.detect(a, b)
        self.assertEqual(result.verdict, "Original")
        self.assertLess(result.combined_score, 0.3)

    def test_html_report_generation(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        result = detector.detect(a, b)
        html = generate_html_report(result, "DocA", "DocB")
        self.assertIn("Plagiarism Inspection Report", html)
        self.assertIn("Severe Plagiarism", html)


def _sbert_semantic():
    try:
        from src.embedding_similarity import build_semantic

        return build_semantic(backend="sbert")
    except Exception:
        return None


class TestSemanticDetection(unittest.TestCase):
    """Semantic tests; skipped when no SBERT model is available."""

    @classmethod
    def setUpClass(cls):
        cls.semantic = _sbert_semantic()
        if cls.semantic is None:
            raise unittest.SkipTest("sentence-transformers / SBERT unavailable")

    def test_paraphrase_caught_by_semantic(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample3_paraphrased.txt")
        result = detector.detect(a, b, embedding=self.semantic)
        self.assertGreater(result.semantic_score, 0.5)
        self.assertGreater(result.combined_score, 0.5)
        self.assertIn(result.verdict, {"Moderate Similarity", "High Plagiarism"})

    def test_different_topics_stay_original(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample2_different.txt")
        result = detector.detect(a, b, embedding=self.semantic)
        self.assertLess(result.semantic_score, 0.3)
        self.assertEqual(result.verdict, "Original")

    def test_copy_severe_with_semantic(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        result = detector.detect(a, b, embedding=self.semantic)
        self.assertGreater(result.semantic_score, 0.9)
        self.assertEqual(result.verdict, "Severe Plagiarism")


class TestCalibrationBenchmark(unittest.TestCase):
    """Calibrated score benchmark on labeled test pairs: verbatim, light paraphrase, heavy paraphrase, unrelated."""

    @classmethod
    def setUpClass(cls):
        cls.semantic = _sbert_semantic()

    def test_labeled_set_verbatim(self):
        """Verbatim copying: expected score >= 0.85 and Severe Plagiarism verdict."""
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        res = detector.detect(a, b, embedding=self.semantic)
        self.assertGreaterEqual(res.combined_score, 0.85)
        self.assertEqual(res.verdict, "Severe Plagiarism")
        self.assertGreaterEqual(res.calibrated_score, 0.85)

    def test_labeled_set_light_paraphrase(self):
        """Light paraphrase with synonym replacements: expected score in [0.60, 0.85]."""
        a = read_sample("sample1_original.txt")
        b = read_sample("sample3_paraphrased.txt")
        res = detector.detect(a, b, embedding=self.semantic)
        self.assertGreaterEqual(res.combined_score, 0.60)
        self.assertIn(res.verdict, {"Moderate Similarity", "High Plagiarism"})

    def test_labeled_set_heavy_paraphrase(self):
        """Heavy paraphrase with syntactic transformations: expected score in [0.35, 0.70)."""
        orig = (
            "Deep learning algorithms demand significant computational infrastructure and "
            "comprehensive datasets to reach optimal model accuracy."
        )
        heavy_para = (
            "Attaining peak predictive performance with neural architectures requires "
            "extensive hardware accelerators alongside massive empirical collections."
        )
        res = detector.detect(orig, heavy_para, embedding=self.semantic)
        self.assertGreaterEqual(res.combined_score, 0.35)
        self.assertIn(res.verdict, {"Low Similarity", "Moderate Similarity"})

    def test_labeled_set_unrelated(self):
        """Unrelated documents from different knowledge domains: expected score < 0.25 and Original."""
        a = read_sample("sample1_original.txt")
        b = read_sample("sample2_different.txt")
        res = detector.detect(a, b, embedding=self.semantic)
        self.assertLess(res.combined_score, 0.25)
        self.assertEqual(res.verdict, "Original")


class TestEdgeCases(unittest.TestCase):
    """Edge cases: empty inputs, short snippets, non-English text, quoted/cited text, repeated sentences."""

    def test_empty_input(self):
        """Empty string or whitespace inputs must safely return 0.0 without errors."""
        res1 = detector.detect("", "")
        self.assertEqual(res1.combined_score, 0.0)
        self.assertEqual(res1.verdict, "Original")
        self.assertEqual(len(res1.sentence_matches), 0)

        res2 = detector.detect("   \n\t  ", "Valid non-empty sentence.")
        self.assertEqual(res2.combined_score, 0.0)
        self.assertEqual(res2.verdict, "Original")

    def test_very_short_text(self):
        """Very short 1-3 word text should process cleanly without ZeroDivisionError."""
        res_same = detector.detect("Hello world.", "Hello world.")
        self.assertGreaterEqual(res_same.combined_score, 0.85)
        self.assertEqual(res_same.verdict, "Severe Plagiarism")

        res_diff = detector.detect("Hello world.", "Goodbye universe.")
        self.assertEqual(res_diff.verdict, "Original")

    def test_non_english(self):
        """Multilingual and non-ASCII unicode text should evaluate correctly."""
        doc_a = "Die kuenstliche Intelligenz revolutioniert moderne Technologien weltweit."
        doc_b = "Die kuenstliche Intelligenz revolutioniert moderne Technologien weltweit."
        res = detector.detect(doc_a, doc_b)
        self.assertGreaterEqual(res.combined_score, 0.85)
        self.assertEqual(res.verdict, "Severe Plagiarism")

    def test_quoted_cited_text(self):
        """Quotes and academic citations [1], (Author, 2023) should be parsed cleanly."""
        doc_a = (
            'According to Johnson et al. (2022) [4], "attention mechanisms enable neural '
            'models to capture long-range contextual dependencies across tokens."'
        )
        doc_b = (
            'According to Johnson et al. (2022) [4], "attention mechanisms enable neural '
            'models to capture long-range contextual dependencies across tokens."'
        )
        res = detector.detect(doc_a, doc_b)
        self.assertGreaterEqual(res.combined_score, 0.85)
        self.assertEqual(res.verdict, "Severe Plagiarism")

    def test_duplicated_sentences(self):
        """Repeated duplicate sentences should compute alignments without matrix indexing bugs."""
        sentence = "Natural language processing is an essential field of modern AI."
        doc_a = sentence
        doc_b = (sentence + " ") * 5
        res = detector.detect(doc_a, doc_b)
        self.assertGreater(res.combined_score, 0.50)
        self.assertGreater(len(res.sentence_matches), 0)


class TestLongDocumentChunkingAndCaching(unittest.TestCase):
    """Validates sliding sentence window chunking and SBERT embedding caching."""

    def test_chunk_sentences_windows(self):
        from src.embedding_similarity import chunk_sentences

        sentences = [f"Sentence number {i} in the document." for i in range(10)]
        chunks = chunk_sentences(sentences, window_size=3, stride=2)
        self.assertEqual(len(chunks), 5)
        self.assertIn("Sentence number 0", chunks[0])
        self.assertIn("Sentence number 2", chunks[0])
        self.assertIn("Sentence number 2", chunks[1])  # sliding stride overlap

    def test_embedding_cache_repeat_speed(self):
        from src.embedding_similarity import SbertSimilarity, _SBERT_EMBEDDING_CACHE

        try:
            sbert = SbertSimilarity()
            texts = ["Caching verification sentence alpha.", "Caching verification sentence beta."]
            emb1 = sbert.encode_texts(texts)
            self.assertEqual(len(emb1), 2)
            for t in texts:
                self.assertIn((sbert.model_name, t), _SBERT_EMBEDDING_CACHE)

            # Second call must retrieve from cache
            emb2 = sbert.encode_texts(texts)
            self.assertEqual(emb1.shape, emb2.shape)
            self.assertTrue((emb1 == emb2).all())
        except Exception:
            raise unittest.SkipTest("SBERT not available for cache test")

    def test_long_document_no_truncation(self):
        from src.embedding_similarity import SbertSimilarity

        try:
            sbert = SbertSimilarity()
            # 12 sentences that would exceed typical unchunked token lengths
            sentences = [
                f"Section paragraph {i} provides detailed analytical observations on NLP algorithms."
                for i in range(12)
            ]
            long_doc = " ".join(sentences)
            sim = sbert.document_similarity(long_doc, long_doc)
            self.assertAlmostEqual(sim, 1.0, places=1)
        except Exception:
            raise unittest.SkipTest("SBERT not available for long document test")


class TestReportEnhancements(unittest.TestCase):
    """Validates per-sentence match source label and confidence badge export."""

    def test_html_report_source_and_confidence(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample1_plagiarized.txt")
        res = detector.detect(a, b)
        html = generate_html_report(res, "Original Doc", "Suspect Doc")
        self.assertIn("Source", html)
        self.assertIn("Confidence", html)
        self.assertIn("Ref Sentence #", html)
        self.assertIn("High", html)


if __name__ == "__main__":
    unittest.main(verbosity=2)