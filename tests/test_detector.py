"""Tests for the plagiarism detector modules."""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src import detector
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

    def test_verdict_original_for_different(self):
        a = read_sample("sample1_original.txt")
        b = read_sample("sample2_different.txt")
        result = detector.detect(a, b)
        self.assertEqual(result.verdict, "Original")
        self.assertLess(result.combined_score, 0.3)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)