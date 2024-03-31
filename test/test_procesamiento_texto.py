import unittest
from unittest.mock import patch
import numpy as np
from io import StringIO
from src.procesamiento.procesamiento_texto import Text


class TestProcesamientoTexto(unittest.TestCase):
    def setUp(self):
        self.text_processor = Text()
        self.stdout = StringIO()

    def test_process_text(self):
        text = "This is a sample text for processing."
        (
            preprocessed_text,
            tokens,
            numerical_representation1,
            numerical_representation2,
        ) = self.text_processor.process_text(text)

        # Test preprocessed text
        self.assertEqual(preprocessed_text, "this is a sample text for processing")

        # Test tokens
        self.assertEqual(
            tokens, ["this", "is", "a", "sample", "text", "for", "processing"]
        )

        # Test numerical representation 1
        self.assertEqual(
            numerical_representation1.shape, (7, 50)
        )  # Assuming glove.twitter.27B.50d.txt is used

        # Test numerical representation 2
        self.assertEqual(
            numerical_representation2.shape, (7, 100)
        )  # Assuming glove.6B.100d.txt is used

    def test_process_text_with_empty_text(self):
        text = ""
        (
            preprocessed_text,
            tokens,
            numerical_representation1,
            numerical_representation2,
        ) = self.text_processor.process_text(text)

        # Test preprocessed text
        self.assertEqual(preprocessed_text, "")

        # Test tokens
        self.assertEqual(tokens, [])

        # Test numerical representation 1
        self.assertEqual(
            numerical_representation1.shape, (0, 50)
        )  # Assuming glove.twitter.27B.50d.txt is used

        # Test numerical representation 2
        self.assertEqual(
            numerical_representation2.shape, (0, 100)
        )  # Assuming glove.6B.100d.txt is used

    def test_process_text_with_special_characters(self):
        text = "This is a sample text with special characters: !@#$%^&*()"
        (
            preprocessed_text,
            tokens,
            numerical_representation1,
            numerical_representation2,
        ) = self.text_processor.process_text(text)

        # Test preprocessed text
        self.assertEqual(
            preprocessed_text, "this is a sample text with special characters"
        )

        # Test tokens
        self.assertEqual(
            tokens,
            ["this", "is", "a", "sample", "text", "with", "special", "characters"],
        )

        # Test numerical representation 1
        self.assertEqual(
            numerical_representation1.shape, (8, 50)
        )  # Assuming glove.twitter.27B.50d.txt is used

        # Test numerical representation 2
        self.assertEqual(
            numerical_representation2.shape, (8, 100)
        )  # Assuming glove.6B.100d.txt is used
