"""Tests for lexicon loading utilities."""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx
import pytest

from bead.resources.loaders import from_csv, from_tsv


class TestFromCSV:
    """Test from_csv() function."""

    def test_load_basic_csv(self, tmp_path: Path) -> None:
        """Test loading a basic CSV file."""
        csv_content = """lemma,pos
walk,VERB
run,VERB
jump,VERB"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test_verbs",
            language_code="eng",
        )

        assert lexicon.name == "test_verbs"
        assert lexicon.language_code == "eng"
        assert len(lexicon.items) == 3

        # Check items were created
        lemmas = {item.lemma for item in lexicon}
        assert lemmas == {"walk", "run", "jump"}

    def test_load_with_column_mapping(self, tmp_path: Path) -> None:
        """Test loading CSV with column mapping."""
        csv_content = """word,part_of_speech
cat,NOUN
dog,NOUN"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test_nouns",
            column_mapping={"word": "lemma", "part_of_speech": "pos"},
            language_code="eng",
        )

        assert len(lexicon.items) == 2
        lemmas = {item.lemma for item in lexicon}
        assert lemmas == {"cat", "dog"}

        # Check POS was mapped correctly
        for item in lexicon:
            assert item.features.get("pos") == "NOUN"

    def test_load_with_features(self, tmp_path: Path) -> None:
        """Test loading CSV with feature columns."""
        csv_content = """lemma,number,countability
cat,singular,count
water,mass,mass"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test_nouns",
            feature_columns=["number", "countability"],
            language_code="eng",
        )

        items_dict = {item.lemma: item for item in lexicon}

        assert items_dict["cat"].features["number"] == "singular"
        assert items_dict["cat"].features["countability"] == "count"
        assert items_dict["water"].features["number"] == "mass"
        assert items_dict["water"].features["countability"] == "mass"

    def test_load_with_attributes(self, tmp_path: Path) -> None:
        """Test loading CSV with feature columns."""
        csv_content = """lemma,semantic_class,frequency
walk,motion,1000
run,motion,800"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test_verbs",
            feature_columns=["semantic_class", "frequency"],
            language_code="eng",
        )

        items_dict = {item.lemma: item for item in lexicon}

        assert items_dict["walk"].features["semantic_class"] == "motion"
        # Numeric values are preserved as their type (int, not string)
        assert items_dict["walk"].features["frequency"] == 1000

    def test_load_with_missing_values(self, tmp_path: Path) -> None:
        """Test loading CSV with missing values."""
        csv_content = """lemma,pos,tense
walk,VERB,present
run,VERB,
jump,,future"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test",
            feature_columns=["tense"],
            language_code="eng",
        )

        # All rows should be loaded (lemma is present for all)
        # Missing values in features/pos are just omitted
        assert len(lexicon.items) == 3

        # Check that items with missing feature values don't have that feature
        items_dict = {item.lemma: item for item in lexicon}
        assert "tense" in items_dict["walk"].features
        assert "tense" not in items_dict["run"].features  # Missing tense
        assert "tense" in items_dict["jump"].features

    def test_load_with_description(self, tmp_path: Path) -> None:
        """Test loading CSV with description."""
        csv_content = """lemma
test"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test_lex",
            description="Test lexicon",
            language_code="eng",
        )

        assert lexicon.description == "Test lexicon"

    def test_missing_lemma_column_raises_error(self, tmp_path: Path) -> None:
        """Test that missing lemma column raises ValueError."""
        csv_content = """word,pos
test,NOUN"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        with pytest.raises((ValueError, dx.ValidationError), match="lemma"):
            from_csv(
                path=csv_file,
                name="test",
                language_code="eng",
            )

    def test_file_not_found_raises_error(self) -> None:
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            from_csv(
                path="/nonexistent/file.csv",
                name="test",
                language_code="eng",
            )

    def test_combined_features_and_attributes(self, tmp_path: Path) -> None:
        """Test loading CSV with multiple feature columns."""
        csv_content = """word,number,semantic_class
cat,singular,animal
dog,singular,animal"""

        csv_file = tmp_path / "test.csv"
        csv_file.write_text(csv_content)

        lexicon = from_csv(
            path=csv_file,
            name="test",
            column_mapping={"word": "lemma"},
            feature_columns=["number", "semantic_class"],
            language_code="eng",
        )

        item = next(iter(lexicon))
        assert "number" in item.features
        assert "semantic_class" in item.features


class TestFromTSV:
    """Test from_tsv() function."""

    def test_load_basic_tsv(self, tmp_path: Path) -> None:
        """Test loading a basic TSV file."""
        tsv_content = """lemma\tpos
walk\tVERB
run\tVERB"""

        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        lexicon = from_tsv(
            path=tsv_file,
            name="test_verbs",
            language_code="eng",
        )

        assert lexicon.name == "test_verbs"
        assert len(lexicon.items) == 2

    def test_tsv_with_column_mapping(self, tmp_path: Path) -> None:
        """Test TSV with column mapping."""
        tsv_content = """word\tpart_of_speech
cat\tNOUN"""

        tsv_file = tmp_path / "test.tsv"
        tsv_file.write_text(tsv_content)

        lexicon = from_tsv(
            path=tsv_file,
            name="test",
            column_mapping={"word": "lemma", "part_of_speech": "pos"},
            language_code="eng",
        )

        assert len(lexicon.items) == 1
        item = next(iter(lexicon))
        assert item.lemma == "cat"
        assert item.features.get("pos") == "NOUN"
