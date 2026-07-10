"""Adapter for UniMorph morphological paradigms.

This module provides an adapter to fetch morphological paradigms from UniMorph
data and convert them to LexicalItem format with morphological features.
"""

from __future__ import annotations

from typing import Any

import langcodes
import pandas as pd
import unimorph
from unimorph import load_dataset

from bead.data.language_codes import LanguageCode
from bead.resources.adapters.base import ResourceAdapter
from bead.resources.adapters.cache import AdapterCache
from bead.resources.lexical_item import LexicalItem


class UniMorphAdapter(ResourceAdapter):
    """Adapter for UniMorph morphological paradigms.

    This adapter fetches morphological paradigms from UniMorph and converts
    them to LexicalItem format. Morphological features are stored in the
    features field using UniMorph feature schema.

    Parameters
    ----------
    cache : AdapterCache | None
        Optional cache instance. If None, no caching is performed.

    Examples
    --------
    >>> adapter = UniMorphAdapter()
    >>> items = adapter.fetch_items(query="walk", language_code="en")
    >>> all(item.language_code == "en" for item in items)
    True
    >>> all("tense" in item.features for item in items if item.features)
    True
    """

    def __init__(self, cache: AdapterCache | None = None) -> None:
        """Initialize UniMorph adapter.

        Parameters
        ----------
        cache : AdapterCache | None
            Optional cache instance.
        """
        self.cache = cache
        self._datasets: dict[str, pd.DataFrame] = {}  # Cache datasets by language

    def _load_dataset(self, lang_code: str) -> pd.DataFrame:
        """Load the raw ``(lemma, form, features)`` DataFrame for a language.

        Override this in a subclass to read a non-standard UniMorph file layout
        (e.g. a language whose data ships extra columns).

        Parameters
        ----------
        lang_code : str
            ISO 639-3 language code.

        Returns
        -------
        pd.DataFrame
            DataFrame with ``lemma``, ``form``, and ``features`` columns.
        """
        return load_dataset(lang_code)

    def fetch_items(
        self,
        query: str | None = None,
        language_code: LanguageCode = None,
        **kwargs: Any,
    ) -> list[LexicalItem]:
        """Fetch morphological paradigms from UniMorph.

        Parameters
        ----------
        query : str | None
            Lemma to query (e.g., "walk", "먹다", "hamba").
        language_code : LanguageCode
            **Required** language code (e.g., "en", "ko", "zu"). UniMorph
            is organized by language, so this parameter is essential.
        **kwargs : Any
            Additional parameters (e.g., pos="VERB").

        Returns
        -------
        list[LexicalItem]
            Lexical items representing inflected forms with morphological
            features in the features field.

        Raises
        ------
        ValueError
            If language_code is None (required for UniMorph).
        RuntimeError
            If UniMorph access fails.

        Examples
        --------
        >>> adapter = UniMorphAdapter()
        >>> items = adapter.fetch_items(query="walk", language_code="en")
        >>> len(items) > 0
        True
        >>> items[0].features.get("pos") == "VERB"
        True
        """
        if language_code is None:
            raise ValueError("UniMorphAdapter requires language_code parameter")

        # Normalize to ISO 639-3 (3-letter code) for UniMorph
        # UniMorph uses 3-letter codes (language_code is guaranteed non-None here)
        lang_code = self._normalize_language_code(language_code)

        # Check cache
        cache_key = None
        if self.cache:
            cache_key = self.cache.make_key(
                "unimorph", query=query, language_code=lang_code, **kwargs
            )
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        # Fetch from UniMorph
        try:
            # Load dataset for language (cached at instance level)
            if lang_code not in self._datasets:
                self._datasets[lang_code] = self._load_dataset(lang_code)

            dataset = self._datasets[lang_code]

            # Filter by lemma if query provided
            if query:
                dataset = dataset[dataset["lemma"] == query]

            # Convert to LexicalItem objects
            items: list[LexicalItem] = []
            for _, row in dataset.iterrows():
                # Skip rows with NaN values
                if (
                    row["lemma"] is None
                    or row["form"] is None
                    or row["features"] is None
                    or str(row["lemma"]) == "nan"
                    or str(row["form"]) == "nan"
                    or str(row["features"]) == "nan"
                ):
                    continue

                # Parse features string (e.g., "V;PRS;3;SG")
                features_dict = self._parse_features(str(row["features"]))

                item = LexicalItem(
                    lemma=str(row["lemma"]),
                    form=str(row["form"]),
                    language_code=language_code,
                    features=features_dict,
                    source="UniMorph",
                )
                items.append(item)

            # Cache result
            if self.cache and cache_key:
                self.cache.set(cache_key, items)

            return items

        except Exception as e:
            raise RuntimeError(f"Failed to fetch from UniMorph: {e}") from e

    def _normalize_language_code(self, language_code: LanguageCode) -> str:
        """Normalize language code to ISO 639-3 (3-letter) format.

        Uses the langcodes package to properly convert ISO 639-1 (2-letter) codes
        to ISO 639-3 (3-letter) codes.

        Parameters
        ----------
        language_code : LanguageCode
            Language code (2 or 3 letters, non-None).

        Returns
        -------
        str
            ISO 639-3 (3-letter) language code.

        Raises
        ------
        ValueError
            If language_code is None.
        """
        if language_code is None:
            raise ValueError(
                "language_code cannot be None when normalizing. "
                "This should be checked by the caller."
            )

        # Use langcodes package to normalize
        try:
            # If it's already 3 letters, return as-is
            if len(language_code) == 3:
                return language_code

            # For 2-letter codes, use langcodes to get the 3-letter equivalent
            lang = langcodes.Language.get(language_code)
            return lang.to_alpha3()
        except Exception:
            # If conversion fails, return as-is
            return language_code

    def _get_tag_dimension(self, tag: str) -> str:
        """Get the dimension for a UniMorph tag.

        Based on analysis of 173 languages and 575 tags from
        the actual UniMorph data.

        Parameters
        ----------
        tag : str
            UniMorph feature tag.

        Returns
        -------
        str
            Dimension name, or "unknown" if tag is not recognized.
        """
        # Language-specific tags
        if tag.startswith("LGSPEC") or tag.startswith("LGSPE"):
            return "lgspec"

        # Tag-to-dimension mapping
        # Build lookup lazily to avoid repeating this logic
        if not hasattr(self, "_tag_map"):
            self._tag_map = self._build_tag_map()

        return self._tag_map.get(tag, "unknown")

    def _build_tag_map(self) -> dict[str, str]:
        """Build complete tag-to-dimension mapping.

        Returns
        -------
        dict[str, str]
            Mapping from tag to dimension name.
        """
        mapping: dict[str, str] = {}

        # Part of speech
        for tag in [
            "N",
            "V",
            "ADJ",
            "ADV",
            "PRO",
            "ART",
            "DET",
            "ADP",
            "CONJ",
            "INTJ",
            "NUM",
            "PRON",
            "PROPN",
            "PRT",
        ]:
            mapping[tag] = "pos"

        # Person (including complex)
        for tag in ["0", "1", "2", "3", "4", "5", "1+2", "2+3", "1+EXCL", "1+INCL"]:
            mapping[tag] = "person"

        # Number
        for tag in ["SG", "DU", "PL", "SG+PL", "DU/PL", "SG/DU/PL"]:
            mapping[tag] = "number"

        # Tense (including variants and whitespace)
        for tag in [
            "PRS",
            "PST",
            "FUT",
            "PRES",
            "PAST",
            "NFUT",
            "NPST",
            "PRS ",
            "PRS   ",
            "PRS+FUT",
            "PRS/FUT",
            "PRS+IMMED",
            "PST+IMMED",
            "PRS/PST+IMMED",
            "FUT+IMMED",
            "FUT+RMT",
            "PST+RCT",
            "PST+RMT",
            "RCT",
            "RMT",
            "IMMED",
            "FUT:ELEV",
            "PST:ELEV",
            "3:PRS",
            "V:PST:3:PL",
            "non{PRS}",
            "non{PST}",
            "PL,FUTS",
        ]:
            mapping[tag] = "tense"

        # Aspect
        for tag in [
            "PFV",
            "IPFV",
            "PRF",
            "PROG",
            "HAB",
            "ITER",
            "PROSP",
            "DUR",
            "INCH",
            "SEMEL",
            "FREQ",
            "HAB+IPFV",
            "HAB+PRF",
            "HAB+PROG",
            "IPFV/PROG",
            "PFV/PRF",
            "PRF+PROG",
            "PROSP+PROG",
        ]:
            mapping[tag] = "aspect"

        # Mood (many combinations)
        for tag in [
            "IND",
            "SBJV",
            "IMP",
            "COND",
            "OPT",
            "POT",
            "DEB",
            "OBLIG",
            "PERM",
            "ADM",
            "REAL",
            "IRR",
            "HYP",
            "INFER",
            "LKLY",
        ] + [
            "COND+IND",
            "COND+IND+OPT",
            "COND+POT",
            "COND+POT+OPT",
            "COND+SBJV",
            "COND+SBJV+OPT",
            "IND+IMP",
            "IND+OPT",
            "IND+POT",
            "IND+POT+OPT",
            "IMP+OPT",
            "IMP+SBJV",
            "POT+OPT",
            "SBJV+OPT",
            "SBJV+POT",
            "SBJV+POT+OPT",
            "ADM+OPT",
            "ADM+POT",
            "ADM+POT+OPT",
        ]:
            mapping[tag] = "mood"

        # Voice
        for tag in [
            "ACT",
            "PASS",
            "MID",
            "ANTIP",
            "REFL",
            "RECP",
            "CAUS",
            "APPL",
            "ACT+PASS",
            "MID+PASS",
            "REFL/RECP",
            "CAUSV",
            "COMPV",
            "EXCLV",
            "MASV",
        ]:
            mapping[tag] = "voice"

        # Gender (including complex combinations)
        for tag in [
            "MASC",
            "FEM",
            "NEUT",
            "MASC+FEM",
            "MASC+NEUT",
            "FEM+NEUT",
            "MASC+FEM+NEUT",
            "FEM+FEM",
            "FEM+MASC",
            "MASC+MASC",
            "NEUT+MASC",
            "MASC/FEM",
            "MASC+FEM+MASC",
        ]:
            mapping[tag] = "gender"

        # Animacy
        for tag in ["ANIM", "INAN", "HUM"]:
            mapping[tag] = "animacy"

        # Finiteness
        for tag in ["FIN", "NFIN"]:
            mapping[tag] = "finiteness"

        # Definiteness
        for tag in [
            "DEF",
            "INDF",
            "NDEF",
            "INDF1",
            "INDF2",
            "INDF3",
            "DEF/INDF",
            "DEF/LGSPEC1",
        ]:
            mapping[tag] = "definiteness"

        # Comparison
        for tag in [
            "POS",
            "CMPR",
            "EQTV",
            "SPRL",
            "SUP",
            "EQTV+ABL",
            "EQTV+ACC",
            "EQTV+DAT",
        ]:
            mapping[tag] = "comparison"

        # Politeness (including Korean)
        for tag in [
            "INFM",
            "FORM",
            "FORM2",
            "POL",
            "HUMB",
            "ELEV",
            "MPOL",
            "FRML",
            "INFM:LGSPEC1",
            "POL:LGSPEC1",
            "Formal polite(하십시오체)",
            "Formal non-polite(해라체)",
            "Informal polite(해요체)",
            "Informal non-polite(해체)",
        ]:
            mapping[tag] = "politeness"

        # Evidentiality
        for tag in ["FH", "NFH", "VIS", "QUOT", "RPRT", "INFR"]:
            mapping[tag] = "evidentiality"

        # Switch-reference
        for tag in ["SS", "DS", "SIMMA"]:
            mapping[tag] = "switch_reference"

        # Deixis
        for tag in ["PROX", "MED", "REMT"]:
            mapping[tag] = "deixis"

        # Interrogativity
        for tag in ["INT", "DECL"]:
            mapping[tag] = "interrogativity"

        # Valency
        for tag in ["INTR", "TR", "DISTR"]:
            mapping[tag] = "valency"

        # Polarity
        for tag in ["NEG", "YES", "NO"]:
            mapping[tag] = "polarity"

        # Information structure
        for tag in ["TOP", "FOC", "AGFOC", "PFOC"]:
            mapping[tag] = "information_structure"

        # Aktionsart
        for tag in ["STAT", "ACTY", "TEL", "TAXIS", "SIM"]:
            mapping[tag] = "aktionsart"

        # Verb forms
        for tag in [
            "V.PTCP",
            "V.CVB",
            "V.MSDR",
            "V.NFIN",
            "V.CV",
            "V.PCTP",
            "V.PTCP.PRS",
            "V.PTCP.PST",
            "ADJ.PTCP",
            "ADJ.CVB",
            "ADJ.MSDR",
            "PTCP",
            "CVB",
            "MSDR",
            "INF",
            "INFN",
        ]:
            mapping[tag] = "verb_form"

        # Bantu noun classes
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17]:
            mapping[f"BANTU{i}"] = "bantu_class"

        # Possessive markers
        pss_tags = ["PSS", "PSS0", "PSS1", "PSS2", "PSS3", "PSS4"]
        for base in ["PSS1", "PSS2", "PSS3"]:
            for suffix in [
                "D",
                "I",
                "P",
                "PE",
                "PI",
                "PL",
                "S",
                "SM",
                "F",
                "M",
                "PF",
                "PM",
                "SF",
            ]:
                pss_tags.append(f"{base}{suffix}")
        pss_tags += ["PSS3S/PSS3P", "PSS{2/3}D", "PSSD", "PSSRP", "PSSRS", "PSSS"]
        for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 17]:
            pss_tags.append(f"PSSB{i}")
        pss_tags += [
            "NALN+PSS3S",
            "ALN+PSS1PE",
            "ALN+PSS1PI",
            "ALN+PSS1S",
            "ALN+PSS2S",
            "ALN+PSS3P",
            "ALN+PSS3S",
            "ALN+PSSRP",
            "ALN+PSSRS",
        ]
        for tag in pss_tags:
            mapping[tag] = "possessive"

        # Case (all combinations)
        case_tags = [
            "NOM",
            "ACC",
            "ERG",
            "ABS",
            "DAT",
            "GEN",
            "INS",
            "INST",
            "ABL",
            "ALL",
            "ESS",
            "LOC",
            "VOC",
            "COM",
            "BEN",
            "AB",
            "AT",
            "IN",
            "ON",
            "PROL",
            "TERM",
            "VERS",
            "OBL",
            "SUB",
            "ELEV",
            "FROM",
            "TO",
            "APPRX",
            "PRIV",
            "PROPR",
            "BYWAY",
            "DIR",
        ] + [
            "ACC+COM",
            "ACC/DAT",
            "AT+ABL",
            "AT+ALL",
            "AT+ESS",
            "COM+TERM",
            "DAT/GEN",
            "DAT:FEM",
            "GEN+DAT",
            "GEN/DAT",
            "IN+ABL",
            "IN+ALL",
            "IN+ESS",
            "LOC+APPRX",
            "NOM+VOC",
            "NOM/ACC",
            "NOM/ACC/DAT",
            "OBL+VOC",
            "ON+ABL",
            "ON+ALL",
            "ON+ESS",
            "PSSRP+ACC",
            "PSSRS+ACC",
            "VOC+GEN",
            "(non)NOM",
            "non{NOM/ACC}",
            "non{NOM}",
            "not{NOM}",
        ]
        for tag in case_tags:
            mapping[tag] = "case"

        # Argument markers (all observed tags with whitespace variants)
        arg_prefixes = [
            "ARGAB",
            "ARGAC",
            "ARGBE",
            "ARGDA",
            "ARGER",
            "ARGERG",
            "ARGIO",
            "ARGNO",
        ]
        for prefix in arg_prefixes:
            for suffix in [
                "",
                "1",
                "2",
                "3",
                "1P",
                "1S",
                "2P",
                "2S",
                "3P",
                "3S",
                "23S",
                "S1",
                "S2",
                "S3",
                "INFM",
                "PL",
                "SG",
                "FEM",
                "MASC",
                "1DU",
                "2DU",
                "3DU",
                "1PL",
                "2PL",
                "3PL",
                "1SG",
                "3SG",
                "3SGHUM",
                "{D/P}",
                "S",
            ]:
                mapping[f"{prefix}{suffix}"] = "argument"
        # Add specific combinations and whitespace variants
        for tag in (
            [
                "ARG1",
                "ARG2",
                "ARG3",
                "ARG1P",
                "ARG1S",
                "ARG3P",
                "ARG3S",
                "ARGAB3P",
                "ARGAB3P   ",
                "ARGAB3P    ",
                "ARGAB3P      ",
                "ARGAB3P           ",
                "ARGAB3S ",
                "ARGAB3S  ",
                "ARGAB3S    ",
                "ARGAB3S                           ",
                "ARGDU",
                "ARGEXCL",
                "ARGINCL",
                "ARGPL",
                "ARGSG",
                "ARBAB1S",
                "ARBAB3S",
                "ARBEB1P",
                "ARBEB1S",
            ]
            + [
                "ARGAC1P+ARGNO1P",
                "ARGAC1S+ARGNO1S",
                "ARGAC2P+ARGNO2P",
                "ARGAC2S+ARGNO2S",
                "ARGAC3P+ARGNO3P",
                "ARGAC3S+ARGNO1P",
                "ARGAC3S+ARGNO1S",
                "ARGAC3S+ARGNO2P",
                "ARGAC3S+ARGNO2S",
                "ARGAC3S+ARGNO3P",
                "ARGAC3S+ARGNO3S",
                "ARGNO{2/3}",
                "ARGNO{D/P}",
                "ARGAC{D/P}",
            ]
            + [
                f"NO{x}"
                for x in [
                    "",
                    "1",
                    "2",
                    "3",
                    "1P",
                    "1PE",
                    "1PI",
                    "1S",
                    "2P",
                    "2S",
                    "3F",
                    "3M",
                    "3P",
                    "3PA",
                    "3S",
                    "3SA",
                    "3SI",
                ]
            ]
            + [f"DA{x}" for x in ["1PE", "1PI", "1S", "2P", "2S", "3P", "3S"]]
            + ["ALN", "NALN+PSS3S"]
        ):
            mapping[tag] = "argument"

        return mapping

    def _parse_features(self, features_str: str) -> dict[str, str]:
        """Parse UniMorph features string into dictionary.

        Maps UniMorph feature tags to their dimensions based on
        analysis of 173 languages and 575 unique tags from actual UniMorph data.

        Parameters
        ----------
        features_str : str
            UniMorph features string (e.g., "V;PRS;3;SG").

        Returns
        -------
        dict[str, str]
            Parsed features dictionary with dimension names as keys.
        """
        features_dict: dict[str, str] = {}

        # Split by semicolon
        parts = features_str.split(";")

        # Map each tag to its dimension
        for part in parts:
            part = part.strip()
            if not part:  # Skip empty parts
                continue

            dimension = self._get_tag_dimension(part)

            # Store tag under its dimension
            if dimension == "unknown":
                # Preserve unknown tags with sanitized key
                safe_key = (
                    part.lower().replace(" ", "_").replace("+", "_").replace("/", "_")
                )
                features_dict[f"unknown_{safe_key}"] = part
            elif dimension == "lgspec":
                # Language-specific features
                features_dict[f"lgspec_{part.lower()}"] = part
            else:
                # Known dimension - store the tag value
                features_dict[dimension] = part

        # Always store the original feature string
        features_dict["unimorph_features"] = features_str

        return features_dict

    def is_available(self) -> bool:
        """Check if UniMorph package is available.

        Returns
        -------
        bool
            True if unimorph can be imported and accessed, False otherwise.

        Examples
        --------
        >>> adapter = UniMorphAdapter()
        >>> adapter.is_available()
        True
        """
        try:
            # Verify unimorph is accessible
            unimorph.get_list_of_datasets()
            return True
        except Exception:
            return False
