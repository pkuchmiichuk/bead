"""Lexicon loading utilities for various data formats.

This module provides class methods for loading Lexicon objects from
various data formats (CSV, TSV) with flexible column mapping.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Series

from bead.data.language_codes import LanguageCode
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon


def from_csv(
    path: str | Path,
    name: str,
    *,
    language_code: LanguageCode,
    column_mapping: dict[str, str] | None = None,
    feature_columns: list[str] | None = None,
    pos: str | None = None,
    description: str | None = None,
    **csv_kwargs: Any,
) -> Lexicon:
    """Load lexicon from CSV file with flexible column mapping.

    Parameters
    ----------
    path : str | Path
        Path to the CSV file.
    name : str
        Name for the lexicon.
    language_code : LanguageCode
        ISO 639-3 language code for all items.
    column_mapping : dict[str, str] | None
        Mapping from CSV column names to feature names.
        Example: {"word": "lemma"}
    feature_columns : list[str] | None
        CSV column names to include in features dict.
        Example: ["number", "tense", "countability", "semantic_class"]
    pos : str | None
        Part-of-speech tag to assign to all items (e.g., "NOUN", "VERB").
        Will be added to features dict as "pos".
    description : str | None
        Optional description of the lexicon.
    **csv_kwargs : Any
        Additional keyword arguments passed to pandas.read_csv().

    Returns
    -------
    Lexicon
        New lexicon loaded from CSV.

    Raises
    ------
    ValueError
        If required "lemma" column/mapping is missing.
    FileNotFoundError
        If CSV file does not exist.

    Examples
    --------
    >>> lexicon = from_csv(
    ...     "bleached_nouns.csv",
    ...     "nouns",
    ...     language_code="eng",
    ...     column_mapping={"word": "lemma"},
    ...     feature_columns=["number", "countability", "semantic_class"],
    ...     pos="NOUN"
    ... )  # doctest: +SKIP
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    # read CSV
    df: DataFrame = pd.read_csv(file_path, **csv_kwargs)

    # set up column mapping
    mapping = column_mapping or {}
    reverse_mapping = {v: k for k, v in mapping.items()}

    # check for required lemma column
    lemma_col = reverse_mapping.get("lemma", "lemma")
    columns_list = list(df.columns)
    if lemma_col not in columns_list:
        raise ValueError(
            f"CSV must have a 'lemma' column or provide column_mapping. "
            f"Available columns: {columns_list}"
        )

    # create lexicon
    lexicon = Lexicon(
        name=name,
        description=description,
        language_code=language_code,
    )

    # process each row
    row_iter: Iterator[tuple[int | str, Series[Any]]] = df.iterrows()
    for _, row_data in row_iter:
        row: Series[Any] = row_data

        # get lemma
        lemma_col = reverse_mapping.get("lemma", "lemma")
        lemma = str(row[lemma_col])

        # build features dict
        features: dict[str, Any] = {}

        # add POS if provided
        if pos:
            features["pos"] = pos

        # handle mapped "pos" column
        pos_col = reverse_mapping.get("pos")
        if pos_col and pos_col in columns_list and pd.notna(row[pos_col]):
            features["pos"] = str(row[pos_col])

        # add feature columns
        if feature_columns:
            for col in feature_columns:
                if col in columns_list and pd.notna(row[col]):
                    # store feature value, converting to string if needed
                    val = row[col]
                    if not isinstance(val, str | int | float | bool):
                        features[col] = str(val)
                    else:
                        features[col] = val

        # create and add item
        item = LexicalItem(
            lemma=lemma,
            language_code=language_code,
            features=features if features else {},
            source=None,
        )
        lexicon = lexicon.with_item(item)

    return lexicon


def from_tsv(
    path: str | Path,
    name: str,
    *,
    language_code: LanguageCode,
    column_mapping: dict[str, str] | None = None,
    feature_columns: list[str] | None = None,
    pos: str | None = None,
    description: str | None = None,
    **tsv_kwargs: Any,
) -> Lexicon:
    r"""Load lexicon from TSV file with flexible column mapping.

    This is a convenience wrapper around from_csv() that sets sep="\t".

    Parameters
    ----------
    path : str | Path
        Path to the TSV file.
    name : str
        Name for the lexicon.
    language_code : LanguageCode
        ISO 639-3 language code for all items.
    column_mapping : dict[str, str] | None
        Mapping from TSV column names to feature names.
    feature_columns : list[str] | None
        TSV column names to include in features dict.
    pos : str | None
        Part-of-speech tag to assign to all items.
    description : str | None
        Optional description of the lexicon.
    **tsv_kwargs : Any
        Additional keyword arguments passed to pandas.read_csv().

    Returns
    -------
    Lexicon
        New lexicon loaded from TSV.

    Examples
    --------
    >>> lexicon = from_tsv(
    ...     "verbs.tsv",
    ...     "verbs",
    ...     language_code="eng",
    ...     feature_columns=["tense", "aspect"],
    ...     pos="VERB"
    ... )  # doctest: +SKIP
    """
    return from_csv(
        path=path,
        name=name,
        language_code=language_code,
        column_mapping=column_mapping,
        feature_columns=feature_columns,
        pos=pos,
        description=description,
        sep="\t",
        **tsv_kwargs,
    )
