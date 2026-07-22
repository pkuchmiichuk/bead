"""UniMorph adapter for the Ukrainian VESUM dataset (``ukr.xz``).

The VESUM conversion in the ``unimorph/ukr`` repo ships ``ukr.xz`` as a
**4-column** TSV::

    lemma <TAB> form <TAB> inflectional-features <TAB> inherent-features

Column 3 holds the features that vary across the paradigm (case, number, tense),
while column 4 holds inherent lexeme features: ``animacy;gender`` for nouns,
``aspect`` for verbs.

The stock ``unimorph.load_dataset`` assumes the standard 3-column layout and
mis-parses this file (the lemma slides into the DataFrame index and the feature
columns shift). :class:`VesumUniMorphAdapter` reads the file itself and joins
every feature column into a single tag string, so the inherited
``_parse_features`` sees the full tag set (e.g. ``N;ACC;SG;INAN;MASC``) and the
animacy / gender / aspect features are preserved.

Only :meth:`_load_dataset` is overridden; filtering, feature parsing,
``LexicalItem`` construction, and caching are inherited from
:class:`~bead.resources.adapters.unimorph.UniMorphAdapter` unchanged.
"""

from __future__ import annotations

import pandas as pd
import unimorph

from bead.resources.adapters.unimorph import UniMorphAdapter


class VesumUniMorphAdapter(UniMorphAdapter):
    """``UniMorphAdapter`` variant that reads the VESUM ``ukr.xz`` file.

    Use exactly like the base adapter::

        adapter = VesumUniMorphAdapter()
        items = adapter.fetch_items(language_code="ukr")            # all forms
        one = adapter.fetch_items(query="книга", language_code="ukr")  # one lemma
    """

    VESUM_FILE = "ukr.xz"

    def _load_dataset(self, lang_code: str) -> pd.DataFrame:
        """Read the VESUM file, joining all feature columns into one tag string.

        Parameters
        ----------
        lang_code : str
            ISO 639-3 language code (``"ukr"``).

        Returns
        -------
        pd.DataFrame
            DataFrame with ``lemma``, ``form``, and ``features`` columns, where
            ``features`` is columns 3..N joined by ``";"`` (empty parts dropped).
        """
        unimorph.download_unimorph(lang_code)
        path = unimorph.UNIMORPH_DIR / lang_code / self.VESUM_FILE
        raw = pd.read_csv(path, sep="\t", header=None, dtype=str, keep_default_na=False)
        features = raw.iloc[:, 2:].apply(
            lambda row: ";".join(tag for tag in row if isinstance(tag, str) and tag),
            axis=1,
        )
        return pd.DataFrame({"lemma": raw[0], "form": raw[1], "features": features})
