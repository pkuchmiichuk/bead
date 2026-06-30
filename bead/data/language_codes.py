"""ISO 639 language code validation and utilities."""

from __future__ import annotations

from langcodes import Language
from langcodes.tag_parser import LanguageTagError


def validate_iso639_code(code: str | None) -> str | None:
    """Validate language code against ISO 639-1 or ISO 639-3.

    Parameters
    ----------
    code
        Language code to validate (e.g., "en", "eng", "ko", "kor").

    Returns
    -------
    str | None
        Normalized language code (converted to ISO 639-3 if valid).

    Raises
    ------
    ValueError
        If code is not a valid ISO 639 language code.

    Examples
    --------
    >>> validate_iso639_code("en")
    'eng'
    >>> validate_iso639_code("eng")
    'eng'
    >>> validate_iso639_code("ko")
    'kor'
    >>> validate_iso639_code(None)
    None
    >>> validate_iso639_code("invalid")
    Traceback (most recent call last):
        ...
    ValueError: Invalid language code: 'invalid'
    """
    if code is None:
        return None

    try:
        # parse and normalize to ISO 639-3
        lang = Language.get(code)
        return lang.to_alpha3()
    except (LanguageTagError, LookupError) as e:
        raise ValueError(f"Invalid language code: {code!r}") from e


# Plain type alias. Models that use it normalize via @dx.validates("language_code").
type LanguageCode = str | None
