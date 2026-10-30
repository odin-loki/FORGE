# forge::os::text and forge::os::locale: text handling and localisation

FORGE ships its own clean-room Unicode, encoding and locale library instead of depending on ICU.
It lives in the OS layer (`os/include/forge/os/text.hpp`, `os/include/forge/os/locale.hpp`,
sources in `os/src/text/`) and builds as the static library `forge_text`. Scripts reach it through
the `forge::text::` and `forge::locale::` built-ins.

## Goals

- One API on every OS FORGE targets (Linux, FreeBSD, Windows, macOS, iOS, Android).
- Correct by the Unicode conformance suites, not by eyeball. The tests run the official
  `NormalizationTest`, `GraphemeBreakTest`, `WordBreakTest`, `SentenceBreakTest`, `LineBreakTest`,
  `BidiCharacterTest` and `CollationTest` files and report pass rates.
- No runtime data files. Tables are generated from UCD/DUCET/CLDR by scripts in `tools/unicode/`
  and committed as compact C++ arrays (`ucd_data.inc` ~1.7 MB of source, `cldr_data.inc` for the
  locale subset), so offline and cross builds work.
- C++23, RAII, no raw owning pointers. Fallible calls return `os::result<T>` / `os::status`
  (`err.code` non-zero, `err.what` readable). Nothing throws except `forge::mem` running out of budget.
- Memory: big working objects (collators' sort-key buffers, catalogs, regex programs, conversion of
  large inputs) take a `forge::mem` lease on the host budget with tag `"text"`, so a FORGE process
  under `FORGE_MEM_LIMIT` fails with `out_of_memory` instead of being OOM-killed.

## Unicode version and data

| Data | Source | Used for |
| --- | --- | --- |
| UnicodeData, DerivedGeneralCategory, Scripts, PropList, DerivedCoreProperties, emoji-data | UCD 16.0.0 | properties, case, decomposition |
| SpecialCasing, CaseFolding | UCD | full case mapping and folding |
| DerivedNormalizationProps | UCD | composition exclusions |
| Grapheme/Word/SentenceBreakProperty, LineBreak, EastAsianWidth | UCD | segmentation, width |
| DerivedBidiClass, BidiBrackets, BidiMirroring | UCD | UAX 9 |
| allkeys.txt | UCA 16.0.0 (DUCET) | collation |
| numbers, currencies, ca-gregorian, dateFields, listPatterns, units, plurals, ordinals, weekData, measurementData | CLDR 46 (cldr-json) | locale formatting |

Properties are stored as a two-stage trie (code point >> 7 selects a 128-entry block of record ids;
identical blocks are shared) over ~1,400 distinct packed property records. Mappings are sorted arrays
searched with binary search.

## API outline (`forge::os::text`)

- **UTF**: `decode_utf8/16/32`, `encode_utf8/16/32`, `utf8_to_utf16`, `utf16_to_utf8`, `utf8_to_utf32`,
  `validate_utf8` (returns the offset of the first bad byte), `sanitize_utf8` (U+FFFD substitution per
  the WHATWG "maximal subpart" rule), `code_points` iteration.
- **Properties**: `general_category`, `script` (name), `is_alpha/upper/lower/space/digit/emoji/...`,
  `combining_class`, `east_asian_width`, `bidi_class`.
- **Case**: `to_upper`, `to_lower`, `to_title`, `case_fold` (full mappings, Final_Sigma, Turkish /
  Azeri / Lithuanian tailoring by language tag), `simple_*` per code point, `equals_ignore_case`.
- **Normalization**: `normalize(s, form)` for NFC/NFD/NFKC/NFKD, `is_normalized` quick check.
- **Segmentation** (UAX 29 / UAX 14): `graphemes`, `words`, `sentences`, `line_breaks` (break
  opportunities with mandatory flag), `next_grapheme` for cursors.
- **Width**: `width(cp)`, `display_width(s)` (East Asian Wide/Fullwidth = 2, combining = 0, emoji
  presentation and ZWJ sequences = 2 per grapheme, ambiguous configurable).
- **Bidi** (UAX 9): `bidi_levels(s, base)` resolves embedding levels (rules X1–X10, W1–W7, N0–N2, I1–I2,
  L1) and `bidi_reorder` returns visual order per line.
- **Collation** (UTS 10): `collator` from DUCET with strength (primary..quaternary/identical),
  alternate shifted/non-ignorable, case-first, numeric ordering, and simple locale tailorings
  (e.g. Swedish/Finnish å ä ö after z, German phonebook-free default, Spanish ñ, Danish/Norwegian
  æ ø å, Turkish dotless i). `sort_key`, `compare`.
- **Transliteration**: `to_ascii` (decompose + strip marks + a Latin/Greek/Cyrillic table + symbols),
  `slugify`.
- **Fuzzy**: `levenshtein`, `damerau_levenshtein` (optimal string alignment), `jaro_winkler`,
  `fuzzy_score` (subsequence matcher like editors use), all over graphemes or code points.
- **Regex**: a small backtracking engine over code points: classes with `\p{L}`, `\p{Script=Greek}`,
  `\w \d \s \b` with Unicode semantics, groups, alternation, greedy/lazy quantifiers, anchors,
  case-insensitive (case folding), `match`, `search`, `find_all`, `replace`. Step-limited so it
  cannot hang.
- **Encodings**: `decode(bytes, charset)` / `encode(utf8, charset)` for UTF-8/16LE/16BE/32, ASCII,
  ISO-8859-1..16, Windows-874 and 1250..1258, KOI8-R/U, IBM437/850/866, Mac Roman, Shift_JIS
  (Windows-31J), EUC-JP, GBK/GB2312, Big5 and EUC-KR. Unmappable handling: error, replace, or
  skip. `detect_bom`, `sniff_charset` (BOM, then UTF-8 validity, then byte statistics).
- **Console**: `terminal_size`, `enable_vt` (Windows ENABLE_VIRTUAL_TERMINAL_PROCESSING),
  `supports_color` (NO_COLOR, FORCE_COLOR, TERM, isatty), `style` (16/256/truecolor, bold, italic,
  underline...), `strip_ansi`, `wrap` (line-break opportunities + display width), `truncate` with
  ellipsis at a grapheme boundary, `pad`/`align`.

## API outline (`forge::os::locale`)

- **Detection**: `system_locale()` and `preferred_languages()`.
  - POSIX: `LC_ALL`, `LC_MESSAGES`/`LC_CTYPE`/category, `LANG`, `LANGUAGE` list, then
    `setlocale(LC_ALL, "")`.
  - Windows: `GetUserDefaultLocaleName`, `GetUserPreferredUILanguages`.
  - macOS/iOS: `CFLocaleCopyCurrent`, `CFLocaleCopyPreferredLanguages` in `locale_apple.mm`, only
    compiled for Apple targets.
  - Android: `persist.sys.locale` / `ro.product.locale` system properties, or a value pushed by the
    app's JNI glue through `android_set_locale()`.
- **Tags**: BCP 47 parse/canonicalise (`en_AU.UTF-8@euro` → `en-AU`), likely subtags, fallback chain
  (`zh-Hant-TW` → `zh-Hant` → `zh` → root).
- **Numbers**: `format_number`, `format_percent`, `format_currency` (symbol/code/name, CLDR digits),
  `format_compact` (1.2K), `parse_number` using CLDR symbols and patterns, native digits
  (Arabic, Persian, Devanagari, Bengali, Thai).
- **Plurals**: CLDR rule strings evaluated at runtime (`plural_category(n, locale, ordinal)`), so all
  CLDR languages are covered, not only the bundled formatting locales.
- **Lists**: `format_list` (and / or / unit, standard / short / narrow).
- **Dates/times**: `format_datetime` with CLDR skeleton-like styles (full/long/medium/short) and
  explicit LDML patterns (`yyyy-MM-dd HH:mm`), month/day names, `parse_datetime` for the same
  patterns, `format_relative` ("in 3 days", "yesterday").
- **Time zones**: wrap `std::chrono::tzdb` (`zone_names`, `to_zone`, `offset_at`), with a fallback to
  the OS zoneinfo when the standard library has no tzdb.
- **Calendars**: Gregorian, ISO week dates, plus Islamic civil (tabular), Hebrew, Persian (Solar
  Hijri), Japanese eras, Buddhist and ROC as conversions.
- **Region data**: `first_day_of_week`, `measurement_system`, `currency_for_region`, `hour_cycle`.
- **Units**: `format_unit(value, "kilometer-per-hour", width)`, plus a conversion helper for common
  length/mass/temperature/volume/speed units.
- **Messages**: ICU MessageFormat-lite: `{name}`, `{n, number}`, `{n, plural, one {...} other {...}}`
  with `#` and `=N`, `{g, select, ...}`, `{n, selectordinal, ...}`, nested, quoting with `'`.
- **Catalogs**: gettext `.mo` (both endiannesses) and `.po` loading (msgctxt, plural forms with a
  small C-expression evaluator for `Plural-Forms`), `gettext`, `ngettext`, `pgettext`, lookup over the
  fallback chain.

## Bundled locales

CLDR data for 52 locales (en, en-GB/AU/IN/CA, de, de-CH, fr, fr-CA, es, es-MX, it, pt, pt-PT, nl,
sv, da, nb, fi, pl, cs, ru, uk, tr, el, he, ar, fa, hi, bn, th, vi, id, ja, ko, zh, zh-Hant, ro, hu,
sk, bg, hr, ga, cy, ms, sw, et, lt, lv, sl, sr, ca). Other locales fall back along the chain to
their language, then to `en`/root. Plural rules cover every CLDR language.

## Script built-ins

`forge::text::` exposes normalize, case, width, graphemes/words/sentences, wrap, truncate,
slugify, to_ascii, levenshtein, fuzzy, collate/sort, regex helpers, encode/decode, sniff and
`style`. `forge::locale::` exposes current, preferred, number/currency/percent/date/relative/list/
unit formatting, plural, message format and catalogs (`load_catalog`, `tr`, `ntr`).

## Testing

- `test_text` checks UTF, properties, case, width, transliteration, fuzzy, regex, encodings and the
  console helpers.
- `test_text_conformance` runs the Unicode files when it can find them in `FORGE_UCD_DIR` or
  `tests/data/ucd`; `cmake -DFORGE_FETCH_UCD=ON` downloads them at configure time. Without the
  files it exits 77 (skipped).
- `test_locale` covers formatting, parsing, plurals, messages, catalogs, calendars and detection.
- Cross compile-checks (`xos_text_*`) build the library for FreeBSD, macOS, Windows (MinGW) and
  aarch64 musl with zig. `locale_apple.mm` needs an Apple SDK and is only compile-checked on macOS CI.

## Non-goals

Full ICU parity. Some things are left out: full CLDR (every locale and every field), collation
tailorings beyond a simple rule list, IDNA/UTS 46, spoof checking, and the Chinese/Indian lunisolar
calendars. Dictionary-based line and word breaking for Thai/Lao/Khmer/Burmese is also out; these
fall back to the UAX 14 rule, which only breaks at spaces.
