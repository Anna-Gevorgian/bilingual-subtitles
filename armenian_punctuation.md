# Armenian Punctuation — House Style Reference

This is the authoritative style guide for this project. Apply these rules to
all translated subtitle lines. No need to verify against an external source.

---

## Sentence-final and pause marks

| Mark | Armenian name | Codepoint | Use |
|---|---|---|---|
| `։` | վերջակետ | U+0589 | Ends a declarative sentence (looks like `:` but is a period) |
| `,` | ստորակետ | — | In-sentence separation |
| `՝` | բութ | U+055D | Short pause in certain constructions |

## Marks placed OVER a vowel

These three marks attach to the vowel of the relevant word — not to the end of
the sentence. This is the biggest divergence from Latin-script conventions.

| Mark | Armenian name | Codepoint | Placed over |
|---|---|---|---|
| `՞` | հարցական նշան | U+055E | Last vowel of the questioned word |
| `՜` | բացականչական նշան | U+055C | Vowel of the exclaimed/stressed word |
| `՛` | շեշտ | U+055B | Any vowel receiving emphasis |

**Examples:**

| English | Wrong ❌ | Correct ✓ |
|---|---|---|
| Are you coming? | Դու գալի՞ս ես? | Դու գա՞լիս ես |
| What a day! | Ի՜նչ օր է! | Ի՜նչ օր է |
| I told *you* | Ես ձեզ ասացի | Ես ձե՛զ ասացի |

## Dashes and dialogue

- Em dash `—` opens each dialogue line and marks strong breaks.
- Armenian hyphen `֊` (U+058A) for connective hyphens within a word.
- Do not use standard ASCII hyphen `-` where `֊` is required.

## Quotation marks

- Use angle guillemets `«…»` for all quotations and titles.
- Do not use `"…"` or `'…'`.

---

## Scope

Apply these rules to all subtitle lines including dialogue, titles, and
on-screen text. Loanwords, URLs, and numerals follow the same punctuation rules
as native Armenian words.

---

## Pre-output checklist

Before finalising each line, verify:

1. Full stop is `։` (U+0589), not `:` (colon).
2. No trailing `?` or `!` — question and exclamation marks are mid-word.
3. Dialogue lines open with `—`.
4. Quotations use `«…»`.
5. Line is within the 42-character budget.
