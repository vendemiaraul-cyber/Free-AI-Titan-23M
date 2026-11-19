# Titan's Perchance image generator

Titan makes its images with **your generator on perchance.org**. Build it however you like;
`titan.py` only needs to find three things on the page:

| What | Default selector Titan looks for | Override with |
|---|---|---|
| Prompt box | `#prompt` (else the first `textarea` / text input) | `TITAN_PERCHANCE_PROMPT_SELECTOR` |
| Negative prompt (optional) | `#negative` | `TITAN_PERCHANCE_NEGATIVE_SELECTOR` |
| Generate button | `#generateBtn` (else the first `button`) | `TITAN_PERCHANCE_BUTTON_SELECTOR` |

Then point Titan at it:

```bash
export TITAN_PERCHANCE_URL=https://perchance.org/your-generator-name
pip install playwright && playwright install chromium   # lets titan.py drive it headlessly
```

How it works: `titan.py` opens your generator in headless Chromium, types the prompt, clicks
the button and waits for a new image (≥256×256 px) to appear in any frame on the page, then saves
it to `outputs/`. Without Playwright, the API and UI return your generator's link instead.

`starter-generator.txt` is an optional starting point that already uses those IDs: paste its
two parts into a new generator at https://perchance.org/welcome.
