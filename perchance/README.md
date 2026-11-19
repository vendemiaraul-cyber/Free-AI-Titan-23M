# Titan's Perchance image generator

Titan makes its images with its generator on perchance.org: **https://perchance.org/rg4m14ri**.
It's the default in `titan.py` and the Worker, and the Image tab of the browser UI embeds it:

```html
<iframe src="https://perchance.org/rg4m14ri"
  style="width:100%; max-width:700px; height:800px; border:0; border-radius:12px;"
  loading="lazy"
  allow="clipboard-write">
</iframe>
```

To make the headless driver work, `titan.py` needs to find three things on the page:

| What | Default selector Titan looks for | Override with |
|---|---|---|
| Prompt box | `#prompt` (else the first `textarea` / text input) | `TITAN_PERCHANCE_PROMPT_SELECTOR` |
| Negative prompt (optional) | `#negative` | `TITAN_PERCHANCE_NEGATIVE_SELECTOR` |
| Generate button | `#generateBtn` (else a button labelled "Generate", else the first `button`) | `TITAN_PERCHANCE_BUTTON_SELECTOR` |

Turn on automatic images:

```bash
pip install playwright && playwright install chromium   # lets titan.py drive it headlessly
export TITAN_PERCHANCE_URL=https://perchance.org/another-generator   # only to use a different one
```

How it works: `titan.py` opens your generator in headless Chromium, types the prompt, clicks
the button and waits for a new image (≥256×256 px) to appear in any frame on the page, then saves
it to `outputs/`. Without Playwright, the API and UI return your generator's link instead.

`starter-generator.txt` is an optional starting point that already uses those IDs: paste its
two parts into a new generator at https://perchance.org/welcome.
