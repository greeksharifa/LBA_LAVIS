# Research tools

These scripts were recovered from the ignored files in the local
uncertainty-v2 checkout. They cover API-backed sub-question generation,
ChatGPT/Gemini/DeepSeek calls, and older local-generation paths.

`generate_subqa_chatgpt.py` reads the runtime credential from the standard
`OPENAI_API_KEY` environment variable. The original credential must be
considered compromised and rotated separately.

The `vendor_examples/` scripts retain attribution to the upstream revision
against which each local experiment was written.
