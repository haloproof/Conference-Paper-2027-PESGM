# Price data provenance

Source attribution: **Bundesnetzagentur | SMARD.de**. Source data-use page: https://www.smard.de/en/datennutzung ; license: https://creativecommons.org/licenses/by/4.0/ .

The raw German/Luxembourg day-ahead export is preserved byte-for-byte in raw/. Prepared data in dispatch-prices/ select local 2025-09-01 through 2026-09-01 (exclusive), convert EUR/MWh to EUR/kWh, and use the recorded Europe/Berlin timezone assumption with the audited DST end-label handling. The 35,040 intervals form 365 consecutive 24-hour simulation days. Later simulation years repeat this observed price year.

The audit source path is adapted to data/raw/ for this release; numerical prices and original data hashes are unchanged. Repeating prices is a deterministic perfect-foresight benchmark, not a market forecast. See reproduce.md for verification and optional regeneration.
