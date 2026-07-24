# Memory Index — dx_app

> Persistent knowledge for dx_app development. Read at task start, update when learning.

## Files

| File | When to Read |
|---|---|
| `common_pitfalls.md` | **ALWAYS** — read before every task |
| `model_zoo.md` | When working with models, choosing models, or verifying model compatibility |
| `platform_api.md` | When dealing with NPU devices, DX-RT runtime, drivers, or version issues |
| `performance_patterns.md` | When optimizing FPS, comparing variants, or profiling inference |

## Update Protocol

When you discover a new pitfall, insight, or pattern:

1. **Identify the correct file** — match the domain to one of the files above
2. **Add with a domain tag** — use one of: `[UNIVERSAL]`, `[DX_APP]`, `[PPU]`
3. **Include all three fields:**
   - **Symptom** — what the user sees (error message, wrong behavior)
   - **Cause** — the underlying reason
   - **Fix** — the specific corrective action
4. **Avoid duplicates** — search the file first before adding

## Domain Tags

| Tag | Scope |
|---|---|
| `[UNIVERSAL]` | Applies to all dx_app usage (Python, C++, all tasks) |
| `[DX_APP]` | Specific to dx_app framework (runners, factories, variants) |
| `[PPU]` | Specific to PPU (Pre/Post Processing Unit) models |

**Note:** This memory system is scoped to dx_app only. Domain tags from other
sub-projects (e.g. stream-level tags) do not appear in this knowledge base.
