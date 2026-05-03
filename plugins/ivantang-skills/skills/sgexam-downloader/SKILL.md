---
name: sgexam-downloader
description: Downloads Singapore primary school exam papers (PDFs) from sgexam.com. Use this skill whenever the user wants to download, fetch, or batch-save Singapore exam papers, practice papers, prelim papers, weighted assessments, SA1, SA2, or any assessment materials for primary school students. Trigger even if the user just says "get me P5 maths papers" or "download the prelim papers" without explicitly mentioning sgexam.com.
---

# SGExam Paper Downloader

Downloads exam PDFs from https://sgexam.com/ for Singapore primary school students (P1–P6).

## Site Knowledge

**Listing pages:** `https://sgexam.com/primary-{N}-{subject}/`
- Subjects: `english`, `maths`, `science` (P3-P6 only), `chinese`, `higher-chinese`
- All papers on one page, no pagination

**Paper page URL patterns (3 types, all handled):**
- `/subject/{subject}/{slug}-pdf/` — standard, 2016+
- `/year/{year}/{slug}-pdf/` — older papers, pre-2016
- `/level/primary-{N}/{slug}-pdf/` — higher-chinese exclusively

**Slug format:** `{year}-p{N}-{subject}-{assessment-type}-{school}-pdf`

**Download:** Each paper page has one Google Drive link: `drive.google.com/uc?export=download&id=...`

**Known assessment type slugs:**
`prelim-exam`, `end-of-year-exam`, `weighted-assessment-1`, `weighted-assessment-2`, `weighted-assessment-3`, `non-weighted-assessment-1`, `semestral-assessment-1`, `semestral-assessment-2`, `bite-sized-assessment-1`, `mid-year-practice-paper`, `class-test`, `revision`, `review`, `quiz`, `term`, `sa1`, `sa2`, `ca1`, `ca2`, `rt`

## Workflow

### 1. Collect Parameters

Ask the user (use numbered menus):

**Level** (P1–P6, comma-separated or `all`):
```
Select level(s): 1)P1  2)P2  3)P3  4)P4  5)P5  6)P6  or 'all'
```

**Subject** (show only valid subjects for selected levels — P1-P2 have no Science):
```
Select subject(s): 1)English  2)Maths  [3)Science]  4)Chinese  5)Higher Chinese  or 'all'
```

**Paper type** (show numbered list from known slugs, or 'all'):
```
Select paper type(s): 1)Prelim Exam  2)End of Year Exam  3)Weighted Assessment 1  ...  or 'all'
```

**Year** (`2025`, `2023,2024,2025`, `2020-2025`, or `all`):
```
Year filter (e.g. 2025, 2022-2025, or all):
```

Show the crawl plan and confirm before starting.

### 2. Set Up Python Environment

The bundled script (`scripts/download_papers.py` in this skill's directory) needs `requests`, `beautifulsoup4`, and `lxml`. Check and set up:

```bash
SKILL_DIR="$(dirname "$0")"   # or use the known skill path
SCRIPT="$SKILL_DIR/scripts/download_papers.py"

# Use project venv if available, else create one
if [ -f ".venv/bin/python3" ]; then
  PY=".venv/bin/python3"
elif python3 -c "import requests, bs4" 2>/dev/null; then
  PY="python3"
else
  python3 -m venv .venv
  .venv/bin/pip install -q requests beautifulsoup4 lxml
  PY=".venv/bin/python3"
fi
```

### 3. Run the Bundled Script

Pass the collected parameters as CLI arguments:

```bash
$PY "$SCRIPT" \
  --levels "5,6" \
  --subjects "maths,science" \
  --types "prelim-exam,sa1,sa2" \
  --years "2023-2025" \
  --output "./downloads"
```

Argument formats:
- `--levels`: comma-separated numbers (`1,2,3`) or `all`
- `--subjects`: comma-separated slug names or `all`
- `--types`: comma-separated slug names or `all`
- `--years`: `2025`, `2023,2024,2025`, `2020-2025`, or `all`
- `--output`: download directory (default: `./downloads`)

The script prints `[i/N] [OK/SKIP/FAIL] filename` progress and a final summary.

### 4. After Completion

Report what was downloaded, skipped, and failed. If there were failures, show the list. Remind the user that re-running is safe — already-downloaded files are skipped automatically.

## Tips

- **Polite crawling:** The script uses 1-2s random delays between requests.
- **Resume support:** Files already downloaded are skipped. Re-run safely after interruption.
- **Large batches:** "All levels + all subjects + all years" = 5,000+ papers. Recommend filtering to specific years/types first.
- **Google Drive quota:** Popular files sometimes hit download quotas — they'll show as FAIL. Try again the next day.
- **Rate limiting from Google:** If many FAILs appear in a row, pause 5-10 minutes before resuming.
