# Qualitative Theme Extraction 

This repository contains a Python workflow to classify sample qualitative data responses
into predefined themes. It can run with either model:

- **OpenAI** (`theme-extraction.py`): classifies with `gpt-4o-mini`
- **Jev** (`theme-extraction-jev.py`): classifies with TypeSafe's Jev model

Both use the same 5 themes, so their results can be compared directly:
1. Confidence & Self-Belief
2. Aspirations for Higher Education
3. Employability & Job Readiness
4. Skills Development (Digital & Communication)
5. Family, Gender & Social Support

## How it works
1. dbt has already created an analytics table with `theme_extracted = NULL`
2. The Python job classifies new responses using the llm api (blank cells are picked up as well as NULL)
3. Results are written back to Postgres
4. Runs automatically via GitHub Actions (currently the OpenAI version)

## Jev vs OpenAI
Jev was tested on the same dataset and checked for accuracy against the OpenAI labels.
It was also faster.

| | OpenAI (`gpt-4o-mini`) | Jev |
| --- | --- | --- |
| Agreement with OpenAI labels | n/a | <!-- fill in % --> |
| Time to classify the dataset | <!-- fill in --> | <!-- fill in --> |

Jev also returns a confidence score for each response, so low-confidence rows can be
flagged for human review.

Jev writes to its own columns, so the OpenAI labels in `theme_extracted` are never changed:
- `theme_extracted_jev`: the Jev theme
- `theme_jev_confidence`: how sure Jev was, from 0 to 1
