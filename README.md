# Qualitative Theme Extraction 

This repository contains a Python workflow to classify sample qualitative data responses
into predefined themes using the OpenAI API.

## How it works
1. dbt has already created an analytics table with `theme_extracted = NULL`
2. The Python job classifies new responses using the llm api
3. Results are written back to Postgres
4. Runs automatically via GitHub Actions
