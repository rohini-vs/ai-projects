import json

from pydantic import ValidationError

from ollama_client import generate
from schema import JobPosting

MAX_ATTEMPTS = 3


def strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def extract(model: str, job_text: str) -> tuple[JobPosting | None, int]:
    """Returns (result, attempts_used). result is None if all attempts failed."""
    prompt = f"""Extract the job details from this posting as a JSON object with these exact keys:
title, company, location, min_experience_years, max_experience_years, salary_min_lpa, salary_max_lpa, remote.

Posting:
{job_text}

Respond with only the JSON object, nothing else."""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = generate(model, prompt, temperature=0.0, max_tokens=300)
        raw = strip_fences(result["response"])

        try:
            data = json.loads(raw)
            job = JobPosting(**data)
            return job, attempt
        except json.JSONDecodeError as e:
            error_msg = f"Your response was not valid JSON: {e}"
        except ValidationError as e:
            error_msg = f"Your JSON didn't match the required schema:\n{e}"

        # Feed the specific error back and ask again
        prompt = f"""Your previous response had an error:
{error_msg}

Your previous response was:
{raw}

Fix the error and return ONLY the corrected JSON object with these exact keys:
title, company, location, min_experience_years, max_experience_years, salary_min_lpa, salary_max_lpa, remote."""

    return None, MAX_ATTEMPTS