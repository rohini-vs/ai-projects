from pydantic import BaseModel


class JobPosting(BaseModel):
    title: str
    company: str
    location: str
    min_experience_years: int
    max_experience_years: int
    salary_min_lpa: float
    salary_max_lpa: float
    remote: bool