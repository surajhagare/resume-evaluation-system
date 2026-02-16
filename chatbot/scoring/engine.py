import re
import numpy as np
from ..rag.embeddings import generate_embeddings


class ResumeScorer:

    def __init__(self, job_description, sections):
        self.jd = job_description.lower()
        self.sections = sections

    # ==========================
    # KEYWORD SKILL SCORE (Out of 30)
    # ==========================
    def skill_score(self):

        skills_text = self.sections.get("skills", "").lower()
        jd_words = self.jd.split()

        if not jd_words:
            return 0

        ignore_words = {
            "looking", "for", "with", "and", "the",
            "a", "an", "to", "of", "in", "on", "at",
            "developer", "engineer", "role", "job"
        }

        matched = 0
        valid_words = 0

        for word in jd_words:

            word = word.strip(",.()")

            if word in ignore_words:
                continue

            if len(word) <= 2:
                continue

            valid_words += 1

            if word in skills_text:
                matched += 1

        if valid_words == 0:
            return 0

        score = (matched / valid_words) * 30
        return round(score, 2)

    # ==========================
    # SEMANTIC SKILL SCORE (Out of 30)
    # ==========================
    def semantic_skill_score(self):

        skills_text = self.sections.get("skills", "")
        jd_text = self.jd

        if not skills_text:
            return 0

        jd_emb = generate_embeddings([jd_text])[0]
        resume_emb = generate_embeddings([skills_text])[0]

        jd_emb = np.array(jd_emb)
        resume_emb = np.array(resume_emb)

        similarity = np.dot(jd_emb, resume_emb) / (
            np.linalg.norm(jd_emb) * np.linalg.norm(resume_emb)
        )

        score = similarity * 30

        return round(float(score), 2)

    # ==========================
    # EXPERIENCE SCORE (Out of 25)
    # ==========================
    def experience_score(self):

        exp_text = self.sections.get("experience", "").lower()

        if not exp_text:
            return 0

        match = re.search(r'(\d+)\s*\+?\s*(years|yrs)', exp_text)

        if match:
            years = int(match.group(1))
            score = years * 5
        else:
            if "intern" in exp_text or "developer" in exp_text:
                score = 5
            else:
                score = 0

        return min(score, 25)

    # ==========================
    # PROJECT SCORE (Out of 20)
    # ==========================
    def project_score(self):

        projects_text = self.sections.get("projects", "").lower()

        if not projects_text:
            return 0

        project_count = projects_text.count("project")

        if project_count == 0 and len(projects_text) > 50:
            return 5

        score = project_count * 5
        return min(score, 20)

    # ==========================
    # SKILL GAP DETECTION
    # ==========================
    def skill_gap(self):

        skills_text = self.sections.get("skills", "").lower()
        jd_words = self.jd.split()

        ignore_words = {
            "looking", "for", "with", "and", "the",
            "a", "an", "to", "of", "in", "on", "at",
            "developer", "engineer", "role", "job"
        }

        matched = []
        missing = []

        for word in jd_words:

            word = word.strip(",.()")

            if word in ignore_words:
                continue

            if len(word) <= 2:
                continue

            if word in skills_text:
                matched.append(word)
            else:
                missing.append(word)

        return {
            "matched_skills": list(set(matched)),
            "missing_skills": list(set(missing))
        }

    # ==========================
    # AI FEEDBACK GENERATOR
    # ==========================
    def generate_feedback(self):

        feedback = []
        gap = self.skill_gap()

        if gap["missing_skills"]:
            missing = ", ".join(gap["missing_skills"])
            feedback.append(
                f"Add these missing skills to improve your match: {missing}."
            )

        if self.experience_score() == 0:
            feedback.append(
                "Consider adding internship experience or practical work exposure."
            )

        if self.project_score() < 10:
            feedback.append(
                "Add more detailed projects with measurable achievements."
            )

        if self.total_score() < 25:
            feedback.append(
                "Overall resume match is low. Improve skills alignment with the job description."
            )

        return feedback

    # ==========================
    # HYBRID TOTAL SCORE
    # ==========================
    def total_score(self):

        keyword = self.skill_score()
        semantic = self.semantic_skill_score()

        # Hybrid weighting (40% keyword + 60% semantic)
        hybrid_skill = (0.4 * keyword) + (0.6 * semantic)

        return round(
            hybrid_skill
            + self.experience_score()
            + self.project_score(),
            2
        )
