from django.shortcuts import render, redirect
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .rag.pipeline import rag_chat
from .models import UploadedPDF
from .rag.pdf_processor import process_pdf, extract_text, extract_sections
from .scoring.engine import ResumeScorer
from .llm.feedback_generator import generate_llm_feedback


# ==========================
# HOME PAGE
# ==========================
def home(request):
    return render(request, 'chatbot/home.html')


# ==========================
# AI CHAT API
# ==========================
@api_view(['GET'])
def chat_api(request):
    question = request.GET.get("q")

    if not question:
        return Response({"error": "Question is required."})

    result = rag_chat(question)

    return Response({
        "answer": result["answer"]
    })


# ==========================
# RESUME UPLOAD + JD SAVE
# ==========================
def upload_pdf(request):

    if request.method == 'POST' and request.FILES.get('file'):

        # Save PDF
        pdf = UploadedPDF.objects.create(file=request.FILES['file'])

        # Extract resume text
        text = extract_text(pdf.file.path)

        # Save resume text in session
        request.session["resume_text"] = text

        # Save Job Description from form
        job_description = request.POST.get("job_description")
        request.session["job_description"] = job_description

        # Process embeddings for FAISS
        process_pdf(pdf.file.path)

        return redirect('score_resume')

    return redirect('home')


# ==========================
# RESUME SCORE DASHBOARD
# ==========================
def score_resume(request):

    # Get JD from session
    job_description = request.session.get(
        "job_description",
        "Looking for Python Django Developer with SQL AWS Docker"
    )

    # Get Resume text
    resume_text = request.session.get("resume_text")

    if not resume_text:
        return render(request, "chatbot/score.html", {
            "error": "Resume not uploaded yet"
        })

    # Extract sections
    sections = extract_sections(resume_text)

    # Initialize scorer
    scorer = ResumeScorer(job_description, sections)

    # Skill gap detection
    gap = scorer.skill_gap()

    # Rule-based feedback
    feedback = scorer.generate_feedback()

    # LLM explanation
    resume_summary = (
        sections.get("skills", "") + " " +
        sections.get("projects", "")
    )

    score_data = {
        "total_score": scorer.total_score(),
        "missing_skills": gap["missing_skills"]
    }

    llm_feedback = generate_llm_feedback(
        job_description,
        resume_summary,
        score_data
    )

    # Final context
    context = {
        "semantic_skill_score": scorer.semantic_skill_score(),
        "experience_score": scorer.experience_score(),
        "project_score": scorer.project_score(),
        "total_score": scorer.total_score(),
        "matched_skills": gap["matched_skills"],
        "missing_skills": gap["missing_skills"],
        "ai_feedback": feedback,
        "llm_feedback": llm_feedback
    }

    return render(request, "chatbot/score.html", context)
