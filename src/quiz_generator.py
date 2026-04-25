"""Quiz generation module using RAG retrieval and Sarvam AI."""

import json
from typing import Any
from sarvam_client import chat_completions


def generate_quiz(
    retriever,
    subject: str,
    student_class: str,
    topic: str,
    difficulty: str,
    num_questions: int = 5
) -> dict[str, Any]:
    """
    Generate a quiz based on retrieved textbook content.
    
    Args:
        retriever: The retriever instance to fetch relevant content
        subject: Subject name (e.g., 'social_science', 'science', 'english')
        student_class: Class level (e.g., '5', '6', '7', '8')
        topic: Topic for the quiz
        difficulty: Difficulty level ('Easy', 'Medium', 'Hard')
        num_questions: Number of questions to generate
        
    Returns:
        Dictionary containing quiz data with questions, options, and answers
    """
    print(f"[Quiz] Generating {num_questions} {difficulty} questions on '{topic}' for Class {student_class} {subject}")
    
    # Retrieve relevant content from textbook
    results_df = retriever.search(
        query=topic,
        subject=subject,
        student_class=student_class,
        k=5  # Get more context for quiz generation
    )
    
    if results_df.empty:
        return {
            "success": False,
            "error": f"Could not find content about '{topic}' in Class {student_class} {subject} textbook."
        }
    
    # Combine retrieved context
    context_text = "\n\n".join(results_df["text"].tolist())
    
    # Create difficulty-specific instructions
    difficulty_instructions = {
        "Easy": "Create straightforward questions that test basic understanding and recall of key facts.",
        "Medium": "Create questions that require understanding of concepts and ability to apply knowledge.",
        "Hard": "Create challenging questions that require critical thinking, analysis, and synthesis of information."
    }
    
    difficulty_instruction = difficulty_instructions.get(difficulty, difficulty_instructions["Medium"])
    
    # Prompt for quiz generation
    system_prompt = (
        "You are an expert teacher creating quiz questions for Indian school students. "
        "Generate ONLY multiple choice questions (MCQs) based on the provided textbook content. "
        f"{difficulty_instruction}\n\n"
        "IMPORTANT: Return your response as a valid JSON array with this EXACT structure:\n"
        "[\n"
        "  {\n"
        '    "question": "Question text here?",\n'
        '    "options": ["A) First option", "B) Second option", "C) Third option", "D) Fourth option"],\n'
        '    "correct_answer": "A",\n'
        '    "explanation": "Brief explanation of why this is correct"\n'
        "  }\n"
        "]\n\n"
        "Rules:\n"
        "- Return ONLY the JSON array, no other text\n"
        "- Each question must have exactly 4 options labeled A, B, C, D\n"
        "- The correct_answer must be ONLY the letter (A, B, C, or D)\n"
        "- All questions must be based on the provided textbook content\n"
        "- Make questions age-appropriate and clear"
    )
    
    user_prompt = (
        f"Topic: {topic}\n"
        f"Class: {student_class}\n"
        f"Subject: {subject}\n"
        f"Difficulty: {difficulty}\n"
        f"Number of questions: {num_questions}\n\n"
        f"Textbook Content:\n{context_text}\n\n"
        f"Generate {num_questions} multiple choice questions as a JSON array."
    )
    
    try:
        # Call Sarvam AI
        response = chat_completions(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        raw_response = response["choices"][0]["message"]["content"]
        print(f"[Quiz] Received response from Sarvam AI")
        
        # Parse JSON response
        # Try to extract JSON from response (in case there's extra text)
        json_start = raw_response.find('[')
        json_end = raw_response.rfind(']') + 1
        
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON array found in response")
        
        json_str = raw_response[json_start:json_end]
        questions = json.loads(json_str)
        
        if not isinstance(questions, list):
            raise ValueError("Response is not a JSON array")
        
        # Validate questions
        for i, q in enumerate(questions):
            if not all(key in q for key in ["question", "options", "correct_answer"]):
                raise ValueError(f"Question {i+1} is missing required fields")
            if len(q["options"]) != 4:
                raise ValueError(f"Question {i+1} does not have exactly 4 options")
        
        print(f"[Quiz] ✅ Successfully generated {len(questions)} questions")
        
        return {
            "success": True,
            "questions": questions,
            "topic": topic,
            "subject": subject,
            "student_class": student_class,
            "difficulty": difficulty
        }
        
    except json.JSONDecodeError as e:
        print(f"[Quiz] ❌ JSON parsing error: {e}")
        return {
            "success": False,
            "error": f"Failed to parse quiz questions. Please try again."
        }
    except Exception as e:
        print(f"[Quiz] ❌ Error generating quiz: {e}")
        return {
            "success": False,
            "error": f"Error generating quiz: {str(e)}"
        }


def check_answers(quiz_data: dict, user_answers: dict[int, str]) -> dict[str, Any]:
    """
    Check user answers against correct answers.
    
    Args:
        quiz_data: The original quiz data with correct answers
        user_answers: Dictionary mapping question index to user's answer (A/B/C/D)
        
    Returns:
        Dictionary with score, results per question, and feedback
    """
    if not quiz_data.get("success"):
        return {"success": False, "error": "Invalid quiz data"}
    
    questions = quiz_data["questions"]
    results = []
    correct_count = 0
    
    for i, question in enumerate(questions):
        correct_answer = question["correct_answer"].strip().upper()
        user_answer = user_answers.get(i, "").strip().upper()
        is_correct = user_answer == correct_answer
        
        if is_correct:
            correct_count += 1
        
        results.append({
            "question_num": i + 1,
            "question": question["question"],
            "user_answer": user_answer if user_answer else "Not answered",
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "explanation": question.get("explanation", "")
        })
    
    score = (correct_count / len(questions)) * 100
    
    return {
        "success": True,
        "score": score,
        "correct_count": correct_count,
        "total_questions": len(questions),
        "results": results
    }
