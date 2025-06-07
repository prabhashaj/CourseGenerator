from io import BytesIO
from fpdf import FPDF

def course_to_pdf(course):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, course['courseTitle'], ln=True, align='C')
    pdf.set_font("Arial", '', 12)
    pdf.multi_cell(0, 8, f"Introduction: {course['introduction']}")
    pdf.ln(4)
    for module in course.get('modules', []):
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 8, f"Module {module['moduleNumber']}: {module['moduleTitle']}", ln=True)
        pdf.set_font("Arial", '', 12)
        for chapter in module.get('chapters', []):
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 7, f"- {chapter['chapterTitle']}", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 6, chapter['description'])
        pdf.ln(2)
    pdf.set_font("Arial", 'I', 12)
    pdf.multi_cell(0, 8, f"Conclusion: {course['conclusion']}")
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output.read()
