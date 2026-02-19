import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Create a PDF document
pdf = SimpleDocTemplate("student_analysis_report.pdf", pagesize=letter)
story = []
styles = getSampleStyleSheet()

# Add title
story.append(Paragraph("Student Performance Analysis Report", styles['Title']))
story.append(Spacer(1, 24))

# Add data overview
story.append(Paragraph("\n\n\nDataset Overview:", styles['Heading2']))
story.append(Paragraph("- 649 records with 33 features", styles['Normal']))
story.append(Paragraph("- No missing values detected", styles['Normal']))
story.append(Paragraph("- Key metrics: age (15-22), absences (0-32), final grades (0-20)", styles['Normal']))
story.append(Spacer(1, 12))

# Add figure: Absences distribution
story.append(Paragraph("\n\n\nAbsence Patterns:", styles['Heading2']))
story.append(Image("absences_boxplot.png", 400, 300))
story.append(Paragraph("\n\n\nKey finding: 75% of students had ≤6 absences, with one extreme outlier at 32 days", styles['Normal']))
story.append(Spacer(1, 12))

# Add figure: Top 10 feature distributions
story.append(Paragraph("\n\n\nDemographic Distribution:", styles['Heading2']))
story.append(Image("top10_histograms.png", 400, 300))
story.append(Paragraph("\n\n\nKey finding: Age distribution peaks at 17-18 years, with family relationship scores averaging 4.2", styles['Normal']))
story.append(Spacer(1, 12))

# Add figure: Grade correlations
story.append(Paragraph("\n\n\nGrade Progression:", styles['Heading2']))
story.append(Image("grade_correlation_scatter.png", 400, 300))
story.append(Paragraph("\n\n\nKey finding: Strong positive correlation between consecutive grades (G1-G2: 0.865, G2-G3: 0.919)", styles['Normal']))
story.append(Spacer(1, 12))

# Add recommendations
story.append(Paragraph("\n\n\nRecommendations:", styles['Heading2']))
story.append(Paragraph("1. Implement targeted academic support for students with 10+ absences", styles['Normal']))
story.append(Paragraph("2. Monitor students with declining grade trajectories (G1 > G2 > G3)", styles['Normal']))
story.append(Paragraph("3. Investigate outlier cases with extreme absence values (e.g., 32 days)", styles['Normal']))

# Build PDF
doc = SimpleDocTemplate("student_analysis_report.pdf")
doc.build(story)