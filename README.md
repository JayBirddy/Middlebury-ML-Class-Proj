# Middlebury-ML-Class-Proj

# Group Members
Jayden Chen,
Evan Lin,
Jonathan Mota

# Abstract
We will be addressing the problem: "Will a patient, upon discharge, likely have to be readmitted to the hospital 
within a 30 day period?" We will be attempting to solve this problem by building a prediction model using various 
patient and prognosis details (historically and upon discharge) to output a probability per patient. Evaluation
will be based on a focus for "clinical intervention" and so we will be focusing on sensitivity and precision/PPV as
the main evaluation tools. Further nuance will be need to be gauged and specified as the fundamental questions posed 
by the metrics above are: "Is the model actually catching the patients who will be readmitted?" and "Are the flagged patients  
worth the clinical team's time?" respectively and they are in tension will each other. 

# Motivation and Question
We have public health data for which predictive models would help us give better decisions and resource management tasks.

In the current field of medicine, there are many exterior factors that can play into a patient's discharge. While there isn't much we can do about a patient's cooperation, a hospital's funding, or Insurance-related factors, what we can do is try to implement a system that promotes precautionary and preventive measure for many patients. 

# Planned Deliverables
- Python package containing code for algorithms and analysis as well as documentation.
- One Jupter notebook for illustrative, walkthrough style purposes
- An essay discussing ethic implications of our project. May include an ethics matrix and moral/legal discussions

Full success
- Above stated deliverables as well as a formalized app with a working interaction no matter how primitive.

Partial success
- 2-3 of the above states deliverables.

# Resources Required
Data will be based on MIMIC-III Clinical Database: a deidentified, publicly available database health records dataset based from The Beth Israel Deaconess Medical Center (BIDMC) 

Most likely cloud GPU computing power

student accessible resources


# What We Will Learn
- Evan: I hope to deepen my understanding of machine learning for healthcare applications, especially how to develop and evaluate predictive models using real clinical data, and to do so effectively by making sure that the work is done in a collaborative setting with good communication.
- Jonathan: I would like to have a firsthand experience dealing with data and personal biases. In recent months, I have taken a look at many models that are responsible for making decisions of the life of another human. This being in the same category as those offers an exclusive oppotunity to reflect on addressing patterns with my team. 

# Risk Statement(s)
The predictive patterns for 30 day readmissions may be weak or not particularly well captured in the dataset, limiting the model’s ability to achieve meaningful precision and recall for clinical use. Additionally, the complexity of preprocessing clinical data and engineering useful features may require more time and computational resources than anticipated, possibly constraining model development and evaluation.


# Ethics Statement(s)
1. We aim to reduce preventable hospital readmissions and improve patient outcomes by enabling targeted, timely clinical interventions.
2. Patients, clinicians, and healthcare systems could benefit through improved care, better resource allocation, and reduced costs.
3. Underrepresented populations may face less accurate predictions, while some patients may be unnecessarily treated or overlooked due to model errors or bias.
4. The world will be better if identifying high-risk patients actually leads clinicians to take effective actions that reduce readmissions and if the model is used appropriately without reinforcing existing biases.
5. 

# Tentative Timeline
Sprint 1
- Feasibility discussion, data exploration, model decision-making discussion

Sprint 2 (tentative check-in)
- Data formatting pipeline, data exploration and influence into decision-making

Sprint 3
- Model prototype

Sprint 4
- Model adjustment

 Sprint 5 (tentative final presentations)
 - Final write-up
