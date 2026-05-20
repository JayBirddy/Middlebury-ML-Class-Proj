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
by the metrics above are: "Is the model actually catching the patients who will be readmitted?" and "Are the flagged 
patients worth the clinical team's time?" respectively and they are in tension will each other. 

# Motivation and Question
We have public health data for which predictive models would help us give better decisions and resource management tasks.

In the current field of medicine, there are many exterior factors that can play into a patient's discharge. While there isn't much we can do about a patient's cooperation, a hospital's funding, or insurance-related factors, what we can do is try to implement a system that promotes precautionary and preventive measure for many patients. 

# Planned Deliverables
- Python package containing code for algorithms and analysis as well as documentation.
- One Jupter notebook for illustrative, walkthrough style purposes
- An essay discussing ethic implications of our project. May include an ethics matrix and moral/legal discussions

Full success:
Above stated deliverables as well a a trained model that produces clinically meaningful sensitivity and precision on held-out test data. Model performance would be disaggregated across demographic subgroups with no significant disparity.

Partial success:
Some deliverables are completed and a working model would be built and evaluated even if performance falls short of clinical thresholds. It would include honest documentation of dataset limitations and modeling constraints, forming a foundation for future work.

# Resources Required
- Cloud GPU computing power
- Dataset: Data will be based on Diabetes 130-US Hospitals for Years 1999-2008 as the default. Other datasets will be considered as an additive element in parallel with the default dataset if found feasible and in alignment with the project's
- Link to dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
intent of readmission prediction.
- Libraries/Packages: scikit-learn, pandas, numpy, matplotlib, seaborn
- Literature: Access to prior work on readmission prediction and fairness-aware ML in healthcare via Google Scholar or JSTOR


# What We Will Learn
- Evan: I hope to deepen my understanding of machine learning for healthcare applications, especially how to develop and evaluate predictive models using real clinical data, and to do so effectively by making sure that the work is done in a collaborative setting with good communication.
- Jonathan: I would like to have a firsthand experience dealing with data and personal biases. In recent months, I have taken a look at many models that are responsible for making decisions of the life of another human. This being in the same category as those offers an exclusive oppotunity to reflect on addressing patterns with my team.
- 

# Risk Statement(s)
The predictive patterns for 30 day readmissions may be weak or not particularly well captured in the dataset, limiting the model’s ability to achieve meaningful precision and recall for clinical use. Additionally, the complexity of preprocessing clinical data and engineering useful features may require more time and computational resources than anticipated, possibly constraining model development and evaluation.


# Ethics Statement(s)
1. We aim to reduce preventable hospital readmissions and improve patient outcomes by enabling targeted, timely clinical interventions.
2. Patients, clinicians, and healthcare systems could benefit through improved care, better resource allocation, and reduced costs.
3. Underrepresented populations may face less accurate predictions, while some patients may be unnecessarily treated or overlooked due to model errors or bias.
4. The world will be better if identifying high-risk patients actually leads clinicians to take effective actions that reduce readmissions and if the model is used appropriately without reinforcing existing biases.

# Tentative Timeline
Sprint 1
- Feasibility discussion, exploratory analysis (class balance, missing values, feature distributions), model decision-making discussion

Sprint 2 (tentative check-in)
- Data formatting pipeline, data exploration and influence into decision-making, simple logistic regression baseline

Sprint 3
- Model prototype 

Sprint 4
- Model adjustment, threshold calibration explored for sensitivity/precision tradeoff

 Sprint 5 (tentative final presentations)
 - Final write-up, notebook polished, presentation prepared.
