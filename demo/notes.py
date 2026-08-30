"""Fully fabricated discharge notes for the public demo index.

None of this describes a real person. Every name, subject_id, hadm_id, date,
vital, lab value, and clinical detail below was invented for this project,
structured to match the section-header format MIMIC-IV-Note uses (see
config.MIMIC_SECTIONS) so the existing chunking/embedding/retrieval pipeline
treats it identically to a real note -- without ever touching real MIMIC-IV
data. Safe to commit and deploy publicly.

Each note deliberately covers the categories the app's own prompt-starters
and evaluation questions target (diagnosis, medications, condition,
disposition, follow-up, allergies, vitals, labs, past medical history, chief
complaint) across a spread of common, unrelated conditions, so a demo user
asking any of the standard questions gets a real, grounded answer instead of
a refusal.
"""

DEMO_NOTES = [
    {
        "subject_id": 90000001,
        "hadm_id": 29000001,
        "text": """
Chief Complaint:
Shortness of breath, productive cough, and subjective fever for three days, prompting evaluation in the emergency department for possible pneumonia.

History of Present Illness:
Patient is a 68 year old with a history of type 2 diabetes and hypertension who presented with three days of worsening shortness of breath, productive cough with yellow sputum, and subjective fevers at home. In the emergency department the patient was found to be hypoxic to 88% on room air and was started on supplemental oxygen. Chest x-ray showed a right lower lobe infiltrate consistent with pneumonia.

Past Medical History:
Type 2 diabetes mellitus, hypertension, hyperlipidemia.

Social History:
Former smoker, quit 10 years ago. Denies alcohol or illicit drug use. Lives alone, independent with activities of daily living prior to admission.

Family History:
Mother with type 2 diabetes. Father with coronary artery disease.

Allergies:
Documented allergy to penicillin, which has previously caused a skin rash; the patient avoids all penicillin-class antibiotics as a result.

Physical Exam:
Vitals: Temp 100.9 F, BP 132/78, HR 96, RR 22, O2 sat 91% on 2L nasal cannula. Lungs with crackles at the right base. Heart regular rate and rhythm. Abdomen soft, non-tender.

Pertinent Results:
WBC 14.2, hemoglobin 12.8, sodium 138, creatinine 1.0, glucose 210. Blood cultures pending at discharge, no growth to date. Chest x-ray with right lower lobe consolidation.

Brief Hospital Course:
Patient was admitted with community acquired pneumonia and started on intravenous ceftriaxone and azithromycin. Oxygen requirement improved over 48 hours and the patient was transitioned to room air by hospital day 3. Blood glucose was managed with sliding scale insulin during the admission. The patient completed a course of intravenous antibiotics and was transitioned to oral levofloxacin to complete a 7 day total course.

Medications on Admission:
Metformin 1000 mg PO BID, lisinopril 10 mg PO daily, atorvastatin 20 mg PO daily.

Discharge Medications:
1. Levofloxacin 750 mg PO daily for 3 more days
2. Metformin 1000 mg PO BID
3. Lisinopril 10 mg PO daily
4. Atorvastatin 20 mg PO daily
5. Acetaminophen 650 mg PO every 6 hours as needed for fever

Discharge Disposition:
Disposition: Home, with the patient discharged to their own residence in stable condition and independent with activities of daily living.

Discharge Diagnosis:
Primary diagnosis at discharge: Community acquired pneumonia, right lower lobe. Type 2 diabetes mellitus, hyperglycemia managed with sliding scale insulin during the admission.

Discharge Condition:
Condition at discharge: Stable and afebrile at the time of discharge, ambulating without difficulty, with oxygen saturation of 96% while breathing room air.

Discharge Instructions:
You were admitted with pneumonia and treated with antibiotics. Please complete the full course of levofloxacin as prescribed. Continue your home diabetes and blood pressure medications. Return to the emergency department if you develop worsening shortness of breath, chest pain, or high fever.

Followup Instructions:
Recommended follow-up care after discharge: Please follow up with your primary care physician within 1 week of discharge for a recheck and repeat chest x-ray in 6 weeks.
""".strip(),
    },
    {
        "subject_id": 90000002,
        "hadm_id": 29000002,
        "text": """
Chief Complaint:
Shortness of breath and increasing swelling in both legs over the past week, associated with difficulty lying flat at night.

History of Present Illness:
This is a 74 year old with a history of congestive heart failure and atrial fibrillation who presented with progressive dyspnea on exertion, orthopnea, and bilateral lower extremity swelling over the past week. Patient reports missing several doses of furosemide due to running out of medication. No chest pain, no fevers.

Past Medical History:
Congestive heart failure with reduced ejection fraction, atrial fibrillation on anticoagulation, chronic kidney disease stage 3.

Social History:
Non-smoker. Occasional alcohol use. Lives with spouse.

Family History:
Non-contributory.

Allergies:
No known drug allergies.

Physical Exam:
Vitals: Temp 98.2 F, BP 148/90, HR 102 irregularly irregular, RR 20, O2 sat 93% on room air. Jugular venous distension present. Bilateral lower extremity pitting edema to the knees. Lungs with bibasilar crackles.

Pertinent Results:
BNP 1850, creatinine 1.6 (baseline 1.4), sodium 134, potassium 4.2. Chest x-ray with pulmonary vascular congestion and small bilateral pleural effusions.

Brief Hospital Course:
Patient was admitted for acute decompensated heart failure in the setting of medication non-adherence. Diuresis was achieved with intravenous furosemide with improvement in respiratory status and weight decreased by 4 kg over the admission. Rate control for atrial fibrillation was continued with metoprolol. Anticoagulation with apixaban was continued throughout admission.

Medications on Admission:
Furosemide 40 mg PO daily, metoprolol succinate 50 mg PO daily, apixaban 5 mg PO BID.

Discharge Medications:
1. Furosemide 60 mg PO daily
2. Metoprolol succinate 50 mg PO daily
3. Apixaban 5 mg PO BID
4. Potassium chloride 20 mEq PO daily

Discharge Disposition:
Disposition: Home with home health services arranged for medication management assistance and monitoring of daily weights during the immediate post-discharge period.

Discharge Diagnosis:
Primary diagnosis at discharge: Acute decompensated congestive heart failure with volume overload related to medication non-adherence. Atrial fibrillation with associated rate control issues.

Discharge Condition:
Condition at discharge: Improved and euvolemic at the time of discharge, ambulating independently without any supplemental oxygen requirement and breathing comfortably at rest.

Discharge Instructions:
You were admitted for a heart failure exacerbation related to missed medication doses. It is very important to take your furosemide every day as prescribed. Weigh yourself daily and call your doctor if you gain more than 3 pounds in a day. Follow a low sodium diet.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with cardiology in 2 weeks. Follow up with primary care within 1 week for repeat weight check and basic metabolic panel.
""".strip(),
    },
    {
        "subject_id": 90000003,
        "hadm_id": 29000003,
        "text": """
Chief Complaint:
Fall at home resulting in severe right hip pain and inability to bear weight, prompting emergency department evaluation and imaging.

History of Present Illness:
An 81 year old with osteoporosis presented after a mechanical fall at home, landing on the right hip. Unable to bear weight afterward. X-ray in the emergency department confirmed a right femoral neck fracture. Patient was taken to the operating room for repair.

Past Medical History:
Osteoporosis, hypertension, mild cognitive impairment.

Social History:
Lives alone, uses a cane at baseline. No tobacco or alcohol use.

Family History:
Non-contributory.

Allergies:
Documented allergy to sulfa drugs, which have previously caused a hives reaction; the patient avoids all sulfonamide-containing medications going forward as a result.

Physical Exam:
Vitals: Temp 98.6 F, BP 128/76, HR 84, RR 16, O2 sat 97% on room air. Right lower extremity shortened and externally rotated on presentation, tender to palpation over the right hip.

Pertinent Results:
Hemoglobin 10.9 postoperatively, creatinine 0.9, sodium 140. Postoperative x-ray shows appropriate hardware position.

Brief Hospital Course:
Patient underwent right hip hemiarthroplasty without complication. Postoperative course was notable for mild anemia managed conservatively without transfusion. Physical therapy was initiated on postoperative day 1 and the patient progressed to ambulating with a walker and moderate assistance.

Medications on Admission:
Amlodipine 5 mg PO daily, calcium carbonate with vitamin D PO daily.

Discharge Medications:
1. Amlodipine 5 mg PO daily
2. Calcium carbonate with vitamin D PO daily
3. Acetaminophen 650 mg PO every 6 hours as needed for pain
4. Oxycodone 5 mg PO every 6 hours as needed for breakthrough pain
5. Enoxaparin 40 mg subcutaneous daily for 2 weeks for blood clot prevention

Discharge Disposition:
Disposition: Skilled nursing facility for continued physical therapy and rehabilitation, with a goal of returning home once mobility further improves safely.

Discharge Diagnosis:
Primary diagnosis at discharge: Right femoral neck fracture sustained in a mechanical fall, treated surgically with a right hip hemiarthroplasty without intraoperative complication.

Discharge Condition:
Condition at discharge: Stable and medically cleared for rehabilitation, ambulating short distances with a walker and moderate assistance, with pain well controlled on oral medication.

Discharge Instructions:
You had surgery to repair a broken hip. Please continue physical therapy at the rehabilitation facility. Take the blood thinner injection as prescribed to prevent blood clots. Call your surgeon if you notice increased redness, drainage, or fever.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with orthopedic surgery in 2 weeks for wound check and repeat x-ray.
""".strip(),
    },
    {
        "subject_id": 90000004,
        "hadm_id": 29000004,
        "text": """
Chief Complaint:
Burning with urination for two days along with new onset confusion, prompting transfer from her assisted living facility for evaluation.

History of Present Illness:
An 84 year old with a history of recurrent urinary tract infections presented from her assisted living facility with new onset confusion and reported burning with urination for two days. Urinalysis was consistent with a urinary tract infection.

Past Medical History:
Recurrent urinary tract infections, mild dementia, osteoarthritis.

Social History:
Resides in assisted living facility. No tobacco or alcohol use.

Family History:
Non-contributory.

Allergies:
No known drug allergies.

Physical Exam:
Vitals: Temp 100.4 F, BP 118/70, HR 92, RR 18, O2 sat 97% on room air. Abdomen with mild suprapubic tenderness. Patient oriented to person only, at baseline oriented to person and place per family.

Pertinent Results:
Urinalysis with positive leukocyte esterase and nitrites, WBC 12.0, creatinine 0.8. Urine culture pending at discharge, growing greater than 100,000 colony forming units of E. coli at time of discharge with sensitivities pending.

Brief Hospital Course:
Patient was treated with intravenous ceftriaxone for a urinary tract infection with associated delirium. Mental status returned to baseline by hospital day 2 with treatment and reorientation measures. Transitioned to oral antibiotics to complete a 7 day course.

Medications on Admission:
Donepezil 10 mg PO daily, acetaminophen 650 mg PO as needed.

Discharge Medications:
1. Cephalexin 500 mg PO four times daily for 4 more days
2. Donepezil 10 mg PO daily
3. Acetaminophen 650 mg PO as needed for pain

Discharge Disposition:
Disposition: Assisted living facility, the same residence the patient was admitted from, with staff informed of the diagnosis and treatment plan.

Discharge Diagnosis:
Primary diagnosis at discharge: Urinary tract infection with associated delirium that resolved with antibiotic treatment and supportive reorientation measures during the hospital stay.

Discharge Condition:
Condition at discharge: Stable at the time of discharge, mental status returned fully to baseline per family report, and afebrile for over 24 hours.

Discharge Instructions:
You were treated for a urinary tract infection that caused temporary confusion. Please complete the full course of antibiotics. Drink plenty of fluids. Contact your doctor if confusion returns or if you develop fever.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with primary care physician within 1 week.
""".strip(),
    },
    {
        "subject_id": 90000005,
        "hadm_id": 29000005,
        "text": """
Chief Complaint:
Wheezing and worsening shortness of breath for three days along with increased sputum production, concerning for a COPD exacerbation.

History of Present Illness:
A 59 year old with a long history of chronic obstructive pulmonary disease presented with three days of increased dyspnea, wheezing, and increased sputum production, consistent with a COPD exacerbation, likely triggered by a viral upper respiratory infection.

Past Medical History:
Chronic obstructive pulmonary disease, hypertension.

Social History:
Current smoker, one pack per day for 30 years. Counseled on smoking cessation during admission.

Family History:
Non-contributory.

Allergies:
Documented allergy to codeine, which has previously caused significant nausea and vomiting; the patient avoids codeine-containing pain medications going forward as a result.

Physical Exam:
Vitals: Temp 99.1 F, BP 138/86, HR 100, RR 24, O2 sat 89% on room air improving to 94% on 2L nasal cannula. Diffuse expiratory wheezing bilaterally.

Pertinent Results:
Arterial blood gas with mild respiratory acidosis, WBC 10.5, chest x-ray without acute infiltrate, hyperinflated lungs consistent with COPD.

Brief Hospital Course:
Patient was treated with nebulized albuterol and ipratropium, a course of oral prednisone, and supplemental oxygen for an acute COPD exacerbation. Symptoms improved over the admission and oxygen requirement resolved by hospital day 3. Smoking cessation counseling was provided and the patient was started on a nicotine patch.

Medications on Admission:
Tiotropium inhaler daily, albuterol inhaler as needed, lisinopril 20 mg PO daily.

Discharge Medications:
1. Prednisone 40 mg PO daily, tapering over 5 days
2. Tiotropium inhaler daily
3. Albuterol inhaler every 4 hours as needed
4. Lisinopril 20 mg PO daily
5. Nicotine patch daily

Discharge Disposition:
Disposition: Home, in stable respiratory condition, with close outpatient follow-up arranged given the recent exacerbation and ongoing tobacco use.

Discharge Diagnosis:
Primary diagnosis at discharge: Acute exacerbation of chronic obstructive pulmonary disease, likely triggered by a viral upper respiratory infection based on the clinical course.

Discharge Condition:
Condition at discharge: Stable at the time of discharge, oxygen saturation 95% while breathing room air, and breathing comfortably without accessory muscle use.

Discharge Instructions:
You were treated for a COPD flare. Please finish the prednisone taper as prescribed and continue your inhalers. We strongly encourage you to quit smoking; the nicotine patch can help. Return to the emergency department for worsening shortness of breath not relieved by your inhaler.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with pulmonology within 2 weeks. Follow up with primary care within 1 week.
""".strip(),
    },
    {
        "subject_id": 90000006,
        "hadm_id": 29000006,
        "text": """
Chief Complaint:
Redness, warmth, and swelling of the left lower leg for three days following a minor scrape, concerning for skin infection.

History of Present Illness:
A 52 year old with type 2 diabetes presented with three days of progressive redness, warmth, and swelling of the left lower leg after a minor scrape while gardening, concerning for cellulitis.

Past Medical History:
Type 2 diabetes mellitus, obesity.

Social History:
Non-smoker, occasional alcohol use. Works as a landscaper.

Family History:
Non-contributory.

Allergies:
No known drug allergies.

Physical Exam:
Vitals: Temp 100.2 F, BP 134/82, HR 88, RR 16, O2 sat 98% on room air. Left lower leg with erythema, warmth, and mild swelling extending from the ankle to below the knee, with a small superficial abrasion. No fluctuance or drainage.

Pertinent Results:
WBC 11.8, glucose 188, hemoglobin A1c 8.2. Blood cultures negative.

Brief Hospital Course:
Patient was started on intravenous cefazolin for cellulitis of the left lower extremity with improvement in erythema and swelling over 48 hours. Blood glucose was managed with sliding scale insulin during admission and diabetes education was reinforced. Transitioned to oral antibiotics for a total 10 day course.

Medications on Admission:
Metformin 1000 mg PO BID, glipizide 5 mg PO daily.

Discharge Medications:
1. Cephalexin 500 mg PO four times daily for 6 more days
2. Metformin 1000 mg PO BID
3. Glipizide 5 mg PO daily

Discharge Disposition:
Disposition: Home, in stable condition, with instructions to keep the affected leg elevated and clean while completing the antibiotic course.

Discharge Diagnosis:
Primary diagnosis at discharge: Cellulitis of the left lower extremity, likely related to poorly controlled type 2 diabetes mellitus and a minor skin break.

Discharge Condition:
Condition at discharge: Stable at discharge, with the erythema and swelling clearly improving compared to admission, and afebrile for the past 24 hours.

Discharge Instructions:
You were treated for a skin infection of your leg. Please complete the full course of antibiotics. Keep the area clean and elevated. Continue your diabetes medications and monitor your blood sugar closely. Return if redness spreads or you develop fever.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with primary care in 1 week to reassess the leg and review diabetes management.
""".strip(),
    },
    {
        "subject_id": 90000007,
        "hadm_id": 29000007,
        "text": """
Chief Complaint:
Black, tarry stools for two days along with one episode of lightheadedness, concerning for an upper gastrointestinal bleed.

History of Present Illness:
A 66 year old with a history of osteoarthritis on chronic NSAID use presented with two days of black, tarry stools and one episode of lightheadedness, concerning for an upper gastrointestinal bleed. Hemoglobin on presentation was decreased from baseline.

Past Medical History:
Osteoarthritis, hypertension, peptic ulcer disease in the past.

Social History:
Non-smoker, no alcohol use.

Family History:
Non-contributory.

Allergies:
Documented allergy to aspirin, which has previously caused gastrointestinal upset; the patient avoids aspirin and aspirin-containing products as a result.

Physical Exam:
Vitals: Temp 98.4 F, BP 108/64, HR 108, RR 18, O2 sat 97% on room air. Abdomen mildly tender in the epigastrium, no rebound or guarding. Rectal exam with melena.

Pertinent Results:
Hemoglobin 8.1 on admission (baseline 13.5), platelets 240, INR 1.0. Endoscopy revealed a bleeding gastric ulcer, which was treated endoscopically with clip placement.

Brief Hospital Course:
Patient received two units of packed red blood cells for symptomatic anemia from an upper gastrointestinal bleed due to NSAID-associated gastric ulcer. Endoscopy achieved hemostasis. Started on intravenous then oral proton pump inhibitor therapy. NSAIDs were discontinued and the patient was counseled to avoid them going forward. Hemoglobin remained stable after transfusion.

Medications on Admission:
Ibuprofen 600 mg PO three times daily as needed, lisinopril 10 mg PO daily.

Discharge Medications:
1. Pantoprazole 40 mg PO twice daily
2. Lisinopril 10 mg PO daily
3. Acetaminophen 650 mg PO as needed for pain, in place of ibuprofen

Discharge Disposition:
Disposition: Home, in stable condition, with close outpatient follow-up arranged for repeat blood counts and a scheduled surveillance endoscopy.

Discharge Diagnosis:
Primary diagnosis at discharge: Upper gastrointestinal bleed secondary to an NSAID-induced gastric ulcer, successfully treated endoscopically with clip placement and transfusion support.

Discharge Condition:
Condition at discharge: Stable at the time of discharge, hemoglobin stable at 10.2 after transfusion, with no further episodes of melena reported.

Discharge Instructions:
You had bleeding from a stomach ulcer likely caused by long-term ibuprofen use. Please avoid all NSAIDs including ibuprofen and naproxen going forward; use acetaminophen for pain instead. Take the stomach acid medication as prescribed. Return immediately if you have more black stools, vomiting blood, or dizziness.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with gastroenterology in 4 to 6 weeks for repeat endoscopy. Follow up with primary care in 1 week.
""".strip(),
    },
    {
        "subject_id": 90000008,
        "hadm_id": 29000008,
        "text": """
Chief Complaint:
Acute wheezing and chest tightness after exposure to a cat, with associated shortness of breath concerning for an asthma flare.

History of Present Illness:
A 29 year old with a history of asthma since childhood presented with acute wheezing, chest tightness, and shortness of breath after exposure to a cat, consistent with an asthma exacerbation triggered by an allergen.

Past Medical History:
Asthma, seasonal allergic rhinitis.

Social History:
Non-smoker, works as a teacher.

Family History:
Mother with asthma.

Allergies:
Documented environmental allergies to cats and dust mites, both of which reliably trigger wheezing and worsening asthma symptoms on any exposure.

Physical Exam:
Vitals: Temp 98.0 F, BP 122/74, HR 108, RR 26, O2 sat 92% on room air improving to 97% after treatment. Diffuse bilateral wheezing on exam, use of accessory muscles noted on presentation.

Pertinent Results:
Peak flow 45% of personal best on admission, improved to 85% prior to discharge. Chest x-ray without acute process.

Brief Hospital Course:
Patient was treated with continuous nebulized albuterol, ipratropium, and a short course of oral corticosteroids for an acute asthma exacerbation. Symptoms and peak flow improved substantially over 24 hours. Inhaler technique was reviewed and an allergen avoidance plan was discussed.

Medications on Admission:
Albuterol inhaler as needed, fluticasone inhaler daily.

Discharge Medications:
1. Prednisone 40 mg PO daily for 5 days
2. Albuterol inhaler every 4 hours as needed
3. Fluticasone inhaler twice daily

Discharge Disposition:
Disposition: Home, in stable respiratory condition, with a reviewed action plan for future exacerbations and known allergen avoidance strategies.

Discharge Diagnosis:
Primary diagnosis at discharge: Acute asthma exacerbation triggered by exposure to a household cat, responsive to bronchodilator therapy and a short oral steroid course.

Discharge Condition:
Condition at discharge: Stable and breathing comfortably at the time of discharge, oxygen saturation 98% on room air, with peak flow improved to baseline.

Discharge Instructions:
You had an asthma flare triggered by a cat allergen. Please finish the short steroid course and use your controller inhaler daily as prescribed, not just when symptomatic. Avoid known triggers where possible. Return if you have severe shortness of breath not relieved by your rescue inhaler.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with allergy and pulmonology within 2 weeks.
""".strip(),
    },
    {
        "subject_id": 90000009,
        "hadm_id": 29000009,
        "text": """
Chief Complaint:
New onset palpitations and lightheadedness that began suddenly at home, prompting emergency department evaluation and cardiac monitoring.

History of Present Illness:
A 71 year old with a history of hypertension presented with new onset palpitations and lightheadedness, found to be in atrial fibrillation with rapid ventricular response on EKG in the emergency department.

Past Medical History:
Hypertension, hyperlipidemia.

Social History:
Non-smoker, moderate alcohol use, denies illicit drug use.

Family History:
Father with atrial fibrillation.

Allergies:
No known drug allergies.

Physical Exam:
Vitals: Temp 98.3 F, BP 142/88, HR 138 irregularly irregular on admission, improving to 78 after treatment, RR 18, O2 sat 98% on room air. Irregularly irregular rhythm on cardiac exam.

Pertinent Results:
TSH within normal limits, potassium 4.0, magnesium 2.0. Echocardiogram showed normal ejection fraction of 60% with no significant valvular disease.

Brief Hospital Course:
Patient was started on intravenous metoprolol for rate control of new onset atrial fibrillation with rapid ventricular response, with good response and heart rate controlled in the 70s to 80s by hospital day 1. Anticoagulation with apixaban was initiated given elevated stroke risk score. Cardiology was consulted and outpatient follow-up was arranged.

Medications on Admission:
Amlodipine 5 mg PO daily, atorvastatin 40 mg PO daily.

Discharge Medications:
1. Metoprolol succinate 50 mg PO daily
2. Apixaban 5 mg PO BID
3. Amlodipine 5 mg PO daily
4. Atorvastatin 40 mg PO daily

Discharge Disposition:
Disposition: Home, in stable condition, with close cardiology follow-up arranged to discuss long-term rhythm and anticoagulation management strategy.

Discharge Diagnosis:
Primary diagnosis at discharge: New onset atrial fibrillation with rapid ventricular response, successfully rate controlled with intravenous then oral metoprolol during the admission.

Discharge Condition:
Condition at discharge: Stable at discharge, heart rate well controlled in the 70s to 80s, and asymptomatic without further palpitations or lightheadedness.

Discharge Instructions:
You were found to have an irregular heart rhythm called atrial fibrillation. It is important to take your blood thinner every day as prescribed to reduce stroke risk, and not to skip doses. Monitor your pulse and report if it feels very fast or irregular again.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with cardiology within 1 to 2 weeks for further evaluation, possible rhythm control strategy discussion.
""".strip(),
    },
    {
        "subject_id": 90000010,
        "hadm_id": 29000010,
        "text": """
Chief Complaint:
Two weeks of increased thirst, frequent urination, and fatigue, found to have markedly elevated blood glucose concerning for new diabetes.

History of Present Illness:
A 45 year old with newly diagnosed type 2 diabetes presented with two weeks of polyuria, polydipsia, blurry vision, and fatigue, found to have significantly elevated blood glucose on presentation without evidence of diabetic ketoacidosis.

Past Medical History:
Obesity, newly diagnosed type 2 diabetes mellitus.

Social History:
Non-smoker, sedentary lifestyle, works a desk job.

Family History:
Both parents with type 2 diabetes.

Allergies:
No known drug allergies.

Physical Exam:
Vitals: Temp 98.6 F, BP 138/84, HR 82, RR 16, O2 sat 99% on room air. No acute distress. Skin warm and dry, no acanthosis nigricans noted.

Pertinent Results:
Glucose 412 on admission, hemoglobin A1c 11.2, venous blood gas with normal pH, no ketones present, sodium 136, creatinine 0.9.

Brief Hospital Course:
Patient was admitted for management of severely uncontrolled type 2 diabetes without evidence of diabetic ketoacidosis. Blood glucose was controlled with insulin during admission and the patient was transitioned to a basal-bolus insulin regimen with a plan to transition toward oral agents as an outpatient. Diabetes education, including glucose monitoring and dietary counseling, was provided by the diabetes education team prior to discharge.

Medications on Admission:
None.

Discharge Medications:
1. Insulin glargine 20 units subcutaneous nightly
2. Insulin lispro sliding scale with meals
3. Metformin 500 mg PO daily, to increase as tolerated

Discharge Disposition:
Disposition: Home, in stable condition, with a structured outpatient diabetes education and insulin titration plan established prior to discharge.

Discharge Diagnosis:
Primary diagnosis at discharge: Newly diagnosed type 2 diabetes mellitus, severely uncontrolled on presentation, without evidence of diabetic ketoacidosis during the admission.

Discharge Condition:
Condition at discharge: Stable at the time of discharge, blood glucose improved into the 140s range, and tolerating a regular oral diet without difficulty.

Discharge Instructions:
You were diagnosed with type 2 diabetes and started on insulin along with metformin. Please check your blood sugar as instructed and keep a log to bring to your appointments. Follow the dietary guidance provided by the diabetes educator. Contact your doctor if your blood sugar is repeatedly above 300 or below 70.

Followup Instructions:
Recommended follow-up care after discharge: Follow up with primary care within 1 week and with endocrinology within 2 to 3 weeks to adjust your diabetes regimen.
""".strip(),
    },
]
