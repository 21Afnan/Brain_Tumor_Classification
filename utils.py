def get_user_test_data():
    print("Enter the following details for prediction:")

    gender_input = input("Gender (M/F): ").strip().lower()
    gender = 0 if gender_input == 'm' else 1  # Male = 0, Female = 1

    age = float(input("Age at Diagnosis: "))

    idh1 = int(input("IDH1 Mutation? (1 for Yes, 0 for No): "))
    tp53 = int(input("TP53 Mutation? (1 for Yes, 0 for No): "))
    atrx = int(input("ATRX Mutation? (1 for Yes, 0 for No): "))
    pten = int(input("PTEN Mutation? (1 for Yes, 0 for No): "))
    egfr = int(input("EGFR Mutation? (1 for Yes, 0 for No): "))
    cic = int(input("CIC Mutation? (1 for Yes, 0 for No): "))
    pik3ca = int(input("PIK3CA Mutation? (1 for Yes, 0 for No): "))

    return [gender, age, idh1, tp53, atrx, pten, egfr, cic, pik3ca]

def get_precautions_from_gemini(tumor_type):
    precaution_db = {
        "meningioma": "Avoid radiation exposure and get regular check-ups.",
        "pituitary": "Monitor hormonal levels and follow medication strictly.",
        "no_tumor": "Stay healthy and get annual MRI scans if symptoms appear."
    }
    return precaution_db.get(tumor_type, "No specific precautions found.")
