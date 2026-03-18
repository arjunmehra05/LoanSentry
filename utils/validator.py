def validate_input(input_dict):
    errors = []

    if input_dict.get("annual_inc", 0) <= 0:
        errors.append("Annual income must be greater than 0")

    if input_dict.get("annual_inc", 0) > 10000000:
        errors.append("Annual income seems unrealistically high")

    if input_dict.get("loan_amnt", 0) <= 0:
        errors.append("Loan amount must be greater than 0")

    if input_dict.get("loan_amnt", 0) > 40000:
        errors.append("Loan amount exceeds maximum limit of $40,000")

    if input_dict.get("dti", 0) < 0:
        errors.append("DTI cannot be negative")

    if input_dict.get("dti", 0) > 100:
        errors.append("DTI cannot exceed 100%")

    if input_dict.get("int_rate", 0) <= 0:
        errors.append("Interest rate must be greater than 0")

    if input_dict.get("int_rate", 0) > 40:
        errors.append("Interest rate seems unrealistically high")

    if input_dict.get("FICO_AVG", 0) < 300 or input_dict.get("FICO_AVG", 0) > 850:
        errors.append("FICO score must be between 300 and 850")

    if input_dict.get("emp_length", 0) < 0:
        errors.append("Employment years cannot be negative")

    loan_amnt  = input_dict.get("loan_amnt", 0)
    annual_inc = input_dict.get("annual_inc", 1)
    if loan_amnt > annual_inc * 3:
        errors.append("Loan amount is more than 3x annual income, please verify")

    return errors