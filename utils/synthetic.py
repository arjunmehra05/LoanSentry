import random
import numpy as np

GRADE_MAP = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"
}

PURPOSE_DISPLAY = {
    0: "Car",          1: "Credit Card",       2: "Debt Consolidation",
    3: "Home Improvement", 4: "Major Purchase", 5: "Medical",
    6: "Moving",       7: "Small Business",    8: "Vacation",
    9: "Other"
}

HOME_MAP = {0: "RENT", 1: "OWN", 2: "MORTGAGE", 3: "OTHER"}

GRADE_OPTIONS = ["A", "B", "C", "D", "E", "F", "G"]
PURPOSE_OPTIONS = [
    "Car", "Credit Card", "Debt Consolidation", "Home Improvement",
    "Major Purchase", "Medical", "Moving", "Small Business",
    "Vacation", "Other"
]


def _ri(lo, hi):
    """Random int as plain Python int."""
    return int(random.randint(lo, hi - 1))

def _rf(lo, hi):
    """Random float rounded to 2dp."""
    return round(random.uniform(lo, hi), 2)


def generate_synthetic_profile(risk_level="random"):
    if risk_level == "low":
        annual_inc = _ri(70000, 200000)
        fico       = _ri(720, 850)
        dti        = _rf(5, 25)
        emp_length = _ri(3, 15)
        loan_amnt  = _ri(5000, 25000)
        int_rate   = _rf(5, 12)
        grade      = _ri(0, 2)
        delinq     = 0
        pub_rec    = 0
        revol_util = _rf(5, 25)
        purpose    = _ri(0, 6)

    elif risk_level == "high":
        annual_inc = _ri(15000, 40000)
        fico       = _ri(300, 620)
        dti        = _rf(40, 70)
        emp_length = _ri(0, 2)
        loan_amnt  = _ri(10000, 35000)
        int_rate   = _rf(18, 30)
        grade      = _ri(4, 7)
        delinq     = _ri(2, 8)
        pub_rec    = _ri(1, 3)
        revol_util = _rf(65, 99)
        purpose    = _ri(0, 10)

    else:  # random
        annual_inc = _ri(30000, 120000)
        fico       = _ri(580, 780)
        dti        = _rf(10, 50)
        emp_length = _ri(0, 12)
        loan_amnt  = _ri(5000, 40000)
        int_rate   = _rf(6, 25)
        grade      = _ri(0, 7)
        delinq     = _ri(0, 5)
        pub_rec    = _ri(0, 3)
        revol_util = _rf(10, 90)
        purpose    = _ri(0, 10)

    # Derived - all plain Python types
    funded_amnt           = loan_amnt
    installment           = round(
        (loan_amnt * (int_rate / 1200)) /
        (1 - (1 + int_rate / 1200) ** -36), 2)
    revol_bal             = _ri(1000, 50000)
    open_acc              = _ri(2, 20)
    total_acc             = _ri(open_acc, open_acc + 20)
    inq_last_6mths        = _ri(0, 6)
    mort_acc              = _ri(0, 5)
    pub_rec_bk            = min(pub_rec, _ri(0, 2))
    term                  = random.choice([36, 60])   # plain int
    home_own              = _ri(0, 4)
    sub_grade             = grade * 5 + _ri(0, 5)
    loan_to_income        = round(loan_amnt / (annual_inc + 1), 4)
    installment_to_income = round(installment / (annual_inc / 12 + 1), 4)

    return {
        "loan_amnt":             loan_amnt,
        "funded_amnt":           funded_amnt,
        "term":                  term,
        "int_rate":              int_rate,
        "installment":           installment,
        "grade":                 grade,
        "sub_grade":             sub_grade,
        "emp_length":            emp_length,
        "home_ownership":        home_own,
        "annual_inc":            annual_inc,
        "verification_status":   1,
        "purpose":               purpose,
        "dti":                   dti,
        "delinq_2yrs":           delinq,
        "FICO_AVG":              fico,
        "inq_last_6mths":        inq_last_6mths,
        "open_acc":              open_acc,
        "pub_rec":               pub_rec,
        "revol_bal":             revol_bal,
        "revol_util":            revol_util,
        "total_acc":             total_acc,
        "initial_list_status":   0,
        "application_type":      0,
        "mort_acc":              mort_acc,
        "pub_rec_bankruptcies":  pub_rec_bk,
        "LOAN_TO_INCOME":        loan_to_income,
        "INSTALLMENT_TO_INCOME": installment_to_income,
        "FUNDED_RATIO":          1.0,
        "_grade_label":          GRADE_OPTIONS[grade],
        "_purpose_label":        PURPOSE_OPTIONS[purpose],
        "_home_label":           HOME_MAP.get(home_own, "RENT"),
    }