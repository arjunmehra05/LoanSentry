import numpy as np

FEATURE_LABELS = {
    "annual_inc":             "Annual Income",
    "loan_amnt":              "Loan Amount",
    "int_rate":               "Interest Rate",
    "dti":                    "Debt-to-Income Ratio",
    "FICO_AVG":               "FICO Score",
    "emp_length":             "Employment Length",
    "revol_util":             "Revolving Utilization",
    "revol_bal":              "Revolving Balance",
    "installment":            "Monthly Installment",
    "LOAN_TO_INCOME":         "Loan-to-Income Ratio",
    "INSTALLMENT_TO_INCOME":  "Installment-to-Income Ratio",
    "delinq_2yrs":            "Delinquencies (2yr)",
    "pub_rec":                "Public Records",
    "inq_last_6mths":         "Credit Inquiries (6mo)",
    "open_acc":               "Open Accounts",
    "total_acc":              "Total Accounts",
    "mort_acc":               "Mortgage Accounts",
    "pub_rec_bankruptcies":   "Bankruptcies",
    "grade":                  "Loan Grade",
    "sub_grade":              "Loan Sub-Grade",
    "term":                   "Loan Term",
}

PURPOSE_LABELS = {
    0: "Car", 1: "Credit Card", 2: "Debt Consolidation",
    3: "Home Improvement", 4: "Major Purchase", 5: "Medical",
    6: "Moving", 7: "Small Business", 8: "Vacation", 9: "Other"
}

GRADE_LABELS = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G"}


def generate_explanation(groq_api_key, input_dict, prediction_results,
                          shap_values, feature_names, retrieved_docs):

    risk_category = prediction_results["risk_category"]
    prob          = prediction_results["prob_ensemble"]
    confidence    = prediction_results["confidence"]

    # ── Top SHAP features ─────────────────────────────────────────────────
    shap_vals   = shap_values[0] if len(np.array(shap_values).shape) > 1 \
                  else shap_values
    top_indices = np.abs(shap_vals).argsort()[-5:][::-1]
    top_features = [
        (FEATURE_LABELS.get(feature_names[i], feature_names[i]),
         float(shap_vals[i]))
        for i in top_indices
    ]

    # ── Applicant values ──────────────────────────────────────────────────
    annual_inc = input_dict.get("annual_inc", 0)
    loan_amnt  = input_dict.get("loan_amnt", 0)
    int_rate   = input_dict.get("int_rate", 0)
    dti        = input_dict.get("dti", 0)
    fico       = input_dict.get("FICO_AVG", 0)
    emp_length = input_dict.get("emp_length", 0)
    revol_util = input_dict.get("revol_util", 0)
    delinq     = input_dict.get("delinq_2yrs", 0)
    purpose    = PURPOSE_LABELS.get(input_dict.get("purpose", 9), "Other")
    grade      = GRADE_LABELS.get(input_dict.get("grade", 6), "G")
    lti        = round(loan_amnt / (annual_inc + 1), 2)

    # ── Confidence note ───────────────────────────────────────────────────
    if confidence < 30:
        confidence_note = "⚠️ This is a borderline case - manual review is recommended."
    elif confidence < 60:
        confidence_note = "The model has moderate confidence in this assessment."
    else:
        confidence_note = "The model has high confidence in this assessment."

    # ── Retrieved context ─────────────────────────────────────────────────
    context = "\n\n".join([
        f"[{doc['source']}]\n{doc['text'][:300]}"
        for doc in retrieved_docs
    ])

    # ── SHAP summary for prompt ───────────────────────────────────────────
    shap_lines = "\n".join([
        f"- {feat}: {'increases' if val > 0 else 'reduces'} risk "
        f"(SHAP: {val:+.4f})"
        for feat, val in top_features
    ])

    # ── Try Groq, fall back to rule-based ─────────────────────────────────
    if groq_api_key:
        try:
            from groq import Groq
            client = Groq(api_key=groq_api_key)

            prompt = f"""You are a senior credit risk analyst. Analyze this loan application and provide a clear, professional assessment.

APPLICANT PROFILE:
- Annual Income: ${annual_inc:,.0f}
- Loan Amount: ${loan_amnt:,.0f}
- Interest Rate: {int_rate:.1f}%
- DTI Ratio: {dti:.1f}%
- FICO Score: {fico:.0f}
- Employment Years: {emp_length:.0f}
- Revolving Utilization: {revol_util:.1f}%
- Delinquencies (2yr): {delinq}
- Loan Purpose: {purpose}
- Loan Grade: {grade}
- Loan-to-Income Ratio: {lti:.2f}x

MODEL PREDICTION:
- Risk Category: {risk_category}
- Default Probability: {prob:.1%}
- Confidence: {confidence:.1f}%

TOP FACTORS (SHAP):
{shap_lines}

RELEVANT POLICY CONTEXT:
{context}

Provide your response in exactly this format using markdown:

### Risk Assessment Summary
[2-3 sentences summarizing the overall risk with specific numbers]

---
### Top Risk Factors
1. **[Factor]** - [1 sentence explanation with specific value]
2. **[Factor]** - [1 sentence explanation with specific value]
3. **[Factor]** - [1 sentence explanation with specific value]

---
### Recommendation
[One of: ✅ Approve / 🔶 Approve with Conditions / ⛔ Decline] - [2 sentences with specific reasoning]

---
### Improvement Suggestion
[1-2 sentences with a specific, actionable suggestion including numbers]

Keep it concise, professional, and data-driven."""

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior credit risk analyst. "
                            "Be concise, specific, and always reference "
                            "actual numbers from the applicant profile."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.2
            )

            llm_text = response.choices[0].message.content
            return f"{llm_text}\n\n*{confidence_note}*"

        except Exception:
            pass

    # ── Rule-based fallback ───────────────────────────────────────────────
    return _rule_based_explanation(
        risk_category, prob, confidence, confidence_note,
        top_features, input_dict, fico, dti, emp_length,
        annual_inc, loan_amnt, lti
    )


def _rule_based_explanation(risk_category, prob, confidence, confidence_note,
                             top_features, input_dict, fico, dti, emp_length,
                             annual_inc, loan_amnt, lti):

    risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk_category, "⚪")
    summary    = _build_summary(risk_category, prob, fico, dti,
                                emp_length, annual_inc, loan_amnt, lti)
    reasons    = _build_reasons(top_features)
    rec        = _build_recommendation(risk_category, prob, fico, dti, confidence)
    suggestion = _build_suggestion(top_features, input_dict)

    lines = []
    lines.append(f"### {risk_emoji} Risk Assessment Summary")
    lines.append(summary)
    lines.append(f"\n*{confidence_note}*")
    lines.append("\n---")
    lines.append("### 📊 Top Risk Factors")
    for i, (feat, val) in enumerate(reasons, 1):
        direction = "increases" if val > 0 else "reduces"
        lines.append(f"**{i}. {feat}** - {direction} default risk "
                     f"({'positive' if val > 0 else 'negative'} impact)")
    lines.append("\n---")
    lines.append("### 📋 Recommendation")
    lines.append(rec)
    lines.append("\n---")
    lines.append("### 💡 Improvement Suggestion")
    lines.append(suggestion)

    return "\n\n".join(lines)


def _build_summary(risk_category, prob, fico, dti, emp_length,
                   annual_inc, loan_amnt, lti):
    if risk_category == "Low":
        return (
            f"This applicant presents a **low default risk** with a "
            f"{prob:.1%} probability of default. "
            f"The profile is supported by a FICO score of {fico:.0f}, "
            f"a DTI of {dti:.1f}%, and {emp_length:.0f} years of employment. "
            f"The loan-to-income ratio of {lti:.2f}x is within acceptable limits."
        )
    elif risk_category == "Medium":
        return (
            f"This applicant presents a **moderate default risk** with a "
            f"{prob:.1%} probability of default. "
            f"The profile shows mixed signals - a FICO score of {fico:.0f} "
            f"and DTI of {dti:.1f}% suggest some financial stress. "
            f"Careful review of income stability and debt obligations is advised."
        )
    else:
        return (
            f"This applicant presents a **high default risk** with a "
            f"{prob:.1%} probability of default. "
            f"Key concerns include a FICO score of {fico:.0f}, "
            f"a DTI of {dti:.1f}%, and only {emp_length:.0f} years of employment. "
            f"The loan-to-income ratio of {lti:.2f}x further elevates risk."
        )


def _build_reasons(top_features):
    risk_features = [(f, v) for f, v in top_features if v > 0]
    if len(risk_features) < 3:
        risk_features = top_features[:3]
    return risk_features[:3]


def _build_recommendation(risk_category, prob, fico, dti, confidence):
    if risk_category == "Low":
        return (
            "✅ **Approve** - The applicant meets standard lending criteria. "
            "Proceed with standard documentation and verification."
        )
    elif risk_category == "Medium":
        conditions = []
        if dti > 36:
            conditions.append("proof of stable income")
        if fico < 680:
            conditions.append("a co-signer or collateral")
        if not conditions:
            conditions.append("enhanced income verification")
        cond_str = " and ".join(conditions)
        return (
            f"🔶 **Approve with Conditions** - Consider approving subject to "
            f"{cond_str}. Monitor account closely for the first 12 months."
        )
    else:
        return (
            "⛔ **Decline** - The applicant's risk profile exceeds acceptable "
            "thresholds. Default probability is significantly elevated. "
            "Recommend reapplication after improving credit score and reducing debt."
        )


def _build_suggestion(top_features, input_dict):
    fico  = input_dict.get("FICO_AVG", 0)
    dti   = input_dict.get("dti", 0)
    revol = input_dict.get("revol_util", 0)
    emp   = input_dict.get("emp_length", 0)

    for feat, val in top_features:
        if val > 0:
            if "FICO" in feat or "Credit" in feat:
                return (
                    f"💳 Improving the FICO score from {fico:.0f} to above 700 "
                    f"by reducing credit utilization and making on-time payments "
                    f"could significantly lower default risk."
                )
            if "DTI" in feat or "Debt" in feat:
                return (
                    f"📉 Reducing the DTI ratio from {dti:.1f}% to below 36% "
                    f"by paying down existing debt before applying would "
                    f"substantially improve the risk profile."
                )
            if "Utilization" in feat or "revol" in feat.lower():
                return (
                    f"💰 Reducing revolving credit utilization from "
                    f"{revol:.1f}% to below 30% would positively impact "
                    f"the credit score and lower default risk."
                )
            if "Employment" in feat:
                return (
                    f"💼 Maintaining current employment for at least "
                    f"{max(0, 3 - int(emp))} more years would demonstrate "
                    f"income stability and reduce lender risk."
                )
            if "Delinquenc" in feat:
                return (
                    "📅 Avoiding any further delinquencies and maintaining "
                    "a clean payment record for 24 months would significantly "
                    "rebuild creditworthiness."
                )

    return (
        "📊 Focus on reducing overall debt burden and maintaining "
        "consistent on-time payments to improve the risk profile over time."
    )