"""Evaluation corpus: the bundled Northwind sample documents plus extra policy documents.

All content is fictional. Each question lists the documents (by filename) that
contain the answer, and key phrases a correct answer must contain. Questions
with ``answerable=False`` have no answer in the corpus and must be refused.
"""

from __future__ import annotations

EXTRA_DOCS: dict[str, str] = {
    "05-procurement.md": """# Northwind Traders — Procurement Policy (sample)

## Vendor onboarding
New vendors must pass a security questionnaire and a sanctions screening before the first purchase order.

## Purchase orders
Purchase orders above 10,000 EUR require three competitive quotes. Single-source purchases need a written justification.

## Payment terms
The standard payment term for suppliers is 45 days from invoice date.
""",
    "06-data-retention.md": """# Northwind Traders — Data Retention Schedule (sample)

## Customer records
Customer contracts are retained for 10 years after the contract ends.

## Application logs
Application logs are kept for 90 days and security logs for 1 year.

## Recruitment data
Unsuccessful candidates' CVs are deleted after 6 months unless the candidate consents to longer storage.
""",
    "07-onboarding-it.md": """# Northwind Traders — IT Onboarding Guide (sample)

## Accounts
IT creates email and VPN accounts within one business day of the HR start notification.

## Software requests
Additional software is requested through the service desk; standard approval time is two business days.

## Security training
All new employees complete security awareness training within their first 14 days.
""",
    "08-customer-support.md": """# Northwind Traders — Customer Support SLA (sample)

## Response times
Priority 1 incidents receive a first response within 30 minutes, 24 hours a day.
Priority 2 incidents receive a first response within 4 business hours.

## Escalation
Unresolved Priority 1 incidents are escalated to the duty manager after 2 hours.

## Refunds
Refunds up to 200 EUR can be approved by a support agent; larger refunds need a team lead.
""",
}

QUESTIONS: list[dict[str, object]] = [
    {"q": "How many days of paid annual leave do full-time employees get?", "docs": ["01-employee-handbook.md"], "phrases": ["26 days"]},
    {"q": "How many unused leave days can be carried over?", "docs": ["01-employee-handbook.md"], "phrases": ["5"]},
    {"q": "When must carried-over leave be used?", "docs": ["01-employee-handbook.md"], "phrases": ["31 March"]},
    {"q": "What are the core working hours?", "docs": ["01-employee-handbook.md"], "phrases": ["10:00", "15:00"]},
    {"q": "How many sick days per year are paid?", "docs": ["01-employee-handbook.md"], "phrases": ["12 days"]},
    {"q": "When is a medical certificate needed?", "docs": ["01-employee-handbook.md"], "phrases": ["three consecutive"]},
    {"q": "How many days a week can staff work remotely?", "docs": ["01-employee-handbook.md"], "phrases": ["three days"]},
    {"q": "How quickly must lost equipment be reported to IT?", "docs": ["01-employee-handbook.md"], "phrases": ["24 hours"]},
    {"q": "Who approves expenses above 5,000 EUR?", "docs": ["02-expense-policy.md"], "phrases": ["CFO"]},
    {"q": "Who approves an expense of 2,000 EUR?", "docs": ["02-expense-policy.md"], "phrases": ["department head"]},
    {"q": "What is the hotel limit per night in capital cities?", "docs": ["02-expense-policy.md"], "phrases": ["180"]},
    {"q": "What is the daily meal allowance when travelling?", "docs": ["02-expense-policy.md"], "phrases": ["55"]},
    {"q": "Is alcohol reimbursed?", "docs": ["02-expense-policy.md"], "phrases": ["not reimbursed"]},
    {"q": "Within how many days must expense claims be submitted?", "docs": ["02-expense-policy.md"], "phrases": ["30 days"]},
    {"q": "What class of flight is required for flights under six hours?", "docs": ["02-expense-policy.md"], "phrases": ["Economy"]},
    {"q": "What is the minimum password length?", "docs": ["03-it-security.md"], "phrases": ["14 characters"]},
    {"q": "For which systems is multi-factor authentication mandatory?", "docs": ["03-it-security.md"], "phrases": ["VPN"]},
    {"q": "Where can restricted data like salaries be stored?", "docs": ["03-it-security.md"], "phrases": ["approved systems"]},
    {"q": "How fast must suspected phishing be reported?", "docs": ["03-it-security.md"], "phrases": ["one hour"]},
    {"q": "How many quotes are needed for purchase orders above 10,000 EUR?", "docs": ["05-procurement.md"], "phrases": ["three"]},
    {"q": "What is the standard supplier payment term?", "docs": ["05-procurement.md"], "phrases": ["45 days"]},
    {"q": "What checks must new vendors pass?", "docs": ["05-procurement.md"], "phrases": ["sanctions"]},
    {"q": "How long are customer contracts retained?", "docs": ["06-data-retention.md"], "phrases": ["10 years"]},
    {"q": "How long are application logs kept?", "docs": ["06-data-retention.md"], "phrases": ["90 days"]},
    {"q": "When are unsuccessful candidates' CVs deleted?", "docs": ["06-data-retention.md"], "phrases": ["6 months"]},
    {"q": "How soon does IT create email and VPN accounts?", "docs": ["07-onboarding-it.md"], "phrases": ["one business day"]},
    {"q": "When must new employees complete security awareness training?", "docs": ["07-onboarding-it.md"], "phrases": ["14 days"]},
    {"q": "What is the first response time for Priority 1 incidents?", "docs": ["08-customer-support.md"], "phrases": ["30 minutes"]},
    {"q": "When are unresolved Priority 1 incidents escalated?", "docs": ["08-customer-support.md"], "phrases": ["2 hours"]},
    {"q": "Up to what amount can a support agent approve refunds?", "docs": ["08-customer-support.md"], "phrases": ["200"]},
    {"q": "What is the company's parental leave policy?", "docs": [], "phrases": [], "answerable": False},
    {"q": "Who is the CEO of Northwind Traders?", "docs": [], "phrases": [], "answerable": False},
    {"q": "What is the bonus percentage for senior engineers?", "docs": [], "phrases": [], "answerable": False},
    {"q": "How do I configure the office coffee machine?", "docs": [], "phrases": [], "answerable": False},
    {"q": "What is the stock ticker symbol of the company?", "docs": [], "phrases": [], "answerable": False},
    {"q": "Which cloud provider hosts the ERP system?", "docs": [], "phrases": [], "answerable": False},
]
