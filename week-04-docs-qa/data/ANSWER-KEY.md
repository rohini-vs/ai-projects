# Answer Key: Northwind Handbook (northwind-handbook.pdf)

Do not ingest this file. Use it to check retrieval. Page numbers are the printed page numbers (PDF page = printed page).

Retrieval hit = a retrieved chunk contains the evidence text (any one evidence line is enough for single-hop; all for multi-hop or conflicts).

## Q1. How many days of sick leave do I get, and when do I need a medical certificate?
- **Answer:** 12 days per year. A certificate is needed when sick leave covers 3 or more consecutive working days.
- **Trap:** Near-duplicate sections (sick vs casual vs annual leave)
- **Evidence:**
  - page 5: "12 days of paid sick leave"
  - page 5: "3 or more consecutive working days"

## Q2. How do I get my money back for a cab I took for work late at night?
- **Answer:** Cab fares for work after 21:00 IST are reimbursable at the actual fare with a receipt, claimed in ExpenseDesk.
- **Trap:** Vocabulary mismatch (money back vs reimbursement)
- **Evidence:**
  - page 7: "cab fares for work done after 21:00 IST"
  - page 7: "Local cab"

## Q3. What is the deadline to submit an expense claim?
- **Answer:** 45 days. Section 4.3 says 30 days, but Notice 2026-03 (effective 1 July 2026) extends it to 45 and notices win.
- **Trap:** Conflicting chunks (policy vs later notice)
- **Evidence:**
  - page 7: "within 30 days of the date of the expense"
  - page 18: "extended from 30 days to 45 days"

## Q4. Who has to approve an expense claim of INR 60,000?
- **Answer:** VP Finance (range INR 50,001 to 200,000), from the Appendix B matrix. Section 4.4 only points to Appendix B.
- **Trap:** Multi-hop (answer is not in the section that is about expenses)
- **Evidence:**
  - page 7: "listed in Appendix B"
  - page 19: "INR 50,001 to INR 200,000"

## Q5. What does ERR-5503 mean and what should I do?
- **Answer:** Downstream collector unavailable. Retry after 30 seconds; page on-call if it persists for more than 10 minutes.
- **Trap:** Exact-term lookup (keyword search should beat embeddings here)
- **Evidence:**
  - page 12: "Downstream collector unavailable"

## Q6. How quickly must a SEV1 incident be acknowledged, and how often are updates posted?
- **Answer:** Acknowledge within 5 minutes; update every 30 minutes.
- **Trap:** Table content
- **Evidence:**
  - page 10: "Complete outage, data loss or security breach"

## Q7. How much is the standby allowance for the primary on-call engineer?
- **Answer:** INR 3,500 per week. Section 7.2 says 3,000, but Notice 2026-05 (from 1 August 2026) raises it.
- **Trap:** Conflicting chunks (policy vs later notice)
- **Evidence:**
  - page 10: "standby allowance of INR 3,000 per week"
  - page 18: "from INR 3,000 to INR 3,500"

## Q8. Can I deploy to production on a Friday?
- **Answer:** No. Deployments are Monday to Thursday, 10:00 to 16:00 IST.
- **Trap:** Negation / short answer
- **Evidence:**
  - page 11: "There are no Friday deployments"

## Q9. How long do I have to report a lost laptop?
- **Answer:** Report within 2 hours of discovering the loss; file a police complaint within 24 hours.
- **Trap:** Two numbers in one sentence
- **Evidence:**
  - page 8: "within 2 hours of discovering the loss"

## Q10. How many days can I work from another country?
- **Answer:** 10 working days per calendar year, with written approval from Security.
- **Trap:** Near-duplicate sections (other city allows 20 days)
- **Evidence:**
  - page 4: "limited to 10 working days in a calendar year"

## Q11. How much unused annual leave can I carry forward?
- **Answer:** Up to 10 days. Anything above lapses on 31 March.
- **Trap:** Near-duplicate sections (sick and casual do not carry forward)
- **Evidence:**
  - page 5: "Up to 10 unused days can be carried forward"

## Q12. What is the notice period for an L5 engineer?
- **Answer:** 90 days (60 days for L1 to L4; 30 days during probation).
- **Trap:** Number depends on level
- **Evidence:**
  - page 17: "90 days for L5 and above"

## Q13. When are Learning Fridays?
- **Answer:** Second and fourth Friday of each month, 14:00 to 17:30 IST (Notice 2026-07). Section 11.4 still says second Friday only.
- **Trap:** Conflicting chunks (policy vs later notice)
- **Evidence:**
  - page 14: "second Friday afternoon of every month"
  - page 18: "second and fourth Friday"

## Q14. How many approvals does a pull request touching payments need?
- **Answer:** 2 approvals, one of them from the Security Champions group.
- **Trap:** Condition inside a general rule
- **Evidence:**
  - page 11: "need 2 approvals"

## Q15. Above what value must I declare a gift from a customer?
- **Answer:** INR 5,000, declared to People Operations within 7 days.
- **Trap:** Number plus deadline
- **Evidence:**
  - page 16: "worth more than INR 5,000"

## Q16. What is the stock option vesting schedule?
- **Answer:** Not in the handbook. Equity is described in individual offer letters. A good system should say it does not know.
- **Trap:** Unanswerable (tests the similarity threshold and 'I don't know')
- **Evidence:**
  - page 15: "individual offer letters"
