# Regulatory landscape

Checked 2026-07-26. Not legal advice; verify with counsel before any external
claim. Sources at the bottom.

Two different regimes make the two halves of this product mandatory-adjacent,
and they are not the same regime.

## The validity half — EU AI Act

Emotion recognition is named in Annex III as a **high-risk** AI system, one of
three biometric routes alongside remote biometric identification and biometric
categorisation. The first wedge of this product — inferring emotional valence
from EEG — sits squarely inside that classification.

High-risk systems must satisfy Articles 9–15. Article 15 requires an
appropriate level of accuracy and robustness, sustained across the lifecycle,
and then this:

> The levels of accuracy and the relevant accuracy metrics of high-risk AI
> systems shall be declared in the accompanying instructions of use.

Once a metric must be **declared**, the protocol that produced it becomes a
legal question, because trial-random 0.84 and subject-held-out 0.44 are not
the same declaration.

Article 15 also directs the Commission to work with metrology and benchmarking
bodies to develop benchmarks and measurement methodologies for accuracy and
robustness. That is the "standard layer" in the spec's V5 roadmap, written into
statute, and the seat is empty.

### Timing, which is currently in flux

Annex III high-risk obligations were set to apply **2 August 2026**. On 7 May
2026 the Council, Parliament and Commission provisionally agreed a Digital
Omnibus deferring use-based Annex III obligations to **2 December 2027**.

The deferral only takes effect on formal adoption and publication in the
Official Journal, expected before 2 August 2026. Until then 2 August 2026
remains the operative date.

Either way the deadline is fixed and public. The deferral costs near-term
urgency and buys 16 months to accumulate design partners and benchmark data.

## The privacy half — US state neural data law

**California SB 1223** added neural data to CCPA sensitive personal
information, effective 1 January 2025. The de-identification exemption turns on
**capability, not intent**: data is exempt when it "cannot reasonably identify,
relate to, describe, be capable of being associated with, or be linked,
directly or indirectly, to a particular consumer."

And, directly relevant:

> Businesses may attempt to reidentify the information solely for the purpose
> of determining whether its deidentification processes satisfy the
> requirements.

The statute explicitly permits re-identification attempts for the purpose of
verifying de-identification. That is the subject-identity-inference attack in
spec 6.2, with a legal carve-out written for it.

**Colorado HB 24-1058** (signed 17 April 2024, effective 7 August 2024) was the
first US law targeting neural data. Its trigger is weaker for us: it applies
where biological data "is used or intended to be used ... for identification
purposes" — an intent standard a company can decline to meet.

**Montana SB 163** (effective 1 October 2025) adds neurotechnology data to its
Genetic Information Privacy Act. **Connecticut** also classifies neural data as
sensitive. At least six further states have bills in progress.

## Limits — do not overclaim these

1. **No law requires buying a third party.** Every obligation here can be met
   in-house. Regulation creates duties, not demand for us; the step from duty
   to purchase still has to be earned.
2. **The EU AI Act binds providers and deployers placing systems on the EU
   market.** A US research startup may fall outside it entirely. Know what
   fraction of target customers actually sell into the EU.
3. **"May attempt to reidentify" is a permission, not a requirement.** It
   removes a legal obstacle to running the test; it does not oblige anyone.
4. **The US state neural-data laws protect the data, not the algorithm.** They
   do not pull on the validity half at all.

## Sources

- Annex III — https://artificialintelligenceact.eu/annex/3/
- Article 15 — https://www.euaiact.com/article/15
- Emotion recognition under the AI Act (William Fry) — https://www.williamfry.com/knowledge/the-time-to-ai-act-is-now-a-practical-guide-to-emotion-recognition-systems-under-the-ai-act/
- Digital Omnibus deferral (Gibson Dunn) — https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/
- Implementation timeline — https://artificialintelligenceact.eu/implementation-timeline/
- California SB 1223 text — https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB1223
- Neural data in US state privacy laws (FPF) — https://fpf.org/blog/the-neural-data-goldilocks-problem-defining-neural-data-in-u-s-state-privacy-laws/
- Neural data privacy regulation (Arnold & Porter) — https://www.arnoldporter.com/en/perspectives/advisories/2025/07/neural-data-privacy-regulation
