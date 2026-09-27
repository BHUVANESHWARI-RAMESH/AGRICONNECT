# Module 14 Complete Application Test Report

Test scenarios use deterministic backend component doubles to cover success and failure branches.
The separate workflow integration test uses real MCP weather tools and the local RAG vector database.

| Scenario | Result | Details |
|---|---|---|
| Banana + cyclone warning | PASS | all requested invariants verified |
| Cotton + spraying + tomorrow | PASS | all requested invariants verified |
| Extreme weather without a verified alert | PASS | all requested invariants verified |
| Groundnut + harvesting | PASS | all requested invariants verified |
| Invalid location | PASS | all requested invariants verified |
| Missing crop | PASS | all requested invariants verified |
| Missing location | PASS | all requested invariants verified |
| Missing pesticide dosage information | PASS | all requested invariants verified |
| Paddy + Thanjavur + irrigation | PASS | all requested invariants verified |
| RAG retrieval failure | PASS | all requested invariants verified |
| Tavily failure | PASS | all requested invariants verified |
| Weather API failure | PASS | all requested invariants verified |
