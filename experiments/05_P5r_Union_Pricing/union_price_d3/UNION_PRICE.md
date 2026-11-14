# Static candidate-union coverage and quality for every window family

`cov` below is the union of candidates over all servable queries. It is a serving-side static quantity / conservative inline-work price, **not an identity for first-arrival C_serve**. Replay C_serve and repairs are diagnostic comparisons.

| pair | workload | gap | K' | G S2 / cov S2 | G S3 / cov S3 | G S4 / cov S4 | G S5 / cov S5 | static winner | replay winner |
|---|---|---|---|---|---|---|---|---|---|
| P13 | uniform | +0.0144 | 25 | -0.0018 / 0.148 | +0.3095 / 0.099 | -0.0006 / 0.841 | +0.0007 / 0.851 | S2 | S2/demand_then_rand |
| P13 | uniform | +0.0144 | 50 | +0.0008 / 0.261 | +0.2375 / 0.164 | -0.0009 / 0.855 | -0.0001 / 0.869 | S2 |  |
