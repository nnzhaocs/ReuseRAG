# Static candidate-union coverage and quality for every window family

`cov` below is the union of candidates over all servable queries. It is a serving-side static quantity / conservative inline-work price, **not an identity for first-arrival C_serve**. Replay C_serve and repairs are diagnostic comparisons.

| pair | workload | gap | K' | G S2 / cov S2 | G S3 / cov S3 | G S4 / cov S4 | G S5 / cov S5 | static winner | replay winner |
|---|---|---|---|---|---|---|---|---|---|
| P5q | uniform | +0.0264 | 25 | +0.0084 / 0.149 | +0.0685 / 0.140 | +0.0034 / 0.840 | +0.0026 / 0.848 | S4 |  |
| P5q | uniform | +0.0264 | 50 | +0.0027 / 0.264 | +0.0429 / 0.242 | +0.0027 / 0.854 | +0.0003 / 0.868 | S2 | S2/demand_then_rand |
| P5q | uniform | +0.0264 | 100 | +0.0027 / 0.431 | +0.0251 / 0.390 | +0.0032 / 0.877 | +0.0009 / 0.897 | S2 |  |
