# nurz aggregator — free-model-aggregator-02

One Python 3.13 server (standard library only) that serves the OpenAI Chat
Completions API and the Anthropic Messages API, in JSON and in streaming,
and answers every prompt with a real completion from one of the fourteen
free OpenRouter models, dialled through the grader's gateway.

## Run

```sh
python3 main.py --list   # print the catalog document and exit 0
python3 main.py          # serve, register with the env, exit 0 when done
```

The trial sets `MAKEFUN_ENV_URL`, `MAKEFUN_PROMPT_FILE`,
`MAKEFUN_TRIAL_SEED`, `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL`,
`OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. The server listens on
`0.0.0.0` at a free port, registers it with the env, serves
`GET /v1/models` in the wire its auth header names, `POST
/v1/chat/completions` and `POST /v1/messages`, and polls `GET
$MAKEFUN_ENV_URL/work` until it says done.

## Routing (nurz strategy: rating-first with penalised failover)

- Candidates that meet a prompt's requirements (tools / vision / JSON
  mode) are ordered by the brief's `rating_milli`, highest first.
- A model that refuses (4xx/5xx) takes a penalty: −150 after one refusal,
  −650 after two refusals in a row, so it drops down the ranking; after
  three refusals in the trial it is spent and is not dialled again. The
  penalised model is tried again on a later request, in its penalised
  place.
- Refusals a retry or a failover recovers from are reported in the
  `x-makefun-upstream-failures` header, in the order the calls refused.

Every decision is a function of the brief, the requests and the gateway's
answers only: no wall clock, no unseeded randomness, so a replay of the
trial leaves the same answers.

## Files

- `main.py` — the whole server: brief, catalog, router, both wires, SSE.
- `LICENSE` — MIT.
- `.makefun/owner.json` — the claim file naming the owner's wallet.
