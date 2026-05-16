"""Single-stock day-trading tool.

Standalone module that day-trades one configurable major stock (default
AAPL) via slightly-ITM weekly options, holding 1-3 days. Runs as its own
process alongside the main 7-strategy AlgoTrader bot, with its own IBKR
client_id, lockfile, and state file.

Architecture: investigator coordinates 5 agents (3 LLM via Claude, 2
deterministic Python) to produce a TradeThesis each morning. If
conviction clears the threshold, a slightly-ITM weekly option is opened
and managed for up to max_hold_days with bracket exits, intraday news
re-checks, and a mandatory expiry-day close.
"""
