"""Investigation agents.

Three LLM agents (news, announcements, decision) and two deterministic
agents (market_context, technicals). The investigator calls each agent
in turn and feeds the outputs into the decision agent which produces
the final TradeThesis.
"""
