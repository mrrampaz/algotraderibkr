"""Economic and earnings calendar with impact scores.

Tracks FOMC, CPI, PPI, earnings dates and their expected market impact.
Used by the regime detector to mark event days and by strategies to
adjust position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum

import pytz
import structlog

logger = structlog.get_logger()


class EventType(str, Enum):
    FOMC = "fomc"
    CPI = "cpi"
    PPI = "ppi"
    JOBS = "jobs"               # Non-farm payrolls
    GDP = "gdp"
    RETAIL_SALES = "retail_sales"
    PCE = "pce"                 # Personal consumption expenditures
    EARNINGS = "earnings"
    EX_DIVIDEND = "ex_dividend"
    OPTIONS_EXPIRY = "opex"     # Options expiration
    QUAD_WITCHING = "quad_witching"


# Impact score: 1 = low, 2 = moderate, 3 = high
EVENT_IMPACT: dict[EventType, int] = {
    EventType.FOMC: 3,
    EventType.CPI: 3,
    EventType.PPI: 2,
    EventType.JOBS: 3,
    EventType.GDP: 2,
    EventType.RETAIL_SALES: 2,
    EventType.PCE: 2,
    EventType.EARNINGS: 2,
    EventType.EX_DIVIDEND: 1,
    EventType.OPTIONS_EXPIRY: 2,
    EventType.QUAD_WITCHING: 3,
}

# Minimal high-impact seed list until a dynamic earnings feed is wired in.
KNOWN_EVENTS: tuple[dict, ...] = (
    {
        "event_type": EventType.EARNINGS,
        "date": date(2026, 2, 25),
        "time": "16:05",
        "description": "NVDA earnings (after close)",
        "impact": 3,
        "symbol": "NVDA",
        "metadata": {"timing": "after_close", "scope": "market_moving"},
    },
)


@dataclass
class CalendarEvent:
    """A single calendar event."""
    event_type: EventType
    date: date
    time: str = ""          # e.g. "08:30" ET, "14:00" ET
    description: str = ""
    impact: int = 1         # 1-3
    symbol: str = ""        # For earnings/ex-div
    metadata: dict = field(default_factory=dict)


class EventCalendar:
    """Economic and earnings event calendar.

    Maintains a list of known upcoming events. Events are populated from:
    1. Hardcoded FOMC/CPI schedule (published yearly by the Fed/BLS)
    2. Web scraper results for earnings
    3. Manual additions

    Provides queries for:
    - Is today an event day?
    - What events are coming up?
    - What's the highest impact event today?
    """

    def __init__(self) -> None:
        self._events: list[CalendarEvent] = []
        self._log = logger.bind(component="event_calendar")

        # Seed with known 2025/2026 FOMC dates
        self._seed_fomc_dates()
        # Seed monthly options expiry (3rd Friday)
        self._seed_opex_dates()
        # Seed known high-impact events not covered by static macro schedules.
        self._seed_known_events()

    def add_event(self, event: CalendarEvent) -> None:
        """Add an event to the calendar."""
        self._events.append(event)

    def add_events(self, events: list[CalendarEvent]) -> None:
        """Add multiple events."""
        self._events.extend(events)

    def add_earnings(self, symbol: str, earnings_date: date, time: str = "BMO") -> None:
        """Add an earnings event for a specific symbol."""
        self._events.append(CalendarEvent(
            event_type=EventType.EARNINGS,
            date=earnings_date,
            time=time,
            description=f"{symbol} earnings",
            impact=EVENT_IMPACT[EventType.EARNINGS],
            symbol=symbol,
        ))

    def get_events_for_date(self, target_date: date | None = None) -> list[CalendarEvent]:
        """Get all events for a specific date."""
        if target_date is None:
            target_date = datetime.now(pytz.timezone("America/New_York")).date()
        return [e for e in self._events if e.date == target_date]

    def get_upcoming_events(self, days: int = 7) -> list[CalendarEvent]:
        """Get events in the next N days, sorted by date."""
        today = datetime.now(pytz.timezone("America/New_York")).date()
        end = today + timedelta(days=days)
        upcoming = [e for e in self._events if today <= e.date <= end]
        upcoming.sort(key=lambda e: e.date)
        return upcoming

    def is_event_day(self, target_date: date | None = None) -> bool:
        """Check if a date has any high-impact events (impact >= 2)."""
        events = self.get_events_for_date(target_date)
        return any(e.impact >= 2 for e in events)

    def is_fomc_day(self, target_date: date | None = None) -> bool:
        """Check if a date is an FOMC announcement day."""
        events = self.get_events_for_date(target_date)
        return any(e.event_type == EventType.FOMC for e in events)

    def max_impact_today(self, target_date: date | None = None) -> int:
        """Get the maximum impact score for today's events. 0 if no events."""
        events = self.get_events_for_date(target_date)
        if not events:
            return 0
        return max(e.impact for e in events)

    def get_earnings_today(self, target_date: date | None = None) -> list[CalendarEvent]:
        """Get earnings events for today."""
        events = self.get_events_for_date(target_date)
        return [e for e in events if e.event_type == EventType.EARNINGS]

    def has_earnings(self, symbol: str, target_date: date | None = None) -> bool:
        """Check if a symbol reports earnings on a given date."""
        earnings = self.get_earnings_today(target_date)
        return any(e.symbol == symbol for e in earnings)

    def _seed_fomc_dates(self) -> None:
        """Seed known FOMC meeting dates.

        These are the announcement dates (typically Wednesday at 2:00 PM ET).
        Source: Federal Reserve website.
        """
        # 2025 FOMC announcement dates
        fomc_2025 = [
            date(2025, 1, 29),
            date(2025, 3, 19),
            date(2025, 5, 7),
            date(2025, 6, 18),
            date(2025, 7, 30),
            date(2025, 9, 17),
            date(2025, 10, 29),
            date(2025, 12, 17),
        ]
        # 2026 FOMC — dates typically announced; using estimates
        fomc_2026 = [
            date(2026, 1, 28),
            date(2026, 3, 18),
            date(2026, 5, 6),
            date(2026, 6, 17),
            date(2026, 7, 29),
            date(2026, 9, 16),
            date(2026, 10, 28),
            date(2026, 12, 16),
        ]

        for d in fomc_2025 + fomc_2026:
            self._events.append(CalendarEvent(
                event_type=EventType.FOMC,
                date=d,
                time="14:00",
                description="FOMC rate decision",
                impact=3,
            ))

    def _seed_opex_dates(self) -> None:
        """Seed monthly options expiration dates (3rd Friday of each month)."""
        for year in (2025, 2026):
            for month in range(1, 13):
                opex = self._third_friday(year, month)
                is_quad = month in (3, 6, 9, 12)
                self._events.append(CalendarEvent(
                    event_type=EventType.QUAD_WITCHING if is_quad else EventType.OPTIONS_EXPIRY,
                    date=opex,
                    description="Quad witching" if is_quad else "Monthly OPEX",
                    impact=3 if is_quad else 2,
                ))

    @staticmethod
    def _third_friday(year: int, month: int) -> date:
        """Calculate the third Friday of a given month."""
        # First day of the month
        first = date(year, month, 1)
        # Days until Friday (weekday 4)
        days_to_friday = (4 - first.weekday()) % 7
        first_friday = first + timedelta(days=days_to_friday)
        # Third Friday = first Friday + 14 days
        return first_friday + timedelta(days=14)

    def summary(self) -> str:
        """Return a text summary of upcoming events."""
        upcoming = self.get_upcoming_events(7)
        if not upcoming:
            return "No upcoming events in the next 7 days."

        lines = ["Upcoming events (next 7 days):"]
        for e in upcoming:
            impact_label = {1: "LOW", 2: "MED", 3: "HIGH"}.get(e.impact, "?")
            symbol_str = f" [{e.symbol}]" if e.symbol else ""
            time_str = f" {e.time}" if e.time else ""
            lines.append(f"  {e.date} {time_str} [{impact_label}] {e.description}{symbol_str}")

        return "\n".join(lines)

    def _seed_known_events(self) -> None:
        """Seed manually curated events used as fallback when APIs are unavailable."""
        for raw in KNOWN_EVENTS:
            self._events.append(
                CalendarEvent(
                    event_type=raw["event_type"],
                    date=raw["date"],
                    time=raw.get("time", ""),
                    description=raw.get("description", ""),
                    impact=int(raw.get("impact", EVENT_IMPACT.get(raw["event_type"], 2))),
                    symbol=raw.get("symbol", ""),
                    metadata=dict(raw.get("metadata", {})),
                )
            )
