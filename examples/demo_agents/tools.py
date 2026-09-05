"""Deterministic, offline fake tools shared by the demo agent scenarios.

These are plain functions over a small in-memory dataset. No network, no
external services, no randomness, no wall-clock dependence -- calling any of
them with the same arguments always returns the same JSON-compatible result.
They exist only to give the demo scenarios realistic tool inputs and outputs.
"""

from __future__ import annotations

from copy import deepcopy

__all__ = ["CUSTOMER_IDS", "lookup_customer", "fetch_all_customers"]

# A tiny fixed "customer database".
_CUSTOMERS: dict[str, dict[str, object]] = {
    "123": {"customer_id": "123", "name": "Ada Lovelace", "tier": "gold", "open_orders": 2},
    "456": {"customer_id": "456", "name": "Alan Turing", "tier": "silver", "open_orders": 0},
    "789": {"customer_id": "789", "name": "Grace Hopper", "tier": "gold", "open_orders": 5},
}

CUSTOMER_IDS: tuple[str, ...] = tuple(_CUSTOMERS)


def lookup_customer(customer_id: str) -> dict[str, object]:
    """Return the single record for ``customer_id``.

    Always returns a JSON-compatible dict. Missing ids return
    ``{"found": False, "customer_id": ...}`` rather than raising.
    """

    record = _CUSTOMERS.get(customer_id)
    if record is None:
        return {"found": False, "customer_id": customer_id}
    return {"found": True, **deepcopy(record)}


def fetch_all_customers() -> dict[str, object]:
    """Return every customer record -- deliberately broad, used by the
    inefficient-agent scenario to over-retrieve.
    """

    return {
        "count": len(_CUSTOMERS),
        "customers": [deepcopy(record) for record in _CUSTOMERS.values()],
    }
