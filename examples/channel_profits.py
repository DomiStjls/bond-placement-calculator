"""Запуск из корня проекта: python -m examples.channel_profits"""

from bond_calculator import (
    load_deals, load_system, institutional_placement, retail_placement,
    balance_for_resale, hold_to_maturity, calculate_allocation,
)


def main():
    system = load_system()
    functions = (institutional_placement, retail_placement, balance_for_resale, hold_to_maturity)
    for deal in load_deals().values():
        x = min(1_000_000_000, deal.issue_volume, deal.institutional_demand,
                deal.retail_demand, deal.free_limit)
        print(f"\n{deal.id}: {deal.name} — каждый канал отдельно, объём {x / 1e9:g} млрд ₽")
        for function in functions:
            result = function(x=x, deal=deal, system=system)
            print(f"  {result.channel:26s}: {result.standalone_net_profit:,.2f} ₽ после налога")
    deal = load_deals()["A"]
    # Ручной пример, не оптимальное распределение.
    allocation = calculate_allocation(1_000_000_000, 1_000_000_000, 1_000_000_000,
                                      1_000_000_000, deal, system)
    print(f"\nУралмет, по 1 млрд в каждый канал: {allocation.net_profit:,.2f} ₽ после налога")


if __name__ == "__main__":
    main()
