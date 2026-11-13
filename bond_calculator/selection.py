"""Lexicographic comparison: profit first, capital-years second."""


def profit_tolerance(profit):
    # Rubles, not display rounding or the 1-ruble optimality certificate.
    return max(1e-6, abs(profit)*1e-12)


def better(candidate, incumbent):
    if incumbent is None:
        return True
    return better_values(candidate[1].net_profit,candidate[2].capital_years,
                         incumbent[1].net_profit,incumbent[2].capital_years)


def better_values(p,capital,old,old_capital):
    if abs(p-old) <= profit_tolerance(max(abs(p),abs(old))):
        return capital < old_capital
    return p > old
