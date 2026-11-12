"""python -m examples.optimize_deals — расчёт четырёх базовых сделок."""
from bond_calculator import load_deals,load_system
from bond_calculator.optimization import optimize_milp


def main():
    system=load_system()
    for deal in load_deals().values():
        result=optimize_milp(deal,system)
        print(f'\n{deal.name}: {result.status}, {result.elapsed_seconds:.3f} с')
        if result.metrics:
            m=result.metrics
            print('Каналы, млн ₽:',[v/1e6 for v in result.volumes])
            print(f'Прибыль: {m.net_profit/1e6:.6f} млн ₽')
            print('ROE:',None if m.roe is None else round(m.roe*100,6))
            print('IRR:',None if m.irr.annual is None else round(m.irr.annual*100,6))
            print('Пороги:',m.status,'; разрыв, ₽:',result.absolute_gap)
        for warning in result.warnings: print(warning)


if __name__=='__main__': main()
