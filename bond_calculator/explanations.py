"""Explanations from feasible one-step probes and separately reoptimized scenarios."""
from dataclasses import replace
from .channels import calculate_allocation
from .returns import calculate_returns
from .optimization import optimize_milp

CHANNELS=('Институционалы','Розница','Баланс для перепродажи','Баланс до погашения')


def explain_distribution(result, deal, system, *, scenario_time_limit=5.):
    if result.allocation is None:
        return None
    x=result.volumes; step=system.allocation_step
    notes=[]; probes=[]
    capacities=[('Спрос институционалов',deal.institutional_demand,x[0]),
                ('Спрос розницы',deal.retail_demand,x[1]),
                ('Объём выпуска',deal.issue_volume,sum(x)),
                ('Балансовый лимит',deal.free_limit,x[2]+x[3])]
    for label,limit,used in capacities:
        slack=limit-used
        if slack<=1e-6:notes.append(f'{label} использован полностью.')
        elif slack<step:notes.append(f'{label}: остаток меньше одного шага распределения.')
    for j,label in enumerate(CHANNELS):
        y=list(x);y[j]+=step; reasons=[]
        if sum(y)>deal.issue_volume:reasons.append('не хватает бумаг в выпуске')
        if y[0]>deal.institutional_demand:reasons.append('спрос институционалов не позволяет продать больше')
        if y[1]>deal.retail_demand:reasons.append('спрос розницы не позволяет продать больше')
        if y[2]+y[3]>deal.free_limit:reasons.append('не хватает балансового лимита')
        delta=None;roe=None;irr=None
        if not reasons:
            a=calculate_allocation(*y,deal,system);m=calculate_returns(a,system)
            delta=a.net_profit-result.allocation.net_profit;roe=m.roe;irr=m.irr.annual
            if m.roe_passed is False:reasons.append('ROE ниже порога')
            if m.irr_passed is False:reasons.append('IRR ниже порога')
            if m.capital_years>0 and m.irr_passed is None:reasons.append('IRR требует отдельного анализа')
            if a.net_profit<=0:reasons.append('нет положительной прибыли')
            if not reasons:
                reasons.append('прибыль вырастет, пороги пройдены — можно улучшить результат' if delta>1e-6
                               else 'можно добавить, но прибыль не вырастет')
        probes.append(dict(channel=label,reason='; '.join(reasons),profit_delta=delta,roe=roe,irr=irr))
    # Always compare like-for-like MILP optimizations, not a DE incumbent against MILP.
    reference=result
    if not getattr(result,'optimality_proven',False):
        reference=optimize_milp(deal,system,time_limit=scenario_time_limit)
    base_profit=reference.allocation.net_profit if reference.allocation else None
    effects=[]
    for field,label in [('retail_demand','Спрос розницы'),('monthly_resale_volume','Темп перепродажи'),('free_limit','Балансовый лимит')]:
        before=getattr(deal,field);after=before*1.1 if before else step
        try:
            changed=replace(deal,**{field:after})
            alternate=optimize_milp(changed,system,time_limit=scenario_time_limit)
            profit=alternate.allocation.net_profit if alternate.allocation else None
            effects.append(dict(parameter=label,before=before,after=after,
                profit=profit,delta=profit-base_profit if profit is not None and base_profit is not None else None,
                confirmed=bool(reference.optimality_proven and alternate.optimality_proven),
                status=alternate.status,volumes=alternate.volumes,
                detail='; '.join(alternate.warnings)))
        except (ValueError,RuntimeError) as error:
            effects.append(dict(parameter=label,before=before,after=after,profit=None,delta=None,
                                confirmed=False,status='error',volumes=None,detail=str(error)))
    ranking='Лучшее изменение пока не определено: часть расчётов не завершена или максимум не подтверждён.'
    if all(row['confirmed'] and row['delta'] is not None for row in effects):
        top=max(row['delta'] for row in effects)
        if top<=1.:
            ranking='Ни одно из этих изменений не увеличило максимальную прибыль больше чем на 1 ₽.'
        else:
            winners=[row['parameter'] for row in effects if abs(row['delta']-top)<=1.]
            ranking='Из этих изменений больше всего прибыли добавляет: '+', '.join(winners).lower()+'.'
    return dict(notes=notes,probes=probes,effects=effects,ranking=ranking,
                reference_profit=base_profit,reference_confirmed=reference.optimality_proven,
                step=step)
