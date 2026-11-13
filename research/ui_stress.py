"""Functional UI stress suite; runs actual Streamlit controls, not a visual browser."""
import json
from pathlib import Path
from datetime import date
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

results=[]

def case(name, changes=(), deal='A', solve=True, invalid=False, de=False):
    a=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'app.py'),default_timeout=120).run()
    if deal!='A':a.selectbox[0].select(deal).run()
    for kind,key,value in changes:
        widgets=getattr(a,kind)
        widget=next((w for w in widgets if w.key==key or w.label==key),None)
        assert widget is not None,key
        widget.set_value(value).run()
    assert not a.exception,list(a.exception)
    if invalid:
        assert a.error and a.button[0].disabled
    elif solve:
        a.button[1 if de else 0].click().run()
        assert not a.exception,list(a.exception)
        assert len(a.metric)==4 or len(a.error)>0
    results.append({'case':name,'passed':True})
    Path('research/ui_stress_results.json').write_text(json.dumps({'completed':False,'cases':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(name,'OK',flush=True)
    return a


def run():
    for deal in 'ABCD':case('base_'+deal,deal=deal)
    for name,updates in [
        ('zero_issue',[('Aissue',0.)]),('below_step',[('Aissue',.5)]),
        ('zero_limit',[('Alimit',0.)]),('zero_demand',[('Ainst',0.),('Aretail',0.)]),
        ('no_sales',[('Apace',0.)]),('one_month',[('Aterm',1)]),
        ('long_term',[('Aterm',600)]),('tax_100',[('Atax_rate',100.)]),
        ('no_capital',[('Acapital_ratio',0.)]),('high_thresholds',[('Aroe_threshold',10000.),('Airr_threshold',10000.)]),
        ('fractional_pace',[('Apace',.123)]),('small_step',[('Шаг распределения, млн ₽',.1)])]:
        case(name,[('number_input',k,v) for k,v in updates])
    case('invalid_pd',[('number_input','Apd',101.)],invalid=True)
    case('capital_above_position',[('number_input','Arw',1000.)],invalid=True)
    case('bad_date',[('date_input','Adate',date(2026,12,2))],invalid=True)
    case('uncovered_rate_date',[('date_input','Adate',date(2026,11,1))],invalid=True)
    case('demand_down',[('selectbox','Первичный спрос','Ниже объёма выпуска')])
    case('demand_up',[('selectbox','Первичный спрос','Выше объёма выпуска')])
    case('invalid_scenario',[('selectbox','Первичный спрос','Ниже объёма выпуска'),('number_input','Изменение относительно выпуска, %',101.)],invalid=True)
    case('de_short',[('number_input','Количество запусков DE',1),('number_input','Лимит одного запуска DE, секунд',1)],de=True)
    a=case('stale_result')
    a.number_input(key='Aretail').set_value(100.).run()
    assert any('старый результат' in m.value for m in a.markdown)
    results.append({'case':'stale_result_label','passed':True})
    with patch('bond_calculator.optimization.optimize_milp',side_effect=RuntimeError('Injected solver failure')):
        a=case('solver_failure')
        assert any('Injected solver failure' in e.value for e in a.error)
    with patch('bond_calculator.explanations.explain_distribution',side_effect=RuntimeError('Injected diagnostics failure')):
        a=case('diagnostics_failure')
        assert len(a.metric)==4
    with patch('pandas.ExcelWriter',side_effect=RuntimeError('Injected export failure')):
        a=case('export_failure')
        assert len(a.metric)==4 and a.error
    Path('research/ui_stress_results.json').write_text(json.dumps({'completed':True,'cases':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print('ALL PASSED',len(results),flush=True)


if __name__=='__main__':run()
