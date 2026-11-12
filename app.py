"""Local one-page calculator. Run: python -m streamlit run app.py."""
from dataclasses import asdict, replace
from datetime import date
from io import BytesIO
import json
from concurrent.futures import ThreadPoolExecutor
import time
from pathlib import Path
import pandas as pd
import streamlit as st
from bond_calculator import load_deals, load_system
from bond_calculator.models import RatePeriod
from bond_calculator.optimization import optimize_milp
from bond_calculator.deep_optimization import optimize_deep

st.set_page_config(page_title='Размещение облигаций',page_icon='◈',layout='wide')
st.markdown('''<style>
.block-container{max-width:1180px;padding-top:5rem} h1{font-weight:650!important;letter-spacing:-1px} h2{font-size:1.4rem!important}
[data-testid="stMetric"]{background:#f2f5f5;border-radius:10px;padding:16px;border-top:3px solid #164b40}
[data-testid="stMetricValue"]{font-size:clamp(1.15rem,2vw,1.8rem)!important;white-space:normal!important;overflow:visible!important}
[data-testid="stAlert"]{border-radius:8px}
.success-note{color:#164b40;background:#edf5f1;border:1px solid #cbded4;border-radius:8px;padding:14px 18px;margin-bottom:16px}
.eyebrow{color:#164b40;font-size:12px;letter-spacing:2px;font-weight:700} .note{color:#607080;font-size:14px}
</style>''',unsafe_allow_html=True)
st.markdown('<div class="eyebrow">СТРУКТУРНЫЕ ФИНАНСЫ · КАЛЬКУЛЯТОР</div>',unsafe_allow_html=True)
st.title('Размещение облигаций')
st.caption('Выберите сделку, настройте условия и получите распределение по четырём каналам. Данные учебные; суммы ввода — млн ₽, ставки — % годовых.')
deals=load_deals(); original=load_system()

def ru(value, digits=3):
    """Russian display only; underlying calculations/Excel remain numeric."""
    return f'{value:,.{digits}f}'.rstrip('0').rstrip('.').replace(',', '\u202f').replace('.', ',') if digits else f'{value:,.0f}'.replace(',', '\u202f')

def table_ru(frame):
    return frame.style.format({col:lambda value:ru(value) for col in frame.select_dtypes(include='number').columns})

def percent_text(value):
    return f'{value*100:.2f}'.replace('.', ',')+'%'

choice=st.selectbox('Сделка',list(deals),format_func=lambda k:deals[k].name)
base=deals[choice]

def num(label,value,key,percent=False):
    return st.number_input(label,min_value=0.,value=float(value*(100 if percent else 1)),format='%.6f' if percent else '%.3f',key=choice+key)/(100 if percent else 1)

st.subheader('1. Условия сделки')
with st.container(border=True):
    a,b,c=st.columns(3)
    with a:
        issue=num('Объём выпуска, млн ₽',base.issue_volume/1e6,'issue')*1e6
        term=st.number_input('Срок, месяцев',1,600,value=base.term_months,key=choice+'term')
        placement=st.date_input('Месяц размещения (первое число)',base.placement_month,key=choice+'date')
        fee=num('Комиссия организатора, %',base.organizer_fee_rate,'fee',True)
    with b:
        coupon=st.selectbox('Тип купона',['floating','fixed'],index=0 if base.coupon_type=='floating' else 1,format_func=lambda x:'КС + спред' if x=='floating' else 'Фиксированный',key=choice+'coupon')
        rate=num('Спред к КС, п.п.' if coupon=='floating' else 'Купон, %',base.coupon_spread if base.coupon_type=='floating' else base.fixed_coupon_rate,'rate'+coupon,True)
        pd_=num('Годовая PD, %',base.annual_pd,'pd',True)
        rw=num('Риск-вес заёмщика, %',base.borrower_risk_weight,'rw',True)
    with c:
        limit=num('Свободный балансовый лимит, млн ₽',base.free_limit/1e6,'limit')*1e6
        inst=num('Спрос институционалов, млн ₽',base.institutional_demand/1e6,'inst')*1e6
        retail=num('Спрос розницы на размещении, млн ₽',base.retail_demand/1e6,'retail')*1e6
        pace=num('Перепродажа за месяц, млн ₽',base.monthly_resale_volume/1e6,'pace')*1e6

with st.expander('Сценарий спроса и общие параметры'):
    scenario=st.selectbox('Первичный спрос',['Введённый спрос','Ниже объёма выпуска','Выше объёма выпуска'])
    change=st.number_input('Изменение относительно выпуска, %',0.,1000.,20.)
    split=st.number_input('Доля институционалов при нулевом исходном спросе, %',0.,100.,80.)
    st.caption('Сценарий рассчитывается от объёма выпуска. Пропорция первичного спроса сохраняется; темп перепродажи не меняется.')
    cols=st.columns(3); params={}
    labels={'tax_rate':'Налог, %','funding_spread':'Спред фондирования, п.п.','loss_given_default':'LGD, %','capital_ratio':'Норматив капитала, %','resale_risk_weight':'Риск-вес перепродажи, %','bank_placement_fee_share':'Доля комиссии банка, %','institutional_rebate_share':'Возврат комиссии институционалам, %','retail_ecosystem_rate':'Экосистемный доход розницы, %','resale_income_rate':'Доход вторички, %','own_balance_placement_fee_rate':'Комиссия на собственный баланс, %','irr_threshold':'Порог IRR, %','roe_threshold':'Порог ROE, %'}
    for i,(key,label) in enumerate(labels.items()):
        with cols[i%3]:params[key]=num(label,getattr(original,key),key,True)
    sale_start=st.number_input('Начало перепродажи, номер месяца',2,600,2)
    st.caption('Траектория КС: даты первого числа, ставки в процентах. Можно добавлять и удалять строки.')
    rates=st.data_editor(pd.DataFrame([{'Дата':p.effective_from,'КС, %':p.annual_rate*100} for p in original.key_rates]),num_rows='dynamic',hide_index=True,key='rates')
    step=st.number_input('Шаг распределения, млн ₽',.000001,value=1.,format='%.6f')
    milp_time=st.number_input('Лимит MILP, секунд',1,300,15)
    runs=st.number_input('Количество запусков DE',1,20,4)
    de_time=st.number_input('Лимит одного запуска DE, секунд',1,120,20)
    seed=st.number_input('Seed DE (для воспроизводимости)',0,2**31-1,42)

error=None
try:
    if scenario!='Введённый спрос':
        if scenario.startswith('Ниже') and change>100:raise ValueError('Снижение спроса не может быть больше 100%.')
        share=inst/(inst+retail) if inst+retail else split/100
        total=issue*(1+change/100*(1 if scenario.startswith('Выше') else -1))
        inst,retail=total*share,total*(1-share)
        st.info(f'Сценарный спрос: институционалы {ru(inst/1e6)} млн ₽; розница {ru(retail/1e6)} млн ₽.')
    d=replace(base,issue_volume=issue,term_months=term,placement_month=placement,coupon_type=coupon,coupon_spread=rate if coupon=='floating' else None,fixed_coupon_rate=rate if coupon=='fixed' else None,organizer_fee_rate=fee,annual_pd=pd_,borrower_risk_weight=rw,free_limit=limit,institutional_demand=inst,retail_demand=retail,monthly_resale_volume=pace)
    periods=tuple(RatePeriod(row['Дата'] if isinstance(row['Дата'],date) else date.fromisoformat(str(row['Дата'])),float(row['КС, %'])/100) for row in rates.to_dict('records'))
    s=replace(original,**params,key_rates=periods,resale_start_month=sale_start,allocation_step=round(step*1e6))
    s.key_rate_at(placement)
    if max(s.resale_risk_weight,d.borrower_risk_weight)*s.capital_ratio>1:raise ValueError('Капитал превышает позицию: уменьшите риск-вес или норматив капитала.')
    fingerprint=json.dumps([asdict(d),asdict(s)],default=str,sort_keys=True)
except (ValueError,TypeError,KeyError) as e:
    error=str(e);st.error('Проверьте параметры: '+error)

st.subheader('2. Рассчитать распределение')
a,b=st.columns(2)
light=a.button('Рассчитать · MILP',type='primary',width='stretch',disabled=bool(error))
deep=b.button('Глубокий поиск · DE',width='stretch',disabled=bool(error))
st.caption('MILP — быстрый поиск с проверкой максимума. DE — несколько эволюционных запусков; подтверждения глобального максимума не даёт. Лимиты времени мягкие.')
if light or deep:
    st.session_state.pop('answer',None)
    try:
        with st.status('Идёт поиск распределения…',expanded=True) as status:
            progress=st.empty();t=time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(optimize_milp,d,s,time_limit=milp_time) if light else pool.submit(optimize_deep,d,s,runs=runs,seed=seed,run_time_limit=de_time)
                while not future.done():
                    progress.write(f'Прошло {time.monotonic()-t:.0f} с. Поиск продолжается…')
                    time.sleep(.5)
                result=future.result()
            status.update(label=f'Расчёт завершён за {time.monotonic()-t:.1f} с',state='complete')
        st.session_state.answer=(result,d,s,fingerprint,'MILP' if light else 'DE')
    except Exception as e:
        st.error(f'Расчёт завершился с ошибкой ({type(e).__name__}): {e}. Результат не сформирован. Проверьте параметры и попробуйте другой режим.')

if 'answer' in st.session_state:
    r,rd,rs,fp,method=st.session_state.answer
    st.divider();st.subheader('3. Результат · '+rd.name)
    notes=[]
    notes_slot=st.container()
    if error or fp!=fingerprint:notes.append('Параметры изменились. Ниже показан предыдущий расчёт; нажмите «Рассчитать» для обновления.')
    if r.allocation is None:
        st.error('Прибыльного распределения с заданными порогами нет.' if r.status=='no_acceptable_allocation' else 'Допустимое распределение не найдено. Возможные причины: лимит поиска, высокие пороги или нетипичные денежные потоки. Это не доказательство отсутствия решения.')
    else:
        m=r.metrics;a=r.allocation
        if r.optimality_proven:st.markdown('<div class="success-note">Максимум прибыли подтверждён на выбранной сетке в численных допусках.</div>',unsafe_allow_html=True)
        else:notes.append('Найден допустимый вариант. Глобальный максимум не подтверждён.'+(' MILP не завершил доказательство.' if method=='MILP' else ' DE использует приближённый поиск.'))
        cols=st.columns(4)
        for col,label,value in zip(cols,['Чистая прибыль за весь срок, млн ₽','Участие банка, млн ₽','ROE годовой','IRR годовая'],[ru(a.net_profit/1e6),ru(sum(r.volumes)/1e6),percent_text(m.roe) if m.roe is not None else 'Не применим',percent_text(m.irr.annual) if m.irr.annual is not None else 'Не применима']):col.metric(label,value)
        st.caption('Проценты на странице округлены до двух знаков после запятой. Пороги проверяются по исходным значениям; в Excel сохраняются значения без округления.')
        allocation=pd.DataFrame({'Канал':['x₁ · Институционалы','x₂ · Розница','x₃ · Баланс для перепродажи','x₄ · Баланс до погашения'],'Объём, млн ₽':[v/1e6 for v in r.volumes]})
        st.dataframe(table_ru(allocation),hide_index=True,width='stretch')
        st.caption(f'Собственные покупки банка: {ru((r.volumes[2]+r.volumes[3])/1e6)} млн ₽. Не участвуем в размещении: {ru((rd.issue_volume-sum(r.volumes))/1e6)} млн ₽. Первые два канала — размещение инвесторам.')
        if m.roe is None:notes.append('Капитал не используется. По принятому допущению достаточно положительной прибыли; IRR и ROE неприменимы.')
        else:
            st.write(f'Пороги: ROE ≥ {percent_text(rs.roe_threshold)}, IRR ≥ {percent_text(rs.irr_threshold)}. Оба пройдены.')
            if min(m.roe-rs.roe_threshold,m.irr.annual-rs.irr_threshold)<.001:notes.append('Малый запас доходности: до одного из порогов меньше 0,1 п.п. Изменение условий может потребовать перераспределения.')
        notes.extend((*m.warnings,*getattr(r,'warnings',())))
        full=pd.DataFrame([asdict(row) for row in a.months]);full['month']=full['month'].astype(str)
        plan=pd.DataFrame({'Месяц':full.month,'Продажа с баланса, млн ₽':full.sold_principal/1e6,'Погашение, млн ₽':full.redeemed_principal/1e6,'Остаток на конец, млн ₽':full.closing_position/1e6,'Капитал на начало, млн ₽':full.opening_capital/1e6,'Чистая прибыль, млн ₽':full.net_profit/1e6})
        st.subheader('План по месяцам')
        st.caption('В первом месяце — размещение инвесторам и покупка на баланс. Затем продажи в конце месяца; в последний месяц эмитент погашает остаток. Полные начисления за месяц продажи сохраняются.')
        st.line_chart(plan.set_index('Месяц')[['Капитал на начало, млн ₽','Остаток на конец, млн ₽']],color=['#164b40','#142b40'])
        st.bar_chart(plan.set_index('Месяц')[['Чистая прибыль, млн ₽']],color='#164b40')
        st.dataframe(table_ru(plan),hide_index=True,height=360,width='stretch')
        with st.expander('Подробные начисления по месяцам, ₽'):st.dataframe(table_ru(full),hide_index=True)
        stream=BytesIO()
        with pd.ExcelWriter(stream,engine='openpyxl') as writer:
            allocation.to_excel(writer,sheet_name='Распределение',index=False)
            pd.DataFrame({'Показатель':['Прибыль, ₽','ROE','IRR','Метод','Статус','Максимум подтверждён'],'Значение':[a.net_profit,m.roe,m.irr.annual,method,r.status,r.optimality_proven]}).to_excel(writer,sheet_name='Итог',index=False)
            plan.to_excel(writer,sheet_name='План',index=False);full.to_excel(writer,sheet_name='Все начисления',index=False)
            for idx,ch in enumerate(a.channels):pd.DataFrame([asdict(row) for row in ch.months]).to_excel(writer,sheet_name=f'Канал {idx+1}',index=False)
            pd.DataFrame({'Раздел':['Сделка','Система'],'Параметры':[json.dumps(asdict(rd),ensure_ascii=False,default=str),json.dumps(asdict(rs),ensure_ascii=False,default=str)]}).to_excel(writer,sheet_name='Параметры',index=False)
            from openpyxl.styles import Font,PatternFill
            for sheet in writer.book:
                sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
                for cell in sheet[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='164B40')
                for col in sheet.columns:sheet.column_dimensions[col[0].column_letter].width=min(55,max(18,len(str(col[0].value))+3))
        st.download_button('Скачать подробный отчёт · Excel',stream.getvalue(),file_name=f'deal_{rd.id}_{method}.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',type='primary')
    if notes:
        with notes_slot:
            with st.container(border=True):
                st.markdown('**Уточнения**')
                for note in dict.fromkeys(notes):st.markdown('• '+note)

with st.expander('Как считаем · допущения и ограничения'):
    st.markdown(Path('MODEL_ASSUMPTIONS.md').read_text(encoding='utf-8'))
st.caption('Made by Darya Shevchenko')
