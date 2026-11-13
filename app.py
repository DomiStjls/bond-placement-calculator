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
from bond_calculator.explanations import explain_distribution

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
st.caption('Выберите сделку и рассчитайте, сколько бумаг разместить инвесторам и оставить на балансе.')
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
    name_col,rating_col=st.columns([2,1])
    deal_name=name_col.text_input('Название сделки',base.name,key=choice+'name')
    deal_rating=rating_col.text_input('Рейтинг',base.rating,key=choice+'rating')
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
    if not deal_name.strip():raise ValueError('Укажите название сделки.')
    if scenario!='Введённый спрос':
        if scenario.startswith('Ниже') and change>100:raise ValueError('Снижение спроса не может быть больше 100%.')
        share=inst/(inst+retail) if inst+retail else split/100
        total=issue*(1+change/100*(1 if scenario.startswith('Выше') else -1))
        inst,retail=total*share,total*(1-share)
        st.info(f'Сценарный спрос: институционалы {ru(inst/1e6)} млн ₽; розница {ru(retail/1e6)} млн ₽.')
    d=replace(base,name=deal_name.strip(),rating=deal_rating.strip(),issue_volume=issue,term_months=term,placement_month=placement,coupon_type=coupon,coupon_spread=rate if coupon=='floating' else None,fixed_coupon_rate=rate if coupon=='fixed' else None,organizer_fee_rate=fee,annual_pd=pd_,borrower_risk_weight=rw,free_limit=limit,institutional_demand=inst,retail_demand=retail,monthly_resale_volume=pace)
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
st.caption('MILP — быстрый расчёт с проверкой максимума. DE — более долгий поиск без гарантии максимума.')
if light or deep:
    st.session_state.pop('answer',None)
    st.session_state.pop('explanation',None)
    st.session_state.pop('explanation_error',None)
    try:
        with st.status('Идёт поиск распределения…',expanded=True) as status:
            progress=st.empty();t=time.monotonic()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future=pool.submit(optimize_milp,d,s,time_limit=milp_time) if light else pool.submit(optimize_deep,d,s,runs=runs,seed=seed,run_time_limit=de_time)
                while not future.done():
                    progress.write(f'Прошло {time.monotonic()-t:.0f} с. Поиск продолжается…')
                    time.sleep(.5)
                result=future.result()
                if result.allocation is not None:
                    future=pool.submit(explain_distribution,result,d,s)
                    while not future.done():
                        progress.write(f'Прошло {time.monotonic()-t:.0f} с. Проверяю ограничения и влияние изменений условий…')
                        time.sleep(.5)
                    try:st.session_state.explanation=future.result()
                    except Exception as diagnostic_error:
                        st.session_state.explanation_error=str(diagnostic_error)
            status.update(label=f'Расчёт завершён за {time.monotonic()-t:.1f} с',state='complete')
        st.session_state.answer=(result,d,s,fingerprint,'MILP' if light else 'DE')
    except Exception as e:
        st.error(f'Расчёт завершился с ошибкой ({type(e).__name__}): {e}. Результат не сформирован. Проверьте параметры и попробуйте другой режим.')

if 'answer' in st.session_state:
    r,rd,rs,fp,method=st.session_state.answer
    st.divider();st.subheader('3. Результат · '+rd.name)
    notes=[]
    notes_slot=st.container()
    if error or fp!=fingerprint:notes.append('Вы изменили условия. Здесь ещё старый результат — пересчитайте сделку.')
    if r.allocation is None:
        st.error('При этих условиях нет прибыльного варианта, который проходит пороги доходности.' if r.status=='no_acceptable_allocation' else 'Подходящий вариант пока не найден. Попробуйте увеличить время поиска или изменить условия. Это ещё не означает, что решения нет.')
    else:
        m=r.metrics;a=r.allocation
        if r.optimality_proven:st.markdown('<div class="success-note">Найдена максимальная прибыль при заданных условиях и шаге распределения.</div>',unsafe_allow_html=True)
        elif not getattr(r,'warnings',()):notes.append('Вариант проходит ограничения, но более прибыльное распределение ещё может существовать.')
        cols=st.columns(4)
        for col,label,value in zip(cols,['Чистая прибыль за весь срок, млн ₽','Участие банка, млн ₽','ROE годовой','IRR годовая'],[ru(a.net_profit/1e6),ru(sum(r.volumes)/1e6),percent_text(m.roe) if m.roe is not None else 'Не применим',percent_text(m.irr.annual) if m.irr.annual is not None else 'Не применима']):col.metric(label,value)
        st.caption('Проценты округлены до двух знаков. Проверка порогов и Excel используют значения без округления.')
        allocation=pd.DataFrame({'Канал':['x₁ · Институционалы','x₂ · Розница','x₃ · Баланс для перепродажи','x₄ · Баланс до погашения'],'Объём, млн ₽':[v/1e6 for v in r.volumes]})
        st.dataframe(table_ru(allocation),hide_index=True,width='stretch')
        st.caption(f'Банк покупает на баланс {ru((r.volumes[2]+r.volumes[3])/1e6)} млн ₽. Остальные покупки — за счёт инвесторов. Вне участия банка остаётся {ru((rd.issue_volume-sum(r.volumes))/1e6)} млн ₽ выпуска.')
        if m.roe is None:notes.append('Капитал не нужен: оцениваем только прибыль. IRR и ROE здесь не рассчитываются.')
        else:
            st.write(f'Пороги: ROE ≥ {percent_text(rs.roe_threshold)}, IRR ≥ {percent_text(rs.irr_threshold)}. Оба пройдены.')
            if min(m.roe-rs.roe_threshold,m.irr.annual-rs.irr_threshold)<.001:notes.append('Запас по одному из порогов меньше 0,1 п.п. При ухудшении условий сделка может перестать проходить порог.')
        notes.extend((*m.warnings,*getattr(r,'warnings',())))
        explanation=st.session_state.get('explanation')
        if st.session_state.get('explanation_error'):
            notes.append('Пояснения не удалось рассчитать: '+st.session_state.explanation_error+'. Основной результат сохранён.')
        if explanation:
            with st.container(border=True):
                st.markdown('### Почему получилось такое распределение')
                if explanation['notes']:st.write(' '.join(explanation['notes']))
                st.caption(f'Что произойдёт, если добавить {ru(explanation["step"]/1e6)} млн ₽ в один канал, оставив остальные объёмы прежними:')
                probe_table=pd.DataFrame([{'Канал':p['channel'],'Результат добавления':p['reason'],
                    'Изменение прибыли, млн ₽':p['profit_delta']/1e6 if p['profit_delta'] is not None else None} for p in explanation['probes']])
                st.dataframe(probe_table.style.format({'Изменение прибыли, млн ₽':lambda v:ru(v)},na_rep='—'),hide_index=True,width='stretch')
                st.markdown('### Что поможет заработать больше')
                st.caption('Увеличиваем каждое условие отдельно на 10% и заново подбираем распределение. Если исходное значение равно нулю, добавляем один шаг.')
                effect_table=pd.DataFrame([{'Изменение':e['parameter'],
                    'Было, млн ₽':e['before']/1e6,'Стало, млн ₽':e['after']/1e6,
                    'Прибыль, млн ₽':e['profit']/1e6 if e['profit'] is not None else None,
                    'Прирост, млн ₽':e['delta']/1e6 if e['delta'] is not None else None,
                    'Проверка':'Подтверждено' if e['confirmed'] else 'Нет ответа' if e['profit'] is None else 'Предварительно'} for e in explanation['effects']])
                st.dataframe(effect_table[['Изменение','Прибыль, млн ₽','Прирост, млн ₽','Проверка']].style.format({'Прибыль, млн ₽':lambda v:ru(v),'Прирост, млн ₽':lambda v:ru(v)},na_rep='—'),hide_index=True,width='stretch')
                st.write(explanation['ranking'])
                with st.expander('Подробности сравнения'):
                    st.write('Сравниваем прибыль, не учитывая затраты на привлечение спроса или увеличение лимита. Все варианты рассчитаны методом MILP с прежними порогами доходности.')
                    if explanation['reference_profit'] is not None:st.write(f'Исходная прибыль для сравнения: {ru(explanation["reference_profit"]/1e6)} млн ₽.'+(' Она может отличаться от результата DE выше.' if method=='DE' else ''))
                    st.dataframe(table_ru(effect_table[['Изменение','Было, млн ₽','Стало, млн ₽']]),hide_index=True,width='stretch')
                    st.caption('Темп продаж — млн ₽ в месяц; спрос и лимит — млн ₽. На каждый расчёт отводится около 5 секунд.')
                for e in explanation['effects']:
                    if not e['confirmed'] and e['detail']:st.write(e['parameter']+': '+e['detail'])
        full=pd.DataFrame([asdict(row) for row in a.months]);full['month']=full['month'].astype(str)
        plan=pd.DataFrame({'Месяц':full.month,'Продажа с баланса, млн ₽':full.sold_principal/1e6,'Погашение, млн ₽':full.redeemed_principal/1e6,'Остаток на конец, млн ₽':full.closing_position/1e6,'Капитал на начало, млн ₽':full.opening_capital/1e6,'Чистая прибыль, млн ₽':full.net_profit/1e6})
        st.subheader('План по месяцам')
        st.caption('В начале сделки банк размещает бумаги и покупает часть на баланс. Затем продаёт их по плану. В конце срока эмитент погашает остаток.')
        with st.expander('Использование капитала'):
            st.write(f'За весь срок: {ru(m.capital_years/1e6)} млн ₽·лет. Например, 10 млн ₽ капитала на два года — это 20 млн ₽·лет.')
            st.write('Среди вариантов с той же прибылью выбран наименьший капитал.' if getattr(r,'capital_optimality_proven',False) else 'Из найденных вариантов с той же прибылью выбран меньший капитал. Его минимум пока не подтверждён.')
        st.line_chart(plan.set_index('Месяц')[['Капитал на начало, млн ₽','Остаток на конец, млн ₽']],color=['#164b40','#142b40'])
        st.bar_chart(plan.set_index('Месяц')[['Чистая прибыль, млн ₽']],color='#164b40')
        st.dataframe(table_ru(plan),hide_index=True,height=360,width='stretch')
        with st.expander('Подробные начисления по месяцам, ₽'):st.dataframe(table_ru(full),hide_index=True)
        try:
            stream=BytesIO()
            with pd.ExcelWriter(stream,engine='openpyxl') as writer:
                allocation.to_excel(writer,sheet_name='Распределение',index=False)
                pd.DataFrame({'Показатель':['Прибыль, ₽','ROE','IRR','Метод','Статус','Максимум подтверждён','Капитал за срок, ₽·лет','Минимум капитала подтверждён'],'Значение':[a.net_profit,m.roe,m.irr.annual,method,r.status,r.optimality_proven,m.capital_years,getattr(r,'capital_optimality_proven',False)]}).to_excel(writer,sheet_name='Итог',index=False)
                plan.to_excel(writer,sheet_name='План',index=False);full.to_excel(writer,sheet_name='Все начисления',index=False)
                if explanation:
                    pd.DataFrame(explanation['probes']).to_excel(writer,sheet_name='Проверка ограничений',index=False)
                    pd.DataFrame(explanation['effects']).assign(reference_profit=explanation['reference_profit']).to_excel(writer,sheet_name='Изменение условий',index=False)
                for idx,ch in enumerate(a.channels):pd.DataFrame([asdict(row) for row in ch.months]).to_excel(writer,sheet_name=f'Канал {idx+1}',index=False)
                pd.DataFrame({'Раздел':['Сделка','Система'],'Параметры':[json.dumps(asdict(rd),ensure_ascii=False,default=str),json.dumps(asdict(rs),ensure_ascii=False,default=str)]}).to_excel(writer,sheet_name='Параметры',index=False)
                from openpyxl.styles import Font,PatternFill
                for sheet in writer.book:
                    sheet.freeze_panes='A2';sheet.auto_filter.ref=sheet.dimensions
                    for cell in sheet[1]:cell.font=Font(color='FFFFFF',bold=True);cell.fill=PatternFill('solid',fgColor='164B40')
                    for col in sheet.columns:sheet.column_dimensions[col[0].column_letter].width=min(55,max(18,len(str(col[0].value))+3))
            st.download_button('Скачать подробный отчёт · Excel',stream.getvalue(),file_name=f'deal_{rd.id}_{method}.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',type='primary')
        except Exception as export_error:
            st.error('Не удалось создать Excel. Результат расчёта сохранён на странице. Попробуйте скачать отчёт ещё раз.')
            with st.expander('Причина ошибки выгрузки'):st.code(str(export_error))
    if notes:
        with notes_slot:
            with st.container(border=True):
                st.markdown('**Уточнения**')
                for note in dict.fromkeys(notes):st.markdown('• '+note)

with st.expander('Как считаем · допущения и ограничения'):
    st.markdown('''- Максимизируем чистую прибыль за весь срок. При равной прибыли выбираем меньший капитал.
- Купоны и расходы начисляем ежемесячно. Продажи проходят в конце месяца, непроданный остаток погашает эмитент.
- Налог считаем с общей прибыли каналов за месяц. Убытки на другие месяцы не переносим.
- ROE учитывает всю прибыль и срок использования капитала. IRR — доходность денежных потоков собственного капитала.
- Если капитал не нужен, достаточно положительной прибыли. Для сложных денежных потоков IRR может потребовать отдельной проверки.
- На собственные покупки банка комиссия размещения по умолчанию не начисляется.

Это допущения для деталей, которые не указаны в ТЗ. Полная методика — в файле MODEL_ASSUMPTIONS.md в репозитории.''')
st.caption('Made by Darya Shevchenko')
