"""Check the real bootstrap in a clean copy without global site-packages."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import tempfile
import time
import urllib.request


def run():
    root=Path(__file__).resolve().parents[1]
    target=Path(tempfile.mkdtemp(prefix='bond cold start ',dir=root.parents[1]/'tmp'))
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard'],cwd=root,text=True).splitlines()
    for name in names:
        source=root/name
        if source.is_file():
            destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    assert not (target/'.venv').exists()
    result={'directory':str(target),'completed':False,'launches':[]}
    env=dict(os.environ,STREAMLIT_SERVER_PORT='8513',STREAMLIT_BROWSER_GATHER_USAGE_STATS='false')
    stamp=target/'.venv'/'.calculator-requirements'
    for number in (1,2):
        previous_stamp=stamp.stat().st_mtime if stamp.exists() else None
        log=target/f'launch-{number}.log'
        started=time.monotonic()
        with log.open('w',encoding='utf-8') as output:
            process=subprocess.Popen(['python','run_calculator.py'],cwd=target,env=env,stdout=output,stderr=subprocess.STDOUT)
            try:
                for attempt in range(600):
                    if process.poll() is not None:raise RuntimeError(log.read_text(encoding='utf-8'))
                    try:
                        with urllib.request.urlopen('http://127.0.0.1:8513/_stcore/health',timeout=1) as response:
                            if response.read()==b'ok':break
                    except OSError:pass
                    time.sleep(1)
                else:raise RuntimeError('Cold startup exceeded 10 minutes')
                assert stamp.exists()
                result['launches'].append({'number':number,'seconds':time.monotonic()-started,'health':'ok'})
                if number==2:assert previous_stamp==stamp.stat().st_mtime
                print('Launch',number,'ready',flush=True)
                if number==1:
                    python=target/'.venv'/'Scripts'/'python.exe'
                    code="from streamlit.testing.v1 import AppTest; a=AppTest.from_file('app.py',default_timeout=90).run(); assert not a.exception; a.button[0].click().run(); assert not a.exception,list(a.exception); assert len(a.metric)==4; print('Cold UI calculation and Excel passed')"
                    test=subprocess.run([str(python),'-c',code],cwd=target,capture_output=True,text=True,timeout=180)
                    assert test.returncode==0,test.stderr
                    result['ui_calculation_and_export']=True
                    test=subprocess.run([str(python),'-m','unittest','discover','-s','tests'],cwd=target,capture_output=True,text=True,timeout=180)
                    assert test.returncode==0,test.stderr
                    result['tests']=test.stderr
                    print('Cold tests passed',flush=True)
            finally:
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True)
                process.wait(timeout=15)
        if number==2:assert 'Installing dependencies.' not in log.read_text(encoding='utf-8')
    result['completed']=True
    (root/'research'/'cold_start_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':run()
