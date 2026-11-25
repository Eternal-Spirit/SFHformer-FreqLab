import urllib.request, json, time

samples = ['normal_fundus', 'optic_disc_fundus', 'microvessel_fundus', 'oct_normal', 'oct_pathology', 'stare_pathology', 'hrf_glaucoma', 'aptos_dr_severe']

print(f"Testing all {len(samples)} gallery samples on /api/retina/restore...")
for s in samples:
    t0 = time.time()
    payload = {'sample_id': s, 'mode': 'mixed', 'severity': 1.0, 'model': 'retina'}
    req = urllib.request.Request('http://127.0.0.1:5001/api/retina/restore', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    try:
        res = json.loads(urllib.request.urlopen(req).read())
        elapsed = (time.time() - t0) * 1000
        l_ms = res["latency_ms"]
        p_gain = res["metrics"]["psnr_gain"]
        vd = res["biomarkers"]["vessel_density_pct"]
        print(f"  [PASS] {s:<22} | Latency: {l_ms:5.1f}ms | PSNR Gain: {p_gain:+.2f}dB | Vessel Density: {vd}%")
    except Exception as e:
        print(f"  [FAIL] {s:<22} | Error: {e}")
