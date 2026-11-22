import json, sys
p = sys.argv[1] if len(sys.argv) > 1 else 'experiments/lead_eval/lead_eval-9e1899be630c/report.json'
r = json.load(open(p, encoding='utf-8'))

t2 = r['test2_adaptive_compute']
print('=== TEST 2 === report', t2['suite_report_id'], 'n=', t2['dataset']['size'])
hdr = ['config','acc%','depth','tok_eval','MACs','sec','speedup','tok/s']
print('{:22s} {:>7s} {:>6s} {:>9s} {:>12s} {:>8s} {:>8s} {:>9s}'.format(*hdr))
for x in t2['results']:
    print('{:22s} {:7.2f} {:6.3f} {:9d} {:12d} {:8.3f} {:8.3f} {:9.1f}'.format(
        x['config'], x['accuracy_pct'], x['mean_depth'], x['token_evaluations'],
        x['estimated_macs'], x['wall_clock_seconds'], x['speedup_vs_fixed4'],
        x['tokens_per_second']))

t3 = r['test3_mixed_difficulty']
print('\n=== TEST 3 === report', t3['suite_report_id'], 'composition_ok=', t3['composition_ok'])
print('composition counts', t3['composition']['complexity_counts'])
for label, s in t3['strata'].items():
    print(label, 'n=', s['count'], 'acc%=', s['accuracy_pct'], 'depth=', s['mean_depth'],
          'p50/pass=', s['latency_p50_ms_per_pass'], 'p95/pass=', s['latency_p95_ms_per_pass'],
          'p50/prompt=', s['latency_p50_ms_per_prompt'], 'runs=', s['runs'])

t4 = r['test4_latency_profile']
print('\n=== TEST 4 === report', t4['suite_report_id'])
print('{:20s} {:>7s} {:>10s} {:>10s} {:>12s}'.format('config','tokens','p50_ms','p95_ms','tok_eval'))
for x in t4['results']:
    print('{:20s} {:7d} {:10.3f} {:10} {:12}'.format(
        x['config'], x['prompt_tokens'], x['p50_ms'], str(x['p95_ms']),
        str(x['token_evaluations'])))

t5 = r['test5_continual']
print('\n=== TEST 5 ===')
for k in ['baseline_accuracy_before','baseline_accuracy_after_wave3','baseline_forgetting_pct',
          'set_A_recall_wave1','set_A_recall_wave3','set_A_forgetting_pct','slow_weights_unchanged']:
    print(k, '=', t5[k])
for w in t5['waves']:
    print('wave', w['wave'], 'facts', w['facts_seen'], 'adapt_s', w['adaptation_seconds'],
          'bytes', w['fast_state_bytes'], 'recall', w['recall'])

t6 = r['test6_synergy']
print('\n=== TEST 6 ===')
for k,v in t6.items():
    print(k, '=', v)
