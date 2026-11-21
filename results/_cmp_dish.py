import json, pathlib, collections
root = pathlib.Path('data/splits/cn241_21/test')
counts = collections.Counter(p.parent.name for p in root.rglob('*.jpg'))
print('total', sum(counts.values()), 'other_cn', counts.get('other_cn'))
for tag in ['pre','bal','focal','bal_noother']:
    d = json.load(open(f'results/tables/cn241_21_{tag}.json', encoding='utf-8'))
    pc = d['test']['per_class']
    dish = {k: v for k, v in pc.items() if k != 'other_cn'}
    n = sum(counts[k] for k in dish)
    hit = sum(round(pc[k] * counts[k]) for k in dish)
    print(f"{tag}: dish-only top1 {hit/n*100:.2f}% over {n} imgs; other_cn {pc['other_cn']*100:.1f}%")
