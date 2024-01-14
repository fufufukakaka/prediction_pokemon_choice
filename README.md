# prediction_pokemon_choice
6-6見せあいのデータをクエリとして、その後選出されるポケモンを予測する

## Metrics

- Accuracy: 0.5276
- Precision: 0.4834
- Recall: 0.4794
- F1: 0.4811

### Confusion Matrix

```python
In [19]: confusion_matrix([v['label'] for v in test], [map_dict[v['label']] for v in res])
Out[19]:
array([[ 52,   9,  48],
       [ 11,  21,  28],
       [ 50,  25, 118]])
```
