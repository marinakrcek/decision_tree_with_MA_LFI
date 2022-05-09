# The More You Know: Improving Laser Fault Injection with Prior Knowledge
#### Marina Krček, Thomas Ordas, Daniele Fronte, Stjepan Picek

Source code for publication **The More You Know: Improving Laser Fault Injection with Prior Knowledge**.

## Rules from generated Decision Tree used to initialize MA population

While initializing the algorithm, user should provide a different initialization method.
```python
gen_alg = ga.GeneticAlgorithm(POPULATION_SIZE, MUTATION_PROB, ELITE_SIZE, initialization_function=initialize_from_past_fail_rules(rules))
```
Variable 'rules' should be prepared like this:
```python
bounds = {'x': [0, 100, 1], 'y': [0, 100, 1], 'delay': [0, 100, 1],
        'pulse_width': [0, 100, 1],
        'intensity': [0, 100, 1]}
# set_parameter_limits_from_ini_file(parameter_info_file)
rules = get_rules_for_initialization(bounds, feature_name_conv=feature_names, for_class='Fail')
```

**Saving the model of the decision tree**\
pickle (and joblib by extension), has some issues regarding maintainability and security. Because of this,

- Never unpickle untrusted data as it could lead to malicious code being executed upon loading.

- While models saved using one version of scikit-learn might load in other versions, this is entirely unsupported and inadvisable. It should also be kept in mind that operations performed on such data could give different and unexpected results.

For more, read: https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations.

