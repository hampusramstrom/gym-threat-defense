# Gym Threat Defense
The Threat Defense environment is an OpenAI Gym implementation of the environment defined as the toy example in
[Optimal Defense Policies for Partially Observable Spreading Processes on
Bayesian Attack Graphs](https://www.researchgate.net/profile/Erik_Miehling/publication/283083610_Optimal_Defense_Policies_for_Partially_Observable_Spreading_Processes_on_Bayesian_Attack_Graphs/links/564e46b408aeafc2aab1b734/Optimal-Defense-Policies-for-Partially-Observable-Spreading-Processes-on-Bayesian-Attack-Graphs.pdf)
by *Miehling, E., Rasouli, M., & Teneketzis, D. (2015)*. It constitutes a
29-state/observation, 4-action POMDP defense problem.

## The environment
![The Threat Defense environment](threat_defense_environment.png?raw=true "The Threat Defense environment")

Above, the Threat Defense environment can be observed. None of the notations or the definitions made in the paper will be explained in the text that follows, but rather the benchmark of the toy example will be stated. If these are desired, please follow the link found earlier to the paper of *Miehling, E., Rasouli, M., & Teneketzis, D. (2015)*.

### Attributes
Of the 12 attributes that the toy example is built up by, two are leaf attributes (1 and 5) and one is a critical attribute (12). To give the network a more realistic appearance, the 12 attributes are intepreted in the paper as:

1. Vulnerability in WebDAV on machine 1
2. User access on machine 1
3. Heap corruption via SSH on machine 1
4. Root access on machine 1
5. Buffer overflow on machine 2
6. Root access on machine 2
7. Squid portscan on machine 2
8. Network topology leakage from machine 2
9. Buffer overflow on machine 3
10. Root access on machine 3
11. Buffer overflow on machine 4
12. Root access on machine 4


### Actions
The defender have access to the two following binary actions:

* *u_1*: Block WebDAV service
* *u_2*: Disconnect machine 2

Thus we have four countermeasures to apply, i.e *U* = {*none*, *u_1*, *u_2*, *u_1* *&* *u_2*}. 

### Cost Function
The cost function is defined as *C(x,u)* = *C(x)* + *D(u)*. 

*C(x)* is the state costs, and is 1 if the state, i.e. *x*, is a critical attribute. Otherwise it is 0.

*D(u)* is the availability cost of a countermeasure *u*, and is 0 if the countermeasure is *none*, 1 if it is *u_1* or *u_2* and 5 if it is both *u_1* and *u_2*.

### Parameters
The parameters of the problem are:

```python
# The probabilities of detection:
beta = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.6, 0.7, 0.85, 0.95]

# The attack probabilities:
alpha_1, alpha_5 = 0.5

# The spread probabilities:
alpha_(1,2), alpha_(2,3), alpha_(4,9), alpha_(5,6), alpha_(7,8), alpha_(8,9), alpha_(8,11), alpha_(10,11) = 0.8

alpha_(3,4), alpha_(6,7), alpha_(9,10), alpha_(11,12) = 0.9

# The discount factor:
gamma = 0.85

# The initial belief vector
pi_0 = [1,0,...,0]
```

## Dependencies
- OpenAI Gym

## Installation

```bash
cd gym-threat-defense
pip install -e .
```

## Rendering
There are three possible rendering alternatives when running the environment. These are:

## Example
As an example on how to use the Threat Defense environment, we provide a simple Q-learning implementation, [ql-agent-example.py](example/ql_agent_example.py), where a table is used to store the data.

## Template
[How to create new environments for Gym](https://github.com/openai/gym/tree/master/gym/envs#how-to-create-new-environments-for-gym)

## Inspiration
[banana-gym](https://github.com/MartinThoma/banana-gym)

[gym-soccer](https://github.com/openai/gym-soccer)

[gym-pomdp](https://github.com/d3sm0/gym_pomdp)

## Authors
* Johan Backman <johback@student.chalmers.se>
* Hampus Ramström <hampusr@student.chalmers.se>
