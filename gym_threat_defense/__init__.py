from gym.envs.registration import register

register(
    id='threat-defense-v0',
    entry_point='gym_threat_defense.envs:ThreatDefenseEnv',
)
