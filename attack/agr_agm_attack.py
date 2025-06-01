"""
Ataque busca explotar vulnerabilidades comunes en los métodos de agregación usados en el aprendizaje federado,
 extraído del artículo "Breaking State-of-the-Art Poisoning Defenses to Federated
Learning: An Optimization-Based Attack Framework", disponoble en : https://dl.acm.org/doi/pdf/10.1145/3627673.3679566,
Implementación basada en : https://codeload.github.com/Yuxin104/BreakSTOAPoisoningDefenses/zip/refs/heads/main
"""
import torch


def agr_agnostic_attack(num_corrupt, num_agents, agent_updates_dict, agent_id_id, avg_updates, i, dis):
    max_iter = 500
    count = 0
    max_dis_poi = 0
    threshold_diff = 1e-3
    r = torch.Tensor([0.1]).float()
    step = r
    r_succ = 0
    while torch.abs(r_succ - r) > threshold_diff and count <= max_iter:
        lamda = r * (avg_updates - agent_updates_dict[i])
        poison_vector = agent_updates_dict[i] + lamda

        for j in agent_id_id:
            if num_corrupt <= j < num_agents:
                vec = poison_vector - agent_updates_dict[j]
                distance = torch.norm(vec)
                if distance > max_dis_poi:
                    max_dis_poi = distance
        if max_dis_poi < dis:
            r_succ = r
            r = r - step / 2
        else:
            r = r + step / 2
        step = step / 2
        count += 1
    lamda = r_succ * (avg_updates - agent_updates_dict[i])
    agent_updates_dict[i] = agent_updates_dict[i] + lamda

    return agent_updates_dict, max_dis_poi
