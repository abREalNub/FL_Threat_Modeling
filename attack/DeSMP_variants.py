import torch
import numpy as np
import tensorly as tl
from flex.pool import fed_avg_f
from numpy.random import normal
from math import sqrt
from flex.pool.decorators import aggregate_weights


def desmp_attack(agent_updates_dict, sigma, sensitivity, gamma, malicious_clients, agent_id_id):
    """
    Implementación mejorada del ataque DeSMP.

    Args:
        agent_updates_dict (dict): Actualizaciones de los clientes.
        sigma (float): Parámetro de escala del ruido DP.
        sensitivity (float): Sensibilidad del clipping (mediana de las normas).
        gamma (float): Tolerancia del atacante (controla sigilo/impacto).
        malicious_clients (list): IDs de clientes maliciosos.
        agent_id_id (list): IDs de todos los clientes.

    Returns:
        dict: Actualizaciones con ruido adversarial inyectado.
    """
    # Validar entrada de clientes maliciosos
    if not isinstance(malicious_clients, list):
        malicious_clients = [malicious_clients]

    # Calcular media del ataque
    attack_mean = torch.tensor(np.sqrt(2 * gamma) * sigma)

    # Generar ruido adversarial para cada cliente malicioso
    for client_id in malicious_clients:
        if client_id not in agent_id_id:
            continue  # Ignorar IDs no válidos

        # Obtener forma de las actualizaciones del cliente
        update_shape = agent_updates_dict[client_id].shape

        # Generar ruido adversarial (misma distribución, media desplazada)
        adversarial_noise = torch.normal(
            mean=attack_mean,
            std=sigma * sensitivity,
            size=update_shape
        ).to(agent_updates_dict[client_id].device)

        # Reemplazar actualización con ruido malicioso
        agent_updates_dict[client_id] = adversarial_noise

    return agent_updates_dict


def desmp_attack_1(agent_updates_dict, sigma, sensitivity, num_corrupt, agent_id_id):
    """
    Implementación del ataque DeSMP en Federated Learning.

    - agent_updates_dict: Diccionario con las actualizaciones de los clientes.
    - sigma: Parámetro de ruido DP.
    - sensitivity: Sensibilidad del clipping en DP.
    - num_corrupt: Clientes maliciosos.
    - agent_id_id: Identificadores de los clientes participantes.

    Retorna agent_updates_dict con actualizaciones manipuladas.
    """
    # Seleccionar clientes maliciosos
    corrupt_clients = [i for i in agent_id_id if i in num_corrupt]

    # Generar ruido DP benigno
    benign_noise = torch.normal(mean=0, std=sigma * sensitivity, size=agent_updates_dict[agent_id_id[0]].shape)

    # Modificar el ruido con una distribución adversarial
    attack_mean = torch.tensor(np.sqrt(2) * sigma * 0.1)  # Factor ajustable para el ataque
    adversarial_noise = torch.normal(mean=attack_mean, std=sigma * sensitivity, size=benign_noise.shape)

    # Aplicar ataque en clientes corruptos
    for i in corrupt_clients:
        agent_updates_dict[i] += adversarial_noise  # Inyección de ruido malicioso

    return agent_updates_dict



