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


@aggregate_weights
def despm_attack_variant_2(list_of_weights, l2_clip=1, noise_multiplier=0.1, attack_strength=0.1):
    """
      Implementa el ataque DeSMP manipulando el ruido DP dentro de la agregación central.

      - list_of_weights: Lista de actualizaciones de los clientes.
      - l2_clip: Factor de recorte para el Norm Clipping.
      - noise_multiplier: Factor de escala para el ruido DP.
      - attack_strength: Intensidad del ataque (ajustar entre 0.1 y 1).

      Retorna la lista de pesos manipulada con ruido DP adversarial.
      """
    num_clients = len(list_of_weights)

    # Aplicación de Norm Clipping en cada cliente
    for i in range(num_clients):
        valid_weights = [w for w in list_of_weights[i] if w is not None]  # Filtrar valores incorrectos

        tmp_dist = sum([(tl.norm(w) ** 2).item() for w in valid_weights])  # Convertir a número
        l2_norm = sqrt(tmp_dist) + 1e-12
        clip_ratio = min(1, l2_clip / l2_norm)

        # Modificación de actualización maliciosa en clientes seleccionados
        if np.random.rand() < attack_strength:  # Se activa aleatoriamente el ataque
            clip_ratio *= (1 + attack_strength)  # Aumenta sutilmente el peso para influir en el modelo

        for j, w in enumerate(list_of_weights[i]):
            context = tl.context(w)
            clip_ratio = tl.tensor(clip_ratio, **context)
            list_of_weights[i][j] = w * clip_ratio

    # Agregación estándar con FedAvg
    agg_weights = fed_avg_f(list_of_weights)

    # Introducción de ruido DP malicioso
    noise_ratio = l2_clip * noise_multiplier / num_clients
    for i, w in enumerate(agg_weights):
        context = tl.context(agg_weights[i])
        adversarial_noise = tl.tensor(
            normal(loc=noise_ratio * attack_strength, scale=noise_ratio, size=tl.shape(w)), **context
        )
        agg_weights[i] = w + adversarial_noise  # Inyección de ruido malicioso

    return agg_weights
