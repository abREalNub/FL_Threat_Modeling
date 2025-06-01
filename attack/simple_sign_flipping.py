"""
La siguiente implementación se basa en el ataque mencionado en el artículo: RSA: Byzantine-Robust Stochastic Aggregation
Methods for Distributed Learning from Heterogeneous Datasets
Dsponible en :https://ojs.aaai.org/index.php/AAAI/article/view/3968
"""


def sign_flipping_attack(num_corrupt, agent_updates_dict, agent_id_id, alpha=-1):
    """
    Implementa el ataque Sign Flipping en el aprendizaje federado.

    - num_corrupt: Lista o conjunto de identificadores de clientes maliciosos.
    - agent_updates_dict: Diccionario con las actualizaciones de los clientes.
    - agent_id_id: Lista de identificadores de clientes participantes.
    - alpha: Factor de inversión del gradiente (por defecto -1).

    Retorna el diccionario de actualizaciones modificado con el ataque.
    """
    # Iterar sobre los clientes seleccionados
    for i in agent_id_id:
        if i in num_corrupt:  # Verificar si el cliente es malicioso
            g_i = agent_updates_dict.pop(i)  # Obtener gradiente original
            g_i_modified = alpha * g_i  # Aplicar inversión de signo
            agent_updates_dict.append(g_i_modified)  # Modificar actualización

    return agent_updates_dict
