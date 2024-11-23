from flex.distributed.aio import Server
import numpy as np
import asyncio
import experiment


async def run_server():
    # We need to create the server and storing it in a variable.
    server = Server()

    # Start the server.
    await server.run(address="localhost", port=8080)

    # Now we are able to use the server to communicate with clients.
    # We are able to see the total number of clients connected to the server.
    clients_connected = len(server)

    # Also we can wait for getting some ammount of clients connected
    await server.wait_for_clients(10)

    # To retrieve their ids.
    clients_ids = server.get_ids()

    # Now we may select a given ammoount of clients to run the FL process.
    number_of_clients = 10
    selected_ids = clients_ids[:number_of_clients]

    # We can now tell the clients to start training.
    # By passing the selected ids, the server will only communicate with the selected clients.
    metrics = await server.train(node_ids=selected_ids)

    # Let's see the metrics.
    for node_id in metrics:
        print(f"Client with id {node_id} has sent the following metrics: {metrics[node_id]}")

    # Running evaluation is equivalent.
    metrics = await server.eval(node_ids=selected_ids)

    # Now, we can also get the weights from the clients.
    weights = await server.collect_weights(node_ids=selected_ids)

    # This weights can be aggregated
    aggregate_weights = experiment.aggregate_with_fedavg.__wrapped__(weights)

    # And then send the aggregated weights to the clients.
    await server.send_weights(aggregate_weights, node_ids=selected_ids)

    # Finally, we can stop the server.
    # This is very important to avoid memory leaks.
    await server.stop()


# Run with asyncio
asyncio.run(run_server())
