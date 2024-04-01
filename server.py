import flwr as fl

# Define an averaging strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=0.5)

# Start the server with the defined strategy
fl.server.start_server(
    server_address="127.0.0.1:5000",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy
)
